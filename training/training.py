import json
import os
from collections import Counter

import datasets
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, \
    PreTrainedTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from data_binding.enumerates import Intents
from dataset_generation.utils import load_json_file
from training.dataset import DatasetCreationArguments, remove_user_profile_from_prompt
from training.training_arguments import ModelArguments, DataArguments, TrainArgs, LoRaArguments, \
    CustomIntervalStrategy
from user_simulator.traits_and_profiles.user_profile import UserTypes


def sft_training(model_arguments: ModelArguments,
                 data_arguments: DataArguments,
                 training_arguments: TrainArgs,
                 lora_arguments: LoRaArguments,
                 only_user: str = None):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_arguments.base_model)  # type: PreTrainedTokenizer

    # during training we padding side to the right
    tokenizer.padding_side = "right"

    # if the tokenizer does not have a pad token, set it to eos token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if model_arguments.truncation_side:
        tokenizer.truncation_side = model_arguments.truncation_side
    print("Finished loading tokenizer")

    # add extra tokens to tokenizer
    total_tokens_added = 0
    if training_arguments.use_special_token_profiles:
        # add special tokens to the tokenizer
        special_tokens = [f"<|{user_type.user_custom_name}|>" for user_type in UserTypes.get_all_user_types()]
        tokens_added = tokenizer.add_tokens(special_tokens)
        if tokens_added:
            print(f"Added special user profile tokens to tokenizer: {special_tokens}")
        total_tokens_added += tokens_added

    if training_arguments.use_special_token_intents:
        # add special tokens to the tokenizer
        special_tokens = [f"<|{intent}|>" for intent in Intents.pretty_names.values()]
        tokens_added = tokenizer.add_tokens(special_tokens)
        if tokens_added:
            print(f"Added special intent tokens to tokenizer: {special_tokens}")
        total_tokens_added += tokens_added

    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_arguments.base_model,  # "meta-llama/Llama-2-7b-hf"
        device_map="auto",
        trust_remote_code=True,
        # use_auth_token=True,
        torch_dtype=torch.bfloat16 if training_arguments.use_bf16 else None,
    )

    print("Finished loading model")
    base_model.config.use_cache = False

    # add LoRA layers on top of the quantized base model
    peft_config = None
    if lora_arguments.lora:
        peft_config = LoraConfig(
            r=lora_arguments.lora_rank,
            lora_alpha=lora_arguments.lora_alpha,
            lora_dropout=lora_arguments.lora_dropout,
            target_modules=lora_arguments.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    if total_tokens_added:
        print("Total tokens added to tokenizer:", total_tokens_added)
        if peft_config and "embed_tokens" not in lora_arguments.target_modules:
            raise ValueError(
                f"When adding special tokens to the tokenizer and using LoRA, the target_modules must include "
                f"'embed_tokens' but only provided {lora_arguments.target_modules}")

        # resize the model to fit the new tokenizer
        base_model.resize_token_embeddings(len(tokenizer))

    specific_args = get_only_training_args(training_arguments)

    training_args = TrainingArguments(**specific_args)

    # get dataset creation arguments from data_path to be able to load later
    train_params = os.path.join(data_arguments.data_path, "dataset_train_config.json")
    dataset_arguments = None
    if os.path.exists(train_params):
        params = load_json_file(train_params)
        # create DatasetCreationArguments avoid non-existing keys
        dataset_arguments = DatasetCreationArguments()
        # fill with the parameters and avoid non-existing keys
        for k in DatasetCreationArguments().__dict__.keys():
            if k in params:
                setattr(dataset_arguments, k, params[k])
        if only_user:
            dataset_arguments.only_user = only_user

    # run_name_additional
    final_run_name = run_name_additional(model_arguments, training_arguments, dataset_arguments)
    training_args.output_dir = os.path.join(training_args.output_dir, final_run_name)

    eval_dataset_text, train_dataset_text = load_sft_dataset(
        data_path=data_arguments.data_path,
        use_debug=training_arguments.use_debug,
        only_user=only_user,
    )

    if len(train_dataset_text) == 0:
        print(f"No data for the specified user {only_user} skipping training and finishing.")
        return

    # save each model (profile) to a different folder
    if only_user:
        training_args.output_dir = os.path.join(training_args.output_dir, only_user)

    print()
    print("Trainer Training Arguments")
    print(training_args)
    print()

    response_template = data_arguments.response_template
    # to resolve problems with interpreting \n as a new line
    response_template = response_template.replace("\\n", "\n")

    collator = get_datacollator(tokenizer, response_template)

    init_wandb(only_user, training_args, final_run_name)

    print("Starting training...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset_text,
        eval_dataset=eval_dataset_text,
        peft_config=peft_config,
        # packing=False,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,  # HF Trainer arguments
        # formatting_func=formatting_prompts_func,
        dataset_text_field="text",
        data_collator=collator,
    )

    if training_arguments.patience:
        trainer.add_callback(EarlyStoppingCallback(training_arguments.patience))

    if peft_config:
        trainer.model.print_trainable_parameters()

    train_results = trainer.train()

    print("Finished training")
    print("Best model checkpoint:", check_path := str(trainer.state.best_model_checkpoint))
    print("Training results:")
    print(train_results)

    # save to outputdir
    config_options = {}
    for arg_type in [model_arguments, data_arguments, training_args, lora_arguments, dataset_arguments]:
        if arg_type:
            config_options.update(arg_type.to_dict())

    # add best checkpoint path to config_options
    config_options["best_checkpoint_path"] = check_path

    with open(os.path.join(training_args.output_dir, "config_options.json"), "w") as f:
        json.dump(config_options, f, indent=2)

    # close wandb
    if "wandb" in training_args.report_to:
        import wandb

        wandb.finish()

    # clean models (to avoid memory leaks)
    del trainer
    del base_model
    torch.cuda.empty_cache()


def load_sft_dataset(data_path: str, use_debug: bool, only_user: str = None):
    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=os.path.join(data_path, "dataset_train.json"))["train"]
    eval_dataset = load_dataset("json", data_files=os.path.join(data_path, "dataset_valid.json"))["train"]
    print("Finished loading dataset")

    if only_user:
        train_dataset = train_dataset.filter(lambda x: x["user_profile"] == only_user)
        eval_dataset = eval_dataset.filter(lambda x: x["user_profile"] == only_user)

    if use_debug:
        train_dataset = train_dataset.select(range(100))
        eval_dataset = eval_dataset.select(range(10))

    train_dataset_text = {"text": [], "user_profile": []}
    for item in train_dataset:
        prompt = item["prompt"]
        if only_user:
            prompt = remove_user_profile_from_prompt(prompt)
        train_dataset_text["text"].append(prompt + item["completion"])
        train_dataset_text["user_profile"].append(item["user_profile"])
    train_dataset_text = datasets.Dataset.from_dict(train_dataset_text)

    eval_dataset_text = {"text": [], "user_profile": []}
    for item in eval_dataset:
        prompt = item["prompt"]
        if only_user:
            prompt = remove_user_profile_from_prompt(prompt)
        eval_dataset_text["text"].append(prompt + item["completion"])
        eval_dataset_text["user_profile"].append(item["user_profile"])
    eval_dataset_text = datasets.Dataset.from_dict(eval_dataset_text)

    print("Train dataset size:", len(train_dataset_text["text"]))
    if len(train_dataset_text["text"]) > 0:
        # first item
        print("Train dataset first item:", train_dataset_text["text"][0])
        print("Eval dataset size:", len(eval_dataset_text["text"]))
        # first item
        print("Eval dataset first item:", eval_dataset_text["text"][0])
        print()

    if len(train_dataset_text["text"]) > 0:
        # count examples for each user_profile
        print("Train dataset user_profile count:")
        print(Counter(train_dataset["user_profile"]))
        print("Eval dataset user_profile count:")
        print(Counter(eval_dataset["user_profile"]))

    return eval_dataset_text, train_dataset_text


def get_only_training_args(training_arguments: TrainArgs):
    specific_args = {}
    training_arguments_dict = training_arguments.to_dict()
    # only add the arguments that are in the TrainingArguments class
    for k in TrainingArguments(output_dir=None).to_dict().keys():
        if k in training_arguments.to_dict():
            specific_args[k] = training_arguments_dict[k]
    return specific_args


def get_datacollator(tokenizer, assistant_str='\nASSISTANT:'):
    # Llama tokenizer is weird so we need to find the ASSISTANT token by adding some \n before it
    response_template_with_context = assistant_str
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    return data_collator


def init_wandb(only_user: str,
               training_args: TrainingArguments,
               run_name: str):
    if "wandb" in training_args.report_to:
        # name project and run
        import wandb
        experiment_name = run_name
        if only_user:
            experiment_name += "_" + only_user

        wandb.init(project="user_simulator", name=experiment_name)


def run_name_additional(model_args: ModelArguments, training_args: TrainArgs, dataset_args: DatasetCreationArguments):
    run_name = model_args.base_model.replace("/", "-")
    if training_args.use_bf16:
        run_name += "_bf16"
    batch_size = training_args.per_device_train_batch_size
    run_name += f"_bs{batch_size}"

    accumulation_steps = training_args.gradient_accumulation_steps
    if accumulation_steps != 1:
        run_name += f"_accum{accumulation_steps}"

    learning_rate = training_args.learning_rate
    run_name += f"_lr{learning_rate}"

    if training_args.save_strategy == CustomIntervalStrategy.EPOCH:
        num_epochs = training_args.num_train_epochs
        run_name += f"_ep{num_epochs}"
    elif training_args.save_strategy == CustomIntervalStrategy.STEPS:
        num_steps = training_args.save_steps
        run_name += f"_steps{num_steps}"

    context_window = dataset_args.context_window
    run_name += f"_cw{context_window}"

    intent_prefix = dataset_args.add_intent_prefix
    if intent_prefix:
        run_name += f"_ip{intent_prefix}"

    stratified = False
    avoid_next_percentage = 0
    turn_level_user_profile_format = None
    if dataset_args:
        stratified = dataset_args.stratify_user_profile
        avoid_next_percentage = dataset_args.avoid_next_percentage
        turn_level_user_profile_format = dataset_args.turn_level_user_profile_format
    else:
        print("No dataset_args found when creating run_name_additional!!!")

    if stratified:
        run_name += "_stratified"

    if avoid_next_percentage:
        run_name += f"_avoid_next_{avoid_next_percentage}"

    if turn_level_user_profile_format:
        run_name += f"_format_{turn_level_user_profile_format}"

    if training_args.run_name:
        run_name += "_" + training_args.run_name

    return run_name
