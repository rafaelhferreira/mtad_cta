import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, \
    StoppingCriteriaList, StoppingCriteria

from data_binding.enumerates import Intents
from data_binding.task_result import TaskResult
from dataset_generation.calc_dialog_stats import calc_dialog_stats, print_and_plot_intent_distribution
from dataset_generation.dialog import Dialog
from dataset_generation.system_responses import ResponseType
from dataset_generation.utils import write_to_json_file, load_json_file
from run_simulation.plangpt_input_creation import build_raw_samples_plangpt
from run_simulation.utils import take_intent_from_text, take_system_response_from_model_output, \
    clean_special_tokens_plangpt, get_step_number_from_system_response, is_system_error, is_ending_conversation
from training.dataset import dialogue_to_prompt, DatasetCreationArguments, get_prompt_and_completion, \
    remove_user_profile_from_prompt
from training.utils import calc_scores_intent_tokens, create_results_dataframe, create_dialog_from_dict
from user_simulator.model_merging.merging_methods import multiple_loras_combination
from user_simulator.model_merging.utils import ModelMergingTypes
from user_simulator.traits_and_profiles.user_profile import UserProfile


def run_sim(simulator_path: str, system_path: str,
            user_profiles: List[UserProfile],
            data_arguments: DatasetCreationArguments,
            tasks: List[TaskResult],
            conversations_per_task: int = 3,
            prompt_version: str = "v5", context_size: int = 3, max_turns_dialogue: int = 20,
            output_path: str = "data/simulated_conversations", use_bf16: bool = False,
            system_tone: str = ResponseType.NEUTRAL.value,
            user_max_length: int = 1024, system_max_length: int = 1024,
            user_truncation_side: str = "left", system_truncation_side: str = "left",
            user_generation_config: Dict = None,
            system_generation_config: Dict = None,
            print_extra_info: bool = False,
            extra_loras_list: List[str] = None,
            extra_loras_weights: List[float] = None,
            merging_type: str = None,
            is_debug: bool = False
            ):
    # assert that the number of profiles - 1 is the same as the number of loras
    if extra_loras_list is not None or len(user_profiles) > 1:
        assert len(user_profiles) - 1 == len(
            extra_loras_list), (f"Number of profiles-1 ({len(user_profiles) - 1}) "
                                f"must be the same as the number of loras ({len(extra_loras_list)})")

    # assert that size of extra_loras_list and extra_loras_weights is the same
    if extra_loras_list is not None and extra_loras_weights is not None:
        # add the default adapter to position zero
        extra_loras_list.insert(0, simulator_path)
        assert len(extra_loras_list) == len(
            extra_loras_weights), (f"Size of extra_loras_list ({len(extra_loras_list)}) "
                                   f"and extra_loras_weights ({len(extra_loras_weights)}) must be the same")

    # assign equal weights to all if not provided
    if extra_loras_list and not extra_loras_weights:
        extra_loras_list.insert(0, simulator_path)
        # extra_loras_weights = [1 / len(extra_loras_list)] * len(extra_loras_list)  # equal weights (float number)
        extra_loras_weights = [1] * len(extra_loras_list)  # equal weights (all ones)

    if system_generation_config is None:
        system_generation_config = {}

    if user_generation_config is None:
        user_generation_config = {}

    print("Loading tokenizer...")
    user_tokenizer = AutoTokenizer.from_pretrained(simulator_path)  # type: PreTrainedTokenizer
    print("Finished loading tokenizer")
    user_tokenizer.truncation_side = user_truncation_side

    base_model_name_or_path = None
    if os.path.exists(adapter_path := os.path.join(simulator_path, "adapter_config.json")):
        base_model_name_or_path = load_json_file(adapter_path).get("base_model_name_or_path")

    # load the simulator model

    # time to load the simulator model
    start_time = time.time()

    print("Loading simulator model...")
    simulator_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path if base_model_name_or_path else simulator_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )

    # resize the model to fit the new tokenizer
    simulator_model.resize_token_embeddings(len(user_tokenizer))

    time_to_load_model = time.time() - start_time
    print("Finished loading simulator model in {:.2f} seconds".format(time_to_load_model))

    # time to load the adater
    start_time = time.time()

    # load the adapter
    simulator_model = PeftModel.from_pretrained(
        simulator_model,
        simulator_path,
    )  # type: PreTrainedModel

    # load the remaining adapters
    if extra_loras_list:
        for lora in extra_loras_list:
            if lora != "default":
                print("Loading adapter", lora)
                simulator_model.load_adapter(lora, adapter_name=lora.replace(".", "_"))

        if merging_type and not ModelMergingTypes.is_decoding_time(merging_type):
            print("Loading weighted adapters (this might take a while...)")
            print("Extra Loras List:", extra_loras_list)
            print("Extra Loras Weights:", extra_loras_weights)
            simulator_model.add_weighted_adapter(
                adapters=[lora.replace(".", "_") for lora in extra_loras_list],
                weights=extra_loras_weights,
                adapter_name="combined_adapters",
                combination_type=merging_type,
                density=0.4,  # could be further optimized
                # majority_sign_method="total",
            )
            simulator_model.set_adapter("combined_adapters")

            print(simulator_model.active_adapter)

            print("Finished loading combined_adapters")

    time_to_load_adapter = time.time() - start_time
    print("Finished loading adapters in {:.2f} seconds".format(time_to_load_adapter))

    # total time to load the model and the adapter
    total_time = time_to_load_model + time_to_load_adapter
    print("Total time to load the model and the adapter: {:.2f} seconds".format(total_time))

    stopping_criteria = set_simulator_end_token_criteria(data_arguments, user_tokenizer,
                                                         "mistral" in simulator_path.lower())
    # stopping_criteria = None

    # put in eval mode
    simulator_model.eval()
    print("Finished loading simulator model")

    # load the system model
    print("Loading system model...")
    system_model = AutoModelForCausalLM.from_pretrained(
        system_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else None
    )  # type: PreTrainedModel
    # put in eval mode
    system_model.eval()
    print("Finished loading system model")

    print("Loading tokenizer...")
    system_tokenizer = AutoTokenizer.from_pretrained(system_path)  # type: PreTrainedTokenizer
    print("Finished loading tokenizer")
    system_tokenizer.truncation_side = system_truncation_side

    # generates a full dialogue and then saves it to a file
    time_to_generate_tasks = []
    time_to_generate_responses = []
    time_per_token_list = []
    generated_tokens_list = []
    with (torch.no_grad()):
        for task in tqdm(tasks, desc="Tasks"):
            for i in range(conversations_per_task):
                end_conversation = False
                generated_dialogue_id = f"{'_'.join([u.user_custom_name for u in user_profiles])}_{i}_{task.get_unique_id()}"
                if print_extra_info:
                    print(f"Conversation ID: {generated_dialogue_id}")

                current_dialogue = Dialog(task=task,
                                          dialog_id=generated_dialogue_id,
                                          system_tone=system_tone, user_profile=user_profiles[0])  # we put the first user profile (because we need one)
                dialogue_intent_scores = []

                current_task_time = time.time()

                while not end_conversation and len(current_dialogue.turns) < max_turns_dialogue:

                    current_dialogue.add_turn(intent="", user_utterance="", system_utterance="", current_step=0)

                    if len(current_dialogue.turns) <= 1:
                        current_dialogue.turns[-1].current_step = 0
                    else:  # get the current step from the last turn
                        current_dialogue.turns[-1].current_step = current_dialogue.turns[-1].current_step

                    prompts, _, _ = dialogue_to_prompt(
                        dialogue=current_dialogue,
                        context_window=data_arguments.context_window,
                        user_token=data_arguments.user_token,
                        system_token=data_arguments.system_token,
                        separator_token=data_arguments.separator_token,
                        add_intent_prefix=data_arguments.add_intent_prefix,
                        intent_prefix_style=data_arguments.intent_prefix_style,
                        use_special_intent_tokens=data_arguments.use_special_intent_tokens,
                        user_final_token=data_arguments.user_final_token,
                        system_final_token=data_arguments.system_final_token,
                        use_before_user_profile=data_arguments.use_before_user_profile,
                        ignore_default_trait_values=data_arguments.ignore_default_trait_values,
                        use_turn_level_user_profile=data_arguments.use_turn_level_user_profile,
                        turn_level_user_profile_only_last_turn=data_arguments.turn_level_user_profile_only_last_turn,
                        turn_level_user_profile_format=data_arguments.turn_level_user_profile_format,
                        before_conversation_prefix=data_arguments.before_conversation_prefix
                    )

                    # the prompt that we want is the last one
                    current_prompt = prompts[-1]

                    current_prompt, _ = get_prompt_and_completion(
                        prompt=current_prompt,
                        user_token=data_arguments.user_token,
                        system_token=data_arguments.system_token,
                        intent_prefix_in_completion=data_arguments.add_intent_start_to_completion,
                        intent_prefix_style=data_arguments.intent_prefix_style,
                        user_final_token=data_arguments.user_final_token,
                        system_final_token=data_arguments.system_final_token,
                    )

                    if data_arguments.only_user:
                        current_prompt = remove_user_profile_from_prompt(current_prompt)

                    if print_extra_info:
                        print(f"Prompt (ignore $): {current_prompt}$")
                        print("\n" + "=" * 20 + "\n")

                    if not extra_loras_list or not ModelMergingTypes.is_decoding_time(merging_type):
                        inputs_user = user_tokenizer([current_prompt], return_tensors="pt",
                                                     max_length=user_max_length, truncation=True)
                        inputs_user = inputs_user.to(simulator_model.device)

                        start_time_generate = time.time()

                        outputs = simulator_model.generate(**inputs_user, **user_generation_config,
                                                           return_dict_in_generate=True, output_scores=True,
                                                           stopping_criteria=stopping_criteria)

                        time_to_generate = time.time() - start_time_generate
                        time_per_token = time_to_generate / len(outputs["sequences"][0])

                        # number of tokens generated
                        if print_extra_info:
                            print("Time to generate:", time_to_generate)
                            print("Number of tokens generated:", len(outputs["sequences"][0]))
                            print("Time per token:", time_per_token)

                        # append to times
                        time_to_generate_responses.append(time_to_generate)
                        time_per_token_list.append(time_per_token)
                        generated_tokens_list.append(len(outputs["sequences"][0]))

                        # get probs for intents
                        intent_scores = None
                        if data_arguments.add_intent_prefix:
                            intent_scores = calc_scores_intent_tokens(
                                k=5,
                                g_steps=1 if data_arguments.add_intent_start_to_completion else 5,
                                scores=outputs["scores"],
                                user_tokenizer=user_tokenizer
                            )

                        dialogue_intent_scores.append(intent_scores)

                        # decode the output
                        generated_text = user_tokenizer.batch_decode(outputs["sequences"][:, inputs_user.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    else:
                        generated_text, intent_scores, extra_loras_weights = multiple_loras_combination(
                            current_dialogue=current_dialogue,
                            user_profiles=user_profiles,
                            simulator_model=simulator_model,
                            user_tokenizer=user_tokenizer,
                            user_max_length=user_max_length,
                            extra_loras_list=extra_loras_list,
                            extra_loras_weights=extra_loras_weights,
                            user_generation_config=user_generation_config,
                            data_arguments=data_arguments,
                            merging_type=merging_type,
                            print_extra_info=print_extra_info,
                        )

                        dialogue_intent_scores.append(intent_scores)

                    if print_extra_info and intent_scores:
                        print(f"Intent Scores: {intent_scores}")
                        print("\n" + "=" * 20 + "\n")

                    if print_extra_info:
                        print(f"Simulator: {generated_text}")
                        print("\n" + "=" * 20 + "\n")

                    user_text, intent = take_intent_from_text(
                        user_text=generated_text,
                        intent_prefix=data_arguments.intent_prefix_style
                    )
                    user_text = take_system_response_from_model_output(user_text=user_text,
                                                                       system_separator=data_arguments.system_token,
                                                                       user_final_token=data_arguments.user_final_token)

                    user_text = user_text.strip()
                    intent = intent.strip()

                    if print_extra_info:
                        print(f"User Text: {user_text}")
                        print(f"Intent Extracted: {intent}")
                        print("\n" + "=" * 20 + "\n")

                    # update the dialogue
                    current_dialogue.turns[-1].intent = intent
                    current_dialogue.turns[-1].user_utterance = user_text
                    current_dialogue.turns[-1].generated_text = generated_text
                    current_dialogue.turns[-1].system_utterance = ""

                    # get system response
                    system_input, _, _, _, _ = build_raw_samples_plangpt(
                        dialog=current_dialogue,
                        tokenizer=system_tokenizer,
                        prompt_version=prompt_version,
                        context_size=context_size
                    )

                    system_input = system_input[-1]

                    if print_extra_info:
                        print(f"System Input: {system_input}")
                        print("\n" + "=" * 20 + "\n")

                    inputs_system = system_tokenizer([system_input], return_tensors="pt", max_length=system_max_length,
                                                     truncation=True)
                    inputs_system = inputs_system.to(system_model.device)

                    outputs = system_model.generate(**inputs_system, **system_generation_config)

                    # decode the output
                    system_text = system_tokenizer.batch_decode(outputs[:, inputs_system.input_ids.shape[1]:],
                                                                skip_special_tokens=True)[0]
                    system_text = clean_special_tokens_plangpt(system_text).strip()

                    if print_extra_info:
                        print(f"System (cleaned tokens): {system_text}")
                        print("\n" + "=" * 20 + "\n")

                    # add system text to dialogue
                    current_dialogue.turns[-1].system_utterance = system_text

                    # replace the current step by analyzing the system response
                    extracted_step = get_step_number_from_system_response(system_text)
                    if extracted_step != -1:
                        current_dialogue.turns[-1].current_step = extracted_step

                    # try to figure out if the system gave an error
                    if is_system_error(system_text):
                        current_dialogue.turns[-1].forced_system_error = True

                    # check if conversation is over
                    end_conversation = is_ending_conversation(intent, user_text, system_text)

                    # save conversation to file
                    if end_conversation or len(current_dialogue.turns) == max_turns_dialogue:

                        # create output folder if it does not exist
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        out_path = os.path.join(output_path, f"{generated_dialogue_id}.json")

                        dialog_dict = current_dialogue.dialog_dict()
                        # add information about models used for future reference
                        dialog_dict["simulator_model"] = simulator_path
                        if extra_loras_list:
                            dialog_dict["extra_loras_list"] = extra_loras_list
                            dialog_dict["extra_loras_weights"] = extra_loras_weights
                            dialog_dict["merging_type"] = merging_type
                            dialog_dict["combined_profiles"] = [u.user_custom_name for u in user_profiles]

                        dialog_dict["system_model"] = system_path

                        # add intent scores to the dialogue
                        dialog_dict["intent_scores"] = dialogue_intent_scores

                        time_to_generate_tasks.append(time.time() - current_task_time)
                        if print_extra_info:
                            print("Task generated in {:.2f} seconds".format(time_to_generate_tasks[-1]))

                        if not is_debug:
                            write_to_json_file(out_path, dialog_dict)
                            if print_extra_info:
                                print(f"Conversation saved to {out_path}\n====================\n")

    # if print_extra_info:
    print("\n==========================\n")
    print("Inference Times:")
    print("Total time to load model {:.4f} seconds".format(time_to_load_model))
    print("Total time to load adapter {:.4f} seconds".format(time_to_load_adapter))
    print("Total time to load model and adapter {:.4f} seconds".format(total_time))
    print("Average time to generate tasks {:.4f} seconds".format(np.mean(time_to_generate_tasks)))
    print("Total time to generate all tasks {:.4f} seconds".format(np.sum(time_to_generate_tasks)))
    print("Average time to generate responses {:.4f} seconds".format(np.mean(time_to_generate_responses)))
    print("Total time to generate all responses {:.4f} seconds".format(np.sum(time_to_generate_responses)))
    print("Average time per token {:.4f} seconds".format(np.mean(time_per_token_list)))
    print("Total time per token {:.4f} seconds".format(np.sum(time_per_token_list)))
    # calculate tokens per second
    average_time_per_token = np.mean(time_per_token_list)
    tokens_per_second = 1 / average_time_per_token
    print("Average tokens per second {:.4f}".format(tokens_per_second))
    print("Average generated tokens {:.4f}".format(np.mean(generated_tokens_list)))
    print("\n==========================\n")

    times_dict = {
        "time_to_load_model": time_to_load_model,
        "time_to_load_adapter": time_to_load_adapter,
        "total_time": total_time,
        "average_time_to_generate_tasks": np.mean(time_to_generate_tasks),
        "total_time_to_generate_tasks": np.sum(time_to_generate_tasks),
        "average_time_to_generate_responses": np.mean(time_to_generate_responses),
        "total_time_to_generate_responses": np.sum(time_to_generate_responses),
        "average_time_per_token": np.mean(time_per_token_list),
        "total_time_per_token": np.sum(time_per_token_list),
        "average_tokens_per_second": tokens_per_second,
        "average_generated_tokens": np.mean(generated_tokens_list),
    }

    # del models and clean gpu cache to avoid problems with successive runs
    del simulator_model
    del user_tokenizer
    del system_model
    del system_tokenizer
    torch.cuda.empty_cache()

    return times_dict


def set_simulator_end_token_criteria(data_arguments: DatasetCreationArguments, user_tokenizer: PreTrainedTokenizer,
                                     ignore_last: bool):
    # add a stop criteria for faster generation (e.g. in mistral the eos_token is not the same as the final token)
    # ignore last is used for a special case with mistral because the way the tokenizer works
    stopping_criteria = None
    if data_arguments.user_final_token and data_arguments.user_final_token.strip():
        stop_list = [data_arguments.user_final_token]
        stop_token_ids = [user_tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in
                          stop_list]

        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids, ignore_last=ignore_last)])

    elif "mistral" in user_tokenizer.name_or_path.lower() or "vicuna" in user_tokenizer.name_or_path.lower():
        # manually add the mistral [/INST] token
        stop_list = ["[/INST]"]
        stop_token_ids = [user_tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in
                          stop_list]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids, ignore_last=ignore_last)])

    return stopping_criteria


def aggregate_and_calc_stats_by_user_type(folder_path: str, output_folder: str, config_path: str):
    # example config_path
    # config_path = f"/data/3.0_{UserTypes.RegularUser.user_custom_name}/all/simulated_conversations_train_manual_distribution_config.json"

    config_dict = load_json_file(config_path)

    first_step_probs = config_dict["first_step_probs"]
    transitions_probs_dict = config_dict["transitions_probs_dict"]

    # list all files in the folder
    files = os.listdir(folder_path)
    # keep only the json files
    files = [f for f in files if f.endswith(".json")]

    dialogues_by_profile = defaultdict(list)
    # load each file and create a dialog
    for f in files:
        dialog_dict = load_json_file(os.path.join(folder_path, f))
        dialog = create_dialog_from_dict(dialog_dict)
        dialogues_by_profile[dialog.user_profile.user_custom_name].append(dialog)

        # change all intents to original dataset name for calculating stats
        for turn in dialog.turns:
            converted_intent_name = Intents.from_pretty_name_to_original(turn.intent)
            if not converted_intent_name:
                print(f"Intent conversion failed for {turn.intent} in {f}")
            else:
                turn.intent = converted_intent_name

    stats_by_profile = {}
    # calculate stats for each profile
    for profile, dialogues in dialogues_by_profile.items():
        # this is an approximation since to measure tolerance we needed to know if the system gave an error
        stats_by_profile[profile] = calc_dialog_stats(
            dialogues=dialogues,
            first_step_distribution=first_step_probs,
            intents_distribution=transitions_probs_dict,
            use_tqdm=True,
        )

        print(f"Stats for {profile}")
        print_and_plot_intent_distribution(
            all_dialogues=dialogues, turn_level=False,
            out_path=os.path.join(output_folder, f"intent_distribution_{profile}.pdf"), log_scale=False
        )

    df = create_results_dataframe(
        stats_by_profile,
        additional_columns=["number_dialogues", "dialogues_intent_distribution", "next_percentage",
                            "number_turns_list", "cooperativeness_list", "exploration_list",
                            "number_words_list", "sentiment_list", "fluency_list", "repetition_list",
                            "conditioned_tolerance_list"]
    )

    # write to csv file
    if output_folder:
        df.to_csv(os.path.join(output_folder, "dialog_stats_by_profile.csv"))
        print("File saved at ", os.path.join(output_folder, "dialog_stats_by_profile.csv"))

    print("Stats by profile")
    print(df)

    # final dict is relevant for user type and all_stats is everything
    return stats_by_profile


class StopOnTokens(StoppingCriteria):

    def __init__(self, stop_token_ids: List[torch.LongTensor], ignore_last: bool):
        super().__init__()
        self.stop_token_ids = stop_token_ids
        self.ignore_last = ignore_last

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        # put stop_token_ids in the same device as input_ids
        for stop_ids in self.stop_token_ids:
            # check if stop_ids is in the same device as input_ids
            if input_ids.device != stop_ids.device:
                stop_ids = stop_ids.to(input_ids.device)

            # sum +1 - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/15
            #  special case for mistral model
            plus_one = 0
            if self.ignore_last:
                plus_one = 1

            # compare the last tokens of input_ids with stop_ids
            if torch.eq(input_ids[0][-len(stop_ids[0]) + plus_one:], stop_ids[0][plus_one:]).all():
                return True
        return False
