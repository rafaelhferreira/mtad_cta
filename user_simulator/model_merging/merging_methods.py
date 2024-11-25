import copy
import random
from typing import List, Dict, Tuple, Union

import torch
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer, LogitsProcessorList

from dataset_generation.dialog import Dialog
from training.dataset import DatasetCreationArguments, dialogue_to_prompt, get_prompt_and_completion, \
    remove_user_profile_from_prompt
from training.utils import calc_scores_intent_tokens
from user_simulator.model_merging.utils import ModelMergingTypes
from user_simulator.traits_and_profiles.user_profile import UserProfile, ProfileLevels


class ScoresReturnLogitsProcessor(LogitsProcessor):
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # basically ignore the inputs and return the scores
        return self.scores


def multiple_loras_combination(current_dialogue: Dialog,
                               user_profiles: List[UserProfile],
                               simulator_model: PreTrainedModel,
                               user_tokenizer: PreTrainedTokenizer,
                               user_max_length: int,
                               extra_loras_list: List[str], extra_loras_weights: List[float],
                               user_generation_config: Dict,
                               data_arguments: DatasetCreationArguments,
                               merging_type: str,  # taken from ModelMergingTypes attributes
                               print_extra_info: bool) -> Tuple[str, Union[List[Dict], None], List[float]]:
    if merging_type in {ModelMergingTypes.mtad, ModelMergingTypes.mtad_level_aware}:
        user_text, intent_scores, _ = multiple_lora_mtad(
            current_dialogue=current_dialogue,
            user_profiles=user_profiles,
            simulator_model=simulator_model,
            user_tokenizer=user_tokenizer,
            user_max_length=user_max_length,
            extra_loras_list=extra_loras_list,
            extra_loras_weights=extra_loras_weights,
            user_generation_config=user_generation_config,
            data_arguments=data_arguments,
            separate_dialogue_utterance_profiles=merging_type == ModelMergingTypes.mtad_level_aware,
            print_extra_info=print_extra_info
        )
        return user_text, intent_scores, extra_loras_weights
    elif merging_type == ModelMergingTypes.sampling:
        return multiple_lora_sampling(
            current_dialogue=current_dialogue,
            user_profiles=user_profiles,
            simulator_model=simulator_model,
            user_tokenizer=user_tokenizer,
            user_max_length=user_max_length,
            extra_loras_list=extra_loras_list,
            extra_loras_weights=extra_loras_weights,
            user_generation_config=user_generation_config,
            data_arguments=data_arguments,
            print_extra_info=print_extra_info
        ) + (extra_loras_weights,)
    else:
        raise ValueError(f"Invalid merging type {merging_type}")


def multiple_lora_mtad(current_dialogue: Dialog,
                       user_profiles: List[UserProfile],
                       simulator_model: PreTrainedModel,
                       user_tokenizer: PreTrainedTokenizer,
                       user_max_length: int,
                       extra_loras_list: List[str], extra_loras_weights: List[float],
                       user_generation_config: Dict,
                       data_arguments: DatasetCreationArguments,
                       separate_dialogue_utterance_profiles: bool,
                       print_extra_info: bool) -> Tuple[str, Union[List[Dict], None], List[float]]:
    user_text = ""
    generation_counter = 0
    intent_scores = []
    generated_intent = False

    # remove the max_new_tokens since we are only generating one token
    user_generation_config_copy = copy.deepcopy(user_generation_config)
    user_generation_config_copy.pop("max_new_tokens", None)

    # get one token for each lora layer
    while True:
        token_scores = []

        generation_counter += 1

        active_profiles = user_profiles
        active_weights = extra_loras_weights
        active_loras = extra_loras_list
        # it only makes sense to separate if we generate the intent first
        if separate_dialogue_utterance_profiles and data_arguments.add_intent_prefix:
            active_profiles = []
            active_weights = []
            active_loras = []
            for u, w, l in zip(user_profiles, extra_loras_weights, extra_loras_list):
                if not generated_intent and u.profile_level == ProfileLevels.dialogue.value:
                    active_profiles.append(u)
                    active_weights.append(w)
                    active_loras.append(l)
                elif generated_intent and u.profile_level == ProfileLevels.utterance.value:
                    active_profiles.append(u)
                    active_weights.append(w)
                    active_loras.append(l)
                elif not u.profile_level:  # since we do not know the origin assume it is mixed and add to both
                    active_profiles.append(u)
                    active_weights.append(w)
                    active_loras.append(l)

            # for example this can happen if all are of the same level
            if not active_profiles:
                # activate all again
                active_profiles = copy.deepcopy(user_profiles)
                active_weights = copy.deepcopy(extra_loras_weights)
                active_loras = copy.deepcopy(extra_loras_list)

            if print_extra_info:
                print("Generated Intent:", generated_intent)
                print("All Profiles", [u.user_custom_name for u in user_profiles])
                print("Active Profiles:", [u.user_custom_name for u in active_profiles])
                print("Active Loras:", active_loras)
                print("Active Weights:", active_weights)
                print()

        for lora, u_prof in zip(active_loras, active_profiles):

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
                before_conversation_prefix=data_arguments.before_conversation_prefix,
                use_different_profile=u_prof,  # use the user profile for the current lora to make sure input is correct
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

            current_prompt += user_text

            # resolve problem with spaces
            current_prompt = current_prompt.replace("[Intent:<|", "[Intent: <|")

            if print_extra_info:
                print("User Profile:", u_prof.user_custom_name)
                print("Lora name:", lora)
                print(current_prompt + "$")

            inputs_user = user_tokenizer([current_prompt], return_tensors="pt",
                                         max_length=user_max_length, truncation=True)
            inputs_user = inputs_user.to(simulator_model.device)

            # we need to load the models every time or it will not work! - probably a bug in the library
            simulator_model.load_adapter(lora, lora.replace(".", "_"))
            # activate the adapter
            simulator_model.set_adapter(lora.replace(".", "_"))

            if print_extra_info:
                print("Active adapter:", simulator_model.active_adapter)

            outputs = simulator_model.generate(**inputs_user, max_new_tokens=1, return_dict_in_generate=True,
                                               output_scores=True, do_sample=False,
                                               pad_token_id=user_tokenizer.pad_token_id)
            token_scores.append(outputs["scores"][0].squeeze(0))

            # top-tokens for lora
            if print_extra_info:
                print("Scores:", calc_scores_intent_tokens(5, 1, outputs["scores"], user_tokenizer))
                print("\n" + "=" * 20 + "\n")

        # create a tensor with the scores
        token_scores = torch.stack(token_scores)
        # apply softmax
        token_scores = torch.softmax(token_scores, dim=-1)
        # apply the weights
        token_scores = token_scores * torch.tensor(active_weights).unsqueeze(1).to(token_scores.device)
        # calculate the mean
        token_scores = torch.mean(token_scores, dim=0)
        # convert to logits again
        token_scores = torch.log(token_scores)
        # add a dimension to the scores
        token_scores = token_scores.unsqueeze(0)

        if data_arguments.add_intent_prefix:
            combined_scores = calc_scores_intent_tokens(5, 1, (token_scores.squeeze(0),), user_tokenizer)
            intent_scores.append(combined_scores[0])
            if print_extra_info:
                print("Combined before generation", combined_scores)

        # create processor that returns the scores
        logits_processor_list = LogitsProcessorList([
            ScoresReturnLogitsProcessor(token_scores),
        ])

        # generate using the decoding strategy using these scores
        outputs = simulator_model.generate(
            input_ids=None,  # input_ids are not needed
            max_new_tokens=1,
            **user_generation_config_copy,
            logits_processor=logits_processor_list,  # use the processor to return the scores
            eos_token_id=user_tokenizer.eos_token_id,
            pad_token_id=user_tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        last_token_id = outputs["sequences"][0][-1:]
        # convert to token (apply special treatment to get spaces out of tokenizer)
        generated_token = user_tokenizer.convert_ids_to_tokens(last_token_id, skip_special_tokens=False)[0].replace("▁",
                                                                                                                    " ")
        user_text += generated_token

        if print_extra_info:
            print("Chosen token:", generated_token)
            print()

        # stop if it reaches max length, max tokens to generate or if it finds the end of the sentence token or the user final token
        if generation_counter >= user_generation_config.get("max_new_tokens", None):
            break

        if user_tokenizer.eos_token_id in outputs["sequences"][0]:
            break

        if data_arguments.user_final_token and user_text.endswith(data_arguments.user_final_token):
            break

        if user_text.endswith("]") or user_text.endswith("] "):
            generated_intent = True

    # remove special tokens from user_text
    for s_token in [user_tokenizer.eos_token, user_tokenizer.pad_token, user_tokenizer.sep_token]:
        if s_token:
            user_text = user_text.replace(s_token, "")

    intent_number = 1 if data_arguments.add_intent_start_to_completion else 5
    if intent_scores:
        intent_scores = intent_scores[:intent_number]
    else:
        intent_scores = None

    return user_text, intent_scores, extra_loras_weights


def multiple_lora_sampling(current_dialogue: Dialog,
                           user_profiles: List[UserProfile],
                           simulator_model: PreTrainedModel,
                           user_tokenizer: PreTrainedTokenizer,
                           user_max_length: int,
                           extra_loras_list: List[str], extra_loras_weights: List[float],
                           user_generation_config: Dict,
                           data_arguments: DatasetCreationArguments,
                           print_extra_info: bool) -> Tuple[str, Union[List[Dict], None]]:

    intent_scores = []

    # get a random indice and using weights to choose the lora
    lora_index = random.choices(range(len(extra_loras_list)), weights=extra_loras_weights, k=1)[0]
    lora = extra_loras_list[lora_index]
    u_prof = user_profiles[lora_index]

    # load the lora
    simulator_model.load_adapter(lora, lora.replace(".", "_"))
    # activate the adapter
    simulator_model.set_adapter(lora.replace(".", "_"))

    if print_extra_info:
        print("Sampled adapter:", simulator_model.active_adapter)

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
        before_conversation_prefix=data_arguments.before_conversation_prefix,
        use_different_profile=u_prof,  # use the user profile for the current lora to make sure input is correct
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
        print("User Profile:", u_prof.user_custom_name)
        print("Lora name:", lora)
        print(current_prompt + "$")

    inputs_user = user_tokenizer([current_prompt], return_tensors="pt",
                                 max_length=user_max_length, truncation=True)
    inputs_user = inputs_user.to(simulator_model.device)

    outputs = simulator_model.generate(**inputs_user, **user_generation_config,
                                       eos_token_id=user_tokenizer.eos_token_id,
                                       pad_token_id=user_tokenizer.pad_token_id,
                                       return_dict_in_generate=True,
                                       output_scores=True,
                                       )

    user_text = user_tokenizer.batch_decode(outputs["sequences"][:, inputs_user.input_ids.shape[1]:],
                                            skip_special_tokens=True)[0]

    if print_extra_info:
        print("Scores:", calc_scores_intent_tokens(5, 1, outputs["scores"], user_tokenizer))
        print("\n" + "=" * 20 + "\n")

    if data_arguments.add_intent_prefix:
        combined_scores = calc_scores_intent_tokens(5, 1, outputs["scores"], user_tokenizer)
        intent_scores.append(combined_scores[0])

    intent_number = 1 if data_arguments.add_intent_start_to_completion else 5
    if intent_scores:
        intent_scores = intent_scores[:intent_number]
    else:
        intent_scores = None

    return user_text, intent_scores
