import random
import re

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import List, Union, Tuple

from data_binding.enumerates import Intents
from dataset_generation.dialog import Dialog
from training.utils import create_dialogs_from_json_file
from user_simulator.traits_and_profiles.user_profile import UserProfile, UserTypes


class IntentPrefix:
    user_intent = "[User Intent: "
    intent = "[Intent: "


class TurnProfileFormats(Enum):
    SingleWord = "single_word"
    SingleWordNoExtraTokens = "single_word_no_extra_tokens"
    Format1 = "1"
    Format2 = "2"
    NaturalLanguage = "natural_language"


@dataclass
class DatasetCreationArguments:

    context_window: int = field(
        default=3, metadata={"help": "Number of turns of context."},
    )

    user_token: str = field(
        default="USER: ", metadata={"help": "User Token."},
    )

    system_token: str = field(
        default="SYSTEM: ", metadata={"help": "System Token."},
    )

    separator_token: str = field(
        default="\n", metadata={"help": "Separator between User and System."},
    )

    add_intent_prefix: bool = field(
        default=True, metadata={"help": "Adds the intent prefix before each user intent"}
    )

    intent_prefix_style: str = field(
        default=IntentPrefix.intent, metadata={"help": "Style of the intent prefix"}
    )

    add_intent_start_to_completion: bool = field(
        default=False, metadata={"help": "Adds the intent prefix [Intent: after the User prompt"}
    )

    use_special_intent_tokens: bool = field(
        default=False, metadata={"help": "If true uses special tokens for intents "
                                         "(adds <| and |> to signify a special token)."},
    )

    system_final_token: str = field(
        default="</s>", metadata={"help": "Add a token at the end of the system turn."},
    )

    user_final_token: str = field(
        default="</s>", metadata={"help": "Add a token at the end of the last user turn "
                                          "(to signify end of generation)."},
    )

    use_before_user_profile: bool = field(
        default=False, metadata={"help": "If true puts user profile information before the conversation"},
    )

    ignore_default_trait_values: bool = field(
        default=True, metadata={"help": "If true does not put default values in the user profile"},
    )

    use_turn_level_user_profile: bool = field(
        default=True, metadata={"help": "If true it add the user profile at a turn level"},
    )

    turn_level_user_profile_only_last_turn: bool = field(
        default=True, metadata={"help": "If true add the user profile at a turn level only to the last turn"},
    )

    turn_level_user_profile_format: str = field(
        default=TurnProfileFormats.Format1.value, metadata={"help": "Format of the user profile at a turn level"},
    )

    before_conversation_prefix: str = field(
        default="A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                "The user interacts with the system following its specific user profile description.\n\n",
        metadata={"help": "Start of the prompt."},
    )

    exclude_ids_sft: List[str] = field(
        default=None, metadata={"help": "List of dialogue ids to exclude from the dataset."},
    )

    input_folder_paths: List[str] = field(
        default=None,
        metadata={"help": "Path to list of folders containing datasets."},
    )

    output_folder_path: str = field(
        default=None,
        metadata={"help": "Path to folder to output the datasets containing datasets."},
    )

    num_train_samples_per_file: int = field(
        default=None,
        metadata={"help": "Number of samples per file."},
    )

    num_eval_samples_per_file: int = field(
        default=None,
        metadata={"help": "Number of samples per file."},
    )

    avoid_next_percentage: float = field(
        default=0.0, metadata={"help": "Percentage of nexts to avoid."},
    )

    stratify_user_profile: bool = field(
        default=False, metadata={"help": "If true it stratifies the dataset by user profile. "
                                         "i.e. same number of samples per user profile"},
    )

    only_user: str = field(
        default=None, metadata={"help": "States that the model is only trained on this user profile."},
    )

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def user_profile_to_prompt_1(user_profile: UserProfile, ignore_default_values: bool = True):
    # if ignore_default_values is True, then only the traits with a value different from 1 are included in the prompt

    # e.g.
    # tolerance: high
    # cooperativeness: low

    scale = ["low", "medium", "high"]
    prompt = ""
    for trait, scale_value in user_profile.get_trait_scale():
        if ignore_default_values and scale_value == 1:
            continue
        else:
            prompt += f"{trait.trait_name}: {scale[scale_value]}\n"

    return prompt


def user_profile_to_turn_prompt_1(user_profile: UserProfile, ignore_default_values: bool = True):

    # e.g. [User Profile: tolerance = high and cooperativeness = low and ...]
    scale = ["low", "medium", "high"]
    prompt = "[User Profile: "
    for trait, scale_value in user_profile.get_trait_scale():
        if ignore_default_values and scale_value == 1:
            continue
        else:
            prompt += f"{trait.trait_name} = {scale[scale_value]} and "

    if prompt.endswith(" and "):
        prompt = prompt[:-5] + "]"  # remove the last "and" and add the closing bracket
    else:  # no traits in the user profile
        prompt += user_profile.user_custom_name + "]"
    return prompt


def user_profile_to_turn_prompt_2(user_profile: UserProfile, ignore_default_values: bool = True):

    # e.g. [User Profile: tolerance = not tolerant and cooperativeness = fairly cooperative and ...]
    scale = ["not", "fairly", "very"]
    prompt = "[User Profile: "
    for trait, scale_value in user_profile.get_trait_scale():
        if ignore_default_values and scale_value == 1:
            continue
        else:
            prompt += f"{trait.trait_name} = {scale[scale_value]} {trait.user_trait_name} and "
    if prompt.endswith(" and "):
        prompt = prompt[:-5] + "]"  # remove the last "and" and add the closing bracket
    else:  # no traits in the user profile
        prompt += user_profile.user_custom_name + "]"
    return prompt


def user_profile_to_turn_prompt_single_word(user_profile: UserProfile, ignore_default_values: bool = True):
    # e.g. [User Profile: <|Tolerant|>
    # uses the custom name given to the user profile (so no combination of traits)
    # add to the tokenizer a special token for each user profile
    prompt = "[User Profile: "
    prompt += f"<|{user_profile.user_custom_name}|>]"
    return prompt


def user_profile_to_turn_prompt_single_word_no_extra_tokens(user_profile: UserProfile, ignore_default_values: bool = True):
    # e.g. [User Profile: Tolerant
    # uses the custom name given to the user profile (so no combination of traits)
    prompt = "[User Profile: "
    prompt += f"{user_profile.user_custom_name}]"
    return prompt


def user_profile_to_turn_prompt_natural_language(user_profile: UserProfile, ignore_default_values: bool = True):
    # e.g. [User Profile: very tolerant]
    scale = ["not", "fairly", "very"]
    prompt = "[User Profile: "
    for trait, scale_value in user_profile.get_trait_scale():
        if ignore_default_values and scale_value == 1:
            continue
        else:
            prompt += f"{scale[scale_value]} {trait.user_trait_name} and "
    if prompt.endswith(" and "):
        prompt = prompt[:-5] + "]"  # remove the last "and" and add the closing bracket
    else:  # no traits in the user profile (e.g. regular user)
        prompt += user_profile.user_custom_name.lower() + "]"
    return prompt


TURN_PROFILE_FORMAT_TO_FUNCTION = {
    TurnProfileFormats.SingleWord.value: user_profile_to_turn_prompt_single_word,
    TurnProfileFormats.SingleWordNoExtraTokens.value: user_profile_to_turn_prompt_single_word_no_extra_tokens,
    TurnProfileFormats.Format1.value: user_profile_to_turn_prompt_1,
    TurnProfileFormats.Format2.value: user_profile_to_turn_prompt_2,
    TurnProfileFormats.NaturalLanguage.value: user_profile_to_turn_prompt_natural_language,
    None: user_profile_to_turn_prompt_1
}


def remove_user_profile_from_prompt(prompt: str):
    # intentionally replacing empty profile because of response template format
    return re.sub(r'\[User Profile:.*?\]\s', '[User Profile] ', prompt)


def dialogue_to_prompt(dialogue: Dialog, context_window: int, user_token: str, system_token: str,
                       separator_token: str, add_intent_prefix: bool, intent_prefix_style: str,
                       use_special_intent_tokens: bool,
                       user_final_token: str, system_final_token: str,
                       use_before_user_profile: bool, ignore_default_trait_values: bool,
                       use_turn_level_user_profile: bool, turn_level_user_profile_only_last_turn: bool,
                       turn_level_user_profile_format: str,
                       before_conversation_prefix: str,
                       use_different_profile: UserProfile = None) \
        -> Tuple[List[str], List[str], List[Union[None, UserProfile]]]:

    prompts = []
    intents = []
    profiles = []

    if use_different_profile:
        # use a different user profile for the conversation
        user_profile = use_different_profile
    else:
        # use the user profile from the dialogue
        user_profile = dialogue.user_profile  # type: UserProfile

    # create user profile
    user_profile_prompt = user_profile_to_prompt_1(
        user_profile=user_profile,
        ignore_default_values=ignore_default_trait_values
    )
    if user_profile_prompt:
        before_conversation_user_profile = "User Profile:\n" + user_profile_prompt
    else:  # nothing in user profile is specific e.g. regular user
        before_conversation_user_profile = "User Profile:\n" + UserTypes.RegularUser.user_custom_name

    for turn in dialogue.turns:

        current_turn_representation = user_token

        if use_turn_level_user_profile:
            user_turn_representation = TURN_PROFILE_FORMAT_TO_FUNCTION.get(turn_level_user_profile_format, user_profile_to_turn_prompt_1)(user_profile=user_profile, ignore_default_values=ignore_default_trait_values)
            current_turn_representation = user_turn_representation + " " + current_turn_representation

        if add_intent_prefix:
            converted_intent = Intents.convert_to_pretty_name_model(turn.intent, add_intent_prefix=False)
            if use_special_intent_tokens:
                converted_intent = f"<|{converted_intent}|>"
            intent_representation = intent_prefix_style + converted_intent + "]"
            current_turn_representation += intent_representation + " "

        turn_representation = current_turn_representation + turn.user_utterance + user_final_token + separator_token + system_token + turn.system_utterance + system_final_token

        prompts.append(turn_representation)
        intents.append(turn.intent)
        profiles.append(user_profile)

    # create a list of prompts from 0 to n in size considering a context window
    p_list = []
    for i in range(len(prompts)):
        s_index = max(0, i - context_window)
        e_index = i + 1

        considered_turns = prompts[s_index:e_index]
        if turn_level_user_profile_only_last_turn:
            # remove the user profile from all the turns except the last one
            for j, t in enumerate(considered_turns):
                if j != len(considered_turns) - 1:
                    considered_turns[j] = re.sub(r'\[User Profile:.*?\]\s', '', t)

        conversation = separator_token.join(considered_turns)

        if use_before_user_profile:
            conversation = before_conversation_user_profile + separator_token + conversation

        if before_conversation_prefix:
            conversation = before_conversation_prefix + conversation

        p_list.append(conversation)

    return p_list, intents, profiles


def get_prompt_and_completion(prompt: str, user_token: str, system_token: str, intent_prefix_in_completion: bool,
                              intent_prefix_style: str, user_final_token: str, system_final_token: str):
    # remove last user and system turn from the prompt using the user and system tokens
    split_prompt = prompt.split(user_token)

    separator_token = system_token if system_token else system_final_token  # this is to work with mistral model

    last_user_turn = split_prompt[-1].split(separator_token)[0]
    rest_prompt = user_token.join(split_prompt[:-1])

    rest_prompt = rest_prompt + user_token
    rest_prompt_new_line = bool(rest_prompt[-1] == "\n")  # if ends with \n e.g. gemma we keep it has is

    if not rest_prompt_new_line:
        rest_prompt = rest_prompt.rstrip()

    if intent_prefix_in_completion:
        if not rest_prompt_new_line:
            rest_prompt += " " + intent_prefix_style.strip()
            last_user_turn = last_user_turn.replace(intent_prefix_style.strip() + " ", "")
        else:
            rest_prompt += intent_prefix_style.strip() + " "
            last_user_turn = last_user_turn.replace(intent_prefix_style.strip() + " ", "")

    # final check if completion is not there still
    if user_final_token and not last_user_turn.endswith(user_final_token):
        last_user_turn = user_final_token.join(last_user_turn.split(user_final_token)[:-1]) + user_final_token

    # this strips and adding spaces might mess up some models so check this if necessary
    if not rest_prompt_new_line:
        return rest_prompt, " " + last_user_turn.strip()
    else:
        return rest_prompt, last_user_turn.strip()


def get_sft_dataset(file_path: str, is_eval_split: bool, script_arguments: DatasetCreationArguments):
    # based on https://huggingface.co/docs/trl/main/en/sft_trainer

    if is_eval_split:
        num_samples = script_arguments.num_eval_samples_per_file
    else:
        num_samples = script_arguments.num_train_samples_per_file

    dialogues = create_dialogs_from_json_file(file_path)[:num_samples]

    sft_dataset_dict = []

    for dialog in dialogues:

        prompts, intents, profiles = dialogue_to_prompt(
            dialogue=dialog,
            context_window=script_arguments.context_window,
            user_token=script_arguments.user_token,
            system_token=script_arguments.system_token,
            separator_token=script_arguments.separator_token,
            add_intent_prefix=script_arguments.add_intent_prefix,
            intent_prefix_style=script_arguments.intent_prefix_style,
            use_special_intent_tokens=script_arguments.use_special_intent_tokens,
            user_final_token=script_arguments.user_final_token,
            system_final_token=script_arguments.system_final_token,
            use_before_user_profile=script_arguments.use_before_user_profile,
            ignore_default_trait_values=script_arguments.ignore_default_trait_values,
            use_turn_level_user_profile=script_arguments.use_turn_level_user_profile,
            turn_level_user_profile_only_last_turn=script_arguments.turn_level_user_profile_only_last_turn,
            turn_level_user_profile_format=script_arguments.turn_level_user_profile_format,
            before_conversation_prefix=script_arguments.before_conversation_prefix
        )
        # for each prompt create a completion
        for i, prompt in enumerate(prompts):

            # do not use nexts in training data
            if script_arguments.avoid_next_percentage and script_arguments.avoid_next_percentage <= random.random() and intents[i] in {Intents.NextStepIntent, Intents.AMAZONNextIntent}:
                continue

            prompt, completion = get_prompt_and_completion(
                prompt=prompt,
                user_token=script_arguments.user_token,
                system_token=script_arguments.system_token,
                intent_prefix_in_completion=script_arguments.add_intent_start_to_completion,
                intent_prefix_style=script_arguments.intent_prefix_style,
                user_final_token=script_arguments.user_final_token,
                system_final_token=script_arguments.system_final_token
            )

            sft_dataset_dict.append({
                "prompt": prompt,
                "completion": completion,
                "user_profile": profiles[i].user_custom_name if profiles[i] else None,
                "intent": intents[i],
                "dialogue_id": dialog.dialog_id,
            })

    return sft_dataset_dict
