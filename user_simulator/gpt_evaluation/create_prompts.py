import json
import os
import random
import uuid
from collections import Counter
from typing import List, Dict

import pandas as pd

from data_binding.enumerates import Intents
from dataset_generation.dialog import Dialog
from dataset_generation.utils import set_seeds
from user_simulator.gpt_evaluation.comparative_eval import create_dialogue_comparison_prompt
from user_simulator.gpt_evaluation.degeneration_and_uniqueness import is_degeneration
from user_simulator.gpt_evaluation.gpt_prompts import SYSTEM_ERROR_QUESTION, INTENT_QUESTION, GENERAL_PREFIX
from user_simulator.gpt_evaluation.utils import convert_dialogue_to_list_data, dialogue_to_text_simple, task_to_text, \
    dialogue_context_to_text
from user_simulator.traits_and_profiles.identifying_traits import user_name_to_trait
from user_simulator.traits_and_profiles.user_profile import UserTypes, get_possible_opposite_user_types


def dialogue_to_intent_prompts(dialogues_list: List[Dialog],
                               collected_utterances: Dict[str, Dict[str, int]],
                               test_dialogues: List[Dialog] = None,  # if test dialogues is None it will use the dialogues_list to choose comparative else uses test_dialogues
                               output_path: str = None, suffix: str = "", number_sheets: int = 1):
    empty_intents, non_matching_intents = [], []
    matching_intents = []
    degenaration_turns = []
    intents_prompts_list = []
    system_error_prompts_list = []
    profile_prompts_list_3_dialogues = []
    profile_prompts_list_2_dialogues = []

    # automatic annotations using simple matching
    current_intent_annotations = []
    current_degeneration_annotations = []
    placeholder_value = ""
    total_turns = 0

    dialogues_dict = []

    for d in dialogues_list:

        dialogue_data = convert_dialogue_to_list_data(d)

        for i, turn in enumerate(d.turns):

            total_turns += 1
            internal_intent = Intents.inverted_pretty_names.get(turn.intent, turn.intent)

            system_error_prompt = fill_prefix(dialogue=d, current_turn_index=i, annotation_question=SYSTEM_ERROR_QUESTION,
                                              use_intent_description=False, use_task_text=True, annotation_id=str(uuid.uuid4()))
            system_error_prompts_list.append(system_error_prompt)

            # this includes special tokens
            if is_degeneration(turn.user_utterance):
                degenaration_turns.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                current_degeneration_annotations.append(1)
            else:
                current_degeneration_annotations.append(placeholder_value)

            # intent is empty
            if not turn.intent or not turn.intent.strip():
                empty_intents.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                current_intent_annotations.append(1)
                continue
            # intent is not recognized
            elif turn.intent not in Intents.pretty_names and turn.intent not in Intents.inverted_pretty_names:
                non_matching_intents.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                current_intent_annotations.append(1)
                continue
            # intent is recognized and in the collected utterances
            elif internal_intent in collected_utterances:
                if turn.user_utterance.lower().strip() in collected_utterances[internal_intent]:
                    # intent is recognized and in the collected utterances so it is correct
                    matching_intents.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                    current_intent_annotations.append(0)
                    continue

            current_intent_annotations.append(placeholder_value)
            intent_prompt = fill_prefix(dialogue=d, current_turn_index=i, annotation_question=INTENT_QUESTION,
                                        use_intent_description=True,
                                        use_task_text=False, annotation_id=str(uuid.uuid4()))
            intents_prompts_list.append(intent_prompt)

        # add last turns annotations to dialogues_dict
        dialogues_dict.append(
            {
                'ids_list': [f"{d.dialog_id}_Turn {i}" for i in range(len(d.turns))],
                'path': [d.simulator_model_path] * len(d.turns),
                'user_profile': [d.user_profile.user_custom_name] * len(d.turns),
                'intents': [turn.intent for turn in d.turns],
                'turn': dialogue_data,
                'Non-Matching Intent': current_intent_annotations[-len(d.turns):],
                'Degeneration': current_degeneration_annotations[-len(d.turns):],
                'System error': [placeholder_value] * len(d.turns),
            })

    # shuffle dialogues_dict
    set_seeds(42)
    random.shuffle(dialogues_dict)

    df = pd.DataFrame(columns=['ids_list', 'path', 'user_profile', 'intents', 'turn',
                               'Non-Matching Intent', 'Degeneration', 'System error',
                               ])

    for d in dialogues_dict:
        df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
        # add an empty row with placeholder value between dialogues
        df = pd.concat([df, pd.DataFrame({k: [placeholder_value] for k in df.columns})], ignore_index=True)

    # create comparative dialogs excel and prompts
    task_to_dialogs = {}
    for d in dialogues_list:
        task_to_dialogs[d.task.get_unique_id()] = task_to_dialogs.get(d.task.get_unique_id(), []) + [d]
    # shuffle task_to_dialogs (because multitrait to get different combinations)
    for k, v in task_to_dialogs.items():
        random.shuffle(v)

    # create comparative dialogs for test dialogues
    test_task_to_dialogs = {}
    if test_dialogues:
        for d in test_dialogues:
            test_task_to_dialogs[d.task.get_unique_id()] = test_task_to_dialogs.get(d.task.get_unique_id(), []) + [d]
        # shuffle test_task_to_dialogs (because multitrait to get different combinations)
        for k, v in test_task_to_dialogs.items():
            random.shuffle(v)

    comparative_dialogs_3_dialogues = []
    comparative_dialogs_2_dialogues = []
    for task_id, dialogs in task_to_dialogs.items():
        for d in dialogs:
            user_profile = d.user_profile.user_custom_name
            if user_profile == UserTypes.RegularUser.user_custom_name:
                continue

            for trait_name, trait_value in d.user_profile.trait_scale.items():
                # get opposite user profile
                # opposite_user_profile = get_opposite_user(user_profile)

                metric = user_name_to_trait.get(trait_name)

                if not metric:
                    raise ValueError(f"Metric not found for {trait_name} or {user_profile}")

                if trait_value == 1:  # medium intensity we skip
                    continue

                possible_user_profiles = {u.user_custom_name for u in get_possible_opposite_user_types(trait_name=trait_name, trait_value=trait_value)}

                if possible_user_profiles:

                    # if test dialogues is not None use the test dialogues to get the opposite dialogues
                    if test_dialogues:
                        opposite_dialogs = test_task_to_dialogs.get(task_id, [])
                    else:
                        opposite_dialogs = task_to_dialogs.get(task_id, [])

                    opposite_dialog = None
                    regular_dialog = None
                    for od in opposite_dialogs:
                        if od.user_profile.user_custom_name in possible_user_profiles:
                            opposite_dialog = od
                        elif od.user_profile.user_custom_name == UserTypes.RegularUser.user_custom_name:
                            regular_dialog = od

                    if not regular_dialog:
                        if d.user_profile.is_multitrait:
                            # multitrait does not have a regular counterpart so we skip
                            continue
                        else:
                            raise ValueError(f"Regular dialog not found for {trait_name} or {user_profile}")

                    # randomize the order of d, opposite_dialog and regular_dialog
                    order_dialogues = random.sample([d, opposite_dialog, regular_dialog], 3)

                    # first if is to avoid repetitions (since we are already comparing the opposites) / the second if is to model both sides of the comparison
                    # if (not test_task_to_dialogs and d.user_profile.user_custom_name in main_name_traits) or test_task_to_dialogs:
                    # if user_name_to_trait.get(user_profile):
                    if True:

                        # generate unique id
                        comparative_3_id = str(uuid.uuid4())

                        comparative_dialogs_3_dialogues.append({
                            "id": comparative_3_id,
                            "u1_path": order_dialogues[0].simulator_model_path,
                            "u2_path": order_dialogues[1].simulator_model_path,
                            "u3_path": order_dialogues[2].simulator_model_path,
                            "task": d.task.get_unique_id(),
                            "u1": order_dialogues[0].user_profile.user_custom_name,
                            "u2": order_dialogues[1].user_profile.user_custom_name,
                            "u3": order_dialogues[2].user_profile.user_custom_name,
                            "1": dialogue_to_text_simple(order_dialogues[0]),
                            "2": dialogue_to_text_simple(order_dialogues[1]),
                            "3": dialogue_to_text_simple(order_dialogues[2]),
                            "Metric": metric,
                            "High Number": "",
                            "Low Number": "",
                        })

                        if trait_value == 0:
                            model_order = [d, regular_dialog, opposite_dialog]
                        else:
                            model_order = [opposite_dialog, regular_dialog, d]

                        profile_prompts_list_3_dialogues.append(
                            create_dialogue_comparison_prompt(order_dialogues, user_profile, add_intents=False,
                                                              model_order=model_order,
                                                              annotation_id=comparative_3_id, metric=metric))

                        if trait_value == 0:
                            model_order = [d, opposite_dialog]
                        else:
                            model_order = [opposite_dialog, d]

                        # make the 3 combinations of 2 traits to compare
                        # generate unique id
                        comparative_2_id = str(uuid.uuid4())
                        order_dialogues = random.sample([d, opposite_dialog], 2)
                        profile_prompts_list_2_dialogues.append(
                            create_dialogue_comparison_prompt(order_dialogues, user_profile, add_intents=False,
                                                              model_order=model_order,
                                                              annotation_id=comparative_2_id, metric=metric))

                        comparative_dialogs_2_dialogues.append({
                            "id": comparative_2_id,
                            "task": d.task.get_unique_id(),
                            "u1_path": order_dialogues[0].simulator_model_path,
                            "u2_path": order_dialogues[1].simulator_model_path,
                            "u1": order_dialogues[0].user_profile.user_custom_name,
                            "u2": order_dialogues[1].user_profile.user_custom_name,
                            "1": dialogue_to_text_simple(order_dialogues[0]),
                            "2": dialogue_to_text_simple(order_dialogues[1]),
                            "Metric": metric,
                            "Higher": "",
                        })

                        if trait_value == 0:
                            model_order = [d, regular_dialog]
                        else:
                            model_order = [regular_dialog, d]

                        comparative_2_id = str(uuid.uuid4())
                        order_dialogues = random.sample([d, regular_dialog], 2)
                        profile_prompts_list_2_dialogues.append(
                            create_dialogue_comparison_prompt(order_dialogues, user_profile, add_intents=False,
                                                              model_order=model_order,
                                                              annotation_id=comparative_2_id, metric=metric))

                        comparative_dialogs_2_dialogues.append({
                            "id": comparative_2_id,
                            "task": d.task.get_unique_id(),
                            "u1_path": order_dialogues[0].simulator_model_path,
                            "u2_path": order_dialogues[1].simulator_model_path,
                            "u1": order_dialogues[0].user_profile.user_custom_name,
                            "u2": order_dialogues[1].user_profile.user_custom_name,
                            "1": dialogue_to_text_simple(order_dialogues[0]),
                            "2": dialogue_to_text_simple(order_dialogues[1]),
                            "Metric": metric,
                            "Higher": "",
                        })

                        # only makes sense to compare the opposites with regular if not using test dialogues
                        if not test_task_to_dialogs:

                            if trait_value == 0:
                                model_order = [regular_dialog, opposite_dialog]
                            else:
                                model_order = [opposite_dialog, regular_dialog]

                            comparative_2_id = str(uuid.uuid4())
                            order_dialogues = random.sample([opposite_dialog, regular_dialog], 2)
                            profile_prompts_list_2_dialogues.append(
                                create_dialogue_comparison_prompt(order_dialogues, user_profile, add_intents=False,
                                                                  model_order=model_order,
                                                                  annotation_id=comparative_2_id, metric=metric))

                            comparative_dialogs_2_dialogues.append({
                                "id": comparative_2_id,
                                "task": d.task.get_unique_id(),
                                "u1_path": order_dialogues[0].simulator_model_path,
                                "u2_path": order_dialogues[1].simulator_model_path,
                                "u1": order_dialogues[0].user_profile.user_custom_name,
                                "u2": order_dialogues[1].user_profile.user_custom_name,
                                "1": dialogue_to_text_simple(order_dialogues[0]),
                                "2": dialogue_to_text_simple(order_dialogues[1]),
                                "Metric": metric,
                                "Higher": "",
                            })

    # create a df with comparative dialogs
    df_comparative_dialogs_3_dialogues = pd.DataFrame(comparative_dialogs_3_dialogues)
    df_comparative_dialogs_2_dialogues = pd.DataFrame(comparative_dialogs_2_dialogues)

    # create a df for the system errors
    df_system_errors = pd.DataFrame(system_error_prompts_list)
    # drop the prompt column
    df_system_errors = df_system_errors.drop(columns=["prompt"])
    # add a column for the system error
    df_system_errors["System Response Quality"] = [placeholder_value] * len(df_system_errors)

    if output_path:

        # save annotation sheet
        out_path = os.path.join(output_path, f"annotation_sheet_{suffix}.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for i in range(0, number_sheets):
                df.to_excel(writer, sheet_name=str(i), index=False, freeze_panes=(1, 0))

        # save comparative dialogs sheet
        out_path = os.path.join(output_path, f"comparative_dialogs_3_dialogues_{suffix}.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            # shuffle the df_comparative_dialogs rows
            df_comparative_dialogs_3_dialogues = df_comparative_dialogs_3_dialogues.sample(frac=1)
            df_comparative_dialogs_3_dialogues.to_excel(writer, sheet_name="Comparative Dialogs", index=False, freeze_panes=(1, 0))

        out_path = os.path.join(output_path, f"comparative_dialogs_2_dialogues_{suffix}.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            # shuffle the df_comparative_dialogs rows
            df_comparative_dialogs_2_dialogues = df_comparative_dialogs_2_dialogues.sample(frac=1)
            df_comparative_dialogs_2_dialogues.to_excel(writer, sheet_name="Comparative Dialogs", index=False, freeze_panes=(1, 0))

        # save the system errors sheet
        out_path = os.path.join(output_path, f"system_errors_{suffix}.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            # shuffle the df_system_errors rows
            df_system_errors = df_system_errors.sample(frac=1)
            df_system_errors.to_excel(writer, sheet_name="System Errors", index=False, freeze_panes=(1, 0))

        # save the prompts to json file
        out_path = os.path.join(output_path, f"prompts_intent_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(intents_prompts_list, f, indent=2)

        out_path = os.path.join(output_path, f"prompts_system_error_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(system_error_prompts_list, f, indent=2)

        out_path = os.path.join(output_path, f"prompts_profile_comparative_3_dialogues_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(profile_prompts_list_3_dialogues, f, indent=2)

        out_path = os.path.join(output_path, f"prompts_profile_comparative_2_dialogues_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(profile_prompts_list_2_dialogues, f, indent=2)

    print("Total turns: ", total_turns)
    print("Degenaration special tokens: ", len(degenaration_turns) / total_turns, len(degenaration_turns))
    print("Empty intents: ", len(empty_intents) / total_turns, len(empty_intents))
    print("Non matching intents: ", len(non_matching_intents) / total_turns, len(non_matching_intents))
    print("Matching intents: ", len(matching_intents) / total_turns, len(matching_intents))
    print("Number of prompts: ", len(intents_prompts_list) / total_turns, len(intents_prompts_list))
    print("Number of dialogues: ", len(dialogues_list))
    print()

    # check different intent values for degenaration special tokens
    degenaration_special_tokens_counter = Counter([i[0] for i in degenaration_turns])
    print("Degenaration special tokens counter: ", degenaration_special_tokens_counter)

    # check different intent values for empty intents
    non_matching_intents_counter = Counter([i[0] for i in non_matching_intents])
    print("Non matching intents counter: ", non_matching_intents_counter)

    # check different intent values for matching intents
    matching_intents_counter = Counter([i[0] for i in matching_intents])
    print("Matching intents counter: ", matching_intents_counter)

    # check different intent values for prompts
    prompts_list_counter = Counter([i["intent"] for i in intents_prompts_list])
    print("Prompts list counter: ", prompts_list_counter)
    print()

    # do the same but with profiles
    degenaration_special_tokens_counter = Counter([i[2] for i in degenaration_turns])
    print("Degenaration special tokens counter: ", degenaration_special_tokens_counter)

    non_matching_profiles_counter = Counter([i[2] for i in non_matching_intents])
    print("Non matching profiles counter: ", non_matching_profiles_counter)

    matching_profiles_counter = Counter([i[2] for i in matching_intents])
    print("Matching profiles counter: ", matching_profiles_counter)

    prompts_profiles_counter = Counter([i["user_profile"] for i in intents_prompts_list])
    print("Prompts profiles counter: ", prompts_profiles_counter)
    print()

    return intents_prompts_list, system_error_prompts_list, profile_prompts_list_3_dialogues, profile_prompts_list_2_dialogues


def fill_prefix(dialogue: Dialog, current_turn_index: int, annotation_question: str,
                use_intent_description: bool, use_task_text: bool, annotation_id: str):
    current_turn = dialogue.turns[current_turn_index]
    internal_intent = Intents.inverted_pretty_names.get(current_turn.intent, current_turn.intent)

    if use_intent_description:
        # here we add the use_intent_description to assistant because of format
        assistant_response = "Assistant: " + current_turn.system_utterance + "\nUser Intent Description: " + Intents.get_intent_description(internal_intent) + "\n"
    else:
        assistant_response = "Assistant: " + current_turn.system_utterance + "\n"

    if use_task_text:
        prompt = GENERAL_PREFIX.format(recipe_text=task_to_text(dialogue.task),
                                       dialog_context=dialogue_context_to_text(dialogue, current_turn_index, None),
                                       user=f"User: {current_turn.user_utterance}",
                                       assistant_response=assistant_response,
                                       annotation_question=annotation_question)
    else:
        prompt = GENERAL_PREFIX.replace("Recipe: {recipe_text}", "").format(
            dialog_context=dialogue_context_to_text(dialogue, current_turn_index, None),
            user=f"User: {current_turn.user_utterance}",
            assistant_response=assistant_response,
            annotation_question=annotation_question
        )

    # add the prompt to the list of prompts
    prompt_dict = {
        "id": annotation_id,
        "simulator_model": dialogue.simulator_model_path,
        "dialog_id": dialogue.dialog_id,
        "user_profile": dialogue.user_profile.user_custom_name,
        "turn_number": current_turn_index,
        "intent": current_turn.intent,
        "user_utterance": current_turn.user_utterance,
        "system_utterance": current_turn.system_utterance,
        "prompt": prompt,
        "dialogue": task_to_text(dialogue.task) + "\n" + dialogue_context_to_text(dialogue, current_turn_index + 1, 4),
    }

    return prompt_dict
