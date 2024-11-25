from typing import List, Dict

import pandas as pd

from data_binding.enumerates import Intents
from dataset_generation.dialog import Dialog
from user_simulator.traits_and_profiles.identifying_traits import user_name_to_trait
from user_simulator.traits_and_profiles.user_profile import get_opposite_user


def is_degeneration(utterance: str):
    degeneration_tokens = {"<|", "|>", "INST", "Assistant", "User", "USER", "SYSTEM", "[Intent", "[User", "endoftext"}
    return any([token in utterance for token in degeneration_tokens])


def has_problem_utterance_level(dialogues_list: List[Dialog]):
    # checks for degeneration and
    degenaration_turns = []
    degeneration_by_profile = {}
    turn_by_profile = {}
    degeneration_by_metric = {}
    number_of_turn_by_metric = {}
    all_turns = 0
    for d in dialogues_list:
        metric = user_name_to_trait.get(d.user_profile.user_custom_name)
        if not metric:
            u_op = get_opposite_user(d.user_profile.user_custom_name)
            if u_op:
                metric = user_name_to_trait.get(u_op.user_custom_name)

        for turn in d.turns:
            all_turns += 1
            number_of_turn_by_metric[metric] = number_of_turn_by_metric.get(metric, 0) + 1
            turn_by_profile[d.user_profile.user_custom_name] = turn_by_profile.get(d.user_profile.user_custom_name, 0) + 1
            if is_degeneration(turn.user_utterance) or turn.intent not in Intents.pretty_names and turn.intent not in Intents.inverted_pretty_names:
                degenaration_turns.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                degeneration_by_profile[d.user_profile.user_custom_name] = degeneration_by_profile.get(d.user_profile.user_custom_name, 0) + 1
                degeneration_by_metric[metric] = degeneration_by_metric.get(metric, 0) + 1

    # order by profile
    degeneration_by_profile = dict(sorted(degeneration_by_profile.items(), key=lambda item: item[1], reverse=True))
    # divide by the total turns for that profile
    degeneration_by_profile = {k: round(v / turn_by_profile[k], 2) for k, v in degeneration_by_profile.items()}

    # divide by the total turns for that metric
    degeneration_by_metric = {k: round(v / number_of_turn_by_metric[k], 2) for k, v in degeneration_by_metric.items()}
    # order by metric
    degeneration_by_metric = dict(sorted(degeneration_by_metric.items(), key=lambda item: item[1], reverse=True))

    print("Total turns: ", all_turns)
    print("Degenaration special tokens: ", len(degenaration_turns) / all_turns, len(degenaration_turns))
    print("Number of dialogues: ", len(dialogues_list))
    print("Degeneration by profile", degeneration_by_profile)
    print("Degeneration by metric", degeneration_by_metric)
    print()

    # put all dialogues into a dataframe
    all_dialogues_dict = []
    # create an entry for each trait scale
    all_trait_names = []
    for d in dialogues_list:
        current_dialog = d.dialog_dict()
        current_dialog["profile"] = d.user_profile.user_custom_name
        trait_scale = d.user_profile.trait_scale
        if trait_scale:
            for trait_name, trait_value in trait_scale.items():
                if trait_value == 2:
                    current_dialog[trait_name] = "High"
                elif trait_value == 0:
                    current_dialog[trait_name] = "Low"
                all_trait_names.append(trait_name)

        current_dialog["number_turns"] = d.number_turns()
        all_dialogues_dict.append(current_dialog)

    dialogues_dict = pd.DataFrame(all_dialogues_dict)
    # the dialog column is a list of turns apply a function that counts the degenerations in the turns
    dialogues_dict["Degeneration"] = dialogues_dict["dialog"].apply(lambda x: sum([is_degeneration(t["user"]) for t in x]))
    # divide by the number of turns
    dialogues_dict["Degeneration"] = dialogues_dict["Degeneration"] / dialogues_dict["number_turns"]

    # calculate average degeneration per trait_name and intensity divided by the number of turns
    degeneration_by_trait = {}
    for trait_name in all_trait_names:
        degeneration_by_trait[trait_name] = round(dialogues_dict.groupby(trait_name)["Degeneration"].mean(), 2)

    return degenaration_turns, degeneration_by_profile


def check_for_unique_utterances(dialogues_list: List[Dialog], collected_utterances: Dict[str, Dict[str, int]]):

    # put all collected_utterances into a set
    collected_utterances_set = set()
    for intent, utterances in collected_utterances.items():
        for u in utterances:
            collected_utterances_set.add(u.lower().strip())

    # check for uniqueness
    unique_utterances = []
    all_turns = 0
    unique_by_profile = {}
    turn_by_profile = {}
    unique_by_metric = {}
    number_of_turn_by_metric = {}
    unique_by_intent = {}
    number_of_turn_by_intent = {}
    for d in dialogues_list:

        metric = user_name_to_trait.get(d.user_profile.user_custom_name)
        if not metric:
            u_op = get_opposite_user(d.user_profile.user_custom_name)
            if u_op:
                metric = user_name_to_trait.get(u_op.user_custom_name)

        for turn in d.turns:
            all_turns += 1
            number_of_turn_by_metric[metric] = number_of_turn_by_metric.get(metric, 0) + 1
            turn_by_profile[d.user_profile.user_custom_name] = turn_by_profile.get(d.user_profile.user_custom_name, 0) + 1
            number_of_turn_by_intent[turn.intent] = number_of_turn_by_intent.get(turn.intent, 0) + 1
            if turn.user_utterance.lower().strip() not in collected_utterances_set:
                unique_utterances.append((turn.intent, turn.user_utterance, d.user_profile.user_custom_name))
                unique_by_profile[d.user_profile.user_custom_name] = unique_by_profile.get(d.user_profile.user_custom_name, 0) + 1
                unique_by_metric[metric] = unique_by_metric.get(metric, 0) + 1
                unique_by_intent[turn.intent] = unique_by_intent.get(turn.intent, 0) + 1

    # order by profile
    unique_by_profile = dict(sorted(unique_by_profile.items(), key=lambda item: item[1], reverse=True))
    # divide by the total turns for that profile
    unique_by_profile = {k: round(v / turn_by_profile[k], 2) for k, v in unique_by_profile.items()}

    # divide by the total turns for that metric
    unique_by_metric = {k: round(v / number_of_turn_by_metric[k], 2) for k, v in unique_by_metric.items()}
    # order by metric
    unique_by_metric = dict(sorted(unique_by_metric.items(), key=lambda item: item[1], reverse=True))

    # divide by the total turns for that intent
    unique_by_intent = {k: round(v / number_of_turn_by_intent[k], 2) for k, v in unique_by_intent.items()}
    # order by intent
    unique_by_intent = dict(sorted(unique_by_intent.items(), key=lambda item: item[1], reverse=True))

    print("Total turns: ", all_turns)
    print("Unique utterances: ", len(unique_utterances) / all_turns, len(unique_utterances))
    print("Number of dialogues: ", len(dialogues_list))
    print("Unique by profile", unique_by_profile)
    print("Unique by metric", unique_by_metric)
    print("Unique by intent", unique_by_intent)
    print()

    return collected_utterances
