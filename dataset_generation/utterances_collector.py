from collections import defaultdict
from typing import List, Dict

import numpy as np

from data_binding.enumerates import Intents
from dataset_generation.utils import apply_smoothing, clean_wake_words_from_text


def clean_transition_prefix(intent: str):
    return intent.replace("user_", "").replace("system_", "")


def calculate_transition_probabilities(transitions_count_dict: Dict[str, int],
                                       considered_intents: List[str] = None) -> Dict[str, Dict[str, float]]:

    # transitions_count_dict: dict[str, int] where each entry is user_NextStepIntent->system_NEXT_RESPONDER": 4349
    # list of strs with the intents (it will only consider the ones passed here or all if None is provided)

    transition_probabilities = defaultdict(dict)
    # go through all parameter combinations
    current_transitions = defaultdict(int)

    # calculate the sum of all transitions
    for transition_key, transition_count in transitions_count_dict.items():
        start, end = transition_key.split("->")
        start = clean_transition_prefix(start)
        end = clean_transition_prefix(end)

        if considered_intents is not None:
            if start in considered_intents and end in considered_intents:
                current_transitions[start] += transition_count
        else:
            current_transitions[start] += transition_count

    # calculate the probability of the transition
    for transition_key, transition_count in transitions_count_dict.items():
        start, end = transition_key.split("->")
        start = clean_transition_prefix(start)
        end = clean_transition_prefix(end)

        if considered_intents is not None:
            if start in considered_intents and end in considered_intents:
                if start not in transition_probabilities:
                    transition_probabilities[start] = {}
                transition_probabilities[start][end] = round(transition_count / current_transitions[start], 3)
        else:
            if start not in transition_probabilities:
                transition_probabilities[start] = {}
            transition_probabilities[start][end] = round(transition_count / current_transitions[start], 3)

    return transition_probabilities


def apply_smoothing_to_collected_utterances(collected_utterances: Dict[str, Dict[str, int]], alpha: float = None):
    new_probs = defaultdict(dict)
    for intent, utterances_dict in collected_utterances.items():
        values = list(utterances_dict.values())
        counts = np.array(values)
        if alpha is None:
            alpha = np.mean(values)

        for k, v in zip(utterances_dict.keys(), apply_smoothing(counts=counts, alpha=alpha).tolist()):
            new_probs[intent][k] = v

    return new_probs


def remove_unrelated_utterances(collected_utterances: Dict[str, Dict[str, int]], clean_wake_words: bool,
                                add_from_annotated_utterances: bool = True):
    annotated_utterances = {}
    # affects collected_utterances
    for intent in list(collected_utterances.keys()):
        if intent not in Intents.QuestionIntent:
            for utterance in list(collected_utterances[intent].keys()):
                # as requested by diogo silva removes "what" from everything that is not question
                if "what" in utterance:
                    collected_utterances[intent].pop(utterance)
                    continue

                # keep only the ones correctly annotated
                if annotated_utterances:
                    if intent in annotated_utterances:
                        if utterance not in annotated_utterances[intent]:
                            collected_utterances[intent].pop(utterance, None)
                            continue

                # cleans wake words from utterances
                if clean_wake_words:
                    clean_utterance = clean_wake_words_from_text(utterance)
                    if clean_utterance != utterance:  # there was a change
                        if clean_utterance in collected_utterances[intent]:
                            # if it exists, sum the counts
                            collected_utterances[intent][clean_utterance] += collected_utterances[intent].pop(utterance)
                        else:
                            # if it does not exist, add the previous value
                            collected_utterances[intent][clean_utterance] = collected_utterances[intent].pop(utterance)

    # if add_from_annotated_utterances we add the corrected intents to the collected utterances
    if annotated_utterances and add_from_annotated_utterances:
        for intent, utterance_dict in annotated_utterances.items():  # go through all the intents
            if intent in collected_utterances:
                for utterance, counts in utterance_dict.items():  # go through all the utterances and counts
                    if clean_wake_words:
                        utterance = clean_wake_words_from_text(utterance)
                    if utterance not in collected_utterances[intent]:  # if it does not exist, add it
                        collected_utterances[intent][utterance] = counts
