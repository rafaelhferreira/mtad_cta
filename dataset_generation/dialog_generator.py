import copy
import os
import random
from typing import List, Dict, Optional
from tqdm import tqdm

from data_binding.enumerates import Intents
from dataset_generation import response_building_utils
from dataset_generation.calc_dialog_stats import calc_dialog_stats
from dataset_generation.dialog import Dialog
from dataset_generation.system_responses import ResponseType, SystemResponses
from dataset_generation.utils import set_seeds, get_tasks_files, CONSIDERED_INTENTS, random_choices_from_dict, \
    get_valid_task, write_to_json_file

from dataset_generation.utterances_collector import calculate_transition_probabilities, \
    apply_smoothing_to_collected_utterances, remove_unrelated_utterances
from user_simulator.traits_and_profiles.user_profile import UserProfile


def remove_additional_intents(probability_dict: Dict[str, Dict[str, float]],
                              considered_intents: List[str]) -> Dict[str, Dict[str, float]]:
    for key in list(probability_dict.keys()):
        if key not in considered_intents:  # remove main transition
            probability_dict.pop(key)
        else:
            for key_2 in list(probability_dict[key].keys()):  # remove secondary transitions
                if key_2 not in considered_intents:
                    probability_dict[key].pop(key_2)

    return probability_dict


def create_dialog_based_on_probs(collected_utterances: Dict[str, Dict[str, int]],
                                 transitions_count_dict: Optional[Dict[str, int]],
                                 first_step_probs: Dict[str, float], tasks_path: str,
                                 use_weight_for_utterance: bool,
                                 considered_intents: List[str] = None,
                                 number_dialogs: int = 10,
                                 max_number_turns: int = 20,
                                 out_path: str = None,
                                 seed: int = 42,
                                 clean_wake_words: bool = True,
                                 transitions_probs_dict: Dict[str, Dict[str, float]] = None,
                                 apply_smoothing_to_utterances: bool = False,
                                 ignore_stop_intent: bool = False,
                                 user_profiles_prob: Dict[UserProfile, float] = None,
                                 system_errors_prob: Dict[str, float] = None,
                                 lower_case_and_remove_punctuation_user: bool = False,
                                 calc_stats: bool = False,
                                 ) -> List[Dialog]:
    if considered_intents is None:
        considered_intents = CONSIDERED_INTENTS

    # if transitions_probs_dict is passed we use it values instead of transitions_count_dict
    if transitions_probs_dict:
        probability_dict = transitions_probs_dict
    else:
        # calculate transition probabilities
        probability_dict = calculate_transition_probabilities(transitions_count_dict, considered_intents)

    # remove non-considered probabilities from first step prob
    for i in list(first_step_probs.keys()):
        if i not in considered_intents:
            first_step_probs.pop(i)
    # remove additional intents from main flow
    probability_dict = remove_additional_intents(probability_dict, considered_intents)

    # remove unrelated utterances (it changes collected_utterances)
    remove_unrelated_utterances(collected_utterances=collected_utterances, clean_wake_words=clean_wake_words)

    if apply_smoothing_to_utterances:
        collected_utterances = apply_smoothing_to_collected_utterances(collected_utterances)

    dialogs = generate_dialogs(first_step_probs=first_step_probs, probability_dict=probability_dict,
                               collected_utterances=collected_utterances, tasks_path=tasks_path,
                               use_weight_for_utterance=use_weight_for_utterance, number_dialogs=number_dialogs,
                               max_number_turns=max_number_turns, seed=seed, ignore_stop_intent=ignore_stop_intent,
                               user_profiles_prob=user_profiles_prob,
                               system_errors_prob=system_errors_prob,
                               lower_case_and_remove_punctuation_user=lower_case_and_remove_punctuation_user)

    if out_path:
        # create dir
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if out_path:
        # write to file and also generate a unique id per conversation
        write_to_json_file(file_path=out_path, data={d.dialog_id: d.dialog_dict() for d in dialogs})

    if out_path:
        # we change out_path to save config
        out_path = out_path.replace(".json", "_config.json")
        write_to_json_file(file_path=out_path, data={
            "seed": seed,
            "collected_utterances": collected_utterances,
            "transitions_count_dict": transitions_count_dict,
            "first_step_probs": first_step_probs,
            "use_weight_for_utterance": use_weight_for_utterance,
            "considered_intents": considered_intents,
            "number_dialogs": number_dialogs,
            "out_path": out_path,
            "transitions_probs_dict": transitions_probs_dict,
            "apply_smoothing_to_utterances": apply_smoothing_to_utterances,
            "ignore_stop_intent": ignore_stop_intent,
            "user_profiles_prob": {user_prof.__str__(): user_prob for user_prof, user_prob in
                                   user_profiles_prob.items()} if user_profiles_prob else None,
        })

    if calc_stats:
        dialogue_stats = calc_dialog_stats(dialogues=dialogs, intents_distribution=probability_dict,
                                           first_step_distribution=first_step_probs)
        print("Dialog Stats")
        print(dialogue_stats)
        write_to_json_file(file_path=out_path.replace(".json", "_stats.json"), data=dialogue_stats)

    return dialogs


def generate_dialogs(first_step_probs: Dict[str, float],
                     probability_dict: Dict[str, Dict[str, float]],
                     collected_utterances: Dict[str, Dict[str, int]], tasks_path: str,
                     use_weight_for_utterance: bool,
                     number_dialogs: int = 10,
                     max_number_turns: int = 10,
                     seed: int = 42,
                     ignore_stop_intent: bool = False,
                     user_profiles_prob: Dict[UserProfile, float] = None,
                     system_errors_prob: Dict[str, float] = None,
                     lower_case_and_remove_punctuation_user: bool = True) -> List[Dialog]:
    # set seeds for reproducibility
    set_seeds(seed)

    # get the task file names
    tasks_file_paths = get_tasks_files(tasks_path)
    # remove stop intents from first turn
    for s_intent in Intents.stop_task_intents():
        first_step_probs.pop(s_intent, None)

    # if ignore stop intent we remove end of conversation intents from the main flow
    # this means that the interaction will only finish when the user reaches the end of task or the max number of turns
    # this also silently removes any intents that do not have a transitions out of each is neat except the fact that it
    # leads to believe that a certain intent is being considered when it isnt
    if ignore_stop_intent:
        to_keep_keys = [i for i in probability_dict.keys() if
                        i not in Intents.stop_task_intents()]
        remove_additional_intents(probability_dict, to_keep_keys)

    dialogs = []

    for i in tqdm(range(number_dialogs)):

        # choose a task from the list
        task = get_valid_task(tasks_file_paths)

        updated_probability_dict = copy.deepcopy(probability_dict)
        updated_first_step_probs = copy.deepcopy(first_step_probs)

        # get a user profile
        user_profile = None
        if user_profiles_prob:
            user_profile = \
                random.choices(list(user_profiles_prob.keys()), weights=list(user_profiles_prob.values()), k=1)[0]  # type: UserProfile
            # we add first_turn as a key because code expects a dict of dicts
            updated_first_step_probs = user_profile.apply_escalation_traits({"first_turn": first_step_probs}).get(
                "first_turn")
            updated_probability_dict = user_profile.apply_escalation_traits(probability_dict)

        current_dialog = Dialog(
            task=task, system_tone=ResponseType.NEUTRAL.value, user_profile=user_profile,
            lower_case_and_remove_punctuation_user=lower_case_and_remove_punctuation_user
        )

        # if we are working with tasks we always start with StartStepsIntent and the first step
        if task:
            # add the first turn which is only system with the first step
            initial_user_req = current_dialog.__get_user_utterance__(
                Intents.StartStepsIntent, collected_utterances, use_weight_for_utterance)
            current_dialog.add_turn(
                intent=Intents.StartStepsIntent, user_utterance=initial_user_req,
                system_utterance=response_building_utils.add_next_step_envelope(
                    task.get_methods()[0].get_steps()[0].get_text(), 0, len(task.get_methods()[0].get_steps())),
                current_step=0, negative_response="",
                lowercase_and_remove_punctuation_user=lower_case_and_remove_punctuation_user
            )

        intent, _, _, _ = current_dialog.create_and_add_turn_from_prob_dict(
            probs_dict=updated_first_step_probs,
            collected_utterances=collected_utterances,
            use_weight_for_utterance=use_weight_for_utterance,
        )

        # add the rest of the turns
        end_conversation = False
        while intent not in Intents.stop_task_intents() and not end_conversation:
            probs_dict = updated_probability_dict[intent]
            intent, utterance, system_response, end_conversation = current_dialog.create_and_add_turn_from_prob_dict(
                probs_dict=probs_dict,
                collected_utterances=collected_utterances,
                use_weight_for_utterance=use_weight_for_utterance,
            )

            # apply tolerance trait since it depends on the number of fallbacks
            if user_profile:
                apply_system_error = system_errors_prob and random.random() <= system_errors_prob.get(intent, -1)
                # on fallback (user caused) and a percentage of time the system makes a mistake on purpose to account for those cases
                if intent == Intents.AMAZONFallbackIntent or (
                        apply_system_error and current_dialog.turns[-1].apply_forced_system_error()):
                    # depends on the turn level to increase more slowly
                    # this makes it so tolerance is y = stop_prob * factor^(turn)
                    updated_probability_dict = user_profile.intolerance_trait.apply_escalation_to_distribution(
                        updated_probability_dict)

            # add a stop intent by force and break
            if current_dialog.number_turns() == max_number_turns - 1 or system_response in SystemResponses.end_of_task:
                utterance = random_choices_from_dict(collected_utterances[Intents.AMAZONStopIntent],
                                                     use_weights=use_weight_for_utterance)[0]
                current_dialog.add_turn(
                    intent=Intents.AMAZONStopIntent, user_utterance=utterance,
                    system_utterance=random.choice(SystemResponses.goodbye),
                    current_step=current_dialog.current_step,
                    lowercase_and_remove_punctuation_user=lower_case_and_remove_punctuation_user
                )
                break

        # add the dialog to the list (NOTE: it is still needed to filter the dialogues to garantee that they are valid)
        dialogs.append(current_dialog)

    # print the first 10 dialogs
    for d in dialogs[:10]:
        for t in d.turns:
            print(t)
        print("###################\n")

    return dialogs
