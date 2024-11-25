from typing import List, Dict, Union

from data_binding.task_result import TaskResult
from dataset_generation.dialog import Dialog
from dataset_generation.utterances_collector import remove_unrelated_utterances
from training.utils import get_folders_profile_per_model, get_dialogues_from_generation_folder
from user_simulator.traits_and_profiles.user_profile import UserProfile


def sample_dialogues(folder_path: str, number_dialogues: int, profiles: List[str] = None):
    profile_folders = get_folders_profile_per_model(path=folder_path, profiles=profiles)
    sampled_dialogues = []
    for profile_folder in profile_folders:
        dialogues = get_dialogues_from_generation_folder(folder_path=profile_folder)
        # order by dialogue id for reproducibility
        dialogues = sorted(dialogues, key=lambda x: x.dialog_id)[:number_dialogues]
        sampled_dialogues.extend(dialogues)

    return sampled_dialogues


def sample_dialogues_multi_trait(folder_paths: List[str], number_dialogues: int, profiles: List[UserProfile]):

    assert len(folder_paths) == len(profiles), "The number of folder paths and profiles should be the same"

    sampled_dialogues = []
    for profile_folder, profile in zip(folder_paths, profiles):
        dialogues = get_dialogues_from_generation_folder(folder_path=profile_folder)
        # order by dialogue id for reproducibility
        dialogues = sorted(dialogues, key=lambda x: x.dialog_id)[:number_dialogues]
        # replace by the user profile
        for d in dialogues:
            d.user_profile = profile

        sampled_dialogues.extend(dialogues)

    return sampled_dialogues


def get_utterances(collected_utterances: Dict[str, Dict[str, int]]):
    remove_unrelated_utterances(collected_utterances=collected_utterances, clean_wake_words=True)
    return collected_utterances


def task_to_text(task: TaskResult):
    task_text = task.get_title() + "\n"
    for i, step in enumerate(task.get_methods()[0].get_steps()):
        task_text += f"{i + 1}. {step.get_text()}\n"
    return task_text


def dialogue_context_to_text(dialogue: Dialog, current_turn: int, context_size: Union[int, None]):
    context = ""

    if context_size is None:
        considered_turns = dialogue.turns[:current_turn]
    else:
        considered_turns = dialogue.turns[max(0, current_turn - context_size):current_turn]

    for turn in considered_turns:
        context += f"User: {turn.user_utterance}\n"
        context += f"Assistant: {turn.system_utterance}\n"
    return context


def dialogue_to_text_simple(dialogue: Dialog):
    dialogue_text = "Task title: " + dialogue.task.get_title() + "\n"
    for i, turn in enumerate(dialogue.turns):
        # dialogue_text += f"Turn {i + 1}\nUser: (Intent: {turn.intent}) - {turn.user_utterance}\nSystem: {turn.system_utterance}\n\n"
        dialogue_text += f"Turn {i + 1}\nUser: {turn.user_utterance}\nSystem: {turn.system_utterance}\n\n"  # without intent
    return dialogue_text.rstrip()


def convert_dialogue_to_list_data(dialogue: Dialog):
    dialogue_data = []
    for i, turn in enumerate(dialogue.turns):
        data = ""
        if i == 0:
            title = dialogue.task.get_title()
            data += f"Task: {title}\n"
        data += f"Turn {i + 1}\nIntent: {turn.intent}\nUser: {turn.user_utterance}\nSystem: {turn.system_utterance}"
        dialogue_data.append(data)
    return dialogue_data
