import json
import os
import random
from typing import List, Union, Dict
import torch
import numpy as np

from data_binding.enumerates import Intents
from data_binding.task_result import DummyTaskResult


def set_seeds(seed: int):
    # set all seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_file(file_path: str):
    with open(file_path) as f_open:
        return json.load(f_open)


def write_to_json_file(file_path: str, data, cls: json.JSONEncoder = None, indent: int = 2):
    with open(file_path, "w") as f_open:
        json.dump(data, f_open, indent=indent, cls=cls)


def get_tasks_files(folder_path: str) -> List[str]:
    task_files = []
    for i in os.listdir(folder_path):
        if i.endswith(".json"):
            task_files.append(os.path.join(folder_path, i))
    return task_files


def get_valid_task(tasks_file_paths: List[str]):
    valid_task = False
    task = None
    while not valid_task:
        task_path = random.choice(tasks_file_paths)
        task = is_valid_task(task_path)
        if task is not None:
            valid_task = True
    return task


def is_valid_task(task_path: str):
    task_json = load_json_file(task_path)
    if "id" in task_json:
        task = DummyTaskResult(task_json)
    else:
        return None

    if task.get_total_number_steps(0) >= 3:
        return task
    return None


def random_choices_from_dict(option_count: Dict[str, Union[float, int]], use_weights: bool, k=1):
    if use_weights:
        return random.choices(list(option_count.keys()), weights=list(option_count.values()), k=k)
    else:
        return random.choices(list(option_count.keys()), k=k)


def apply_smoothing(counts: np.ndarray, alpha: float) -> np.ndarray:
    total_counts = np.sum(counts)
    num_classes = counts.shape[0]
    smoothed_probs = (counts + alpha) / (total_counts + alpha * num_classes)

    return smoothed_probs


def calculate_word_diversity(utterances: List[str]):
    # returns the ratio of unique words to total words
    total_words = sum(len(utterance.split()) for utterance in utterances)
    unique_words = len(set(word for utterance in utterances for word in utterance.split()))
    word_diversity_score = unique_words / total_words if total_words > 0 else 0
    return word_diversity_score


def calculate_token_overlap(current_turn: str, previous_turn: str) -> Union[float, None]:
    if current_turn and previous_turn:
        current_tokens = set(current_turn.split())
        previous_tokens = set(previous_turn.split())
        return len(current_tokens.intersection(previous_tokens)) / len(current_tokens)
    else:
        return 0


def clean_wake_words_from_text(text: str) -> str:
    # cleans all instances of str in the sentence (not only the beginning)
    list_of_wake_words = ["alexa", "ziggy", "echo", "astro"]
    for wake_word in list_of_wake_words:
        text = text.replace(wake_word, "").strip()
    if text.startswith("computer"):  # computer is another wake word
        text = text.replace("computer", "", 1).strip()  # only removes the first ocurrence
    if text.startswith("amazon"):
        text = text.replace("amazon", "", 1).strip()
    return text


def lowercase_and_remove_punctuation_user_utterance(text: str):
    text = text.lower()
    # replace !, , and ?, ;
    text = text.replace("!", "").replace(",", "").replace("?", "").replace(";", "")

    # if ends with a period we remove it (only remove from the end to avoid removing from abbreviations)
    if text.endswith("."):
        text = text[:-1]
    return text


# keep only the ones that make sense
CONSIDERED_INTENTS = [
    Intents.AMAZONFallbackIntent,
    Intents.AMAZONNoIntent,
    # Intents.AMAZONPauseIntent,  # commented because it always transitions to None
    Intents.AMAZONPreviousIntent,
    Intents.AMAZONRepeatIntent,
    # Intents.AMAZONSelectIntent,
    Intents.AMAZONYesIntent,
    Intents.CommonChitChatIntent,
    Intents.GetCuriositiesIntent,
    # Intents.GoToStepIntent,
    Intents.IdentifyProcessIntent,
    # Intents.IngredientsConfirmationIntent,
    # Intents.LaunchRequestIntent,
    # Intents.MoreDetailIntent,
    Intents.NextStepIntent,
    # Intents.NumberIntent,
    # Intents.PlayMusicIntent,
    Intents.PreviousStepIntent,
    Intents.QuestionIntent,
    Intents.ResumeTaskIntent,
    Intents.CompleteTaskIntent,
    Intents.IngredientsReplacementIntent,
    # Intents.StartStepsIntent,
    # Intents.UserEvent,
    Intents.AMAZONStopIntent,

    # artificial intents (not in the bot originally)
    # ArtificialIntents.DefinitionQuestionIntent,
    # ArtificialIntents.SensitiveIntent,
]
