import os.path
from typing import Union, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset_generation.utils import load_json_file, write_to_json_file, clean_wake_words_from_text


class SequenceClassificationModel:

    def __init__(self, model_name: Union[str, None] = "model name here",
                 scores_cache: Union[Dict[str, float], Dict[str, Dict[str, float]]] = None,
                 default_value: float = 0.5, round_places: Union[int, None] = 2,
                 ):

        self.model_name = model_name
        self.scores_cache = scores_cache
        self.default_value = default_value
        self.round_places = round_places
        self.model = None
        self.tokenizer = None

    def calculate_value_function(self, text: str):
        # analyze
        value = self.analyze(text)
        if isinstance(value, dict):
            value = self.convert_to_0_1_scale(value)

        if self.round_places is not None:
            value = round(value, self.round_places)

        return value

    def analyze(self, utterance):

        # check cache
        if self.scores_cache is not None and utterance in self.scores_cache:
            return self.scores_cache[utterance]

        if not self.model_name:
            return self.default_value

        # load the first time it needs to analyze
        if self.model is None:
            self.load_model()

        # analyze
        with torch.no_grad():
            inputs = self.tokenizer(utterance, return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # get id2label
        id2label = self.model.config.id2label

        value = {}
        if id2label:
            for i, j in zip(id2label.values(), probs[0]):
                value[i] = j.item()

        # save to cache
        if self.scores_cache is None:
            self.scores_cache = {}
        self.scores_cache[utterance] = value

        return value

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Finished Loading: {self.model_name}")

        # move to gpu
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def convert_to_0_1_scale(value: Dict[str, float]):
        # convert to 0 to 1 scale the scores in already_seen
        current_score = 0
        for i, (label, score) in enumerate(value.items()):
            current_score += score * i

        current_score = current_score / (len(value) - 1)

        return current_score


emotion_model = SequenceClassificationModel(
    scores_cache=load_json_file("./data/emotion_cache.json") if os.path.exists("./data/emotion_cache.json") else None,
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    default_value=0.5,
    round_places=2,
)


fluency_model = SequenceClassificationModel(
    scores_cache=load_json_file("./data/fluency_cache.json") if os.path.exists("./data/fluency_cache.json") else None,
    model_name="gchhablani/bert-base-cased-finetuned-cola",
    default_value=0.7,
    round_places=2,
)


def calc_cache(collected_utterances_file: str):
    # calculates the cache for the emotion and fluency traits

    collected_utterances = load_json_file(collected_utterances_file)

    utterances_set = set()
    for intent, utterances_dict in collected_utterances.items():
        for utterance, count in utterances_dict.items():
            utterances_set.add(utterance)
            utterances_set.add(clean_wake_words_from_text(utterance))  # also add the utterance without the wake words

    for utterance in tqdm(utterances_set):
        emotion_model.analyze(utterance)

    write_to_json_file("./data/emotion_cache.json", emotion_model.scores_cache)

    for utterance in tqdm(utterances_set):
        fluency_model.analyze(utterance)
        utterances_set.add(clean_wake_words_from_text(utterance))  # also add the utterance without the wake words

    write_to_json_file("./data/fluency_cache.json", fluency_model.scores_cache)
