import copy
from typing import Dict, List, Union

import numpy as np

from data_binding.enumerates import Intents
from dataset_generation.dialog import Dialog
from dataset_generation.utils import apply_smoothing, calculate_token_overlap
from user_simulator.trait_analysis_models.classifier_models import SequenceClassificationModel
from user_simulator.traits_and_profiles.utils import get_by_top_p, get_by_top_k, get_values_in_percentage_range, \
    get_list_in_percentage_range, nltk_stopwords


class EscalationTrait:

    trait_name = None
    user_trait_name = None

    def __init__(self, factor: float = 1.0):
        self.factor = factor
        self.considered_intents = None

    def apply_escalation_to_distribution(self, distribution: Dict[str, Dict[str, float]]):
        normalized_distribution = copy.deepcopy(distribution)
        for current_intent, intent_distribution in distribution.items():

            for intent, value in intent_distribution.items():
                if intent in self.considered_intents:
                    normalized_distribution[current_intent][intent] *= self.factor

            # Normalize the probabilities to ensure they sum up to 1
            total_probability = sum(normalized_distribution[current_intent].values())
            normalized_distribution[current_intent] = {intent: prob / total_probability for intent, prob in
                                                       normalized_distribution[current_intent].items()}

        return normalized_distribution


class ImpatienceEscalationTrait(EscalationTrait):

    trait_name = "patience"
    user_trait_name = "patient"

    def __init__(self, impatience_factor: float = 1.0):
        super().__init__(factor=impatience_factor)
        self.considered_intents = Intents.impatience_escalation_intents()


class IntoleranceEscalationTrait(EscalationTrait):

    trait_name = "tolerance"
    user_trait_name = "tolerant"

    def __init__(self, tolerance_factor: float = 1.1):
        super().__init__(factor=tolerance_factor)
        self.considered_intents = Intents.intolerance_escalation_intents()


# multiplies non-top intents by factor
class ExplorationTraitV1(EscalationTrait):

    trait_name = "exploration"
    user_trait_name = "explorative"

    def __init__(self, explorative_factor: float = 1.0, top_p: Union[None, float] = 0.5):
        super().__init__(factor=explorative_factor)
        self.top_p = top_p

    def apply_escalation_to_distribution(self, distribution: Dict[str, Dict[str, float]] = None):

        normalized_distribution = copy.deepcopy(distribution)

        for current_intent, intent_distribution in distribution.items():

            # get the top-p intents
            if self.top_p is not None:
                top_p_intents = get_by_top_p(top_p=self.top_p, counts_dict=intent_distribution)
            else:  # consider as top_intent the max value
                top_p_intents = get_by_top_k(top_k=1, counts_dict=intent_distribution)

            # min value in top intents
            min_value = min(top_p_intents.values())

            for intent, value in intent_distribution.items():
                if intent not in top_p_intents and intent not in Intents.non_explorative_intents():
                    new_value = normalized_distribution[current_intent][intent] * self.factor
                    new_value = min(new_value, min_value)  # do not increase more than the min value in top intents
                    normalized_distribution[current_intent][intent] = new_value

            # Normalize the probabilities to ensure they sum up to 1
            total_probability = sum(normalized_distribution[current_intent].values())
            normalized_distribution[current_intent] = {intent: prob / total_probability for intent, prob in
                                                       normalized_distribution[current_intent].items()}
            pass
        return normalized_distribution


# distribs part of the probability of top intents by the others
class ExplorationTraitV2(EscalationTrait):

    trait_name = "exploration"
    user_trait_name = "explorative"

    # exploration factor is not really used here
    def __init__(self, increase_exploration: bool, top_p: Union[None, float] = 0.5, distribution_factor: float = 0.2):
        super().__init__(factor=None)
        self.top_p = top_p
        self.distribution_factor = distribution_factor

        # if true it increases prob of explorative else takes from others to give to top intents
        self.increase_exploration = increase_exploration

    def apply_escalation_to_distribution(self, distribution: Dict[str, Dict[str, float]] = None):
        normalized_distribution = copy.deepcopy(distribution)

        # do nothing if we do not want to redistribute (general user)
        if not self.distribution_factor:
            return normalized_distribution

        for base_intent, intent_distribution in distribution.items():

            # put intent distribution in zero to one scale
            total_probability = sum(normalized_distribution[base_intent].values())
            normalized_distribution[base_intent] = {intent: prob / total_probability for intent, prob in
                                                    intent_distribution.items()}

            # get the top-p intents
            if self.top_p is not None:
                top_p_intents = get_by_top_p(top_p=self.top_p, counts_dict=normalized_distribution[base_intent])
            else:  # consider as top_intent the max value
                top_p_intents = get_by_top_k(top_k=1, counts_dict=normalized_distribution[base_intent])

            if not self.increase_exploration:
                # get the contrary of the top-p intents not counting the non-explorative intents
                top_p_intents = {intent: prob for intent, prob in normalized_distribution[base_intent].items() if intent not in top_p_intents and intent not in Intents.non_explorative_intents()}

            # redistribute the probability of the top-p intents to the other intents
            p_top_intents, p_explorative_intents = 0, 0
            for intent, value in normalized_distribution[base_intent].items():
                if intent in top_p_intents:
                    p_top_intents += value
                else:
                    if intent not in Intents.non_explorative_intents():
                        p_explorative_intents += value

            p_to_distrib = p_top_intents * self.distribution_factor

            # redistribute by the others
            for intent, value in normalized_distribution[base_intent].items():
                if intent in top_p_intents:
                    # remove prob from top intents
                    normalized_distribution[base_intent][intent] -= p_to_distrib * (value / p_top_intents)
                else:
                    # increase prob of explorative intents
                    if intent not in Intents.non_explorative_intents():
                        normalized_distribution[base_intent][intent] += p_to_distrib * (value / p_explorative_intents)
                    else:
                        # do not change prob of the others
                        pass

            # put again in the same scale as before
            total_probability = sum(normalized_distribution[base_intent].values())
            normalized_distribution[base_intent] = {intent: prob / total_probability for intent, prob in
                                                    normalized_distribution[base_intent].items()}

        return normalized_distribution


class UnCooperativenessTrait(EscalationTrait):

    trait_name = "cooperativeness"
    user_trait_name = "cooperative"

    def __init__(self, uncooperativeness_factor: float = 1.0):
        super().__init__(uncooperativeness_factor)
        self.considered_intents = Intents.non_cooperative_intents()


class GeneralTrait:

    trait_name = None
    user_trait_name = None

    def __init__(self, top_p: float = 0.0, bottom_p: float = 1.0,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 consider_distrib_by_value: bool = False):
        self.top_p = top_p
        self.bottom_p = bottom_p
        self.use_smoothing_alpha = use_smoothing_alpha
        self.smoothing_alpha = smoothing_alpha
        self.consider_distrib_by_value = consider_distrib_by_value
        self.top_k = None  # ignored for now

    @staticmethod
    def get_intervals(max_value: int, number_intervals: int):
        # get equally sized spaces for the intervals
        spaces = np.linspace(start=0, stop=max_value, num=number_intervals + 1)

        # create the intervals
        intervals = []
        for i in range(len(spaces) - 1):
            intervals.append((spaces[i], spaces[i + 1]))

        print(intervals)
        return intervals

    def get_new_distribution(self, current_dialogue: Dialog, current_intent: str, utterance_counts: Dict[str, int]):

        if not self.top_k and (self.top_p is None or self.bottom_p is None):
            raise ValueError("At least one of the parameters must be set")
        if self.top_k and (self.top_p or self.bottom_p):
            raise ValueError("Only one of the parameters must be set")

        # account for frequency and consiseness

        # measure consiness in number of words
        utterance_value_dict = {}
        unique_values = set()
        for utterance, count in utterance_counts.items():
            value = self.calculate_value_function(text=utterance)
            utterance_value_dict[utterance] = value

            if self.consider_distrib_by_value:
                unique_values.add(value)

        values = None
        # get the top k or top p by consiness
        if self.top_k:
            values = get_by_top_k(self.top_k, utterance_value_dict)

        if self.top_p is not None and self.bottom_p is not None:
            assert 0 <= self.bottom_p <= 1
            assert 0 <= self.top_p <= 1
            assert self.bottom_p <= self.top_p

            if self.consider_distrib_by_value:
                values_considered = get_list_in_percentage_range(unique_values, bottom_percentage=self.bottom_p,
                                                                 top_percentage=self.top_p)
                # get utterances that have the values in values_considered
                values = {k: v for k, v in utterance_value_dict.items() if v in values_considered}
            else:
                values = get_values_in_percentage_range(utterance_value_dict,
                                                        bottom_percentage=self.bottom_p, top_percentage=self.top_p)

        if not values:
            raise ValueError("No values were found")

        criteria_values = self.apply_smoothing_distribution(utterance_counts, values)

        return criteria_values

    def apply_smoothing_distribution(self, utterance_counts: Dict[str, int], values: Dict[str, int]):
        # get the corresponding values from the utterance counts
        criteria_values = {}
        for utterance, count in values.items():
            criteria_values[utterance] = utterance_counts[utterance]
        if self.use_smoothing_alpha:
            values = list(criteria_values.values())
            counts = np.array(values)

            smoothing_alpha = self.smoothing_alpha
            if self.smoothing_alpha is None:
                smoothing_alpha = np.mean(values)

            for k, v in zip(criteria_values.keys(), apply_smoothing(counts=counts, alpha=smoothing_alpha).tolist()):
                criteria_values[k] = v
        return criteria_values

    def calculate_value_function(self, text: str):
        raise NotImplementedError


class UnconcisenessTrait(GeneralTrait):

    trait_name = "verbosity"
    user_trait_name = "verbose"

    def __init__(self, top_p: float = 0.0, bottom_p: float = 1.0,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 consider_distrib_by_value: bool = False):
        super().__init__(top_p=top_p, bottom_p=bottom_p, use_smoothing_alpha=use_smoothing_alpha,
                         smoothing_alpha=smoothing_alpha, consider_distrib_by_value=consider_distrib_by_value)

    def calculate_value_function(self, text: str):
        return len(text.split())


class ModelBasedTrait(GeneralTrait):

    def __init__(self, model: SequenceClassificationModel,
                 top_p: float = 0.0, bottom_p: float = 1.0,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 ):

        self.model = model
        super().__init__(top_p=top_p, bottom_p=bottom_p, use_smoothing_alpha=use_smoothing_alpha,
                         smoothing_alpha=smoothing_alpha, consider_distrib_by_value=False)

    def calculate_value_function(self, text: str):
        return self.model.calculate_value_function(text)


class EmotionTrait(ModelBasedTrait):

    trait_name = "emotion"
    user_trait_name = "positive"

    def __init__(self, model: SequenceClassificationModel,
                 top_p: float = 0.0, bottom_p: float = 1.0,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 ):
        super().__init__(model=model, top_p=top_p, bottom_p=bottom_p, use_smoothing_alpha=use_smoothing_alpha,
                         smoothing_alpha=smoothing_alpha)


class FluencyTrait(ModelBasedTrait):

    trait_name = "fluency"
    user_trait_name = "fluent"

    # DO NOT use error_rate != 0.0 since it changes the strings and it affects a bunch of other stuff
    def __init__(self, model: SequenceClassificationModel,
                 top_p: float = 0.0, bottom_p: float = 1.0,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 error_rate: float = 0.0):
        self.error_rate = error_rate
        super().__init__(model=model, top_p=top_p, bottom_p=bottom_p, use_smoothing_alpha=use_smoothing_alpha,
                         smoothing_alpha=smoothing_alpha)

    def get_new_distribution(self, current_dialogue: Dialog, current_intent: str, utterance_counts: Dict[str, int]):
        new_distribution = super().get_new_distribution(current_dialogue, current_intent, utterance_counts)

        if self.error_rate:
            distribution_with_errors = {}

            for utterance, count in new_distribution.items():
                utterance = self.add_erros(utterance, self.error_rate)
                distribution_with_errors[utterance] = count

            return distribution_with_errors
        else:
            return new_distribution

    @staticmethod
    def add_erros(utterance: str, prob_remove: float):
        # currently it adds errors by removing with prob_remove each stopword from the utterance
        words = utterance.split()
        new_utterance = ""
        for word in words:
            if word.lower() in nltk_stopwords and np.random.random() <= prob_remove:
                pass
            else:
                new_utterance += word + " "
        return new_utterance.strip()


class RepetitionTrait(GeneralTrait):

    trait_name = "repetition"
    user_trait_name = "repetitive"

    def __init__(self,
                 use_smoothing_alpha: bool = True, smoothing_alpha: Union[None, float] = None,
                 exact_match_p: Union[None, float] = None, overlap_match_p: Union[None, float] = None,
                 ignore_intents: List[str] = None):
        super().__init__(use_smoothing_alpha=use_smoothing_alpha, smoothing_alpha=smoothing_alpha,
                         consider_distrib_by_value=False)
        self.exact_match_p = exact_match_p
        self.overlap_match_p = overlap_match_p
        if ignore_intents is None:
            self.ignore_intents = []
        else:
            self.ignore_intents = ignore_intents

    def get_new_distribution(self, current_dialogue: Dialog, current_intent: str, utterance_counts: Dict[str, int]):
        exact_matches = {}
        overlap_matches = {}
        non_overlap_matches = {}
        overlap_macthes_counts = {}

        # if we do not want to consider repetition
        if self.exact_match_p is None and self.overlap_match_p is None:
            return self.apply_smoothing_distribution(utterance_counts, utterance_counts)

        # ignore repetitions for some intents
        if current_dialogue.turns and current_dialogue.turns[-1].intent in self.ignore_intents:
            return self.apply_smoothing_distribution(utterance_counts, utterance_counts)

        for turn in current_dialogue.turns:
            if turn.intent == current_intent:
                if turn.user_utterance in utterance_counts:
                    exact_matches[turn.user_utterance] = utterance_counts[turn.user_utterance]

                for utterance in utterance_counts.keys():
                    word_overlap = calculate_token_overlap(current_turn=utterance,
                                                           previous_turn=turn.user_utterance)
                    if word_overlap:
                        overlap_matches[utterance] = word_overlap
                        overlap_macthes_counts[utterance] = utterance_counts[utterance]
                    else:
                        non_overlap_matches[utterance] = utterance_counts[utterance]

        # avoid repeating by getting values at the end of the matches
        if len(overlap_matches) > 0 and self.exact_match_p == 0 and self.overlap_match_p == 0:
            # get one with no overlap
            if len(non_overlap_matches) > 0:
                return self.apply_smoothing_distribution(utterance_counts, non_overlap_matches)
            else:
                lower_bounds = get_values_in_percentage_range(
                    data=overlap_matches, bottom_percentage=0.0, top_percentage=0.2
                )
                # return the ones in the lower bounds with the values of overlap_macthes_counts
                distrib = {k: overlap_macthes_counts[k] for k, v in lower_bounds.items() if k in overlap_macthes_counts}
                return self.apply_smoothing_distribution(utterance_counts, distrib)

        # with a percent chance we choose from the exact matches
        selected_matches = utterance_counts
        if len(exact_matches) > 0 and np.random.random() <= self.exact_match_p:
            selected_matches = exact_matches

        # otherwise with a percent we choose from the overlap matches
        if len(overlap_matches) > 0 and np.random.random() <= self.overlap_match_p:
            selected_matches = overlap_macthes_counts

        # otherwise we choose from the utterance counts
        return self.apply_smoothing_distribution(utterance_counts, selected_matches)
