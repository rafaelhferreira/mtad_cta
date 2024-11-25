import copy
import json
from enum import Enum
from typing import Dict, Union, List

from user_simulator.trait_analysis_models.classifier_models import emotion_model, fluency_model
from user_simulator.traits_and_profiles.user_traits import ImpatienceEscalationTrait, IntoleranceEscalationTrait, \
    UnCooperativenessTrait, UnconcisenessTrait, EmotionTrait, FluencyTrait, RepetitionTrait, ExplorationTraitV2


class ProfileLevels(Enum):
    dialogue = "dialogue"  # affects intent trasitions
    utterance = "utterance"  # affects utterance generation
    mixed = "mixed"  # affects both intent trasitions and utterance generation


# Basic trait meanings:
# patience - increases or decreases the probability of stop intent in the distributions dict
# cooperativeness - increases the probability of fallback intents (and responses), sensitive or non-contextual requests (music, lights, weather, other skill, etc)
# exploration - increases the probability of other intents (can use top_p or just increase non-majority intents) (except stop and fallback)
# verbosity - increases the probability of choosing user utterances with more words
# emotion - increases the probability of choosing user utterances with more positive sentiment
# fluency - increases the probability of choosing user utterances with more fluency (also check or artificialy add problems to the utterances)
# repetition - increases the probability of repeating the same utterance (in or not in consecutive intents)
# tolerance - increases the probability of stop intent in the distributions dict when a fallback is given by the system (worse after each error)


# Scales meanings:
# 0-2 - not trait - average - very trait


class UserProfile:

    def __init__(self, user_profile_dict: Union[Dict[str, Union[float, int, str, Dict]], None] = None):

        self.user_custom_name = user_profile_dict.get("user_custom_name", None) if user_profile_dict else None
        self.order = user_profile_dict.get("order", 0.0) if user_profile_dict else 0.0
        assert "_" not in self.user_custom_name, "user_custom_name should not contain underscores"

        self.impatience = user_profile_dict.get("impatience", 1.0) if user_profile_dict else 1.0
        self.min_number_turns = user_profile_dict.get("min_number_turns", 3) if user_profile_dict else 3
        self.uncooperativeness = user_profile_dict.get("uncooperativeness", 1.0) if user_profile_dict else 1.0

        # V1 version of exploration
        # self.exploration_factor = user_profile_dict.get("exploration_factor", 1.0) if user_profile_dict else 1.0
        # self.top_p_exploration = user_profile_dict.get("top_p_exploration", 0.001) if user_profile_dict else 0.001

        # V2 version of exploration
        self.top_p_exploration = user_profile_dict.get("top_p_exploration", 0.001) if user_profile_dict else 0.001
        self.increase_exploration = user_profile_dict.get("increase_exploration", True) if user_profile_dict else True
        self.exploration_distrib_factor = user_profile_dict.get("exploration_distrib_factor",
                                                                0.0) if user_profile_dict else 0.0

        self.intolerance = user_profile_dict.get("intolerance", 3.0) if user_profile_dict else 3.0
        self.bottom_unconciseness = user_profile_dict.get("bottom_unconciseness", 0.0) if user_profile_dict else 0.0
        self.top_unconciseness = user_profile_dict.get("top_unconciseness", 1.0) if user_profile_dict else 1.0
        self.bottom_emotion = user_profile_dict.get("bottom_emotion", 0.0) if user_profile_dict else 0.0
        self.top_emotion = user_profile_dict.get("top_emotion", 1.0) if user_profile_dict else 1.0
        self.bottom_fluency = user_profile_dict.get("bottom_fluency", 0.0) if user_profile_dict else 0.0
        self.top_fluency = user_profile_dict.get("top_fluency", 1.0) if user_profile_dict else 1.0
        self.fluency_error_rate = user_profile_dict.get("fluency_error_rate", 0.0) if user_profile_dict else 0.0
        self.repetition_exact_match_p = user_profile_dict.get("repetition_exact_match_p",
                                                              0.15) if user_profile_dict else 0.15
        self.repetition_overlap_match_p = user_profile_dict.get("repetition_overlap_match_p",
                                                                0.15) if user_profile_dict else 0.15

        # escalation traits
        self.impatience_trait = ImpatienceEscalationTrait(impatience_factor=self.impatience)
        self.intolerance_trait = IntoleranceEscalationTrait(tolerance_factor=self.intolerance)

        # V1 version of exploration
        '''
        self.exploration_trait = ExplorationTraitV1(explorative_factor=self.exploration_factor,
                                                    top_p=self.top_p_exploration)
        '''

        # V2 version of exploration
        self.exploration_trait = ExplorationTraitV2(increase_exploration=self.increase_exploration,
                                                    distribution_factor=self.exploration_distrib_factor,
                                                    top_p=self.top_p_exploration)

        self.uncooperativeness_trait = UnCooperativenessTrait(uncooperativeness_factor=self.uncooperativeness)

        # utterance level traits
        self.unconciseness_trait = UnconcisenessTrait(bottom_p=self.bottom_unconciseness, top_p=self.top_unconciseness,
                                                      consider_distrib_by_value=True)
        self.emotion_trait = EmotionTrait(model=emotion_model,
                                          bottom_p=self.bottom_emotion, top_p=self.top_emotion)
        self.fluency_trait = FluencyTrait(model=fluency_model,
                                          bottom_p=self.bottom_fluency, top_p=self.top_fluency,
                                          error_rate=self.fluency_error_rate)
        self.repetition_trait = RepetitionTrait(exact_match_p=self.repetition_exact_match_p,
                                                overlap_match_p=self.repetition_overlap_match_p)

        # trait scale is a dict in format {trait_name: trait_value} where trait_value is a int between 0 and 2
        self.trait_scale = user_profile_dict.get("trait_scale", None) if user_profile_dict else None

        self.user_profile_description = user_profile_dict.get("user_profile_description",
                                                              "") if user_profile_dict else ""

        self.profile_level = user_profile_dict.get("profile_level", None) if user_profile_dict else None
        self.is_multitrait = user_profile_dict.get("is_multitrait", False) if user_profile_dict else False

    def to_dict(self):
        return {
            "user_custom_name": self.user_custom_name,
            "impatience": self.impatience,
            "min_number_turns": self.min_number_turns,
            "uncooperativeness": self.uncooperativeness,
            # "exploration_distrib_min_prob": self.exploration_distrib_min_prob,
            "top_p_exploration": self.top_p_exploration,
            "increase_exploration": self.increase_exploration,
            "exploration_distrib_factor": self.exploration_distrib_factor,
            "intolerance": self.intolerance,
            "bottom_unconciseness": self.bottom_unconciseness,
            "top_unconciseness": self.top_unconciseness,
            "bottom_emotion": self.bottom_emotion,
            "top_emotion": self.top_emotion,
            "bottom_fluency": self.bottom_fluency,
            "top_fluency": self.top_fluency,
            "fluency_error_rate": self.fluency_error_rate,
            "repetition_exact_match_p": self.repetition_exact_match_p,
            "repetition_overlap_match_p": self.repetition_overlap_match_p,
            "trait_scale": self.trait_scale,
            "user_profile_description": self.user_profile_description,
            "profile_level": self.profile_level,
            "is_multitrait": self.is_multitrait,
        }

    def __repr__(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        # Define a hash based on the value attribute
        return hash(self.__str__())

    def __eq__(self, other):
        # Define equality based on the value attribute
        return isinstance(other, UserProfile) and self.__str__() == other.__str__()

    def get_escalation_traits(self):
        # intolerance is not here since it is a special case applied depending on the current dialog state
        return [self.impatience_trait, self.exploration_trait, self.uncooperativeness_trait]

    def get_utterance_traits(self):
        # IMPORTANT WARNING
        # the order in which the traits are applied influencess final result
        # e.g. apply fluency -> conciseness != apply conciseness -> fluency
        # because way we get the utterances from top_p and bottom_p distributions
        return [self.unconciseness_trait, self.emotion_trait, self.fluency_trait, self.repetition_trait]

    def apply_escalation_traits(self, probs_dict: Dict[str, Dict[str, float]]):
        new_probs_dict = copy.deepcopy(probs_dict)
        for escalation_trait in self.get_escalation_traits():
            new_probs_dict = escalation_trait.apply_escalation_to_distribution(new_probs_dict)
        return new_probs_dict

    from dataset_generation.dialog import Dialog

    def apply_utterance_traits(self, current_dialog: Union[Dialog, None], current_intent: Union[str, None],
                               probs_dict: Dict[str, Union[int, float]]):
        new_probs_dict = copy.deepcopy(probs_dict)
        for utterance_trait in self.get_utterance_traits():
            new_probs_dict = utterance_trait.get_new_distribution(
                current_dialogue=current_dialog,
                current_intent=current_intent,
                utterance_counts=new_probs_dict
            )
        return new_probs_dict

    def get_trait_scale(self):
        trait_scale = []
        for trait in self.get_escalation_traits():
            trait_scale.append((trait, self.get_trait_value(trait.trait_name)))
        for trait in self.get_utterance_traits():
            trait_scale.append((trait, self.get_trait_value(trait.trait_name)))
        return trait_scale

    def get_trait_value(self, trait: str):
        if self.trait_scale:
            return self.trait_scale.get(trait, 1)  # 1 is the default value since we are using 0-2 scales
        else:
            return 1


class UserTypes:
    RegularUser = UserProfile(
        dict(user_custom_name="Regular", order=0.0, profile_level=None))

    PatientUser = UserProfile(
        dict(user_custom_name="Patient", impatience=0.5, min_number_turns=4,
             trait_scale={ImpatienceEscalationTrait.trait_name: 2}, order=1.0,
             user_profile_description="A patient user demonstrates a willingness to engage with the system over an "
                                      "extended period.",
             profile_level=ProfileLevels.dialogue.value))
    ImpatientUser = UserProfile(
        dict(user_custom_name="Impatient", impatience=2.0, min_number_turns=2,
             trait_scale={ImpatienceEscalationTrait.trait_name: 0}, order=1.1,
             user_profile_description="An impatient user seeks quick and concise interactions with the system often "
                                      "finishing the interaction before the end of the task.",
             profile_level=ProfileLevels.dialogue.value))

    # tolerance is a special case since it multiplies the probability of stop intent each time a fallback intent occurs
    TolerantUser = UserProfile(
        dict(intolerance=1.0, user_custom_name="Tolerant",
             trait_scale={IntoleranceEscalationTrait.trait_name: 2}, order=4.0,
             user_profile_description="A tolerant user is willing to give the system multiple chances to recover from "
                                      "errors.",
             profile_level=ProfileLevels.dialogue.value))
    IntolerantUser = UserProfile(
        dict(intolerance=10.0, user_custom_name="Intolerant",
             trait_scale={IntoleranceEscalationTrait.trait_name: 0}, order=4.1,
             user_profile_description="An intolerant user has little patience for system mistakes and tends to "
                                      "conclude the interaction sooner when errors occur.",
             profile_level=ProfileLevels.dialogue.value))

    ExplorativeUser = UserProfile(
        dict(top_p_exploration=0.01, increase_exploration=True,
             exploration_distrib_factor=0.2, user_custom_name="Explorative",
             trait_scale={ExplorationTraitV2.trait_name: 2}, order=3.0,
             user_profile_description="An explorative user is curious and seeks to explore the system's capabilities "
                                      "and the task at hand.",
             profile_level=ProfileLevels.dialogue.value))
    NonExplorativeUser = UserProfile(
        dict(top_p_exploration=0.01, increase_exploration=False,
             exploration_distrib_factor=0.2, user_custom_name="NonExplorative",
             trait_scale={ExplorationTraitV2.trait_name: 0}, order=3.1,
             user_profile_description="A non-explorative shows little interest in engaging with different system's "
                                      "features besides moving forward with the task.",
             profile_level=ProfileLevels.dialogue.value))

    CooperativeUser = UserProfile(
        dict(uncooperativeness=0.5, user_custom_name="Cooperative",
             trait_scale={UnCooperativenessTrait.trait_name: 2}, order=2.0,
             user_profile_description="A cooperative user follows the system's instructions and engages with the "
                                      "system to accomplish the task effectively.",
             profile_level=ProfileLevels.dialogue.value))
    UnCooperativeUser = UserProfile(
        dict(uncooperativeness=2.0, user_custom_name="UnCooperative",
             trait_scale={UnCooperativenessTrait.trait_name: 0}, order=2.1,
             user_profile_description="An uncooperative user is less likely to follow the system's instructions and "
                                      "may provide irrelevant or unhelpful requests.",
             profile_level=ProfileLevels.dialogue.value))

    ConciseUser = UserProfile(
        dict(bottom_unconciseness=0.0, top_unconciseness=0.5, user_custom_name="Concise",
             trait_scale={UnconcisenessTrait.trait_name: 0}, order=5.1,
             user_profile_description="A concise user communicates using short requests, using minimal words to "
                                      "convey their requests to the system.",
             profile_level=ProfileLevels.utterance.value))
    VerboseUser = UserProfile(
        dict(bottom_unconciseness=0.5, top_unconciseness=1.0, user_custom_name="Verbose",
             trait_scale={UnconcisenessTrait.trait_name: 2}, order=5.0,
             user_profile_description="A verbose user communicates using long requests, using many words to convey "
                                      "their requests to the system.",
             profile_level=ProfileLevels.utterance.value))

    PositiveUser = UserProfile(
        dict(bottom_emotion=0.5, top_emotion=1.0, user_custom_name="Positive",
             trait_scale={EmotionTrait.trait_name: 2}, order=6.0,
             user_profile_description="A positive user communicates with a positive sentiment, expressing "
                                      "satisfaction and enthusiasm.",
             profile_level=ProfileLevels.utterance.value))
    NegativeUser = UserProfile(
        dict(bottom_emotion=0.0, top_emotion=0.5, user_custom_name="Negative",
             trait_scale={EmotionTrait.trait_name: 0}, order=6.1,
             user_profile_description="A negative user communicates with a negative sentiment, expressing "
                                      "dissatisfaction and frustration.",
             profile_level=ProfileLevels.utterance.value))

    FluentUser = UserProfile(
        dict(bottom_fluency=0.5, top_fluency=1.0, user_custom_name="Fluent",
             trait_scale={FluencyTrait.trait_name: 2}, order=7.0,
             user_profile_description="A fluent user communicates with a high level of fluency, using proper grammar "
                                      "or vocabulary.",
             profile_level=ProfileLevels.utterance.value))
    NonFluentUser = UserProfile(
        dict(bottom_fluency=0.0, top_fluency=0.5, fluency_error_rate=0.0, user_custom_name="NonFluent",
             trait_scale={FluencyTrait.trait_name: 0}, order=7.1,
             user_profile_description="A non-fluent user communicates with a low level of fluency, using improper "
                                      "grammar or vocabulary.",
             profile_level=ProfileLevels.utterance.value))

    RepetitiveUser = UserProfile(
        dict(repetition_exact_match_p=1.0, repetition_overlap_match_p=1.0, user_custom_name="Repetitive",
             trait_scale={RepetitionTrait.trait_name: 2}, order=8.0,
             user_profile_description="A repetitive user tends to communicate using the same vocabulary or phrases "
                                      "for similar actions.",
             profile_level=ProfileLevels.utterance.value))
    NonRepetitiveUser = UserProfile(
        dict(repetition_exact_match_p=0.0, repetition_overlap_match_p=0.0, user_custom_name="NonRepetitive",
             trait_scale={RepetitionTrait.trait_name: 0}, order=8.1,
             user_profile_description="A non-repetitive user tends to communicate using different vocabulary or "
                                      "phrases for similar actions.",
             profile_level=ProfileLevels.utterance.value))

    # MULTI TRAIT PROFILES
    PatientVerboseUser = UserProfile(
        dict(user_custom_name="MultiPatientVerbose", impatience=0.5, min_number_turns=4,
             trait_scale={UnconcisenessTrait.trait_name: 2,
                          ImpatienceEscalationTrait.trait_name: 2},
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=10.1,
             user_profile_description="A patient verbose user demonstrates a willingness to engage with the system "
                                      "over an extended period and communicates using long requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    PatientConciseUser = UserProfile(
        dict(user_custom_name="MultiPatientConcise", impatience=0.5, min_number_turns=4,
             trait_scale={UnconcisenessTrait.trait_name: 0,
                          ImpatienceEscalationTrait.trait_name: 2},
             bottom_unconciseness=0.0, top_unconciseness=0.5, order=10.2,
             user_profile_description="A patient concise user demonstrates a willingness to engage with the system "
                                      "over an extended period and communicates using short requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    ImpatientVerboseUser = UserProfile(
        dict(user_custom_name="MultiImpatientVerbose", impatience=2.0, min_number_turns=2,
             trait_scale={UnconcisenessTrait.trait_name: 2, ImpatienceEscalationTrait.trait_name: 0},
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=10.3,
             user_profile_description="An impatient verbose user seeks quick and concise interactions with the system "
                                      "often finishing the interaction before the end of the task and communicates "
                                      "using long requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    ImpatientConciseUser = UserProfile(
        dict(user_custom_name="MultiImpatientConcise", impatience=2.0, min_number_turns=2,
             trait_scale={UnconcisenessTrait.trait_name: 0, ImpatienceEscalationTrait.trait_name: 0},
             bottom_unconciseness=0.0, top_unconciseness=0.5, order=10.4,
             user_profile_description="An impatient concise user seeks quick and concise interactions with the system "
                                      "often finishing the interaction before the end of the task and communicates "
                                      "using short requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    # some more examples
    CooperativeNonFluentUser = UserProfile(
        dict(user_custom_name="MultiCooperativeNonFluent", uncooperativeness=0.5, bottom_fluency=0.0, top_fluency=0.5,
             trait_scale={UnCooperativenessTrait.trait_name: 2, FluencyTrait.trait_name: 0},
             order=10.5,
             user_profile_description="A cooperative non-fluent user follows the system's instructions and engages "
                                      "with the system to accomplish the task effectively and communicates with a low "
                                      "level of fluency, using improper grammar or vocabulary.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    VerboseFluentUser = UserProfile(
        dict(user_custom_name="MultiVerboseFluent", bottom_fluency=0.5, top_fluency=1.0,
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=10.6,
             trait_scale={FluencyTrait.trait_name: 2, UnconcisenessTrait.trait_name: 2},
             user_profile_description="A verbose fluent user communicates using long requests, using many words to "
                                      "convey their requests to the system and communicates with a high level of "
                                      "fluency, using proper grammar or vocabulary.",
             profile_level=ProfileLevels.utterance.value,
             is_multitrait=True))  # here trait application order matters

    # more profiles
    ExplorativeImpatientUser = UserProfile(
        dict(user_custom_name="MultiExplorativeImpatient", top_p_exploration=0.01, increase_exploration=True,
             exploration_distrib_factor=0.2, impatience=2.0, min_number_turns=2,
             trait_scale={ExplorationTraitV2.trait_name: 2, ImpatienceEscalationTrait.trait_name: 0},
             order=10.7,
             user_profile_description="An explorative impatient user is curious and seeks to explore the system's "
                                      "capabilities and the task at hand but seeks quick and concise interactions "
                                      "with the system often finishing the interaction before the end of the task.",
             profile_level=ProfileLevels.dialogue.value, is_multitrait=True))

    ExplorativeCooperativeUser = UserProfile(
        dict(user_custom_name="MultiExplorativeCooperative", top_p_exploration=0.01, increase_exploration=True,
             exploration_distrib_factor=0.2, uncooperativeness=0.5,
             trait_scale={ExplorationTraitV2.trait_name: 2, UnCooperativenessTrait.trait_name: 2},
             order=10.8,
             user_profile_description="An explorative cooperative user is curious and seeks to explore the system's "
                                      "capabilities and the task at hand and follows the system's instructions and "
                                      "engages with the system to accomplish the task effectively.",
             profile_level=ProfileLevels.dialogue.value, is_multitrait=True))

    FluentRepetitiveUser = UserProfile(
        dict(user_custom_name="MultiFluentRepetitive", bottom_fluency=0.5, top_fluency=1.0,
             repetition_exact_match_p=1.0, repetition_overlap_match_p=1.0, order=10.9,
             trait_scale={FluencyTrait.trait_name: 2, RepetitionTrait.trait_name: 2},
             user_profile_description="A fluent repetitive user communicates with a high level of fluency, "
                                      "using proper grammar or vocabulary and tends to communicate using the same "
                                      "vocabulary or phrases for similar actions.",
             profile_level=ProfileLevels.utterance.value, is_multitrait=True))

    VerbosePositiveUser = UserProfile(
        dict(user_custom_name="MultiVerbosePositive", bottom_emotion=0.5, top_emotion=1.0,
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=11.0,
             trait_scale={EmotionTrait.trait_name: 2, UnconcisenessTrait.trait_name: 2},
             user_profile_description="A verbose positive user communicates using long requests, using many words to "
                                      "convey their requests to the system and communicates with a positive "
                                      "sentiment, expressing satisfaction and enthusiasm.",
             profile_level=ProfileLevels.utterance.value, is_multitrait=True))

    ExplorativeConciseUser = UserProfile(
        dict(user_custom_name="MultiExplorativeConcise", top_p_exploration=0.01, increase_exploration=True,
             exploration_distrib_factor=0.2, bottom_unconciseness=0.0, top_unconciseness=0.5, order=11.1,
             trait_scale={ExplorationTraitV2.trait_name: 2, UnconcisenessTrait.trait_name: 0},
             user_profile_description="An explorative concise user is curious and seeks to explore the system's "
                                      "capabilities and the task at hand and communicates using short requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    UncooperativeNonFluentUser = UserProfile(
        dict(user_custom_name="MultiUncooperativeNonFluent", uncooperativeness=2.0, bottom_fluency=0.0, top_fluency=0.5,
             trait_scale={UnCooperativenessTrait.trait_name: 0, FluencyTrait.trait_name: 0},
             order=11.2,
             user_profile_description="An uncooperative non-fluent user is less likely to follow the system's "
                                      "instructions and may provide irrelevant or unhelpful requests and communicates "
                                      "with a low level of fluency, using improper grammar or vocabulary.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    # combine 3 traits
    ImpatientConciseNegativeUser = UserProfile(
        dict(user_custom_name="MultiImpatientConciseNegative", impatience=2.0, min_number_turns=2,
             bottom_emotion=0.0, top_emotion=0.5, bottom_unconciseness=0.0, top_unconciseness=0.5, order=11.3,
             trait_scale={ImpatienceEscalationTrait.trait_name: 0, EmotionTrait.trait_name: 0,
                          UnconcisenessTrait.trait_name: 0},
             user_profile_description="An impatient concise negative user seeks quick and concise interactions with "
                                      "the system often finishing the interaction before the end of the task and "
                                      "communicates with a negative sentiment, expressing dissatisfaction and "
                                      "frustration.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    CooperativeFluentRepetitiveUser = UserProfile(
        dict(user_custom_name="MultiCooperativeFluentRepetitive", uncooperativeness=0.5, bottom_fluency=0.5,
             top_fluency=1.0,
             repetition_exact_match_p=1.0, repetition_overlap_match_p=1.0, order=11.4,
             trait_scale={UnCooperativenessTrait.trait_name: 2, FluencyTrait.trait_name: 2,
                          RepetitionTrait.trait_name: 2},
             user_profile_description="A cooperative fluent repetitive user follows the system's instructions and "
                                      "engages with the system to accomplish the task effectively and communicates "
                                      "with a high level of fluency, using proper grammar or vocabulary and tends to "
                                      "communicate using the same vocabulary or phrases for similar actions.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    PatientExplorativeVerboseUser = UserProfile(
        dict(user_custom_name="MultiPatientExplorativeVerbose", impatience=0.5, min_number_turns=4,
             top_p_exploration=0.01, increase_exploration=True, exploration_distrib_factor=0.2,
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=11.5,
             trait_scale={ImpatienceEscalationTrait.trait_name: 2, ExplorationTraitV2.trait_name: 2,
                          UnconcisenessTrait.trait_name: 2},
             user_profile_description="A patient explorative verbose user demonstrates a willingness to engage with "
                                      "the system over an extended period and is curious and seeks to explore the "
                                      "system's capabilities and the task at hand and communicates using long "
                                      "requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    ImpatientNonExplorativeConciseUser = UserProfile(
        dict(user_custom_name="MultiImpatientNonExplorativeConcise", impatience=2.0, min_number_turns=2,
             top_p_exploration=0.01, increase_exploration=False, exploration_distrib_factor=0.2,
             bottom_unconciseness=0.0, top_unconciseness=0.5, order=11.6,
             trait_scale={ImpatienceEscalationTrait.trait_name: 0, ExplorationTraitV2.trait_name: 0,
                          UnconcisenessTrait.trait_name: 0},
             user_profile_description="An impatient non-explorative concise user seeks quick and concise interactions "
                                      "with the system often finishing the interaction before the end of the task and "
                                      "shows little interest in engaging with different system's features besides "
                                      "moving forward with the task and communicates using short requests.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    # combine 4 traits

    # do NOT use NonExplorativeTolerantVerboseRepetitiveUser since it has too little dialogs
    NonExplorativeTolerantVerboseRepetitiveUser = UserProfile(
        dict(user_custom_name="MultiNonExplorativeTolerantVerboseRepetitive", top_p_exploration=0.01,
             increase_exploration=False,
             exploration_distrib_factor=0.2, intolerance=1.0, repetition_exact_match_p=1.0,
             repetition_overlap_match_p=1.0,
             bottom_unconciseness=0.5, top_unconciseness=1.0, order=11.7,
             trait_scale={ExplorationTraitV2.trait_name: 0, IntoleranceEscalationTrait.trait_name: 2,
                          UnconcisenessTrait.trait_name: 2, RepetitionTrait.trait_name: 2},
             user_profile_description="A non-explorative tolerant verbose repetitive user shows little interest in "
                                      "engaging with different system's features besides moving forward with the task "
                                      "and is willing to give the system multiple chances to recover from errors and "
                                      "communicates using long requests and tends to communicate using the same "
                                      "vocabulary or phrases for similar actions.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    PatientExplorativePositiveFluentUser = UserProfile(
        dict(user_custom_name="MultiPatientExplorativePositiveFluent", impatience=0.5, min_number_turns=4,
             top_p_exploration=0.01, increase_exploration=True, exploration_distrib_factor=0.2,
             bottom_emotion=0.5, top_emotion=1.0, bottom_fluency=0.5, top_fluency=1.0, order=11.8,
             trait_scale={ImpatienceEscalationTrait.trait_name: 2, ExplorationTraitV2.trait_name: 2,
                          EmotionTrait.trait_name: 2, FluencyTrait.trait_name: 2},
             user_profile_description="A patient explorative positive fluent user demonstrates a willingness to "
                                      "engage with the system over an extended period and is curious and seeks to "
                                      "explore the system's capabilities and the task at hand and communicates with a "
                                      "positive sentiment, expressing satisfaction and enthusiasm and communicates "
                                      "with a high level of fluency, using proper grammar or vocabulary.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    PatientNonCooperativeNegativeNonFluentUser = UserProfile(
        dict(user_custom_name="MultiPatientNonCooperativeNegativeNonFluent", impatience=0.5, min_number_turns=4,
             uncooperativeness=2.0, bottom_emotion=0.0, top_emotion=0.5, bottom_fluency=0.0, top_fluency=0.5,
             order=11.9,
             trait_scale={ImpatienceEscalationTrait.trait_name: 2, UnCooperativenessTrait.trait_name: 0,
                          EmotionTrait.trait_name: 0, FluencyTrait.trait_name: 0},
             user_profile_description="A patient non-cooperative negative non-fluent user demonstrates a willingness "
                                      "to engage with the system over an extended period and is less likely to follow "
                                      "the system's instructions and may provide irrelevant or unhelpful requests and "
                                      "communicates with a negative sentiment, expressing dissatisfaction and "
                                      "frustration and communicates with a low level of fluency, using improper "
                                      "grammar or vocabulary.",
             profile_level=ProfileLevels.mixed.value, is_multitrait=True))

    @staticmethod
    def get_all_user_types() -> List[UserProfile]:
        all_types = []
        for user_type in dir(UserTypes):
            if not user_type.startswith("__") and isinstance(getattr(UserTypes, user_type), UserProfile):
                all_types.append(getattr(UserTypes, user_type))
        return all_types

    @staticmethod
    def get_all_single_trait_user_types() -> List[UserProfile]:
        all_types = []
        for user_type in dir(UserTypes):
            if not user_type.startswith("__") and isinstance(getattr(UserTypes, user_type), UserProfile):
                if not getattr(UserTypes, user_type).is_multitrait:
                    all_types.append(getattr(UserTypes, user_type))
        return all_types

    @staticmethod
    def get_user_type_by_name(user_profile_name: str) -> Union[UserProfile, None]:
        name_to_profile = {user_type.user_custom_name: user_type for user_type in UserTypes.get_all_user_types()}
        return name_to_profile.get(user_profile_name, None)


def get_oposite_user_utterance_level(user_type: str):
    return {
        # self.RegularUser: self.RegularUser,  # regular does not have an opposite

        UserTypes.ConciseUser.user_custom_name: UserTypes.VerboseUser,
        UserTypes.VerboseUser.user_custom_name: UserTypes.ConciseUser,
        UserTypes.PositiveUser.user_custom_name: UserTypes.NegativeUser,
        UserTypes.NegativeUser.user_custom_name: UserTypes.PositiveUser,
        UserTypes.FluentUser.user_custom_name: UserTypes.NonFluentUser,
        UserTypes.NonFluentUser.user_custom_name: UserTypes.FluentUser,
        UserTypes.RepetitiveUser.user_custom_name: UserTypes.NonRepetitiveUser,
        UserTypes.NonRepetitiveUser.user_custom_name: UserTypes.RepetitiveUser
    }.get(user_type, None)


def get_oposite_user_intents_level(user_type: str):
    return {
        # self.RegularUser: self.RegularUser,  # regular does not have an opposite

        UserTypes.PatientUser.user_custom_name: UserTypes.ImpatientUser,
        UserTypes.ImpatientUser.user_custom_name: UserTypes.PatientUser,
        UserTypes.TolerantUser.user_custom_name: UserTypes.IntolerantUser,
        UserTypes.IntolerantUser.user_custom_name: UserTypes.TolerantUser,
        UserTypes.ExplorativeUser.user_custom_name: UserTypes.NonExplorativeUser,
        UserTypes.NonExplorativeUser.user_custom_name: UserTypes.ExplorativeUser,
        UserTypes.CooperativeUser.user_custom_name: UserTypes.UnCooperativeUser,
        UserTypes.UnCooperativeUser.user_custom_name: UserTypes.CooperativeUser,
    }.get(user_type, None)


def get_opposite_user(user_type: str):
    opposite_user = get_oposite_user_utterance_level(user_type)
    if opposite_user:
        return opposite_user
    else:
        return get_oposite_user_intents_level(user_type)


def get_possible_opposite_user_types(trait_name: str, trait_value: int):
    # use trait scale to get opposite user types
    opposite_user_types = []

    # go through the trait scale
    for user_type in UserTypes.get_all_user_types():
        if user_type == UserTypes.RegularUser:
            continue

        user_type_trait_scale = user_type.trait_scale.get(trait_name, None)
        # if the user type has the same trait but with opposite value
        if user_type_trait_scale is not None and abs(trait_value - user_type_trait_scale) == 2:
            opposite_user_types.append(user_type)

    return opposite_user_types
