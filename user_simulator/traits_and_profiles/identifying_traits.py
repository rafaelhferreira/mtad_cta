from typing import List, Dict, Union

from user_simulator.traits_and_profiles.user_profile import UserProfile, UserTypes
from user_simulator.traits_and_profiles.user_traits import ImpatienceEscalationTrait, IntoleranceEscalationTrait, \
    ExplorationTraitV2, UnCooperativenessTrait, UnconcisenessTrait, EmotionTrait, FluencyTrait, RepetitionTrait


class IdentifyingTrait:

    def __init__(self, stat_name: str, list_based_stat_name: str, is_higher: bool,
                 other_traits: List[str] = None, other_traits_is_higher: List[bool] = None):
        self.stat_name = stat_name
        self.list_based_stat_name = list_based_stat_name
        self.is_higher = is_higher
        self.other_traits = other_traits
        self.other_traits_is_higher = other_traits_is_higher

        if self.other_traits:
            assert len(self.other_traits) == len(self.other_traits_is_higher), ("Other traits and other traits is "
                                                                                "higher must have the same length")

    @staticmethod
    def get_identifying_trait(user_type: UserProfile):
        return profile_to_trait.get(user_type, None)

    # a dialog is valid if the stat is higher/lower than the regular user stat +/- stdev
    def is_dialog_valid(self, regular_user_stats: Dict[str, Union[float, int]],
                        current_dialog_stats: Dict[str, Union[float, int]]):

        considered_traits = [self.stat_name]
        is_higher_list = [self.is_higher]
        if self.other_traits:
            considered_traits += self.other_traits
            is_higher_list += self.other_traits_is_higher

        for trait, is_higher in zip(considered_traits, is_higher_list):
            dialog_stat = regular_user_stats.get(trait, None)
            dialog_stat_stdev = regular_user_stats.get(trait + "_stdev", None)
            if dialog_stat is None or dialog_stat_stdev is None:
                raise Exception("Could not find stat: " + trait + " in regular user stats")

            if is_higher:
                is_valid = current_dialog_stats.get(trait, 0) >= dialog_stat + dialog_stat_stdev / 2
            else:
                is_valid = current_dialog_stats.get(trait, 0) <= dialog_stat - dialog_stat_stdev / 2

            if not is_valid:
                return False

        return True


profile_to_trait = {
    # self.RegularUser: self.RegularUser,  # regular user does not have any defining trait

    UserTypes.PatientUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True),
    UserTypes.ImpatientUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", False),
    UserTypes.TolerantUser.user_custom_name: IdentifyingTrait("dialogue_conditioned_tolerance", "conditioned_tolerance_list", True),
    UserTypes.IntolerantUser.user_custom_name: IdentifyingTrait("dialogue_conditioned_tolerance", "conditioned_tolerance_list", False),
    UserTypes.ExplorativeUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", True),
    UserTypes.NonExplorativeUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", False),
    UserTypes.CooperativeUser.user_custom_name: IdentifyingTrait("dialogue_cooperativeness_rate", "cooperativeness_list", True),
    UserTypes.UnCooperativeUser.user_custom_name: IdentifyingTrait("dialogue_cooperativeness_rate", "cooperativeness_list", False),

    UserTypes.ConciseUser.user_custom_name: IdentifyingTrait("dialogue_avg_number_words", "number_words_list", False),
    UserTypes.VerboseUser.user_custom_name: IdentifyingTrait("dialogue_avg_number_words", "number_words_list", True),
    UserTypes.PositiveUser.user_custom_name: IdentifyingTrait("dialogue_avg_sentiment", "sentiment_list", True),
    UserTypes.NegativeUser.user_custom_name: IdentifyingTrait("dialogue_avg_sentiment", "sentiment_list", False),
    UserTypes.FluentUser.user_custom_name: IdentifyingTrait("dialogue_avg_fluency", "fluency_list", True),
    UserTypes.NonFluentUser.user_custom_name: IdentifyingTrait("dialogue_avg_fluency", "fluency_list", False),
    UserTypes.RepetitiveUser.user_custom_name: IdentifyingTrait("dialogue_avg_repetition", "repetition_list", True),
    UserTypes.NonRepetitiveUser.user_custom_name: IdentifyingTrait("dialogue_avg_repetition", "repetition_list", False),


    # Multi Trait profiles
    UserTypes.PatientVerboseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True, ["dialogue_avg_number_words"], [True]),
    UserTypes.PatientConciseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True, ["dialogue_avg_number_words"], [False]),
    UserTypes.ImpatientVerboseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", False, ["dialogue_avg_number_words"], [True]),
    UserTypes.ImpatientConciseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", False, ["dialogue_avg_number_words"], [False]),
    UserTypes.CooperativeNonFluentUser.user_custom_name: IdentifyingTrait("dialogue_cooperativeness_rate", "cooperativeness_list", True, ["dialogue_avg_fluency"], [False]),
    UserTypes.ExplorativeImpatientUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", True, ["avg_number_turns"], [False]),
    UserTypes.VerboseFluentUser.user_custom_name: IdentifyingTrait("dialogue_avg_fluency", "fluency_list", True, ["dialogue_avg_number_words"], [True]),

    # more multi trait profiles
    UserTypes.ExplorativeCooperativeUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", True, ["dialogue_cooperativeness_rate"], [True]),
    UserTypes.FluentRepetitiveUser.user_custom_name: IdentifyingTrait("dialogue_avg_fluency", "fluency_list", True, ["dialogue_avg_repetition"], [True]),
    UserTypes.VerbosePositiveUser.user_custom_name: IdentifyingTrait("dialogue_avg_sentiment", "sentiment_list", True, ["dialogue_avg_number_words"], [True]),
    UserTypes.ExplorativeConciseUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", True, ["dialogue_avg_number_words"], [False]),
    UserTypes.UncooperativeNonFluentUser.user_custom_name: IdentifyingTrait("dialogue_cooperativeness_rate", "cooperativeness_list", False, ["dialogue_avg_fluency"], [False]),

    # 3 traits
    UserTypes.ImpatientConciseNegativeUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", False, ["dialogue_avg_number_words", "dialogue_avg_sentiment"], [False, False]),
    UserTypes.CooperativeFluentRepetitiveUser.user_custom_name: IdentifyingTrait("dialogue_cooperativeness_rate", "cooperativeness_list", True, ["dialogue_avg_fluency", "dialogue_avg_repetition"], [True, True]),
    UserTypes.PatientExplorativeVerboseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True, ["dialogue_exploration_rate", "dialogue_avg_number_words"], [True, True]),
    UserTypes.ImpatientNonExplorativeConciseUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", False, ["dialogue_exploration_rate", "dialogue_avg_number_words"], [False, False]),

    # 4 traits
    UserTypes.NonExplorativeTolerantVerboseRepetitiveUser.user_custom_name: IdentifyingTrait("dialogue_exploration_rate", "exploration_list", False, ["dialogue_conditioned_tolerance", "dialogue_avg_number_words", "dialogue_avg_repetition"], [True, True, True]),
    UserTypes.PatientExplorativePositiveFluentUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True, ["dialogue_exploration_rate", "dialogue_avg_sentiment", "dialogue_avg_fluency"], [True, True, True]),
    UserTypes.PatientNonCooperativeNegativeNonFluentUser.user_custom_name: IdentifyingTrait("avg_number_turns", "number_turns_list", True, ["dialogue_cooperativeness_rate", "dialogue_avg_sentiment", "dialogue_avg_fluency"], [False, False, False]),

}


user_name_to_trait = {
    'Patience': 'Engagement',
    'Engagement': 'Engagement',
    UserTypes.PatientUser.user_custom_name: 'Engagement',
    UserTypes.ImpatientUser.user_custom_name: 'Engagement',
    UserTypes.CooperativeUser.user_custom_name: 'Cooperativeness',
    UserTypes.UnCooperativeUser.user_custom_name: 'Cooperativeness',
    UserTypes.TolerantUser.user_custom_name: 'Tolerance',
    UserTypes.IntolerantUser.user_custom_name: 'Tolerance',
    'Exploration': 'Exploration',
    UserTypes.ExplorativeUser.user_custom_name: 'Exploration',
    UserTypes.NonExplorativeUser.user_custom_name: 'Exploration',
    UserTypes.VerboseUser.user_custom_name: 'Verbosity',
    UserTypes.ConciseUser.user_custom_name: 'Verbosity',
    UserTypes.PositiveUser.user_custom_name: 'Emotion',
    UserTypes.NegativeUser.user_custom_name: 'Emotion',
    UserTypes.FluentUser.user_custom_name: 'Fluency',
    UserTypes.NonFluentUser.user_custom_name: 'Fluency',
    UserTypes.RepetitiveUser.user_custom_name: 'Repetition',
    UserTypes.NonRepetitiveUser.user_custom_name: 'Repetition',

    # using trait names instead of user types
    ImpatienceEscalationTrait.trait_name: "Engagement",
    IntoleranceEscalationTrait.trait_name: "Tolerance",
    ExplorationTraitV2.trait_name: "Exploration",
    UnCooperativenessTrait.trait_name: "Cooperativeness",
    UnconcisenessTrait.trait_name: "Verbosity",
    EmotionTrait.trait_name: "Emotion",
    FluencyTrait.trait_name: "Fluency",
    RepetitionTrait.trait_name: "Repetition",
}

metric_to_trait_name = {
    'Engagement': ImpatienceEscalationTrait.trait_name,
    'Tolerance': IntoleranceEscalationTrait.trait_name,
    'Exploration': ExplorationTraitV2.trait_name,
    'Cooperativeness': UnCooperativenessTrait.trait_name,
    'Verbosity': UnconcisenessTrait.trait_name,
    'Emotion': EmotionTrait.trait_name,
    'Fluency': FluencyTrait.trait_name,
    'Repetition': RepetitionTrait.trait_name,
}


metric_to_description = {
    'Engagement': "the user's willingness to engage with the system for a longer period of time",
    'Cooperativeness': "the user's tendency to cooperate with the system's requests by following its instructions and decreasing unrelated interactions",
    'Tolerance': "the user's tolerance for system mistakes, where less tolerant users conclude the interaction sooner",
    'Exploration': "the user's propensity to interact with diverse system features",
    'Explorative': "the user's propensity to interact with diverse system features",
    'Verbosity': "the length of the user's requests",
    'Emotion': "the user's overall positive tone expressed towards the system",
    'Fluency': "the user's ability to express oneself clearly and coherently. It involves the natural flow of language without hesitations, disruptions, or ASR errors",
    'Repetition': "The user consistently repeats the same vocabulary for certain actions",
}
