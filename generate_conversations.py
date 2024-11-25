import argparse
import os
from typing import Dict

from tqdm import tqdm

from data_binding.enumerates import Intents, ArtificialIntents
from dataset_generation.dialog_generator import create_dialog_based_on_probs
from dataset_generation.prob_distribution import MANUAL_PROBS_REVISED
from dataset_generation.utils import CONSIDERED_INTENTS, load_json_file
from global_constants import GlobalConstants
from training.dialog_filtering import dialog_filtering_and_stats_writing
from dataset_generation.calc_dialog_stats import check_stats
from user_simulator.trait_analysis_models.classifier_models import calc_cache
from user_simulator.traits_and_profiles.user_profile import UserProfile, UserTypes


def create_train_valid_test_conversations_considered_intents(
    collected_utterances_file: str,
    first_steps_prob_file: str,
    tasks_folder_path: str,
    number_dialogs: int = 10000,
    version: str = '1.5',
    user_profiles_prob: Dict[UserProfile, float] = None,
    seed: int = 42, calc_stats: bool = False
):

    # collected utterances
    collected_utterances = load_json_file(collected_utterances_file)

    # first steps probability
    first_steps_prob = load_json_file(first_steps_prob_file)

    # this is used to simulate system errors to better handle the tolerance param
    # some we do not put because they change the flow of the dialogue or it would be hard
    # for the user simulator to understand the error (e.g. next step intent)
    system_errors_prob = {
        # Intents.NextStepIntent: 0.1,
        # Intents.PreviousStepIntent: 0.1,
        # Intents.ResumeTaskIntent: 0.1,
        # Intents.AMAZONStopIntent: 0.1,
        # Intents.AMAZONRepeatIntent: 0.1,
        # Intents.AMAZONFallbackIntent: 0.1,
        Intents.IngredientsConfirmationIntent: 0.2,
        # Intents.PlayMusicIntent: 0.1,
        Intents.GetCuriositiesIntent: 0.2,
        Intents.QuestionIntent: 0.2,
        # Intents.IdentifyProcessIntent: 0.1,
        ArtificialIntents.DefinitionQuestionIntent: 0.2,
        # ArtificialIntents.SensitiveIntent: 0.1,
    }

    # ignore warning
    create_dialog_based_on_probs(
        collected_utterances=collected_utterances,
        transitions_count_dict=None,
        first_step_probs=first_steps_prob,
        tasks_path=os.path.join(tasks_folder_path, "train"),
        use_weight_for_utterance=True,
        considered_intents=CONSIDERED_INTENTS,
        number_dialogs=max(int(number_dialogs * 0.9), 1),
        transitions_probs_dict=MANUAL_PROBS_REVISED,
        apply_smoothing_to_utterances=True,
        ignore_stop_intent=False,
        out_path=f"data/dataset_versions/{version}/all/simulated_conversations_train_manual_distribution.json",
        seed=seed,
        user_profiles_prob=user_profiles_prob,
        system_errors_prob=system_errors_prob,
        lower_case_and_remove_punctuation_user=True,
        calc_stats=calc_stats
    )

    create_dialog_based_on_probs(
        collected_utterances=collected_utterances,
        transitions_count_dict=None,
        first_step_probs=first_steps_prob,
        tasks_path=os.path.join(tasks_folder_path, "valid"),
        use_weight_for_utterance=True,
        considered_intents=CONSIDERED_INTENTS,
        number_dialogs=max(int(number_dialogs * 0.05), 1),
        transitions_probs_dict=MANUAL_PROBS_REVISED,
        apply_smoothing_to_utterances=True,
        ignore_stop_intent=False,
        out_path=f"data/dataset_versions/{version}/all/simulated_conversations_valid_manual_distribution.json",
        seed=seed,
        user_profiles_prob=user_profiles_prob,
        system_errors_prob=system_errors_prob,
        lower_case_and_remove_punctuation_user=True,
        calc_stats=calc_stats
    )

    create_dialog_based_on_probs(
        collected_utterances=collected_utterances,
        transitions_count_dict=None,
        first_step_probs=first_steps_prob,
        tasks_path=os.path.join(tasks_folder_path, "test"),
        use_weight_for_utterance=True,
        considered_intents=CONSIDERED_INTENTS,
        number_dialogs=max(int(number_dialogs * 0.05), 1),
        transitions_probs_dict=MANUAL_PROBS_REVISED,
        apply_smoothing_to_utterances=True,
        ignore_stop_intent=False,
        out_path=f"data/dataset_versions/{version}/all/simulated_conversations_test_manual_distribution.json",
        seed=seed,
        user_profiles_prob=user_profiles_prob,
        system_errors_prob=system_errors_prob,
        lower_case_and_remove_punctuation_user=True,
        calc_stats=calc_stats
    )


def generate_single_trait_dataset():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--collected_utterances_file", type=str, required=True)
    argparser.add_argument("--first_steps_prob_file", type=str, required=True)
    argparser.add_argument("--tasks_folder_path", type=str, required=True)
    argparser.add_argument("--number_of_dialogs", type=int, default=10000)
    argparser.add_argument("--calculate_cache", action='store_true')

    # one argument that is a list of strs representing the user profiles
    default_user_profiles = [u.user_custom_name for u in UserTypes.get_all_single_trait_user_types()]
    argparser.add_argument('--user_profiles', type=str, nargs='+', default=default_user_profiles)
    argparser.add_argument('--version_number', type=str, default=GlobalConstants.dataset_version)
    argparser.add_argument('--not_generate_dialogs', action='store_true')
    argparser.add_argument('--not_calc_stats', action='store_true')
    argparser.add_argument('--not_filter', action='store_true')
    args = argparser.parse_known_args()[0]

    # do not allow to calculate stats and filter if not all user profiles are provided
    if set(args.user_profiles) != set(default_user_profiles) and not args.not_filter:
        raise ValueError("You need to provide all user profiles to filter")

    if args.calculate_cache:
        calc_cache(args.collected_utterances_file)

    version_number = args.version_number
    # we only create the dataset if we are not only calculating stats and filtering
    if not args.not_generate_dialogs:
        for i in args.user_profiles:
            i = UserTypes.get_user_type_by_name(i)
            print(i.user_custom_name)
            create_train_valid_test_conversations_considered_intents(
                collected_utterances_file=args.collected_utterances_file,
                first_steps_prob_file=args.first_steps_prob_file,
                tasks_folder_path=args.tasks_folder_path,
                number_dialogs=args.number_of_dialogs, version=f'{version_number}_{i.user_custom_name}',
                user_profiles_prob={i: 1.0},
                calc_stats=True
            )
            print()

    # only calculate stats if all profiles are provided
    if not args.not_calc_stats:
        print()
        check_stats(version_prefix=f'{version_number}_')
        print()

    if not args.not_filter and default_user_profiles == args.user_profiles:
        print("Filtering Dialogues")
        dialog_filtering_and_stats_writing(version_prefix=f'{version_number}_')
        print()


def generate_multi_trait_dataset():
    multi_trait_profiles = [
        # 2 traits
        UserTypes.PatientVerboseUser, UserTypes.PatientConciseUser,
        UserTypes.ImpatientVerboseUser, UserTypes.ImpatientConciseUser,
        UserTypes.CooperativeNonFluentUser, UserTypes.ExplorativeImpatientUser,
        UserTypes.VerboseFluentUser,
        # 3 traits
        UserTypes.ImpatientConciseNegativeUser, UserTypes.CooperativeFluentRepetitiveUser,
        UserTypes.PatientExplorativeVerboseUser, UserTypes.ImpatientNonExplorativeConciseUser,
        # 4 traits
        UserTypes.NonExplorativeTolerantVerboseRepetitiveUser, UserTypes.PatientExplorativePositiveFluentUser,
    ]

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--collected_utterances_file", type=str, required=True)
    argparser.add_argument("--first_steps_prob_file", type=str, required=True)
    argparser.add_argument("--tasks_folder_path", type=str, required=True)
    argparser.add_argument("--number_of_dialogs", type=int, default=10000)
    argparser.add_argument("--calculate_cache", action='store_true')

    multi_trait_profiles_names = [u.user_custom_name for u in multi_trait_profiles]
    argparser.add_argument('--user_profiles', type=str, nargs='+', default=multi_trait_profiles_names)
    argparser.add_argument('--version_number', type=str, default=GlobalConstants.dataset_version)
    args = argparser.parse_known_args()[0]

    if args.calculate_cache:
        calc_cache(args.collected_utterances_file)

    folders_to_filter = []
    for i in tqdm(args.user_profiles, desc="Generating Multi-Trait Datasets"):
        i = UserTypes.get_user_type_by_name(i)
        print(i.user_custom_name)
        create_train_valid_test_conversations_considered_intents(
            collected_utterances_file=args.collected_utterances_file,
            first_steps_prob_file=args.first_steps_prob_file,
            tasks_folder_path=args.tasks_folder_path,
            number_dialogs=args.number_of_dialogs, version=f'{args.version_number}_{i.user_custom_name}',
            user_profiles_prob={i: 1.0},
            calc_stats=True
        )
        folders_to_filter.append(f"{args.version_number}_{i.user_custom_name}")
        print()

    print("Filtering Dialogues")
    dialog_filtering_and_stats_writing(version_prefix=f'{GlobalConstants.dataset_version}_', folders=folders_to_filter)
    print()


if __name__ == "__main__":

    first_argparser = argparse.ArgumentParser()
    first_argparser.add_argument('--single_trait', action='store_true')
    first_argparser.add_argument('--multi_trait', action='store_true')

    # ignore unknown arguments
    known_args = first_argparser.parse_known_args()[0]

    if known_args.single_trait:
        generate_single_trait_dataset()
    elif known_args.multi_trait:
        generate_multi_trait_dataset()
    else:
        raise ValueError("You need to provide either --single_trait or --multi_trait")
