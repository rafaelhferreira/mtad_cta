from typing import List

import argparse

from dataset_generation.utils import load_json_file
from training.utils import get_dialogues_from_generation_folder
from user_simulator.gpt_evaluation.open_ai_gpt import OpenAIParams
from user_simulator.gpt_evaluation.degeneration_and_uniqueness import \
    has_problem_utterance_level, check_for_unique_utterances
from user_simulator.gpt_evaluation.create_prompts import dialogue_to_intent_prompts
from user_simulator.gpt_evaluation.utils import sample_dialogues, sample_dialogues_multi_trait, get_utterances
from user_simulator.gpt_evaluation.system_quality_eval import call_open_ai_analyze_system_quality
from user_simulator.gpt_evaluation.comparative_eval import call_open_ai_comparative_dialogues
from user_simulator.traits_and_profiles.user_profile import UserTypes


def create_prompts_single_trait(utterances_file_path: str,  # it is a dict in format as in collected_utterances.json
                                dialogues_folder: str,
                                test_dialogues_path: str,
                                # path to folder with json files with dialogues from the test set
                                output_path: str,
                                suffix: str = "single_trait",  # suffix for the output file
                                number_dialogues: int = 10):

    collected_utterances = load_json_file(utterances_file_path)

    # alters the collected utterances with the annotated ones
    get_utterances(collected_utterances=collected_utterances)

    dialogues = sample_dialogues(
        folder_path=dialogues_folder, number_dialogues=number_dialogues, profiles=None
    )

    # calculate non-gpt metrics
    has_problem_utterance_level(dialogues)
    check_for_unique_utterances(dialogues, collected_utterances=collected_utterances)

    # get the test dialogues
    test_dialogues = get_dialogues_from_generation_folder(test_dialogues_path)

    # create the prompts to be annotated by GPT
    prompts = dialogue_to_intent_prompts(
        dialogues_list=dialogues,
        test_dialogues=None,
        collected_utterances=collected_utterances,
        output_path=output_path,
        suffix=suffix,
        number_sheets=1
    )

    prompts_test_dialogues = dialogue_to_intent_prompts(
        dialogues_list=dialogues,
        test_dialogues=test_dialogues,
        collected_utterances=collected_utterances,
        output_path=output_path,
        suffix=f"{suffix}_with_test_dialogues",
        number_sheets=1
    )

    return prompts, prompts_test_dialogues


def create_prompts_multi_trait(utterances_file_path: str,  # it is a dict in format as in collected_utterances.json
                               folder_paths: List[str], profile_names: List[str],
                               test_dialogues_path: str,
                               # path to folder with json files with dialogues from the test set
                               output_path: str,
                               suffix: str = "multi_trait",  # suffix for the output file
                               number_dialogues: int = 10):

    collected_utterances = load_json_file(utterances_file_path)

    # alters the collected utterances with the annotated ones
    get_utterances(collected_utterances=collected_utterances)

    # create the profiles from the profile_names
    profiles = [UserTypes.get_user_type_by_name(p) for p in profile_names]

    dialogues = sample_dialogues_multi_trait(
        folder_paths=folder_paths, number_dialogues=number_dialogues, profiles=profiles
    )

    # calculate non-gpt metrics
    has_problem_utterance_level(dialogues)
    check_for_unique_utterances(dialogues, collected_utterances=collected_utterances)

    # get the test dialogues
    test_dialogues = get_dialogues_from_generation_folder(test_dialogues_path)

    # create the prompts to be annotated by GPT
    prompts = dialogue_to_intent_prompts(
        dialogues_list=dialogues,
        test_dialogues=None,
        collected_utterances=collected_utterances,
        output_path=output_path,
        suffix=suffix,
        number_sheets=1
    )

    prompts_test_dialogues = dialogue_to_intent_prompts(
        dialogues_list=dialogues,
        test_dialogues=test_dialogues,
        collected_utterances=collected_utterances,
        output_path=output_path,
        suffix=f"{suffix}_with_test_dialogues",
        number_sheets=1
    )

    return prompts, prompts_test_dialogues


def get_results_open_ai_comparative_2_dialogues(prompts_json_path: str, output_path: str, model_name: str):

    open_ai_params = OpenAIParams(MODEL_NAME=model_name, TEMPERATURE=0.0, MAX_TOKENS=250, TOP_P=0.0,
                                  STOP_SEQUENCES=None)

    call_open_ai_comparative_dialogues(json_path=prompts_json_path,
                                       output_path=output_path,
                                       open_ai_params=open_ai_params)


def get_results_open_ai_system_quality(prompts_json_path: str, output_path: str, model_name: str):

    open_ai_params = OpenAIParams(MODEL_NAME=model_name, TEMPERATURE=0.0, MAX_TOKENS=300, TOP_P=0.0,
                                  STOP_SEQUENCES=None)

    call_open_ai_analyze_system_quality(json_path=prompts_json_path,
                                        output_path=output_path,
                                        open_ai_params=open_ai_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances_file_path", type=str, required=True)
    parser.add_argument("--dialogues_folders", type=str, nargs="+", required=True)
    parser.add_argument("--test_dialogues_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--is_multi_trait", action="store_true")
    parser.add_argument("--profile_names", type=str, nargs="+", default=None, required=False)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--number_dialogues", type=int, default=10)
    parser.add_argument("--calc_trait_modeling_accuracy", action="store_true")
    parser.add_argument("--calc_system_response_quality", action="store_true")
    args = parser.parse_args()

    if not args.is_multi_trait:
        _, prompts_test_dialogues = create_prompts_single_trait(
            utterances_file_path=args.utterances_file_path,
            dialogues_folder=args.dialogues_folders[0],
            test_dialogues_path=args.test_dialogues_path,
            output_path=args.output_path,
            suffix="single_trait",
            number_dialogues=args.number_dialogues
        )
    else:

        assert len(args.profile_names) == len(args.dialogues_folders), ("Number of profile names should be equal to "
                                                                        "the number of dialogues folders")

        _, prompts_test_dialogues = create_prompts_multi_trait(
            utterances_file_path=args.utterances_file_path,
            folder_paths=[args.dialogues_folder],
            profile_names=args.profile_names,
            test_dialogues_path=args.test_dialogues_path,
            output_path=args.output_path,
            suffix="multi_trait",
            number_dialogues=args.number_dialogues
        )

    if args.calc_trait_modeling_accuracy:
        get_results_open_ai_comparative_2_dialogues(
            prompts_json_path=prompts_test_dialogues,
            output_path=args.output_path,
            model_name=args.model_name
        )

    if args.calc_system_response_quality:
        get_results_open_ai_system_quality(
            prompts_json_path=prompts_test_dialogues,
            output_path=args.output_path,
            model_name=args.model_name
        )


if __name__ == '__main__':
    main()
