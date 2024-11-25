import json
import os
from collections import Counter, defaultdict

from transformers import HfArgumentParser

from dataset_generation.utils import write_to_json_file, load_json_file
from training.dataset import DatasetCreationArguments, get_sft_dataset


def create_training_data():

    parser = HfArgumentParser((DatasetCreationArguments,))
    dataset_arguments = parser.parse_args_into_dataclasses()[0]  # type: DatasetCreationArguments

    print("Dataset arguments:")
    print(json.dumps(dataset_arguments.to_dict(), indent=2))

    if dataset_arguments.exclude_ids_sft:
        exclude_ids = set()
        # open json file iterate and collect the ids
        for f in dataset_arguments.exclude_ids_sft:
            sft_dataset = load_json_file(f)
            for example in sft_dataset:
                exclude_ids.add(example["dialogue_id"])
        print("Excluding " + str(len(exclude_ids)) + " ids from SFT dataset")

    for split in ["train", "valid", "test"]:
        dataset_examples = []

        for folder_path in dataset_arguments.input_folder_paths:

            file_path = os.path.join(folder_path, "simulated_conversations_" + split + "_manual_distribution.json")
            is_eval_split = split == "valid" or split == "test"

            dataset_examples += get_sft_dataset(file_path, is_eval_split, dataset_arguments)

        # stratify user profile i.e. make sure that each user profile has the same number of examples
        # because for examples there a many more examples of patient than impatient
        if dataset_arguments.stratify_user_profile:
            # it is a list of dicts
            examples_per_profile = Counter([example.get("user_profile") for example in dataset_examples])
            print(examples_per_profile)
            # get min from all profiles
            min_examples = min(examples_per_profile.values())
            stratified_examples = []
            current_counts = defaultdict(int)
            for example in dataset_examples:
                if current_counts[example.get("user_profile")] < min_examples:
                    stratified_examples.append(example)
                    current_counts[example.get("user_profile")] += 1
            dataset_examples = stratified_examples
            print("Stratified counts", Counter([example.get("user_profile") for example in dataset_examples]))

        if dataset_arguments.output_folder_path:

            os.makedirs(dataset_arguments.output_folder_path, exist_ok=True)

            write_to_json_file(os.path.join(dataset_arguments.output_folder_path, "dataset_" + split + ".json"),
                               dataset_examples)

            # write config file
            config_dict = dataset_arguments.to_dict()
            config_dict["split"] = split
            write_to_json_file(os.path.join(dataset_arguments.output_folder_path, "dataset_" + split + "_config.json"),
                               config_dict)

        # print len
        print("Number of examples in " + split + " split: " + str(len(dataset_examples)))


if __name__ == "__main__":
    create_training_data()
    pass
