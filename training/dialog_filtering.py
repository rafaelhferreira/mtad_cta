import os
from typing import List, Dict, Union

from tqdm import tqdm

from dataset_generation.calc_dialog_stats import calc_dialog_stats, \
    check_stats
from dataset_generation.dialog import Dialog
from dataset_generation.utils import load_json_file, write_to_json_file
from training.utils import create_dialogs_from_json_file
from user_simulator.traits_and_profiles.identifying_traits import IdentifyingTrait
from user_simulator.traits_and_profiles.user_profile import UserTypes


def get_filter_dialogues(dialogues_files: List[str],
                         regular_user_stats: Dict[str, Union[float, int]]):

    all_dialogues = []  # type: List[Dialog]
    for i in dialogues_files:
        all_dialogues += create_dialogs_from_json_file(i)

    dialogues_config = load_json_file(dialogues_files[0].replace(".json", "_config.json"))

    # filter dialogues
    print(f"Initial Dialogues {len(all_dialogues)}")
    filtered_dialogues = []
    for dialog in tqdm(all_dialogues):
        current_user_type = dialog.user_profile
        current_dialog_stats = calc_dialog_stats(
            dialogues=[dialog],
            intents_distribution=dialogues_config.get("transitions_probs_dict"),
            first_step_distribution=dialogues_config.get("first_step_probs"),
            use_tqdm=False
        )
        identifying_trait = IdentifyingTrait.get_identifying_trait(current_user_type.user_custom_name)
        if not identifying_trait:
            raise Exception(f"Could not find identifying trait for user type: {current_user_type.user_custom_name}")

        if identifying_trait.is_dialog_valid(regular_user_stats, current_dialog_stats):
            filtered_dialogues.append(dialog)

    print(f"Final Dialogues {len(filtered_dialogues)}")
    return filtered_dialogues


def dialog_filtering_and_stats_writing(version_prefix: str = "3.0_", ignore_prefix: str = "filtered",
                                       folders: List[str] = None):

    regular_user_stats = load_json_file(f"data/dataset_versions/{version_prefix}{UserTypes.RegularUser.user_custom_name}/all/simulated_conversations_train_manual_distribution_config_stats.json")

    if not folders:
        folders = os.listdir("data/dataset_versions")
        folders = [i for i in folders if i.startswith(version_prefix) and UserTypes.RegularUser.user_custom_name not in i
                   and os.path.isdir(os.path.join("data/dataset_versions", i)) and ignore_prefix not in i]

    dialogues_per_user_type = {}
    for f in folders:
        print(f"Processing {f}")
        files = [
            os.path.join("data/dataset_versions", f, "all/simulated_conversations_train_manual_distribution.json"),
            os.path.join("data/dataset_versions", f, "all/simulated_conversations_valid_manual_distribution.json"),
            os.path.join("data/dataset_versions", f, "all/simulated_conversations_test_manual_distribution.json"),
        ]

        for split in files:
            filtered_dialogues = get_filter_dialogues([split], regular_user_stats)
            dialogues_per_user_type[split] = filtered_dialogues

            # save in another folder the filtered dialogues
            new_file_name = split.replace(version_prefix, version_prefix + "filtered_")
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

            if filtered_dialogues:

                # write filtered dialogues
                write_to_json_file(new_file_name, [i.dialog_dict() for i in filtered_dialogues], indent=2)

                # write config file (is the same)
                config_file = split.replace(".json", "_config.json")
                dialogues_config = load_json_file(config_file)
                write_to_json_file(new_file_name.replace(".json", "_config.json"), dialogues_config, indent=2)

                # write filtered stats
                filtered_stats = calc_dialog_stats(filtered_dialogues, dialogues_config.get("transitions_probs_dict"),
                                                   dialogues_config.get("first_step_probs"), use_tqdm=False)
                write_to_json_file(new_file_name.replace(".json", "_config_stats.json"), filtered_stats, indent=2)
            else:
                print("WARNING - No dialogues to calculate stats after filtering for ", new_file_name)

    # print sizes
    for k, v in dialogues_per_user_type.items():
        print(f"{k}: {len(v)}")

    # calculate stats for the filtered dialogues
    check_stats(version_prefix=version_prefix + "filtered_", ignore_prefix="---------------")
