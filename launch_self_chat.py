import json
import os
import sys

from typing import List
from tqdm import tqdm
from transformers import HfArgumentParser

from dataset_generation.utils import get_tasks_files, is_valid_task, set_seeds
from training.dataset import DatasetCreationArguments
from run_simulation.self_chat import run_sim, aggregate_and_calc_stats_by_user_type
from run_simulation.simulation_arguments import RunSimArguments, get_data_arguments
from training.utils import create_latex_table, already_exists_get_path
from user_simulator.traits_and_profiles.user_profile import UserTypes


def launch_self_chat_eval():

    # print arguments used to run this python script
    print("Arguments:", " ".join(sys.argv))

    parser = HfArgumentParser((RunSimArguments,))
    self_chat_arguments = parser.parse_args_into_dataclasses()[0]  # type: RunSimArguments

    print("Self-chat Arguments:")
    print(json.dumps(self_chat_arguments.to_dict(), indent=2))

    if not self_chat_arguments.user_profiles or self_chat_arguments.user_profiles == [""]:
        user_profiles = UserTypes.get_all_single_trait_user_types()  # use all user types
    else:
        # check if it is a list
        if self_chat_arguments.user_profiles[0][0] == "[":  # we need to parse it has a list
            # this is for the case where we combine profiles
            user_profiles_names = [i.replace("[", "").replace("]", "").replace(" ", "").split(",") for i in self_chat_arguments.user_profiles]  # type: List[List[str]]
            user_profiles = []
            for combination_profile in user_profiles_names:
                profile_objs = [UserTypes.get_user_type_by_name(user_profile) for user_profile in combination_profile]
                if None in profile_objs:
                    raise ValueError(f"User profile {combination_profile} not found")
                user_profiles.append(profile_objs)

            print("User Profiles:")
            print(json.dumps([[user.to_dict() for user in user_profile] for user_profile in user_profiles], indent=2))
        else:
            user_profiles = [UserTypes.get_user_type_by_name(user_profile) for user_profile in self_chat_arguments.user_profiles]
            for u, u_arg in zip(user_profiles, self_chat_arguments.user_profiles):
                if u is None:
                    raise ValueError(f"User profile {u_arg} not found")

            print("User Profiles:")
            print(json.dumps([user.to_dict() for user in user_profiles], indent=2))

    if not self_chat_arguments.data_arguments:
        # check if config_options.json in simulator_path
        if os.path.exists(os.path.join(self_chat_arguments.simulator_path, "config_options.json")):
            data_arguments = get_data_arguments(self_chat_arguments.simulator_path)
        # and in folder before basename
        elif os.path.exists(os.path.join(os.path.dirname(self_chat_arguments.simulator_path), "config_options.json")):
            data_arguments = get_data_arguments(os.path.dirname(self_chat_arguments.simulator_path))
        else:
            print("No data_arguments and no config_options.json found in simulator_path using default data_arguments")
            data_arguments = DatasetCreationArguments()
    else:
        data_arguments = DatasetCreationArguments(**json.loads(self_chat_arguments.data_arguments))

    # add max new tokens to data user_generation_config
    self_chat_arguments.user_generation_config["max_new_tokens"] = self_chat_arguments.max_new_tokens_user
    self_chat_arguments.system_generation_config["max_new_tokens"] = self_chat_arguments.max_new_tokens_system

    print("Data Arguments:")
    print(json.dumps(data_arguments.to_dict(), indent=2))

    path_number = 0
    # is the last two parts of the path (so name of folder and checkpoint number)
    name_path = os.path.join(*self_chat_arguments.simulator_path.split("/")[-2 + path_number:])
    if len(user_profiles) == 1:  # use the last 3 parts of the path to account for the user
        name_path = os.path.join(*self_chat_arguments.simulator_path.split("/")[-3 + path_number:])

    # add simulator path to be able to identify model
    output_path = os.path.join(self_chat_arguments.output_path, name_path)

    if self_chat_arguments.eval_suffix:
        output_path += f"_{self_chat_arguments.eval_suffix}"

    output_path = already_exists_get_path(output_path)

    # load the tasks
    tasks = []
    for task_path in get_tasks_files(self_chat_arguments.tasks_path):
        task = is_valid_task(task_path)
        if task:
            tasks.append(task)
    tasks = tasks[:self_chat_arguments.max_tasks]

    all_times_dict = []

    for user_type in tqdm(user_profiles, desc="User Profiles"):

        # more than one profile so it is combination change output path to consider this
        if isinstance(user_type, list):
            # if last part of outputpath is checkpoint- we need to remove it
            if os.path.basename(output_path).startswith("checkpoint-"):
                output_path = os.path.dirname(output_path)

            # concat the name of profiles and weights
            if self_chat_arguments.extra_loras_weights:
                output_path = os.path.join(output_path, "_".join([f"{i.user_custom_name}_{w}" for i, w in zip(user_type, self_chat_arguments.extra_loras_weights)]))
            else:  # no weights use just the names
                output_path = os.path.join(output_path, "_".join([i.user_custom_name for i in user_type]))
            output_path = already_exists_get_path(output_path)
            print("Output Path:", output_path)

        current_profiles = [user_type] if not isinstance(user_type, list) else user_type

        # set seeds for reproducibility
        set_seeds(self_chat_arguments.seed)

        print("Running self-chat for user type:", [i.user_custom_name for i in current_profiles])
        times_dict = run_sim(
            simulator_path=self_chat_arguments.simulator_path,
            system_path=self_chat_arguments.system_path,
            user_profiles=current_profiles,
            data_arguments=data_arguments,
            conversations_per_task=self_chat_arguments.conversations_per_task,
            tasks=tasks,
            prompt_version=self_chat_arguments.prompt_version,
            context_size=self_chat_arguments.context_size,
            max_turns_dialogue=self_chat_arguments.max_turns_dialogue,
            output_path=output_path,
            use_bf16=self_chat_arguments.use_bf16,
            system_tone=self_chat_arguments.system_tone,
            user_generation_config=self_chat_arguments.user_generation_config,
            system_generation_config=self_chat_arguments.system_generation_config,
            user_max_length=self_chat_arguments.user_max_length,
            system_max_length=self_chat_arguments.system_max_length,
            user_truncation_side=self_chat_arguments.user_truncation_side,
            system_truncation_side=self_chat_arguments.system_truncation_side,
            print_extra_info=self_chat_arguments.print_extra_info,
            extra_loras_list=self_chat_arguments.extra_loras_list,
            extra_loras_weights=self_chat_arguments.extra_loras_weights,
            merging_type=self_chat_arguments.merging_type,
            is_debug=self_chat_arguments.is_debug,
        )

        if times_dict:
            all_times_dict.append(times_dict)

    # aggregate all times by averaging over the same key
    all_times_dict_aggregated = {}
    for times_dict in all_times_dict:
        for key, value in times_dict.items():
            if key not in all_times_dict_aggregated:
                all_times_dict_aggregated[key] = []
            all_times_dict_aggregated[key].append(value)

    # calculate the average
    all_times_dict_aggregated = {key: sum(value) / len(value) for key, value in all_times_dict_aggregated.items()}

    # print the average times
    print("\n====================================\n")
    print("Times Stats for all:")
    print(all_times_dict_aggregated)
    print(json.dumps(all_times_dict_aggregated, indent=2))
    print("\n====================================\n")

    # aggregate and calculate stats
    print("Calculating stats")
    aggregate_and_calc_stats_by_user_type(
        folder_path=output_path,
        output_folder=output_path,
        config_path=self_chat_arguments.dataset_generator_config_path,
    )
    print("\n====================================\n")
    print("Create Latex Table")
    create_latex_table(
        stats_csv_location=os.path.join(output_path, "dialog_stats_by_profile.csv"),
        label=f"tab_dialogue_level_{self_chat_arguments.data_arguments}",
        caption=f"Dialogue level using model {self_chat_arguments.simulator_path}",
    )


if __name__ == "__main__":
    launch_self_chat_eval()
    pass
