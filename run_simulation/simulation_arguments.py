import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict

from dataset_generation.system_responses import ResponseType
from dataset_generation.utils import load_json_file
from global_constants import GlobalConstants
from training.dataset import DatasetCreationArguments
from training.training_arguments import GENERATION_PRESETS
from user_simulator.traits_and_profiles.user_profile import UserProfile


@dataclass
class RunSimArguments:

    simulator_path: str = field(
        metadata={"help": "The path to the simulator model"}
    )
    system_path: str = field(
        metadata={"help": "The path to the system model"}
    )

    output_path: str = field(
        metadata={"help": "The path to save the simulated conversations"}
    )

    user_profiles: List[str] = field(
        default=None,
        metadata={"help": "The user profiles to be used"}
    )
    data_arguments: str = field(
        default=None,
        metadata={"help": "The arguments to be used to create the dataset"}
    )
    conversations_per_task: int = field(
        default=2,
        metadata={"help": "The number of conversations to be generated for each task"}
    )
    tasks_path: str = field(
        default="data/tasks_2/test",
        metadata={"help": "The path to the tasks files"}
    )

    max_tasks: int = field(
        default=10,
        metadata={"help": "The maximum number of tasks to be used"}
    )

    prompt_version: str = field(
        default="v5",
        metadata={"help": "The version of the prompt"}
    )
    context_size: int = field(
        default=5,
        metadata={"help": "The size of the context"}
    )
    max_turns_dialogue: int = field(
        default=15,
        metadata={"help": "The maximum number of turns in a dialogue"}
    )

    use_bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16 precision"}
    )

    max_new_tokens_user: int = field(
        default=50,
        metadata={"help": "The maximum number of new tokens to generate in a turn"}
    )

    max_new_tokens_system: int = field(
        default=150,
        metadata={"help": "The maximum number of new tokens to generate in a turn"}
    )

    system_tone: str = field(
        default=ResponseType.NEUTRAL.value,
        metadata={"help": "The tone of the system"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "The seed to be used"}
    )

    user_generation_config: Dict = field(
        default_factory=lambda: GENERATION_PRESETS["sampling"],
        metadata={"help": "The generation config to be used for the user"}
    )

    system_generation_config: Dict = field(
        default_factory=lambda: GENERATION_PRESETS["greedy"],
        metadata={"help": "The generation config to be used for the system"}
    )

    user_max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length of the user input"}
    )

    system_max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length of the system input"}
    )

    user_truncation_side: str = field(
        default="left",
        metadata={"help": "The side to truncate the user input"}
    )

    system_truncation_side: str = field(
        default="left",
        metadata={"help": "The side to truncate the system input"}
    )

    dataset_generator_config_path: str = field(
        default=f"data/dataset_versions/{GlobalConstants.dataset_version}_Regular/all/simulated_conversations_train_manual_distribution_config.json",
        metadata={"help": "The path to the dataset generator config"}
    )

    print_extra_info: bool = field(
        default=False,
        metadata={"help": "Print extra information"}
    )

    extra_loras_list: List[str] = field(
        default=None,
        metadata={"help": "The list of extra loras to be used"}
    )

    extra_loras_weights: List[float] = field(
        default=None,
        metadata={"help": "The weights for the extra loras"}
    )

    merging_type: str = field(
        default=None,
        metadata={"help": "The type of merging to be used"}
    )

    is_debug: bool = field(
        default=False,
        metadata={"help": "If is debug it does not save the outputs to file"}
    )

    eval_suffix: str = field(
        default="",
        metadata={"help": "The suffix to be used in the output folder"}
    )

    def to_dict(self):
        fields_dict = {}
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, DatasetCreationArguments):
                fields_dict[field_name] = field_value.to_dict()
            elif isinstance(field_value, UserProfile):
                fields_dict[field_name] = field_value.to_dict()
            else:
                fields_dict[field_name] = field_value
        return fields_dict


def get_data_arguments(str_path: str):
    # this is used to get the data_arguments from the config_options.json file
    # useful for evaluation to get the correct data_arguments instead of needing to pass them as arguments

    # load config file
    train_config = load_json_file(os.path.join(str_path, "config_options.json"))

    # check if the data_arguments are in the config file
    if "DatasetCreationArguments" in train_config:
        # load from here
        data_arguments = DatasetCreationArguments()
        # fill with the parameters and avoid non-existing keys
        for k in DatasetCreationArguments().__dict__.keys():
            if k in train_config["DatasetCreationArguments"]:
                setattr(data_arguments, k, train_config["DatasetCreationArguments"][k])
        return data_arguments
    else:
        # older version support
        data_arguments = DatasetCreationArguments()
        found_key = False
        # fill with the parameters and avoid non-existing keys
        for k in DatasetCreationArguments().__dict__.keys():
            if k in train_config:
                found_key = True
                setattr(data_arguments, k, train_config[k])
        if found_key:
            return data_arguments

        # load from the data_path
        train_data_path = train_config.get("data_path", None)

        train_params = os.path.join(train_data_path, "dataset_train_config.json")
        data_arguments = None
        if os.path.exists(train_params):
            params = load_json_file(train_params)
            # create DatasetCreationArguments avoid non-existing keys
            data_arguments = DatasetCreationArguments()
            # fill with the parameters and avoid non-existing keys
            for k in DatasetCreationArguments().__dict__.keys():
                if k in params:
                    setattr(data_arguments, k, params[k])
        return data_arguments
