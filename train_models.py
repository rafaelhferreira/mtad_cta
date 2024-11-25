import json

from transformers import HfArgumentParser

from training.training import sft_training
from training.training_arguments import ModelArguments, DataArguments, TrainArgs, LoRaArguments
from user_simulator.traits_and_profiles.user_profile import UserTypes


def train_model():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArgs, LoRaArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()  # type: (ModelArguments, DataArguments, TrainArgs, LoRaArguments)

    print("Training arguments:")
    for arg_type in [model_args, data_args, training_args, lora_args]:
        print(json.dumps(arg_type.to_dict(), indent=2))

    # sft
    if training_args.one_model_per_profile:
        # get all user profiles
        if not training_args.only_profiles:
            profiles = [user_profile.user_custom_name for user_profile in UserTypes.get_all_single_trait_user_types()]
        else:
            profiles = [UserTypes.get_user_type_by_name(user_profile) for user_profile in training_args.only_profiles]
            for u, u_arg in zip(profiles, training_args.only_profiles):
                if u is None:
                    raise ValueError(f"User profile {u_arg} not found")
            profiles = [profile.user_custom_name for profile in profiles]
        # sort
        profiles.sort()
        for i, profile in enumerate(profiles):
            print(f"Training for profile {profile} ({i+1} out of {len(profiles)})")
            sft_training(model_args, data_args, training_args, lora_args, only_user=profile)
    else:
        sft_training(model_args, data_args, training_args, lora_args)


if __name__ == "__main__":
    train_model()
    pass
