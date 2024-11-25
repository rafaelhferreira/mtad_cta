import json
import os
from typing import Union, Dict, List, Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from data_binding.task_result import DummyTaskResult

from dataset_generation.dialog import Dialog
from dataset_generation.utils import load_json_file
from user_simulator.traits_and_profiles.user_profile import UserTypes, get_opposite_user, UserProfile


def calc_scores_intent_tokens(k: int, g_steps: int, scores: torch.FloatTensor, user_tokenizer: PreTrainedTokenizer):
    intent_scores = []
    # if intent already there just consider next token else consider 5
    # print top-k for the first generation steps (intent)
    for l, scores in enumerate(scores[:g_steps]):
        intent_scores.append({})

        # apply softxmax to get the probabilities
        scores = torch.softmax(scores.squeeze(0), dim=-1)
        # get top-5 and its indices
        top_k_scores, top_k_indices = torch.topk(scores, k)

        for x, y in zip(top_k_scores, top_k_indices):
            i_score = round(x.item(), 3)
            if i_score > 0.0:
                intent_scores[l][user_tokenizer.convert_ids_to_tokens(y.item())] = i_score
    return intent_scores


def create_latex_table(stats_csv_location: Union[str, pd.DataFrame], label: str, caption: str, caption_below: bool = True,
                       resize_box: bool = True, two_column: bool = True):
    # creates a latex table given a dialog_stats_by_profile csv file with the stats

    if isinstance(stats_csv_location, pd.DataFrame):
        df_generated = stats_csv_location
    else:
        # load the csv file
        df_generated = pd.read_csv(stats_csv_location, index_col=0, header=0)

    # get orders
    order_of_profiles = {i.user_custom_name: i.order for i in UserTypes.get_all_user_types()}
    # order the row using order of profiles
    df_generated["order"] = df_generated.index.map(lambda x: order_of_profiles[x])
    df_generated = df_generated.sort_values(by="order")
    df_generated = df_generated.drop(columns="order")

    # round the values to 2 decimal places
    df_generated = df_generated.round(2)

    # concat columns with its respective stdev
    for i in df_generated.columns:
        if "stdev" not in i and i + "_stdev" in df_generated.columns:
            df_generated[i] = df_generated[i].astype(str) + " $\pm$ " + df_generated[i + "_stdev"].astype(str)

    # pretty column_names
    column_mapping = {
        'avg_number_turns': '\\# Turns',
        'dialogue_conditioned_tolerance': 'Tolerance',
        'dialogue_exploration_rate': 'Exploration',
        'dialogue_cooperativeness_rate': 'Cooperativeness',
        'dialogue_avg_number_words': '\\# Words',
        'dialogue_avg_sentiment': 'Sentiment',
        'dialogue_avg_fluency': 'Fluency',
        'dialogue_avg_repetition': 'Repetition',
        'number_dialogues': '\\# Dial',
        # 'next_percentage': 'Next \%',
    }

    # Rename columns
    df_generated = df_generated.rename(columns=column_mapping)

    # keep only the columns we want
    df_generated = df_generated[list(column_mapping.values())]

    # create latex table
    latex_code = df_generated.to_latex(
        escape=False, float_format=lambda x: f'{x:.2f}',
        position="tbhp",
        column_format="l" + "c" * len(df_generated.columns),
        label=label,
        caption=caption,
    )

    # center the table
    latex_code = latex_code.replace("\\begin{table}[tbhp]\n", "\\begin{table}[tbhp]\n\\centering")

    if resize_box:
        latex_code = latex_code.replace("\\begin{table}[tbhp]\n\\centering", "\\begin{table}[tbhp]\n\\centering\n\\resizebox{\\linewidth}{!}{%")

    if caption_below:
        # get current caption via regex
        caption = latex_code.split("\\caption{")[1].split("}")[0]
        # remove caption from table
        latex_code = latex_code.replace("\\caption{" + caption + "}", "")
        # add caption below table
        latex_code = latex_code.replace("\\end{tabular}\n\\end{table}", "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}")

    if resize_box:
        latex_code = latex_code.replace("\\bottomrule\n\end{tabular}", "\\bottomrule\n\end{tabular}\n}%")

    if two_column:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table*}")
        latex_code = latex_code.replace("\\end{table}", "\\end{table*}")

    print(latex_code)
    print()

    return latex_code, df_generated


def create_results_dataframe(stats_by_profile: Dict[str, Dict], additional_columns: List[str] = None):

    from user_simulator.traits_and_profiles.identifying_traits import profile_to_trait

    # create a dataframe with all stats
    df = pd.DataFrame(stats_by_profile)
    # transpose
    df = df.T

    # get only columns in profile_to_trait
    traits_columns = [trait.stat_name for trait in profile_to_trait.values()]

    # add stdev columns
    traits_columns += [f"{trait.stat_name}_stdev" for trait in profile_to_trait.values()]

    # add other columns
    if additional_columns:
        traits_columns += additional_columns

    # keep only existing columns
    traits_columns = [c for c in traits_columns if c in df.columns]
    df = df[traits_columns]

    # remove repeated columns
    df = df.loc[:, ~df.columns.duplicated()]

    # get orders
    order_of_profiles = {i.user_custom_name: i.order for i in UserTypes.get_all_user_types()}
    # order the row using order of profiles
    df["order"] = df.index.map(lambda x: order_of_profiles.get(x, 100))
    df = df.sort_values(by="order")
    df = df.drop(columns="order")

    return df


def order_df_based_on_profiles(df: pd.DataFrame):
    # index is the profile name and columns are the stats

    # get orders
    order_of_profiles = {i.user_custom_name: i.order for i in UserTypes.get_all_user_types()}
    # order the row using order of profiles
    df["order"] = df.index.map(lambda x: order_of_profiles.get(x, 100))
    df = df.sort_values(by="order")
    df = df.drop(columns="order")
    return df


def create_dataset_simple_latex_table(stats: Dict[str, Dict[str, float]], add_identifying_characteristic: bool,
                                      label: str, caption: str, caption_below: bool = True,
                                      resize_box: bool = True, two_column: bool = True) -> Tuple[str, pd.DataFrame]:
    # creates a latex table given a dialog_stats_by_profile csv file with the stats

    df_generated = opposite_dict_to_simplified_df(stats)

    # get orders
    order_of_profiles = {i.user_custom_name: i.order for i in UserTypes.get_all_user_types()}
    # order the row using order of profiles
    df_generated["order"] = df_generated.index.map(lambda x: order_of_profiles[x])
    df_generated = df_generated.sort_values(by="order")
    df_generated = df_generated.drop(columns="order")

    # round the values to 2 decimal places
    df_generated = df_generated.round(2)

    # concat columns with its respective stdev
    for i in df_generated.columns:
        if "stdev" not in i and i + "_stdev" in df_generated.columns:
            df_generated[i] = df_generated[i].astype(str) + " $\pm$ " + df_generated[i + "_stdev"].astype(str)

    # drop stdev columns
    df_generated = df_generated.drop(columns=[c for c in df_generated.columns if "stdev" in c])

    # pretty column_names
    column_mapping = {
        'Patient': 'Patience',
        'Cooperative': 'Cooperativeness',
        'Tolerant': 'Tolerance',
        'Exploration': 'Explorative',
        'Verbose': 'Verbosity',
        'Positive': 'Emotion',
        'Fluent': 'Fluency',
        'Repetitive': 'Repetition',
    }

    # Rename row in index
    df_generated = df_generated.rename(index=column_mapping)

    # keep only the rows we want
    df_generated = df_generated.loc[list(column_mapping.values())]

    # rename columns
    new_names = {"Index": "Trait", "Trait": "High", "Regular": "Regular", "Other": "Low"}
    df_generated = df_generated.rename(columns=new_names)

    # add a new column with this information
    df_generated["Identifying Characteristic"] = [
        "Avg \# Turns",
        "Avg Cooperativeness",
        "Avg Tolerance",
        "Avg Exploration",
        "Avg \# Words per Turn",
        "Avg Emotion Level",
        "Avg Fluency Level",
        "Avg Consecutive Word Overlap",
    ]

    # change order of columns
    considered_columns = ["Low", "Regular", "High"]
    if add_identifying_characteristic:
        considered_columns = ["Identifying Characteristic"] + considered_columns

    df_generated = df_generated[considered_columns]

    # create latex table
    latex_code = df_generated.to_latex(
        escape=False, float_format=lambda x: f'{x:.2f}',
        position="tbhp",
        column_format="ll" + "c" * (len(df_generated.columns)-1),
        label=label,
        caption=caption,
    )

    # center the table
    latex_code = latex_code.replace("\\begin{table}[tbhp]\n", "\\begin{table}[tbhp]\n\\centering")

    if resize_box:
        latex_code = latex_code.replace("\\begin{table}[tbhp]\n\\centering", "\\begin{table}[tbhp]\n\\centering\n\\resizebox{\\linewidth}{!}{%")

    if caption_below:
        # get current caption via regex
        caption = latex_code.split("\\caption{")[1].split("}")[0]
        # remove caption from table
        latex_code = latex_code.replace("\\caption{" + caption + "}", "")
        # add caption below table
        latex_code = latex_code.replace("\\end{tabular}\n\\end{table}", "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}")

    if resize_box:
        latex_code = latex_code.replace("\\bottomrule\n\end{tabular}", "\\bottomrule\n\end{tabular}\n}%")

    if two_column:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table*}")
        latex_code = latex_code.replace("\\end{table}", "\\end{table*}")

    print(latex_code)
    print()

    return latex_code, df_generated


def opposite_dict_to_simplified_df(stats: Dict[str, Dict[str, float]]):
    # stats can be obtained from df_to_opposite_results_dict using the original csv file with the stats

    # Initialize empty lists to store the data
    index_column = []
    trait_column = []
    regular_column = []
    other_column = []
    trait_column_stdev = []
    regular_column_stdev = []
    other_column_stdev = []
    # Iterate through the dictionary to extract data
    for key, values in stats.items():
        index_column.append(key)
        for trait, value in values.items():
            trait = trait.lower()
            is_stdev = "stdev" in trait
            if "regular" in trait:
                if is_stdev:
                    regular_column_stdev.append(value)
                else:
                    regular_column.append(value)
            elif trait.startswith(key.lower()):
                if is_stdev:
                    trait_column_stdev.append(value)
                else:
                    trait_column.append(value)
            else:
                if is_stdev:
                    other_column_stdev.append(value)
                else:
                    other_column.append(value)
    # Create a DataFrame
    df_generated = pd.DataFrame({
        'Index': index_column,
        'Trait': trait_column,
        'Regular': regular_column,
        'Other': other_column,
        'Trait_stdev': trait_column_stdev,
        'Regular_stdev': regular_column_stdev,
        'Other_stdev': other_column_stdev
    })
    # set the index
    df_generated = df_generated.set_index('Index')

    return df_generated


def get_metric_simplified_table(df: pd.DataFrame, caption: str, label: str,
                                resize_box: bool, caption_below: bool, two_column: bool):

    from user_simulator.traits_and_profiles.identifying_traits import profile_to_trait

    df = order_df_based_on_profiles(df)
    final_df = {}
    # for each row, get the opposite user value
    for i, row in df.iterrows():
        if i == UserTypes.RegularUser.user_custom_name:
            continue

        trait_name = profile_to_trait[i].list_based_stat_name
        trait_value = row[trait_name]

        # opposite user
        opposite_user = get_opposite_user(i).user_custom_name
        opposite_trait_value = df.loc[opposite_user][trait_name]

        # regular user
        regular_trait = df.loc[UserTypes.RegularUser.user_custom_name][trait_name]

        final_df[i] = {
            # "Trait": trait_name,
            "Regular Value": regular_trait,
            "Opposite Value": opposite_trait_value,
            "Value": trait_value,
        }
    # create df
    df = pd.DataFrame(final_df).T
    df = order_df_based_on_profiles(df)
    # keep only main profiles
    column_mapping = {
        'Patient': 'Patience',
        'Cooperative': 'Cooperativeness',
        'Tolerant': 'Tolerance',
        'Explorative': 'Exploration',
        'Verbose': 'Verbosity',
        'Positive': 'Emotion',
        'Fluent': 'Fluency',
        'Repetitive': 'Repetition',
    }
    # drop index not in column mapping
    df = df.drop(index=[i for i in df.index if i not in column_mapping])
    # rename based on column mapping
    df = df.rename(index=column_mapping)
    column_mapping = {
        "Opposite Value": "Low",
        "Regular Value": "Regular",
        "Value": "High",
    }
    df = df.rename(columns=column_mapping)
    # reorder columns
    df = df[column_mapping.values()]

    # print md
    latex_code = df.to_latex(
        escape=False, float_format=lambda x: f'{x:.2f}',
        position="tbhp",
        column_format="l" + "c" * (len(df.columns)),
        label=label,
        caption=caption,
    )

    # center the table
    latex_code = latex_code.replace("\\begin{table}[tbhp]\n", "\\begin{table}[tbhp]\n\\centering")

    if resize_box:
        latex_code = latex_code.replace("\\begin{table}[tbhp]\n\\centering",
                                        "\\begin{table}[tbhp]\n\\centering\n\\resizebox{\\linewidth}{!}{%")

    if caption_below:
        # get current caption via regex
        caption = latex_code.split("\\caption{")[1].split("}")[0]
        # remove caption from table
        latex_code = latex_code.replace("\\caption{" + caption + "}", "")
        # add caption below table
        latex_code = latex_code.replace("\\end{tabular}\n\\end{table}",
                                        "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}")

    if resize_box:
        latex_code = latex_code.replace("\\bottomrule\n\end{tabular}", "\\bottomrule\n\end{tabular}\n}%")

    if two_column:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table*}")
        latex_code = latex_code.replace("\\end{table}", "\\end{table*}")

    print(latex_code)
    print()

    return latex_code, df


def already_exists_get_path(output_path):
    # if output_path exists check if it is empty
    if os.path.exists(output_path):
        if os.listdir(output_path):
            # add number to the end of the folder
            # this way we avoid confusion with the results
            i = 1
            while os.path.exists(output_path + f"_{i}"):
                i += 1
            output_path += f"_{i}"
    return output_path


def create_dialogs_from_json_file(json_file_path: str) -> List[Dialog]:
    dialogs = []

    dialogs_json = load_json_file(json_file_path)

    # support new and old version of the dataset
    if isinstance(dialogs_json, dict):
        dialogs_json = dialogs_json.items()

    for dialog in dialogs_json:
        # support new and old version of the dataset
        if isinstance(dialog, Tuple):
            dialog_id, dialog = dialog
        else:
            dialog_id = ""  # this is an older version where dialog_id did not exist

        dialog_obj = create_dialog_from_dict(dialog)
        dialogs.append(dialog_obj)
    return dialogs


def create_dialog_from_dict(dialog: Dict) -> Dialog:
    dialog_id = dialog.get("dialog_id", "")
    # create the dialog object
    user_profile = dialog.get("user_profile", None)
    if user_profile:
        # due to bug in previous version we check if it is a str and convert it to a dict before
        if isinstance(user_profile, str):
            user_profile = user_profile.replace("'", "\"").replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
            user_profile = json.loads(user_profile)
        user_profile = UserProfile(user_profile)

    if "id" in dialog.get("task"):
        task = DummyTaskResult(dialog.get("task"))
    else:
        raise ValueError("Task type not recognized")

    dialog_obj = Dialog(task=task, dialog_id=dialog_id,
                        system_tone=dialog.get("system_tone", None), user_profile=user_profile,
                        simulator_model_path=dialog.get("simulator_model", None))
    # add the turns
    for turn in dialog.get("dialog"):
        dialog_obj.add_turn(intent=turn.get("intent"),
                            user_utterance=turn.get("user"),
                            system_utterance=turn.get("system"),
                            current_step=turn.get("current_step"),
                            negative_response=turn.get("negative_response", ""),
                            forced_system_error=turn.get("forced_system_error", False),
                            )
    return dialog_obj


def get_dialogues_from_generation_folder(folder_path: str):
    # point to folder with multiple json files e.g. result from a simulation

    # list all files in folder
    files = os.listdir(folder_path)
    files = [i for i in files if i.endswith(".json")]
    # turn into dialogues
    all_dialogues = []  # type: List[Dialog]
    for i in files:
        raw_dialogues = load_json_file(os.path.join(folder_path, i))
        if isinstance(raw_dialogues, list):
            all_dialogues += [create_dialog_from_dict(i) for i in raw_dialogues]
        else:
            if raw_dialogues.get("dialog_id"):
                all_dialogues.append(create_dialog_from_dict(raw_dialogues))
            else:
                all_dialogues += [create_dialog_from_dict(i) for i in raw_dialogues.values()]

    return all_dialogues


def get_folders_profile_per_model(path: str, profiles: List[str] = None):
    list_all_folders_in_dir = os.listdir(path)
    if profiles is None:
        profiles = [u.user_custom_name for u in UserTypes.get_all_single_trait_user_types()]
    # iterate over all folders and check if it is is in profiles list
    folders_list = []
    for folder in list_all_folders_in_dir:
        if folder in profiles:
            checkpoints = os.listdir(os.path.join(path, folder))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoint in folder please check the folder structure for {path}/{folder}")
            if len(checkpoints) > 1:
                raise ValueError(
                    f"More than one checkpoint in folder please check the folder structure for {path}/{folder}")

            # use the only checkpoint in the folder and append to the list the csv name
            csv_location = os.path.join(path, folder, checkpoints[0])
            folders_list.append(csv_location)
    return folders_list


def get_folder_single_model(path: str):
    list_all_folders_in_dir = os.listdir(path)
    checkpoint_folders = []
    for folder in list_all_folders_in_dir:
        # if starts with checkpoint
        if folder.startswith("checkpoint"):
            checkpoint_folders.append(folder)

    if len(checkpoint_folders) == 0:
        raise ValueError(f"No folder in path please check the folder structure for {path}")
    if len(checkpoint_folders) > 1:
        raise ValueError(f"More than one folder in path please check the folder structure for {path}")

    # use the only checkpoint in the folder and append to the list the csv name
    csv_location = os.path.join(path, checkpoint_folders[0])
    return csv_location
