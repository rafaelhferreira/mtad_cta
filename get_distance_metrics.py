import os
from typing import List, Dict

import argparse
import pandas as pd

from dataset_generation.calc_dialog_stats import get_distance_metrics_by_profile
from training.utils import order_df_based_on_profiles, \
    get_metric_simplified_table, get_folders_profile_per_model, get_folder_single_model
from user_simulator.model_merging.utils import ModelMergingTypes
from user_simulator.traits_and_profiles.user_profile import UserTypes


def get_results_as_tables(all_stats: Dict[str, Dict[str, Dict[str, float]]], caption: str, label: str,
                          path: str, single_model_kls: Dict[str, float]):
    # create df
    df = pd.DataFrame(single_model_kls).T
    complete_table = df.to_latex(
        escape=False, float_format=lambda x: f'{x:.2f}',
        position="tbhp",
        column_format="l" + "c" * (len(df.columns)),
        label=label,
        caption=path,
    )
    print(complete_table)
    get_metric_simplified_table(df=df, caption=caption, label=label, caption_below=True, resize_box=False,
                                two_column=True)
    # now get simplified table with all stats
    profile_metric_dict = {}
    for profile, trait_stats_dict in all_stats.items():
        for trait, stats in trait_stats_dict.items():
            for metric, value in stats.items():
                if metric not in profile_metric_dict:
                    profile_metric_dict[metric] = {}
                if profile not in profile_metric_dict[metric]:
                    profile_metric_dict[metric][profile] = {}
                profile_metric_dict[metric][profile][trait] = value
    # create a df for each metric
    for metric, profile_trait_dict in profile_metric_dict.items():
        df = pd.DataFrame(profile_trait_dict).T
        df = order_df_based_on_profiles(df)

        get_metric_simplified_table(df=df, caption=f"{caption} {metric}", label=f"{label}_{metric}",
                                    caption_below=True, resize_box=False, two_column=False)
    return df


def calc_divergence_single_model(path: str, label: str, caption: str, csv_file_name: str):

    single_model = get_folder_single_model(path)
    single_model_kls, all_stats = get_distance_metrics_by_profile(single_model, csv_file_name, profiles=None)

    return get_results_as_tables(all_stats, caption, label, path, single_model_kls)


def calc_divergence_per_profile(path: str, label: str, caption: str, csv_file_name: str):

    folders = get_folders_profile_per_model(path)
    kl_divergences = {}
    all_stats = {}
    profile = None
    for f in folders:
        # get profile name
        for u in UserTypes.get_all_single_trait_user_types():
            if "/" + u.user_custom_name + "/" in f:
                profile = u.user_custom_name
                break

        if profile:
            divergence, all_stats_temp = get_distance_metrics_by_profile(f, csv_file_name, profiles=[profile])
            kl_divergences.update(divergence)
            all_stats.update(all_stats_temp)
        else:
            print("Profile not found for folder: ", f)

    return get_results_as_tables(all_stats, caption, label, path, kl_divergences)


def from_profile_str_to_trait_scale(profile: str):
    profile_class = UserTypes.get_user_type_by_name(profile)
    profile_combination_name = ""
    for trait_name, trait_value in profile_class.trait_scale.items():
        # upper case first letter of trait name
        trait_name = trait_name[0].upper() + trait_name[1:]
        profile_combination_name += trait_name + "="

        if trait_value == 0:
            profile_combination_name += "Low"
        elif trait_value == 2:
            profile_combination_name += "High"
        else:
            print(trait_value)
            raise ValueError("Trait value should be 0 or 1")
    return profile_combination_name


def calc_divergence_profile_combination(path: str, profile: str, merging_methods: List[str],
                                        csv_file_name: str, is_final_path: bool = False):

    df_results = {}
    df_results_all_stats = {}
    for merging_method in merging_methods:
        if not is_final_path:
            final_path = os.path.join(f"data/plangpt_simulated_conversations_combined_{merging_method}", path)
        else:
            final_path = path

        kl_dict, all_stats = get_distance_metrics_by_profile(
            path=final_path,
            csv_file=csv_file_name,
            profiles=[profile],
            plot_kl=False,
            graph_suffix=f"{profile}_{merging_method}",
        )

        # get first key in all_stats
        first_key = list(all_stats.keys())[0]
        # create a dict with the first key
        all_stats_dict_flatten = {}
        for metric_name, metrics in all_stats[first_key].items():
            for metric, value in metrics.items():
                all_stats_dict_flatten[metric_name + "_" + metric] = value

        final_dict = {
            profile: all_stats_dict_flatten
        }

        # create df
        df = pd.DataFrame(kl_dict).T
        merging_method = ModelMergingTypes.merging_method_to_pretty_name(merging_method)
        df_results[merging_method] = df

        # other stats df
        df_all_stats = pd.DataFrame(final_dict).T
        df_results_all_stats[merging_method] = df_all_stats

    # concat all dataframes
    df = pd.concat(df_results, axis=0)

    trait_names = [
        "number_turns_list", "cooperativeness_list", "conditioned_tolerance_list", "exploration_list",
        "number_words_list", "sentiment_list", "fluency_list", "repetition_list",
    ]

    final_columns = []
    for c in df.columns:
        if c in trait_names:
            final_columns.append(c)
    if len(final_columns) != len(trait_names):
        print(f"READ THIS (exception) - Final columns should have the same length as trait names "
              f"but has {len(final_columns)}, instead of {len(trait_names)}")
        missing_columns = set(trait_names) - set(final_columns)
        print("Missing columns: ", missing_columns)

    # order columns by trait names
    df = df[final_columns]
    # transform index into columns
    df.reset_index(inplace=True)
    # rename columns
    df.rename(columns={"level_0": "Merging Method"}, inplace=True)
    df.rename(columns={"level_1": "Profile"}, inplace=True)

    # apply function from_profile_str_to_trait_scale to Profile column
    df["Profile"] = df["Profile"].apply(from_profile_str_to_trait_scale)

    # set profile as index
    df.set_index("Profile", inplace=True)

    # get trait name
    trait_names_to_pretty_name = {
        "number_turns_list": "# Turns",
        "cooperativeness_list": "Cooperativeness",
        "conditioned_tolerance_list": "Tolerance",
        "exploration_list": "Exploration",
        "number_words_list": "# Words",
        "sentiment_list": "Sentiment",
        "fluency_list": "Fluency",
        "repetition_list": "Repetition",
    }

    # rename columns
    df.rename(columns=trait_names_to_pretty_name, inplace=True)

    complete_table = df.to_latex(
        escape=True, float_format=lambda x: f'{x:.2f}',
        position="tbhp",
        column_format="l" + "c" * (len(df.columns)),
        label=f"kl_divergence_{profile}_merging_methods",
        caption=f"KL Divergence {profile} Merging Methods",
    )
    print(complete_table)

    # complete table for all stats
    df_all_stats = pd.concat(df_results_all_stats, axis=0)
    single_metric_data = print_all_table_metrics(df_all_stats, profile, final_columns, trait_names_to_pretty_name)

    return single_metric_data


def print_all_table_metrics(df_all_stats: pd.DataFrame, profile: str,
                            trait_names: List[str],
                            trait_names_to_pretty_name: Dict[str, str]):
    single_metric_data = {}
    '''
    for metric in ["mean_training", "mean_generated", "variance_training", "variance_generated", "skew_training", 
                   "skew_generated", "kurtosis_training", "kurtosis_generated", "ks_statistic", "p_value", 
                   "wasserstein_dist", "js_divergence", "kl_divergence"]:
    '''
    # for each identyfying metric print a table
    for metric in ["mean_abs_difference", "ks_statistic", "p_value",
                   "wasserstein_dist", "js_divergence", "kl_divergence"]:
        # select only columns that contain the metric
        df_all_stats_metric = df_all_stats.filter(regex=metric)

        # rename columns using trait_names_to_pretty_name
        for column in df_all_stats_metric.columns:
            for trait_name, pretty_name in trait_names_to_pretty_name.items():
                if trait_name in column:
                    df_all_stats_metric.rename(columns={column: trait_name}, inplace=True)
        # order columns by trait names
        df_all_stats_metric = df_all_stats_metric[trait_names]
        # rename again columns using trait_names_to_pretty_name
        df_all_stats_metric.rename(columns=trait_names_to_pretty_name, inplace=True)

        complete_table = df_all_stats_metric.to_latex(
            escape=True, float_format=lambda x: f'{x:.2f}',
            position="tbhp",
            column_format="l" + "c" * (len(df_all_stats_metric.columns)),
            label=f"{metric}_{profile}_merging_methods",
            caption=f"{metric} {profile} Merging Methods",
        )
        print(complete_table)

        # add to single_metric_data
        single_metric_data[metric] = df_all_stats_metric
    return single_metric_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # for JTS and STS
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--is_single_model", action="store_true")
    parser.add_argument("--csv_file_name", type=str, required=False, default="dialog_stats_by_profile.csv")

    # for combination methods
    parser.add_argument("--is_combination", action="store_true")
    parser.add_argument("--profile", type=str, required=False)
    parser.add_argument("--merging_methods", type=str, nargs="+", required=False)
    parser.add_argument("--is_final_path", action="store_true")

    args = parser.parse_args()

    # cannot be single model and combination at the same time
    if args.is_single_model and args.is_combination:
        raise ValueError("You need to provide either --is_single_model or --is_combination")

    if not args.is_combination:
        if args.is_single_model:  # for JTS
            calc_divergence_single_model(
                path=args.path, label=args.label,
                caption=args.caption,  csv_file_name=args.csv_file_name
            )
        else:  # for STS
            calc_divergence_per_profile(
                path=args.path, label=args.label,
                caption=args.caption, csv_file_name=args.csv_file_name
            )
    else:  # for mTAD and derived methods
        calc_divergence_profile_combination(
            path=args.path, profile=args.profile, csv_file_name=args.csv_file_name,
            merging_methods=args.merging_methods, is_final_path=args.is_final_path
        )
