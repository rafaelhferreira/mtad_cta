import json
import os
import traceback
from collections import Counter, defaultdict
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
from lexical_diversity import lex_div as ld
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import kde, entropy, skew, kurtosis, ks_2samp, wasserstein_distance
from tqdm import tqdm

from data_binding.enumerates import Intents
from dataset_generation.dialog import Dialog
from dataset_generation.utils import calculate_word_diversity, calculate_token_overlap, load_json_file, \
    write_to_json_file
from global_constants import GlobalConstants
from training.utils import create_dataset_simple_latex_table, create_results_dataframe, create_latex_table
from user_simulator.trait_analysis_models.classifier_models import emotion_model, fluency_model
from user_simulator.traits_and_profiles.identifying_traits import profile_to_trait
from user_simulator.traits_and_profiles.user_profile import UserTypes, get_oposite_user_utterance_level, \
    get_oposite_user_intents_level
from user_simulator.traits_and_profiles.utils import get_by_top_k


def calc_dialog_stats(dialogues: List[Dialog], intents_distribution: Dict[str, Dict[str, float]],
                      first_step_distribution: Dict[str, float],
                      use_tqdm: bool = False,
                      convert_from_pretty_to_internal_intent: bool = False):
    dialogues_intent_distribution = []
    total_number_turns = []
    patience_relative = []
    cooperativeness_rate = []
    exploration_rate = []
    avg_number_words = []
    avg_sentiment = []
    avg_fluency = []
    avg_repetition = []
    avg_tolerance = []
    avg_word_diversity = []
    conditioned_tolerance = []
    conditioned_tolerance_turns = []
    avg_exact_matches = []
    avg_mtld = []
    avg_ttr = []
    all_user_utterances = ""

    if use_tqdm:
        iter_list = tqdm(dialogues)
    else:
        iter_list = dialogues

    for dialog in iter_list:

        current_number_turns = dialog.number_turns()
        total_number_turns.append(current_number_turns)

        current_cooperativeness_rate = 0
        current_exploration_rate = 0
        current_number_words = 0
        current_sentiment = 0
        current_fluency = 0
        current_repetition = 0
        current_word_diversity = []
        current_tolerance = 0
        current_exact_matches = 0
        has_one_tolerance = False
        for i, turn in enumerate(dialog.turns):

            if convert_from_pretty_to_internal_intent:
                converted_intent = Intents.from_pretty_name_to_original(intent=turn.intent)
                if converted_intent:
                    turn.intent = converted_intent

            dialogues_intent_distribution.append(turn.intent)

            if turn.intent == Intents.AMAZONFallbackIntent or turn.forced_system_error:
                has_one_tolerance = True

            current_number_words += len(turn.user_utterance.split())
            current_sentiment += emotion_model.calculate_value_function(turn.user_utterance)
            current_fluency += fluency_model.calculate_value_function(turn.user_utterance)

            previous_turn = dialog.turns[i - 1] if i > 0 else None

            current_word_diversity.append(turn.user_utterance)

            if turn.user_utterance in current_word_diversity:
                current_exact_matches += 1

            # this not exactly calculated as in generation time since it does not account for the intent
            if previous_turn is not None:
                current_repetition += calculate_token_overlap(current_turn=turn.user_utterance,
                                                              previous_turn=previous_turn.user_utterance)

                # if turn is 1 we use the first step distribution else we use the intents distribution
                dict_to_use = first_step_distribution if (
                        i == 1 and previous_turn.intent == Intents.StartStepsIntent) else intents_distribution.get(
                    previous_turn.intent)
                if not dict_to_use:
                    print(
                        f"WARNING ({dialog.user_profile.user_custom_name}): Intent not found in distribution turn {i}: {previous_turn.intent}")

                # since we apply a lot of calculation and exploration ends up depending on prob and factor here we relax
                # that assumption and consider explorative all that are not in the top-1
                if dict_to_use and turn.intent not in Intents.non_explorative_intents() and turn.intent not in get_by_top_k(
                        1, dict_to_use):
                    current_exploration_rate += 1

            if turn.intent not in Intents.non_cooperative_intents():
                current_cooperativeness_rate += 1

            # tolerant of previous turn was fallback or forced system error and is not the last turn
            if previous_turn and (previous_turn.intent in {
                Intents.AMAZONFallbackIntent} or previous_turn.forced_system_error) and turn.intent not in Intents.stop_task_intents():
                current_tolerance += 1

        cooperativeness_rate.append(current_cooperativeness_rate)
        exploration_rate.append(current_exploration_rate)
        avg_number_words.append(current_number_words)
        avg_sentiment.append(current_sentiment)
        avg_fluency.append(current_fluency)
        avg_repetition.append(current_repetition)
        avg_tolerance.append(current_tolerance)
        avg_word_diversity.append(calculate_word_diversity(current_word_diversity))
        avg_exact_matches.append(current_exact_matches)
        patience_relative.append(current_number_turns / len(dialog.task.get_methods()[0].get_steps()))

        # this makes it so we only consider the tolerance if there was one attempt
        if has_one_tolerance:
            conditioned_tolerance.append(current_tolerance)
            conditioned_tolerance_turns.append(current_number_turns)

        concat_turns = " ".join(turn.user_utterance for turn in dialog.turns)
        tokens = ld.flemmatize(concat_turns)
        avg_ttr.append(ld.ttr(tokens))
        avg_mtld.append(ld.mtld(tokens))

        # add to calculate metrics for entire set of dialogues
        all_user_utterances += concat_turns + " "

    sum_turns = np.sum(total_number_turns)

    # more metrics
    try:
        all_utterances_tokens = ld.flemmatize(all_user_utterances)
        all_ttr = ld.ttr(all_utterances_tokens)
        all_mtld = ld.mtld(all_utterances_tokens)
    except:
        print("Error calculating TTR and MTLD")
        all_ttr = -100
        all_mtld = -100

    # order dict by value
    intent_counts = dict(Counter(dialogues_intent_distribution))
    intent_counts = dict(sorted(intent_counts.items(), key=lambda item: item[1], reverse=True))

    # percentage of next intents
    next_percentage = 0.0
    for i, v in intent_counts.items():
        if "next" in i.lower():
            next_percentage += v

    intent_counts_sum = sum(intent_counts.values())
    if intent_counts_sum > 0:
        next_percentage /= intent_counts_sum
    else:
        next_percentage = 0.0

    stats = {

        "number_dialogues": len(dialogues),
        "avg_number_turns": np.mean(total_number_turns),
        "avg_number_turns_stdev": np.std(total_number_turns),
        "all_ttr": all_ttr,
        "all_mtld": all_mtld,

        # dialogue level
        "dialogue_cooperativeness_rate": np.mean(np.asarray(cooperativeness_rate) / np.asarray(total_number_turns)),
        "dialogue_exploration_rate": np.mean(np.asarray(exploration_rate) / np.asarray(total_number_turns)),
        "dialogue_avg_number_words": np.mean(np.asarray(avg_number_words) / np.asarray(total_number_turns)),
        "dialogue_avg_sentiment": np.mean(np.asarray(avg_sentiment) / np.asarray(total_number_turns)),
        "dialogue_avg_fluency": np.mean(np.asarray(avg_fluency) / np.asarray(total_number_turns)),
        "dialogue_avg_repetition": np.mean(np.asarray(avg_repetition) / np.asarray(total_number_turns)),
        "dialogue_avg_tolerance": np.mean(np.asarray(avg_tolerance) / np.asarray(total_number_turns)),
        "dialogue_avg_word_diversity": np.mean(np.asarray(avg_word_diversity)),
        "dialogue_avg_exact_matches": np.mean(np.asarray(avg_exact_matches) / np.asarray(total_number_turns)),
        "dialogue_patience_relative": np.mean(patience_relative),
        "dialogue_avg_ttr": np.mean(avg_ttr),
        "dialogue_avg_mtld": np.mean(avg_mtld),

        # stdev
        "dialogue_cooperativeness_rate_stdev": np.std(
            np.asarray(cooperativeness_rate) / np.asarray(total_number_turns)),
        "dialogue_exploration_rate_stdev": np.std(np.asarray(exploration_rate) / np.asarray(total_number_turns)),
        "dialogue_avg_number_words_stdev": np.std(np.asarray(avg_number_words) / np.asarray(total_number_turns)),
        "dialogue_avg_sentiment_stdev": np.std(np.asarray(avg_sentiment) / np.asarray(total_number_turns)),
        "dialogue_avg_fluency_stdev": np.std(np.asarray(avg_fluency) / np.asarray(total_number_turns)),
        "dialogue_avg_repetition_stdev": np.std(np.asarray(avg_repetition) / np.asarray(total_number_turns)),
        "dialogue_avg_tolerance_stdev": np.std(np.asarray(avg_tolerance) / np.asarray(total_number_turns)),
        "dialogue_avg_number_turns_stdev": np.std(total_number_turns),
        "dialogue_avg_word_diversity_stdev": np.std(avg_word_diversity),
        "dialogue_avg_exact_matches_stdev": np.std(np.asarray(avg_exact_matches) / np.asarray(total_number_turns)),
        "dialogue_patience_relative_stdev": np.std(patience_relative),
        "dialogue_avg_ttr_stdev": np.std(avg_ttr),
        "dialogue_avg_mtld_stdev": np.std(avg_mtld),

        # sum over list and divide by number of turns
        # "turn_avg_number_turns": np.sum(total_number_turns) / sum_turns,
        "turn_cooperativeness_rate": np.sum(cooperativeness_rate) / sum_turns,
        "turn_exploration_rate": np.sum(exploration_rate) / sum_turns,
        "turn_avg_number_words": np.sum(avg_number_words) / sum_turns,
        "turn_avg_sentiment": np.sum(avg_sentiment) / sum_turns,
        "turn_avg_fluency": np.sum(avg_fluency) / sum_turns,
        "turn_avg_repetition": np.sum(avg_repetition) / sum_turns,
        "turn_avg_tolerance": np.sum(avg_tolerance) / sum_turns,
        "turn_avg_word_diversity": np.mean(avg_word_diversity),
        "turn_avg_exact_matches": np.sum(avg_exact_matches) / sum_turns,

        # stdev
        "turn_cooperativeness_rate_stdev": np.std(cooperativeness_rate),
        "turn_exploration_rate_stdev": np.std(exploration_rate),
        "turn_avg_number_words_stdev": np.std(avg_number_words),
        "turn_avg_sentiment_stdev": np.std(avg_sentiment),
        "turn_avg_fluency_stdev": np.std(avg_fluency),
        "turn_avg_repetition_stdev": np.std(avg_repetition),
        "turn_avg_tolerance_stdev": np.std(avg_tolerance),
        "turn_avg_number_turns_stdev": np.std(total_number_turns),
        "turn_avg_word_diversity_stdev": np.std(avg_word_diversity),
        "turn_avg_exact_matches_stdev": np.std(avg_exact_matches),

        # sum divided by number of dialogues
        "number_cooperativeness": np.mean(cooperativeness_rate),
        "number_avg_number_words": np.mean(avg_number_words),
        "number_avg_tolerance": np.mean(avg_tolerance),

        # conditioned tolerance
        "dialogue_conditioned_tolerance": np.mean(
            np.asarray(conditioned_tolerance) / np.asarray(conditioned_tolerance_turns)),
        "dialogue_conditioned_tolerance_stdev": np.std(
            np.asarray(conditioned_tolerance) / np.asarray(conditioned_tolerance_turns)),
        "turn_conditioned_tolerance": np.sum(conditioned_tolerance) / np.sum(conditioned_tolerance_turns),
        "turn_conditioned_tolerance_stdev": np.std(conditioned_tolerance),

        # other stats
        "dialogues_intent_distribution": intent_counts,
        "next_percentage": next_percentage,

        # raw values (good to plot information)
        "number_turns_list": np.asarray(total_number_turns),
        "cooperativeness_list": np.asarray(cooperativeness_rate) / np.asarray(total_number_turns),
        "exploration_list": np.asarray(exploration_rate) / np.asarray(total_number_turns),
        "number_words_list": np.asarray(avg_number_words) / np.asarray(total_number_turns),
        "sentiment_list": np.asarray(avg_sentiment) / np.asarray(total_number_turns),
        "fluency_list": np.asarray(avg_fluency) / np.asarray(total_number_turns),
        "repetition_list": np.asarray(avg_repetition) / np.asarray(total_number_turns),
        "conditioned_tolerance_list": np.asarray(conditioned_tolerance) / np.asarray(conditioned_tolerance_turns),
        "ttr_list": avg_ttr,
        "mtld_list": avg_mtld
    }

    # round all values
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            stats[key] = round(value, 3)
        # if is np array make it a list
        elif isinstance(value, np.ndarray):
            stats[key] = value.tolist()

    return stats


def calc_distance_metrics(training_stats: List[Union[int, float]],
                          model_stats: List[Union[int, float]],
                          combine_mins_maxs: bool = False,
                          bandwidth: Union[str, float] = "scott",
                          n_samples: int = 100,
                          caption: str = None,
                          plot: bool = False,
                          graph_suffix: str = None):
    other_stats = plot_stats_brev(training_turns=training_stats, generated_turns=model_stats, suffix=graph_suffix)
    try:
        kde_p = kde.gaussian_kde(training_stats, bw_method=bandwidth)
        kde_q = kde.gaussian_kde(model_stats, bw_method=bandwidth)
    except Exception as e:
        other_stats["kl_divergence"] = np.inf
        return other_stats

    if combine_mins_maxs:  # use the full range of values
        min_value = min(min(training_stats), min(model_stats))
        max_value = max(max(training_stats), max(model_stats))
    else:  # use the range of values from the training data
        min_value = min(training_stats)
        max_value = max(training_stats)

    grid = np.linspace(min_value, max_value, n_samples)

    # Evaluate the KDEs on the grid (probability densities)
    p_densities = kde_p(grid)
    q_densities = kde_q(grid)

    if plot:
        # plot the distributions
        plt.plot(grid, p_densities, label="Training Stats")
        plt.plot(grid, q_densities, label="Model Stats")
        plt.xlabel(caption)
        plt.ylabel('Density')
        plt.legend()
        if graph_suffix:
            plt.savefig(f"data/visualizations/kde_plots/{graph_suffix}.pdf", bbox_inches='tight')
        plt.show()

    # Clip zeros from p_densities to avoid division by zero errors
    # p_densities = np.clip(p_densities, np.finfo(float).eps, np.inf)

    # normalize the densities
    p_densities /= np.sum(p_densities)
    q_densities /= np.sum(q_densities)

    # handle zeros
    p_densities = np.clip(p_densities, np.finfo(float).eps, np.inf)
    q_densities = np.clip(q_densities, np.finfo(float).eps, np.inf)

    kl_divergence = entropy(p_densities, q_densities)

    other_stats["kl_divergence"] = kl_divergence

    return other_stats


def plot_stats_brev(training_turns: List[Union[int, float]], generated_turns: List[Union[int, float]],
                    suffix: str, print_stats: bool = False):
    training_turns = np.array(training_turns)
    generated_turns = np.array(generated_turns)

    # Mean and variance
    mean_training = np.mean(training_turns)
    variance_training = np.var(training_turns)
    mean_generated = np.mean(generated_turns)
    variance_generated = np.var(generated_turns)

    mean_abs_difference = np.abs(mean_training - mean_generated)

    # Skewness and kurtosis
    skew_training = skew(training_turns)
    kurtosis_training = kurtosis(training_turns)
    skew_generated = skew(generated_turns)
    kurtosis_generated = kurtosis(generated_turns)

    ks_statistic, p_value = ks_2samp(training_turns, generated_turns)
    wasserstein_dist = wasserstein_distance(training_turns, generated_turns)

    # Normalize histograms to get probability distributions
    training_hist, bin_edges = np.histogram(training_turns, bins=30, density=True)
    generated_hist, _ = np.histogram(generated_turns, bins=bin_edges, density=True)

    js_divergence = jensenshannon(training_hist, generated_hist)

    # print all stats
    if print_stats:
        print()
        print("Stats for:", suffix)
        print(f"Mean (training, generated): {mean_training}, {mean_generated}")
        print(f"Variance (training, generated): {variance_training}, {variance_generated}")
        print(f"Skewness (training, generated): {skew_training}, {skew_generated}")
        print(f"Kurtosis (training, generated): {kurtosis_training}, {kurtosis_generated}")
        print(f"KS Statistic: {ks_statistic}, p-value: {p_value}")
        print(f"Wasserstein Distance: {wasserstein_dist}")
        print(f"JS Divergence: {js_divergence}")

    return {
        "mean_training": mean_training,
        "mean_generated": mean_generated,
        "mean_abs_difference": mean_abs_difference,
        "variance_training": variance_training,
        "variance_generated": variance_generated,
        "skew_training": skew_training,
        "skew_generated": skew_generated,
        "kurtosis_training": kurtosis_training,
        "kurtosis_generated": kurtosis_generated,
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "wasserstein_dist": wasserstein_dist,
        "js_divergence": js_divergence,
        "kl_divergence": None,  # will be calculated later
    }


def check_stats(version_prefix: str = "3.0_", ignore_prefix: str = "filtered"):
    # get all folders with user types in format 3.0_{user_type}
    folders = os.listdir("data/dataset_versions")
    folders = [i for i in folders if i.startswith(version_prefix) and os.path.isdir(
        os.path.join("data/dataset_versions", i)) and ignore_prefix not in i]

    # we remove filtered from regular since it does not make much sense
    regular_user_stats = load_json_file(
        f"data/dataset_versions/{version_prefix.replace('filtered_', '')}{UserTypes.RegularUser.user_custom_name}/all/simulated_conversations_train_manual_distribution_config_stats.json")

    collected_stats = {}
    all_stats = {}
    for f in folders:
        user_type = os.path.basename(f).replace("filtered_", "").split("_")[1]

        files = [
            os.path.join("data/dataset_versions", f,
                         "all/simulated_conversations_train_manual_distribution_config_stats.json"),
            # os.path.join("data/dataset_versions", f, "all/simulated_conversations_valid_manual_distribution.json"),
            # os.path.join("data/dataset_versions", f, "all/simulated_conversations_test_manual_distribution.json"),
        ]

        try:
            stats = load_json_file(files[0])
            all_stats[user_type] = stats

            for user, trait in profile_to_trait.items():
                if user == user_type:
                    collected_stats[user_type] = {
                        f"{user_type}_{trait.stat_name}": stats[trait.stat_name],
                        f"{user_type}_{trait.stat_name}_stdev": stats[trait.stat_name + "_stdev"],
                        f"{UserTypes.RegularUser.user_custom_name}_{trait.stat_name}": regular_user_stats[
                            trait.stat_name],
                        f"{UserTypes.RegularUser.user_custom_name}_{trait.stat_name}_stdev": regular_user_stats[
                            trait.stat_name + "_stdev"],
                    }
        except FileNotFoundError as e:
            print(f"Could not find file for {user_type}")
            continue

    final_dict = {}
    # add the opposite user stats to the same dict
    for user_type, stats in collected_stats.items():
        opposite_user = get_oposite_user_utterance_level(user_type)
        if not opposite_user:
            opposite_user = get_oposite_user_intents_level(user_type)

        if opposite_user:
            final_dict[user_type] = stats
            final_dict[user_type].update(collected_stats[opposite_user.user_custom_name])

    print(json.dumps(final_dict, indent=4))

    create_dataset_simple_latex_table(final_dict,
                                      add_identifying_characteristic=True,
                                      label="tab_dataset_stats",
                                      caption="Identifying Characteristics and Traits.",
                                      resize_box=False)

    # add regular user stats
    all_stats[UserTypes.RegularUser.user_custom_name] = regular_user_stats

    df = create_results_dataframe(
        all_stats,
        additional_columns=["number_dialogues", "dialogues_intent_distribution", "next_percentage",
                            "number_turns_list", "cooperativeness_list", "exploration_list",
                            "number_words_list", "sentiment_list", "fluency_list", "repetition_list",
                            "conditioned_tolerance_list"])

    # write to csv
    csv_location = f"data/dataset_versions/{version_prefix}all_stats.csv"
    df.to_csv(csv_location)

    # create latex table
    create_latex_table(csv_location, label="tab_filtered_stats", caption="Training data statistics.")

    # final dict is relevant for user type and all_stats is everything
    return final_dict, all_stats


def print_and_plot_intent_distribution(all_dialogues: List[Dialog], turn_level: bool, out_path: str = None,
                                       log_scale: bool = False):
    intent_distribution = defaultdict(int)
    intent_utterance_distribution = {}
    # iterate all dialogues
    for d in all_dialogues:
        if not turn_level:
            for turn in d.turns:
                intent = turn.intent
                utterance = turn.user_utterance
                intent_distribution[intent] += 1

                if intent not in intent_utterance_distribution:
                    intent_utterance_distribution[intent] = {}
                if utterance not in intent_utterance_distribution[intent]:
                    intent_utterance_distribution[intent][utterance] = 0
                intent_utterance_distribution[intent][utterance] += 1
        else:
            # considers only the last turn
            turn = d.turns[-1]
            intent = turn.intent
            utterance = turn.user_utterance

            if not intent:  # avoid problems with empty or null intent
                intent = "None"

            intent_distribution[intent] += 1

            if intent not in intent_utterance_distribution:
                intent_utterance_distribution[intent] = {}
            if utterance not in intent_utterance_distribution[intent]:
                intent_utterance_distribution[intent][utterance] = 0
            intent_utterance_distribution[intent][utterance] += 1

    # sort intent distribution by value
    intent_distribution = dict(sorted(intent_distribution.items(), key=lambda item: item[1], reverse=True))

    # print intent distribution
    print("Intent Distribution")
    print(json.dumps(intent_distribution, indent=2))
    if out_path:
        write_to_json_file(out_path.replace(".pdf", ".intent_distrib"), intent_distribution, indent=2)

    # plot intent distribution
    plt.bar(list(intent_distribution.keys()), list(intent_distribution.values()), log=log_scale)
    plt.xlabel('Intent')
    plt.ylabel('Number')
    plt.title('Intent Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()
    plt.close()

    # sort intent utterance distribution first by key and then by value
    for k, v in intent_utterance_distribution.items():
        intent_utterance_distribution[k] = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))

    # print intent utterance distribution
    print("Intent Utterance Distribution")
    print(json.dumps(intent_utterance_distribution, indent=2))

    if out_path:
        write_to_json_file(out_path.replace(".pdf", "_utterance_distrib"), intent_utterance_distribution, indent=2)

    return intent_distribution, intent_utterance_distribution


def get_distance_metrics_by_profile(path: str, csv_file: str, profiles: Optional[List[str]],
                                    plot_kl: bool = False, graph_suffix: str = ""):
    csv_location = os.path.join(path, csv_file)

    # read the csv file
    df = pd.read_csv(csv_location, index_col=0, header=0)

    if profiles is None:
        profiles = [i.user_custom_name for i in UserTypes.get_all_single_trait_user_types()]

    kl_divegences = {}
    all_stats = {}
    for p in profiles:
        kl_divegences[p] = {}
        all_stats[p] = {}

        x = f"data/dataset_versions/{GlobalConstants.dataset_version}_filtered_{p}/all/simulated_conversations_train_manual_distribution_config_stats.json"
        if p == UserTypes.RegularUser.user_custom_name:
            x = x.replace(f"filtered_{UserTypes.RegularUser.user_custom_name}", UserTypes.RegularUser.user_custom_name)

        # read json
        x_json = load_json_file(x)

        # get corresponding in df
        if "multi" in p.lower() and len(df) == 1:
            # special case for multi since name of row does not match name of profile
            df_row = df.iloc[0]
        else:
            df_row = df.loc[p]

        # get trait name
        trait_names = [
            "number_turns_list", "cooperativeness_list", "conditioned_tolerance_list", "exploration_list",
            "number_words_list", "sentiment_list", "fluency_list", "repetition_list",
        ]

        for trait_name in trait_names:

            # get value from df and json
            x_value = x_json[trait_name]

            value = df_row[trait_name]
            # intrepret str as list of floats
            value = json.loads(value)

            # calculate kl divergence
            try:
                all_stats_temp = calc_distance_metrics(training_stats=x_value, model_stats=value,
                                                       caption=trait_name, plot=plot_kl,
                                                       graph_suffix=graph_suffix + f"_{trait_name}")
                kl_value = all_stats_temp["kl_divergence"]
                # it is is infinty set to None
                if kl_value == np.inf:
                    print(f"Exception - Error calculating kl divergence for {p} and {trait_name} "
                          f"(len {len(value)} and {len(x_value)})")
                    print("\n#########################\n")

                kl_divegences[p][trait_name] = kl_value
                all_stats[p][trait_name] = all_stats_temp

            except Exception:
                traceback.print_exc()
                print("\n#########################\n")

    # return kl divergences and all stats
    return kl_divegences, all_stats
