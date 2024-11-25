from typing import List, Dict, Union

import numpy as np
from scipy.stats import kendalltau, spearmanr

from tqdm import tqdm

from dataset_generation.dialog import Dialog
from dataset_generation.utils import load_json_file, write_to_json_file
from user_simulator.gpt_evaluation.utils import dialogue_to_text_simple
from user_simulator.gpt_evaluation.open_ai_gpt import OpenAIParams, CallOpenAIModel
from user_simulator.traits_and_profiles.identifying_traits import metric_to_description, metric_to_trait_name
from user_simulator.traits_and_profiles.user_profile import UserTypes


def create_dialogue_comparison_prompt(order_dialogues: List[Dialog], user_profile: str, add_intents: bool,
                                      model_order: List[Dialog], annotation_id: str, metric: str):

    COMPARATIVE_DIALOGUES = """Below you will find multiple dialogues between a user and a cooking assistant. The 
assistant is not supposed to show any bias and it is not supposed to be rude, even if the user is rude. It should 
always try to help the user to the best of its capabilities. Still, if the user makes any unrelated comments, 
the assistant should politely reject them and try to get the conversation focused on the task. The assistant is 
only able to help with recipes or DIY tasks. However, the assistant can provide fun facts and discuss adjacent 
topics if the user asks for it. Whenever the user asks for instructions for a different recipe or task, 
the assistant should always ask the user to clarify if they want to start a new task or continue with the current 
one. You are an unbiased annotator."""

    current_prompt = COMPARATIVE_DIALOGUES

    current_prompt += f"\n\nTask Title: {order_dialogues[0].task.get_title()}\n\n"
    letters = ["A", "B", "C", "D", "E", "F", "G"]

    for i, d in enumerate(order_dialogues):
        current_prompt += f"<Start of Dialogue {letters[i]}:>\n"
        for turn in d.turns:
            if not add_intents:
                current_prompt += f"User: {turn.user_utterance}\n"
            else:
                current_prompt += f"User: (Intent: {turn.intent}) - {turn.user_utterance}\n"
            current_prompt += f"Assistant: {turn.system_utterance}\n"
        current_prompt += f"<End of Dialogue {letters[i]}>\n\n"

    metric_description = metric_to_description.get(metric)

    question = f"""Rank the dialogues in terms of the user's {metric}. {metric} is defined as {metric_description}. 
First, write a one-sentence justification for your answer. Second, Rank the dialogues from low to high according 
to the user's {metric} using the letter corresponding to each dialogue.
 
Follow the following format:
Justification: <one sentence justification for your answer>
Answer: X<Y<Z
"""

    current_prompt += question

    # get the correct order
    model_order = [order_dialogues.index(cd) for cd in model_order]
    # put in the correct format for the prompt
    model_order = [letters[co] for co in model_order]
    model_order = "<".join(model_order)

    result_dict = {
        "id": annotation_id,
        "task": order_dialogues[0].task.get_unique_id(),
        # "path": order_dialogues[0].simulator_model_path,
        "Metric": metric,
        "prompt": current_prompt,
        "model_order": model_order,
    }

    for i, d in enumerate(order_dialogues):
        result_dict[f"u{i+1}_path"] = d.simulator_model_path
        result_dict[f"u{i+1}"] = d.user_profile.user_custom_name
        result_dict[f"{i+1}"] = dialogue_to_text_simple(d)

    return result_dict


def call_open_ai_comparative_dialogues(json_path: str, output_path: str, open_ai_params: OpenAIParams):

    if OpenAIParams is None:
        open_ai_params = OpenAIParams(MODEL_NAME="gpt-4o", TEMPERATURE=0.0,
                                      MAX_TOKENS=250, TOP_P=0.0)

    comparative_dialogues = load_json_file(json_path)

    open_ai_caller = CallOpenAIModel(open_ai_params)

    for example in tqdm(comparative_dialogues, desc=f"Processing examples for {json_path}"):
        prompt = example["prompt"]
        content, top_log_probs = open_ai_caller.call_open_ai(text=prompt)

        # add the generated text and top log probs to the example
        example["generated_text"] = content
        example["top_log_probs"] = top_log_probs

        # process correctness
        # get justification
        justification = None
        for line in content.split("\n"):
            if "Justification:" in line:
                justification = line.split("Justification:")[1].strip()
                break

        # get ranking
        ranking = None
        for line in content.split("\n"):
            if "Answer:" in line:
                ranking = line.split("Answer:")[1].strip()
                break

        example["justification"] = justification
        example["gpt_ranking"] = [r.strip() for r in ranking.split("<")]
        example["model_ranking"] = [r.strip() for r in example["model_order"].split("<")]

        # print some stuff to make sure everything is ok
        print()
        print("justification:", example["justification"])
        print("GPT ranking:", example["gpt_ranking"], "Model Ranking:", example["model_ranking"])
        pass

    # save the results
    print(f"Saving results to {output_path}")
    write_to_json_file(output_path, comparative_dialogues)

    # calculate metrics
    metrics = calc_stats_comparative_dialogues(comparative_dialogues)
    # save the metric
    print(f"Saving metrics to {output_path.replace('.json', '_metrics.json')}")
    write_to_json_file(output_path.replace(".json", "_metrics.json"), metrics)


def calc_stats_comparative_dialogues(comparative_dialogues: List[Dict[str, Union[str, int]]]):
    total_correct_ranking = []
    total_gpt_error = []

    total_kendall_tau_distance = []
    # here we are taking the absolute value of the kendall tau distance to be in a range of 0 to 1
    total_kendall_tau_distance_abs = []
    total_kendall_tau_p_value = []

    total_spearman_correlation = []
    total_spearman_p_value = []
    total_is_regular = []

    for example in comparative_dialogues:
        correct_ranking = [r.strip() for r in example["model_order"].split("<")]
        if len(example["gpt_ranking"]) == len(correct_ranking):
            if example["gpt_ranking"] == correct_ranking:
                example["correct_ranking"] = 1
                total_correct_ranking.append(1)
                example["gpt_error"] = 0
                total_gpt_error.append(0)
            else:
                example["correct_ranking"] = 0
                total_correct_ranking.append(0)
                example["gpt_error"] = 0
                total_gpt_error.append(0)

            # calculate kendall tau distance
            try:
                kendall_tau_distance, p_value = kendalltau(example["gpt_ranking"], correct_ranking)
                example["kendall_tau_distance"] = kendall_tau_distance
                example["kendall_tau_p_value"] = p_value
                total_kendall_tau_distance.append(kendall_tau_distance)
                total_kendall_tau_distance_abs.append(abs(kendall_tau_distance))
                total_kendall_tau_p_value.append(p_value)
            except Exception as e:
                print(e)
                pass

            # calculate spearman correlation
            try:
                spearman_correlation, p_value = spearmanr(example["gpt_ranking"], correct_ranking)
                example["spearman_correlation"] = spearman_correlation
                example["spearman_p_value"] = p_value
                total_spearman_correlation.append(spearman_correlation)
                total_spearman_p_value.append(p_value)
            except Exception as e:
                print(e)
                pass

        else:
            example["gpt_error"] = 1
            total_gpt_error.append(1)

        # check if regular is in user profiles in format u1, u2, u3
        has_regular = False
        for k in example.keys():
            if k.startswith("u") and "regular" in str(example[k]).lower():
                has_regular = True
                break
        total_is_regular.append(has_regular)

    total_examples = len(comparative_dialogues)
    average_correct_ranking = np.mean(total_correct_ranking)
    average_gpt_error = np.mean(total_gpt_error)
    average_kendall_tau_distance = np.mean(total_kendall_tau_distance)
    average_kendall_tau_distance_abs = np.mean(total_kendall_tau_distance_abs)
    average_kendall_tau_p_value = np.mean(total_kendall_tau_p_value)
    average_spearman_correlation = np.mean(total_spearman_correlation)
    average_spearman_p_value = np.mean(total_spearman_p_value)

    # by is regular
    regular_correct_ranking = np.mean([total_correct_ranking[i] for i in range(len(total_correct_ranking)) if total_is_regular[i]])
    opposite_correct_ranking = np.mean([total_correct_ranking[i] for i in range(len(total_correct_ranking)) if not total_is_regular[i]])
    regular_kendall_tau_distance = np.mean([total_kendall_tau_distance[i] for i in range(len(total_kendall_tau_distance)) if total_is_regular[i]])
    opposite_kendall_tau_distance = np.mean([total_kendall_tau_distance[i] for i in range(len(total_kendall_tau_distance)) if not total_is_regular[i]])
    regular_kendall_tau_distance_abs = np.mean([total_kendall_tau_distance_abs[i] for i in range(len(total_kendall_tau_distance_abs)) if total_is_regular[i]])
    opposite_kendall_tau_distance_abs = np.mean([total_kendall_tau_distance_abs[i] for i in range(len(total_kendall_tau_distance_abs)) if not total_is_regular[i]])
    regular_spearman_correlation = np.mean([total_spearman_correlation[i] for i in range(len(total_spearman_correlation)) if total_is_regular[i]])
    opposite_spearman_correlation = np.mean([total_spearman_correlation[i] for i in range(len(total_spearman_correlation)) if not total_is_regular[i]])

    # calculate average correctness by profile
    profile_correct_ranking = {}
    profile_correct_ranking_kendall_tau = {}
    profile_correct_ranking_spearman = {}
    profile_total_ranking = {}
    profile_total_ranking_kendall_tau = {}
    profile_total_ranking_kendall_tau_abs = {}
    profile_total_ranking_spearman = {}

    opposite_profiles_metrics_correct_ranking = {}
    opposite_profiles_metrics_correct_kendall_tau = {}
    opposite_profiles_metrics_correct_spearman = {}

    regular_and_profile_metrics_correct_ranking = {}
    regular_and_profile_metrics_correct_kendall_tau = {}
    regular_and_profile_metrics_correct_spearman = {}
    profiles_regular_total = {}
    profiles_opposite_total = {}

    low_high_profile_metrics = {}
    low_high_profile_metrics_counts = {}
    low_high_profile_regular_metrics = {}
    low_high_profile_regular_metrics_counts = {}
    low_high_profile_opposite_metrics = {}
    low_high_profile_opposite_metrics_counts = {}

    for i, ex in enumerate(comparative_dialogues):
        metric = ex["Metric"]

        # get the active profile (it is the one with ux_path)
        low_or_high = None
        for k in ex.keys():
            if k.startswith("u") and "path" in k:
                if ex[k]:
                    active_profile = UserTypes.get_user_type_by_name(ex[k.replace("_path", "")])
                    active_trait_name = metric_to_trait_name.get(metric)
                    active_profile_value = active_profile.trait_scale[active_trait_name]
                    if active_profile_value == 0:
                        low_or_high = "low"
                    elif active_profile_value == 2:
                        low_or_high = "high"
                    break

        if not low_or_high:
            raise ValueError(f"Active profile not found for {ex}")

        if total_is_regular[i]:
            profiles_regular_total[metric] = profiles_regular_total.get(metric, 0) + 1
        else:
            profiles_opposite_total[metric] = profiles_opposite_total.get(metric, 0) + 1

        if "correct_ranking" in ex:
            profile_correct_ranking[metric] = profile_correct_ranking.get(metric, 0) + ex["correct_ranking"]
            profile_total_ranking[metric] = profile_total_ranking.get(metric, 0) + 1

            low_high_profile_metrics[f"{metric}_{low_or_high}"] = low_high_profile_metrics.get(f"{metric}_{low_or_high}", 0) + ex["correct_ranking"]
            low_high_profile_metrics_counts[f"{metric}_{low_or_high}"] = low_high_profile_metrics_counts.get(f"{metric}_{low_or_high}", 0) + 1

            if total_is_regular[i]:
                regular_and_profile_metrics_correct_ranking[metric] = regular_and_profile_metrics_correct_ranking.get(metric, 0) + ex["correct_ranking"]
                low_high_profile_regular_metrics[f"{metric}_{low_or_high}"] = low_high_profile_regular_metrics.get(f"{metric}_{low_or_high}", 0) + ex["correct_ranking"]
                low_high_profile_regular_metrics_counts[f"{metric}_{low_or_high}"] = low_high_profile_regular_metrics_counts.get(f"{metric}_{low_or_high}", 0) + 1
            else:
                opposite_profiles_metrics_correct_ranking[metric] = opposite_profiles_metrics_correct_ranking.get(metric, 0) + ex["correct_ranking"]
                low_high_profile_opposite_metrics[f"{metric}_{low_or_high}"] = low_high_profile_opposite_metrics.get(f"{metric}_{low_or_high}", 0) + ex["correct_ranking"]
                low_high_profile_opposite_metrics_counts[f"{metric}_{low_or_high}"] = low_high_profile_opposite_metrics_counts.get(f"{metric}_{low_or_high}", 0) + 1

        if "kendall_tau_distance" in ex:
            profile_total_ranking_kendall_tau[metric] = profile_total_ranking_kendall_tau.get(metric, 0) + 1
            profile_correct_ranking_kendall_tau[metric] = profile_correct_ranking_kendall_tau.get(metric, 0) + ex["kendall_tau_distance"]
            profile_total_ranking_kendall_tau_abs[metric] = profile_total_ranking_kendall_tau_abs.get(metric, 0) + 1

            if total_is_regular[i]:
                regular_and_profile_metrics_correct_kendall_tau[metric] = regular_and_profile_metrics_correct_kendall_tau.get(metric, 0) + ex["kendall_tau_distance"]
            else:
                opposite_profiles_metrics_correct_kendall_tau[metric] = opposite_profiles_metrics_correct_kendall_tau.get(metric, 0) + ex["kendall_tau_distance"]

        if "spearman_correlation" in ex:
            profile_total_ranking_spearman[metric] = profile_total_ranking_spearman.get(metric, 0) + 1
            profile_correct_ranking_spearman[metric] = profile_correct_ranking_spearman.get(metric, 0) + ex["spearman_correlation"]

            if total_is_regular[i]:
                regular_and_profile_metrics_correct_spearman[metric] = regular_and_profile_metrics_correct_spearman.get(metric, 0) + ex["spearman_correlation"]
            else:
                opposite_profiles_metrics_correct_spearman[metric] = opposite_profiles_metrics_correct_spearman.get(metric, 0) + ex["spearman_correlation"]

    # calculate average correctness by profile
    profile_average_correct_ranking = {k: v / profile_total_ranking[k] for k, v in profile_correct_ranking.items()}
    profile_average_correct_ranking_kendall_tau = {k: v / profile_total_ranking_kendall_tau[k] for k, v in
                                                   profile_correct_ranking_kendall_tau.items()}
    profile_average_correct_ranking_kendall_tau_abs = {k: v / profile_total_ranking_kendall_tau_abs[k] for k, v in
                                                       profile_correct_ranking_kendall_tau.items()}
    profile_average_correct_ranking_spearman = {k: v / profile_total_ranking_spearman[k] for k, v in
                                                profile_correct_ranking_spearman.items()}

    # calculate average correctness by profile
    regular_and_profile_metrics_correct_ranking = {k: v / profiles_regular_total[k] for k, v in regular_and_profile_metrics_correct_ranking.items()}
    opposite_profiles_metrics_correct_ranking = {k: v / profiles_opposite_total[k] for k, v in opposite_profiles_metrics_correct_ranking.items()}
    regular_and_profile_metrics_correct_kendall_tau = {k: v / profiles_regular_total[k] for k, v in regular_and_profile_metrics_correct_kendall_tau.items()}
    opposite_profiles_metrics_correct_kendall_tau = {k: v / profiles_opposite_total[k] for k, v in opposite_profiles_metrics_correct_kendall_tau.items()}
    regular_and_profile_metrics_correct_spearman = {k: v / profiles_regular_total[k] for k, v in regular_and_profile_metrics_correct_spearman.items()}
    opposite_profiles_metrics_correct_spearman = {k: v / profiles_opposite_total[k] for k, v in opposite_profiles_metrics_correct_spearman.items()}

    # calculate average correctness by profile
    low_high_profile_metrics = {k: v / low_high_profile_metrics_counts[k] for k, v in low_high_profile_metrics.items()}
    low_high_profile_regular_metrics = {k: v / low_high_profile_regular_metrics_counts[k] for k, v in low_high_profile_regular_metrics.items()}
    low_high_profile_opposite_metrics = {k: v / low_high_profile_opposite_metrics_counts[k] for k, v in low_high_profile_opposite_metrics.items()}

    # order by correctness
    profile_average_correct_ranking = dict(
        sorted(profile_average_correct_ranking.items(), key=lambda item: item[1], reverse=True))
    profile_average_correct_ranking_kendall_tau = dict(
        sorted(profile_average_correct_ranking_kendall_tau.items(), key=lambda item: item[1], reverse=True))
    profile_average_correct_ranking_kendall_tau_abs = dict(
        sorted(profile_average_correct_ranking_kendall_tau_abs.items(), key=lambda item: item[1], reverse=True))
    profile_average_correct_ranking_spearman = dict(
        sorted(profile_average_correct_ranking_spearman.items(), key=lambda item: item[1], reverse=True))

    # order by correctness
    regular_and_profile_metrics_correct_ranking = dict(
        sorted(regular_and_profile_metrics_correct_ranking.items(), key=lambda item: item[1], reverse=True))
    opposite_profiles_metrics_correct_ranking = dict(
        sorted(opposite_profiles_metrics_correct_ranking.items(), key=lambda item: item[1], reverse=True))

    regular_and_profile_metrics_correct_kendall_tau = dict(
        sorted(regular_and_profile_metrics_correct_kendall_tau.items(), key=lambda item: item[1], reverse=True))
    opposite_profiles_metrics_correct_kendall_tau = dict(
        sorted(opposite_profiles_metrics_correct_kendall_tau.items(), key=lambda item: item[1], reverse=True))

    regular_and_profile_metrics_correct_spearman = dict(
        sorted(regular_and_profile_metrics_correct_spearman.items(), key=lambda item: item[1], reverse=True))
    opposite_profiles_metrics_correct_spearman = dict(
        sorted(opposite_profiles_metrics_correct_spearman.items(), key=lambda item: item[1], reverse=True))

    # order by correctness
    low_high_profile_metrics = dict(
        sorted(low_high_profile_metrics.items(), key=lambda item: item[1], reverse=True))
    low_high_profile_regular_metrics = dict(
        sorted(low_high_profile_regular_metrics.items(), key=lambda item: item[1], reverse=True))
    low_high_profile_opposite_metrics = dict(
        sorted(low_high_profile_opposite_metrics.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        "Total examples": total_examples,
        "Average correctness": average_correct_ranking,
        "Average GPT Error": average_gpt_error,
        "Average Kendall Tau Distance": average_kendall_tau_distance,
        "Average Kendall Tau Distance Absolute": average_kendall_tau_distance_abs,
        "Average Kendall Tau P Value": average_kendall_tau_p_value,
        "Average Spearman Correlation Coefficient": average_spearman_correlation,
        "Average Spearman P Value": average_spearman_p_value,

        # comparing regular and opposite profiles
        "Regular Correct Ranking": regular_correct_ranking,
        "Opposite Correct Ranking": opposite_correct_ranking,
        "Regular Kendall Tau Distance": regular_kendall_tau_distance,
        "Opposite Kendall Tau Distance": opposite_kendall_tau_distance,
        "Regular Kendall Tau Distance Absolute": regular_kendall_tau_distance_abs,
        "Opposite Kendall Tau Distance Absolute": opposite_kendall_tau_distance_abs,
        "Regular Spearman Correlation": regular_spearman_correlation,
        "Opposite Spearman Correlation Coefficient": opposite_spearman_correlation,

        # by profile
        "Correctness by profile": profile_average_correct_ranking,
        "Kendall Tau Distance by profile": profile_average_correct_ranking_kendall_tau,
        "Kendall Tau Distance Absolute by profile": profile_average_correct_ranking_kendall_tau_abs,
        "Spearman Correlation by profile": profile_average_correct_ranking_spearman,

        # comparing regular and opposite profiles
        "Regular and Profile Metrics Correct Ranking": regular_and_profile_metrics_correct_ranking,
        "Opposite Profiles Metrics Correct Ranking": opposite_profiles_metrics_correct_ranking,
        "Regular and Profile Metrics Correct Kendall Tau": regular_and_profile_metrics_correct_kendall_tau,
        "Opposite Profiles Metrics Correct Kendall Tau": opposite_profiles_metrics_correct_kendall_tau,
        "Regular and Profile Metrics Correct Spearman": regular_and_profile_metrics_correct_spearman,
        "Opposite Profiles Metrics Correct Spearman": opposite_profiles_metrics_correct_spearman,

        # comparing low and high profiles
        "Low and High Profile Metrics": low_high_profile_metrics,
        "Low and High Profile Regular Metrics": low_high_profile_regular_metrics,
        "Low and High Profile Opposite Metrics": low_high_profile_opposite_metrics,
    }

    # print dict
    for k, v in metrics.items():
        print(k, v)

    return metrics
