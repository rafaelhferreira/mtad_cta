from collections import Counter
from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm

from dataset_generation.utils import load_json_file, write_to_json_file
from user_simulator.gpt_evaluation.open_ai_gpt import OpenAIParams, CallOpenAIModel
from user_simulator.traits_and_profiles.identifying_traits import user_name_to_trait
from user_simulator.traits_and_profiles.user_profile import get_opposite_user, UserTypes


def call_open_ai_analyze_system_quality(json_path: str, output_path: str, open_ai_params: OpenAIParams):

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

        # get score
        score = None
        for line in content.split("\n"):
            if "Answer:" in line:
                score = line.split("Answer:")[1].strip()
                break

        example["justification"] = justification

        try:
            example["gpt_score"] = int(score.strip())
        except Exception as e:
            print(e)
            example["gpt_score"] = None

        # print some stuff to make sure everything is ok
        print()
        print("User Utterance:", example["user_utterance"])
        print("System Utterance:", example["system_utterance"])
        print("justification:", example["justification"])
        print("GPT score:", example["gpt_score"])
        pass

    # save the results
    print(f"Saving results to {output_path}")
    write_to_json_file(output_path, comparative_dialogues)

    # calculate metrics
    metrics = calc_stats_system_quality(comparative_dialogues)
    # save the metric
    print(f"Saving metrics to {output_path.replace('.json', '_metrics.json')}")
    write_to_json_file(output_path.replace(".json", "_metrics.json"), metrics)


def calc_stats_system_quality(system_quality_dialogues: List[Dict[str, Union[str, int]]]):
    # measure average quality, number of times quality is none, and calculate stats by profile and metric
    average_quality = []
    total_none_quality = 0

    average_quality_by_profile = {}
    count_quality_by_profile = {}
    average_quality_by_metric = {}
    count_quality_by_metric = {}
    average_quality_by_intent = {}
    count_quality_by_intent = {}
    average_quality_by_id = {}

    # highs and lows
    average_quality_low_high = {}

    for example in system_quality_dialogues:
        if example["gpt_score"] is not None:
            # average quality
            average_quality.append(example["gpt_score"])

            # by profile
            profile = example["user_profile"]
            average_quality_by_profile[profile] = average_quality_by_profile.get(profile, []) + [example["gpt_score"]]
            count_quality_by_profile[profile] = count_quality_by_profile.get(profile, 0) + 1

            # by metric
            metric = user_name_to_trait.get(profile)
            if not metric:
                u_op = get_opposite_user(profile)
                if u_op:
                    metric = user_name_to_trait.get(u_op.user_custom_name)

            if metric:
                average_quality_by_metric[metric] = average_quality_by_metric.get(metric, []) + [example["gpt_score"]]
                count_quality_by_metric[metric] = count_quality_by_metric.get(metric, 0) + 1

            # by intent
            intent = example["intent"]
            average_quality_by_intent[intent] = average_quality_by_intent.get(intent, []) + [example["gpt_score"]]
            count_quality_by_intent[intent] = count_quality_by_intent.get(intent, 0) + 1

            # by id
            average_quality_by_id[example["dialog_id"]] = average_quality_by_id.get(example["dialog_id"], []) + [example["gpt_score"]]

            # by low and high
            low_or_high = None
            trait_scale = UserTypes.get_user_type_by_name(profile).trait_scale
            if trait_scale:  # avoid regular
                for trait, value in trait_scale.items():
                    if value == 0:
                        low_or_high = "low"
                    elif value == 2:
                        low_or_high = "high"

                    if low_or_high:
                        average_quality_low_high[f"{metric}_{low_or_high}"] = average_quality_low_high.get(f"{metric}_{low_or_high}", []) + [example["gpt_score"]]

        else:
            total_none_quality += 1

    # get counts for each quality
    quality_counts = Counter(average_quality)
    percentage_quality_counts = {k: v / quality_counts.total() for k, v in quality_counts.items()}

    # calculate average quality
    average_quality = np.mean(average_quality)
    # calculate average quality by profile
    average_quality_by_profile = {k: np.mean(v) for k, v in average_quality_by_profile.items()}
    # order by correctness
    average_quality_by_profile = dict(sorted(average_quality_by_profile.items(), key=lambda item: item[1], reverse=True))
    # calculate average quality by metric
    average_quality_by_metric = {k: np.mean(v) for k, v in average_quality_by_metric.items()}
    # order by correctness
    average_quality_by_metric = dict(sorted(average_quality_by_metric.items(), key=lambda item: item[1], reverse=True))
    # calculate average quality by intent
    average_quality_by_intent = {k: np.mean(v) for k, v in average_quality_by_intent.items()}
    # order by correctness
    average_quality_by_intent = dict(sorted(average_quality_by_intent.items(), key=lambda item: item[1], reverse=True))

    # by dialog id first average over the dialogue and then average over all dialogues
    average_quality_by_id = {k: np.mean(v) for k, v in average_quality_by_id.items()}
    average_quality_by_id = np.mean(list(average_quality_by_id.values()))

    # by low and high
    average_quality_low_high = {k: np.mean(v) for k, v in average_quality_low_high.items()}
    average_quality_low_high = dict(sorted(average_quality_low_high.items(), key=lambda item: item[1], reverse=True))

    # metrics
    metrics = {
        "Quality Counts": quality_counts,
        "Percentage Quality Counts": percentage_quality_counts,
        "Average Quality": average_quality,
        "Average Quality by Profile": average_quality_by_profile,
        "Average Quality by Metric": average_quality_by_metric,
        "Average Quality by Intent": average_quality_by_intent,
        "Average Quality by Dialog": average_quality_by_id,
        "Average Quality Low High": average_quality_low_high,
        "Total None Quality": total_none_quality / len(system_quality_dialogues),
    }

    # put the metrics in 0-1 scale by dividing by two
    metrics_keys = list(metrics.keys())
    avoid_keys = {"Quality Counts", "Percentage Quality Counts", "Total None Quality"}
    for k in metrics_keys:
        if k not in avoid_keys:
            new_key = k + " 0-1 scale"
            if isinstance(metrics[k], dict):
                metrics[new_key] = {}
                for k2 in metrics[k].keys():
                    if k2 not in avoid_keys:
                        metrics[new_key][k2] = metrics[k][k2] / 2
            else:
                metrics[new_key] = metrics[k] / 2

    # print dict
    for k, v in metrics.items():
        print(k, v)

    return metrics
