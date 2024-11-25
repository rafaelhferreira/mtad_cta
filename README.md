# Multi-trait User Simulation with Adaptive Decoding for Conversational Task Assistants

This is the repository for the paper "Multi-trait User Simulation with Adaptive Decoding for Conversational Task Assistants" published in EMNLP 2024 Findings [here](https://aclanthology.org/2024.findings-emnlp.945/).

## Creating the Environment
If you use conda you can create the env using the [environment.yml](environment.yml) file (depending on the hardware some versions might need to be different):


```bash
conda env create -f environment.yml
```

Activate environment with:

```bash
conda activate simulator_cta
```


## Creating the Generated Dialogues
In the paper, to evaluate our models, we use data collected in the first edition of the [Alexa Prize TaskBot challenge](https://www.amazon.science/alexa-prize/proceedings/twiz-a-conversational-task-wizard-with-multimodal-curiosity-exploration).\
Due to Alexa Prize Challenge rules the data cannot be shared, but you can provide your own data to train the models.

[generate_conversations.py](generate_conversations.py) launches our profile-aware data generation pipeline.

To make it work with your data there are a few things you need to have:
- **collected_utterances_file** - a json file with the collected utterances in this format:
```json
{
    "intent_1": {
        "utterance_1": "count_of_utterance_1",
        "utterance_2": "count_of_utterance_2",
        ...
  },
    "intent_2": {
        "utterance_1": "count_of_utterance_1",
        "utterance_2": "count_of_utterance_2",
        ...
    },
    ...
}
```
where `intent_1` is the name of the intent and `utterance_1` is the utterance itself and `count_of_utterance_1` is the number of times that utterance was collected.

- **first_steps_prob_file** - a json file with the first steps probabilities in this format:
```json
{
    "intent_1": "probability_1",
    "intent_2": "probability_2",
    ...
}
```
where `intent_1` is the name of the intent and `probability_1` is the probability of that intent being the first step.

- **tasks_folder_path** - a path to a **folder** with **3 folders**: **train**, **valid**, and **test**, each one with various json files in the format
  (if your data does not use tasks you can ignore this):
```json
{
    "id": "id",  
    "title": "title",
    "rating": 2.5,
    "methods": ["Pre-heat the oven...", "Mix in the ingredients...", "Bake for 30 minutes..."],
    "thumbnail_img_url": "thumbnail_img_url",
    "data_source": "recipe",   
    "description": "description"
}
```

- Update the **intents** to be used in the dataset generation:
  - Add your intents to the `Intents` class in [enumerates.py](data_binding%2Fenumerates.py) and fill the appropriate methods with the intents you want to use.
  - Change the `CONSIDERED_INTENTS` variable in [utils.py](dataset_generation%2Futils.py) for the intens you want to consider during dataset generation.
  - Change `MANUAL_PROBS_REVISED` in [prob_distribution.py](dataset_generation%2Fprob_distribution.py) to the intent transition probabilities you want to use.
  
After doing all of this run the following command to generate the dataset for a single trait:
```bash
python3 generate_conversations.py --collected_utterances_file "data/collected_utterances.json" --first_steps_prob_file "data/first_steps_prob.json" --tasks_folder_path "data/tasks" --single_trait 
```

If you want to generate the dataset for multiple traits, you can use the following command:
```bash
python3 generate_conversations.py --collected_utterances_file "data/collected_utterances.json" --first_steps_prob_file "data/first_steps_prob.json" --tasks_folder_path "data/tasks" --multi_trait 
```

This might take some time to generate given that it uses 2 Transformer models to calculate emotion and fluency.
Use a GPU if possible. 
To speed up this process you can pre-process all utterances of your dataset and save them in a cache file using the `calculate_cache` argument.

The script will create a folder for each profile with the following files:
- `simulated_conversations_test_manual_distribution.json`
- `simulated_conversations_test_manual_distribution_config.json`
- `simulated_conversations_test_manual_distribution_config_stats.json`
- `simulated_conversations_train_manual_distribution.json`
- `simulated_conversations_train_manual_distribution_config.json`
- `simulated_conversations_train_manual_distribution_config_stats.json`
- `simulated_conversations_valid_manual_distribution.json`
- `simulated_conversations_valid_manual_distribution_config.json`
- `simulated_conversations_valid_manual_distribution_config_stats.json`

After this it will filter the dialogues according to the statistics of the **Regular** profile and create another set of folders for each profile with the prefix **"filtered"**.
This way, we can garantee that the final dialogues are indeed respecting the characteristics of the profile.


## Creating the Training Dataset

[create_dataset_models.py](create_dataset_models.py) creates the dataset to train, validate, and test the models using the previous step data.

Example usage
```bash
python3 create_dataset_models.py --context_window 5 --avoid_next_percentage 0.5 --turn_level_user_profile_format "single_word" --use_special_intent_tokens True --input_folder_path "data/dataset_versions/1.0_filtered_Concise/all" "data/dataset_versions/1.0_filtered_Cooperative/all" "data/dataset_versions/1.0_filtered_Explorative/all" "data/dataset_versions/1.0_filtered_Fluent/all" "data/dataset_versions/1.0_filtered_Impatient/all" "data/dataset_versions/1.0_filtered_Intolerant/all" "data/dataset_versions/1.0_filtered_Negative/all" "data/dataset_versions/1.0_filtered_NonExplorative/all" "data/dataset_versions/1.0_filtered_NonFluent/all" "data/dataset_versions/1.0_filtered_NonRepetitive/all" "data/dataset_versions/1.0_filtered_Patient/all" "data/dataset_versions/1.0_filtered_Positive/all" "data/dataset_versions/1.0_filtered_Repetitive/all" "data/dataset_versions/1.0_filtered_Tolerant/all" "data/dataset_versions/1.0_filtered_UnCooperative/all" "data/dataset_versions/1.0_filtered_Verbose/all" "data/dataset_versions/1.0_Regular/all" --output_folder_path "data/json_dataset_stratify" --num_train_samples_per_file 1000 --num_eval_samples_per_file 100 --add_intent_start_to_completion True --stratify_user_profile True

```

Check `DatasetCreationArguments` class for more information on the arguments.


Replace the `input_folder_path` with folder paths to the dataset versions you want to use to create the dataset
It should be a folder with at least these **3 json files** (train, valid, and test):
- `simulated_conversations_train_manual_distribution.json`
- `simulated_conversations_valid_manual_distribution.json`
- `simulated_conversations_test_manual_distribution.json`

#### Data Format

Each json file should follow the structure in the example below.
(which is automatically generated by the previous step)
```json
[
   {
      "dialog_id":"cc947e42-4836-4132-8a35-a1dfccf77281",
      "task":{
         ...
      },
      "dialog":[
         {
            "intent":"StartStepsIntent",
            "current_step":0,
            "user":"let's get started",
            "system":"Time to work your culinary magic! Step 1: Mix together Olive Oil, Garlic, Orange, Lemon, Granulated Sugar, Italian Seasoning, Smoked Paprika, Crushed Red Pepper Flakes, and Salt and Pepper.",
            "negative_response":"",
            "forced_system_error":false
         },
         {
            "intent":"NextStepIntent",
            "current_step":1,
            "user":"what's next",
            "system":"Let's keep the momentum going, Step 2: Place Chicken Drumstick in an ovenproof roasting dish. Carefully pour marinade over the chicken. Using tongs, flip the chicken pieces to coat on all sides.",
            "negative_response":"Making progress! Step 0: Place the chicken in the oven for 1 hour, or until the chicken is cooked all the way through and the juices run clear.",
            "forced_system_error":false
         }
      ],
      "system_tone":"neutral",
      "user_profile":{
         "user_custom_name":"Fluent",
         "impatience":1.0,
         "min_number_turns":3,
         "uncooperativeness":1.0,
         "top_p_exploration":0.001,
         "increase_exploration":true,
         "exploration_distrib_factor":0.0,
         "intolerance":3.0,
         "bottom_unconciseness":0.0,
         "top_unconciseness":1.0,
         "bottom_emotion":0.0,
         "top_emotion":1.0,
         "bottom_fluency":0.5,
         "top_fluency":1.0,
         "fluency_error_rate":0.0,
         "repetition_exact_match_p":0.15,
         "repetition_overlap_match_p":0.15,
         "trait_scale":{
            "fluency":2
         }
      }
   }
]
```

i.e., a list of dialogues, each one with a dialog_id, a task, a list of dialog turns, a system tone and a user profile.

At the end you should have a new folder with the following files:
- `dataset_test.json`
- `dataset_test_config.json`
- `dataset_train.json`
- `dataset_train_config.json`
- `dataset_valid.json`
- `dataset_valid_config.json`

Each item in the file is a prompt to the model and the completion it should generate:
```json
{
    "prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The user interacts with the system following its specific user profile description.\n\n[User Profile: <|Concise|>] USER: [Intent:",
    "completion": " <|start|>] start</s>",
    "user_profile": "Concise",
    "intent": "StartStepsIntent",
    "dialogue_id": "10d6d5cf-45db-4e6f-ab84-9da8b8f61bc6"
}
```

## Training the Models

[train_models.py](train_models.py) trains the models and evaluates using the previous step data.

Example usage
```bash
python3 train_models.py --base_model "lmsys/vicuna-7b-v1.5" --output_dir "user_simulator_models" --data_path "data/json_dataset_stratify" --use_bf16 True --lora True --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --save_total_limit 5 --truncation_side "left" --response_template "USER: [Intent:" --save_strategy "epoch" --evaluation_strategy "epoch" --num_train_epochs 15 --use_special_token_profiles True --use_special_token_intents True --target_modules q_proj v_proj k_proj o_proj lm_head embed_tokens
```

Make sure the parameter: `response_template` is in accordance **with the dataset** you are using.

Relevant argument: `one_model_per_profile`
- set to **False** - the model will be trained with all traits and intensities (equivalent to **JTS** in the paper)
- set to **True** - it will create a single model for each trait-intensity pair (equivalent to **STS** in the paper)

Check `ModelArguments`, `DataArguments`, `TrainArgs` and `LoRaArguments` classes for more information on the arguments.


## Evaluating the Models

[launch_self_chat.py](launch_self_chat.py) launches a self-chat evaluation using the trained models.
So it will launch a conversation between the **user simulator** and the specified **system model**.

The code is made to work with [PlanLLM](https://aclanthology.org/2024.eacl-long.77/) change the code in the `run_sim` function if you want to use another model.

Example usage:
```bash
python3 launch_self_chat.py --simulator_path "user_simulator_models/<replace_by_path>" --system_path "dmgcsilva/PlanLLM" --output_path "data/simulated_conversation" --conversations_per_task 1 --tasks_path "data/tasks/test" --max_tasks 100 --prompt_version "v5" --context_size 5 --max_turns_dialogue 20 --use_bf16 True --max_new_tokens_user 50 --max_new_tokens_system 120 --seed 42
```
This will run the system on all main traits. 

If you specify the `user_profiles` argument it will make the evaluation use only the user profiles in the list. 

Example usage:

```bash
python3 launch_self_chat.py --simulator_path "user_simulator_models" --system_path "system_models" --output_path "data/simulated_conversation" --conversations_per_task 1 --tasks_path "data/tasks/test" --max_tasks 100 --prompt_version "v5" --context_size 5 --max_turns_dialogue 20 --use_bf16 True --max_new_tokens_user 50 --max_new_tokens_system 120 --seed 42 --user_profiles "[Explorative]"
```

Make sure the model you are using was **trained on the same profiles** you are evaluating.

Check `RunSimArguments` class for more information on the arguments.

### Multi-trait Adaptive Decoding (mTAD)
To test the mTAD decoding strategy, you can use following:
```bash
python3 launch_self_chat.py --simulator_path "user_simulator_models" --system_path "system_models" --output_path "data/simulated_conversation" --conversations_per_task 1 --tasks_path "data/tasks/test" --max_tasks 100 --prompt_version "v5" --context_size 5 --max_turns_dialogue 20 --use_bf16 True --max_new_tokens_user 50 --max_new_tokens_system 120 --seed 42 --user_profiles "["profile_1", "profile_2"]" --merging_type "mtad" --extra_loras_list "simulator_path_1 simulator_path_2" --extra_loras_weights 0.33 0.33 0.33
```

You can test various combination strategies using the following arguments.
- `user_profiles` - list of user profiles to be used - use format \"[\"profile_1\", \"profile_2\"]\" - use the quotes as shown 
- `merging_type` - check in [ModelMergingTypes](user_simulator%2Fmodel_merging%2Futils.py) for more merging strategies  
- `extra_loras_list` - varius paths to other models to be used in the merging
- `extra_loras_weights` - weights to be used in the combination - should have the same length as the extra_loras_list + 1 (+1 is for the simulator_path), if not specified it will uniformly distribute the weights


### Calculating Distance-based Metrics
Distance based metrics compare the distribution given in the generated dialogues with the distribution of the reference distribution.\
Use [get_distance_metrics.py](get_distance_metrics.py) to calculate the distance-based metrics.


Example usage fo JTS:
```bash
python3 get_distance_metrics.py --path "data/simulated_conversations/model" --label "jts" --caption "jts" --is_single_model
```

Example usage for STS:
```bash
python get_distance_metrics.py --path "data/simulated_conversations/model" --label "sts" --caption "sts"
```

Example usage for combination methods (e.g. mTAD):
```bash
python get_distance_metrics.py --path "data/simulated_conversations_mtad/model/trait/trait_combination" --label "mtad" --caption "mtad" --profile "multi_trait_custom_name" --merging_methods "mtad" --is_combination --is_final_path
```

Here `path` is the path to a folder with the generated dialogues.


## Evaluating using OpenAI Models

Replace the **API key** for OpenAI with your own.
To do this add an environmental variable called `OPEN_AI_API_KEY`.

[run_gpt_eval.py](run_gpt_eval.py) evaluates the models performance on **system response quality** and **trait modeling accuracy**.

Example usage:
```bash
python3 run_gpt_eval.py --utterances_file_path "data/collected_utterances.json" --dialogues_folders "data/simulated_conversations/model" --test_dialogues_path "data/test_dialogues" --output_path "data/annotations" --model_name "gpt-4o" --number_dialogues 10 --calc_trait_modeling_accuracy --calc_system_response_quality
```

When considering **multi-trait** evaluation, use the following arguments:
```bash
python3 run_gpt_eval.py --utterances_file_path "data/collected_utterances.json" --dialogues_folders "profile_1_dialogues" "profile_2_dialogues" --test_dialogues_path "data/test_dialogues" --output_path "data/annotations" --profile_names "profile1" "profile2" --model_name "gpt-4o" --number_dialogues 10 --calc_trait_modeling_accuracy --calc_system_response_quality --is_multi_trait
```


## Citation
If you find it useful please cite our work:
```
@inproceedings{ferreira-etal-2024-multi,
    title = "Multi-trait User Simulation with Adaptive Decoding for Conversational Task Assistants",
    author = "Ferreira, Rafael  and
      Semedo, David  and
      Magalhaes, Joao",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.945",
    pages = "16105--16130"
}
```