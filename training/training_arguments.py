import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, Union, List

from transformers.utils import ExplicitEnum


# ======== DEFAULT ARGUMENTS ========
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_WARMUP_STEPS = 100
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 1000
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_SAVE_LIMIT = 4
BLOCK_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-5
MAX_LEN = 512
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_ACC_STEPS = 1
DEFAULT_TEMPERATURE = 1.0
DEFAULT_STOP_CRITERIA = -1.0
DEFAULT_SEED = 11731
DEFAULT_LR_STEP_SIZE = 250
DEFAULT_LR_NUM_CYCLES = 10
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_LEN = 1024
DEFAULT_GRAD_CLIP = 0.5


class CustomIntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
    BEST_EVAL = "best_eval"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, CustomIntervalStrategy):
            return self.value == other.value
        return False


class RunTypes(ExplicitEnum):
    pretrain = 'pretrain'
    train = 'train'
    inference = 'inference'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return False


class DecodingStrategies(ExplicitEnum):
    greedy = 'greedy'
    beam = 'beam'
    sampling = 'sampling'
    creative = 'creative'
    precise = 'precise'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return False


GENERATION_PRESETS = {
    "greedy": {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "early_stopping": False,
        "repetition_penalty": 1.0,
    },
    "beam": {
        "do_sample": False,
        "num_beams": 4,
        "num_return_sequences": 1,
        "early_stopping": True,
        "repetition_penalty": 1.0,
    },
    "sampling": {
        "do_sample": True,
        "num_return_sequences": 1,
        "early_stopping": False,
        "repetition_penalty": 1.0,
    },
    "precise": {
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    },
    "creative": {
        "do_sample": True,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    }
}


class PrecisionType(ExplicitEnum):
    BF16 = 'BF16'
    FP16 = 'FP16'
    FP32 = 'FP32'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, PrecisionType):
            return self.value.lower() == other.value.lower()
        return False


class CustomSchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    CYCLICAL = "cyclical"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        if isinstance(other, CustomSchedulerType):
            return self.value == other.value
        return False


class ParallelType(ExplicitEnum):
    NO = "no"
    DP = "dp"
    FSDP = "fsdp"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, ParallelType):
            return self.value.lower() == other.value.lower()
        return False


@dataclass
class ModelArguments:
    base_model: str = field(
        metadata={"help": "The name of the model used."},
    )
    ckpt_path: str = field(
        default=None,
        metadata={"help": "The path of the model to be loaded."},
    )
    truncation_side: str = field(
        default=None,
        metadata={"help": "The side to truncate the sequences."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data parent folder."})
    dataset_name: str = field(default=None, metadata={"help": "The name of the dataset folder"})
    dataset_kwargs: str = field(default="{}", metadata={"help": "The arguments for the dataset"})

    response_template: str = field(
        default="USER:",
        metadata={"help": "The response template to be used during training. Responsible for separating prompt and "
                          "completion."}
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class TrainArgs:
    experiment_name: str = field(
        default="debug",
        metadata={"help": "The name of this specific run to be used in WANDB."}
    )
    reload_optimizer: bool = field(
        default=False,
        metadata={"help": "If the optimizer should be reloaded from checkpoint."},
    )
    save_optimizer: bool = field(
        default=False,
        metadata={"help": "If the optimizer should be saved on checkpoint."},
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    run_name: Optional[str] = field(
        default="",
        metadata={"help": "Run name to be used in wandb"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether or not to load the best model at the end of training."}
    )

    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

    optim: str = field(
        default="adamw_torch"
    )
    model_max_length: int = field(
        default=DEFAULT_MAX_LEN,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(
        default='',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    project_name: str = field(
        default='debug',
        metadata={"help": "The name of the project in WANDB."},
    )
    num_train_epochs: float = field(
        default=DEFAULT_EPOCHS,
        metadata={"help": "The number of epochs to train for."},
    )
    per_device_train_batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"help": "The batch size used for each GPU during training."},
    )
    per_device_eval_batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"help": "The batch size used for each GPU during evaluation."},
    )

    eval_accumulation_steps: int = field(
        default=100,
        metadata={"help": "Number of predictions steps to accumulate the output tensors for, before moving the "
                          "results to the CPU"},
    )

    save_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The checkpoint save strategy to use."},
    )
    evaluation_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The evaluation strategy to use."},
    )
    run_type: Union[RunTypes, str] = field(
        default='train',
        metadata={"help": "Whether we are running train, inference or pretrain"},
    )
    report_to_wandb: bool = field(
        default=False,
        metadata={"help": "If the logs should be reported to WANDB"},
    )
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    stop_criteria: float = field(
        default=DEFAULT_STOP_CRITERIA,
        metadata={"help": "The eval loss lower bound."},
    )
    lr_scheduler_type: Union[CustomSchedulerType, str] = field(
        default=CustomSchedulerType.LINEAR.value,
        metadata={"help": "The scheduler type to use."},
    )
    lr_step_size: int = field(
        default=DEFAULT_LR_STEP_SIZE,
        metadata={"help": "For cyclical lr, 2 / the number of steps between peaks."},
    )
    lr_num_cycles: int = field(
        default=DEFAULT_LR_NUM_CYCLES,
        metadata={"help": "For cyclical lr, 2 / the number of steps between peaks."},
    )
    perpetual: bool = field(
        default=False,
        metadata={"help": "If set then the training will never stop. Only when killed."},
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    infer_checkpoints: bool = field(
        default=False,
        metadata={"help": "If set then the inference will be performed on all the checkpoints in the output dir."},
    )
    learning_rate: float = field(default=DEFAULT_LEARNING_RATE,
                                 metadata={"help": "The initial learning rate for AdamW."})

    '''
    mixed_precision: Union[PrecisionType, str] = field(
        default=PrecisionType.FP32.value,
        metadata={"help": "The mixed precision type to use."},
    )
    '''

    patience: int = field(
        default=None,
        metadata={"help": "The number of evaluations to wait for improvement before stopping the training."},
    )

    use_bf16: bool = field(
        default=False,
        metadata={"help": "If the model should be trained in bf16 precision."},
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "If the model should be trained."},
    )

    do_eval: bool = field(
        default=True,
        metadata={"help": "If the model should be evaluated."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=DEFAULT_LOGGING_STEPS, metadata={"help": "Log every X updates steps."})
    eval_steps: Optional[float] = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    save_steps: Optional[int] = field(default=None, metadata={"help": "Save the model every X steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_before_inference: int = field(
        default=0, metadata={"help": "Number of steps before queueing the first inference."}
    )
    param_update_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The hyperparameter update strategy to use."},
    )
    param_update_interval: int = field(
        default=500, metadata={"help": "Number of steps between hyperparameter updates."}
    )
    parallel_type: str = field(
        default=ParallelType.NO.value, metadata={"help": "The type of parallelism to use (NO, DP, FDSP)."}
    )
    load_dtype: str = field(
        default=PrecisionType.FP32.value, metadata={"help": "The data type to load the model weights into."}
    )
    load_in_8bits: bool = field(
        default=False, metadata={"help": "If the model weights should be loaded in 8 bits."}
    )
    max_grad_norm: float = field(
        default=DEFAULT_GRAD_CLIP, metadata={"help": "The gradient clipping value."}
    )
    resume_from_checkpoint: bool = field(
        default=False, metadata={"help": "If the training should resume from the latest checkpoint."}
    )
    resume_from_step: int = field(
        default=0, metadata={"help": "If the training should resume from the given step."}
    )
    no_cuda: bool = field(
        default=False, metadata={"help": "If the model should ignore CUDA."}
    )

    use_debug: bool = field(
        default=False, metadata={"help": "Runs a small sample of the examples for debugging purposes."}
    )

    run_on_test_set_dialogue_level: bool = field(
        default=False, metadata={"help": "Runs the test set on dialog level."}
    )

    one_model_per_profile: bool = field(
        default=False,
        metadata={"help": "If true it creates and saves a one model for each profile else it uses the same model for "
                          "all profiles"},
    )

    only_profiles: Optional[List[str]] = field(
        default=None,
        metadata={"help": "If set, it only trains the profiles in the list. Must also use one_model_per_profile to be "
                          "True"},
    )

    use_special_token_profiles: bool = field(
        default=False,
        metadata={"help": "If true adds all profile names as new tokens to the tokenizer"},
    )

    use_special_token_intents: bool = field(
        default=False,
        metadata={"help": "If true adds all intents as new tokens to the tokenizer"},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


@dataclass
class LoRaArguments:
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load model with lora"
            )
        },
    )
    lora_rank: int = field(
        default=16, metadata={"help": "The rank value for LoRA"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "The alpha value for LoRA"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "The dropout value for LoRA model"}
    )
    lora_merge_adapter: bool = field(
        default=False,
        metadata={
            "help": (
                "If the loaded model is already trained with LoRA, whether or not to merge the adapter"
            )
        },
    )

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        metadata={"help": "The target modules to be used during training."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class InferenceArguments:
    test_file: str = field(
        default=None,
        metadata={"help": "The path to the test file to be used during inference."},
    )
    decoding_strategy: Union[DecodingStrategies, str] = field(
        default=DecodingStrategies.beam.value,
        metadata={"help": "The decoding strategy to be used during inference."},
    )
    score_metrics: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute BLUE, METEOR, and ROUGE after inference."},
    )
    max_new_tokens: int = field(
        default=DEFAULT_MAX_NEW_TOKENS,
        metadata={"help": "The maximum number of new tokens to be generated during inference."},
    )
    max_samples: int = field(
        default=100000000000000,
        metadata={"help": "The maximum number of samples to be considered during inference."},
    )
    log_metrics: bool = field(
        default=False,
        metadata={"help": "If score_metrics is set, it will store the metrics results in a csv on the parent folder."},
    )
    fp32: bool = field(
        default=True,
        metadata={"help": "Whether or not to use fp32 precision during inference."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to use bf16 precision during inference."},
    )
    seperator_token: str = field(
        default="",
        metadata={"help": "The seperator token to be used during inference."},
    )
    inference_batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"help": "The batch size to be used during inference."},
    )
