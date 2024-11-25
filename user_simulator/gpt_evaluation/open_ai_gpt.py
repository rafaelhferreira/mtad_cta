import os
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Union

import numpy as np
import openai
from enum import Enum

from openai import OpenAI


# https://platform.openai.com/docs/models/overview
class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"
    GPT_4_TURBO = "gpt-4-0125-preview"


def default_stop_sequences():
    stop_sequences = ["User:", "System:", "USER:", "SYSTEM:", "Assistant:", "ASSISTANT:"]
    return stop_sequences


@dataclass
class OpenAIParams:

    OPEN_AI_API_KEY: str = field(
        default=os.getenv("OPEN_AI_API_KEY", ""),
        metadata={"help": "Open AI API key"}
    )

    MAX_API_RETRY: int = field(
        default=2,
        metadata={"help": "The number tries to call the OpenAI API when it fails"}
    )

    MODEL_NAME: str = field(
        default=OpenAIModels.GPT_3_5_TURBO.value,
        metadata={"help": "The model name to be used"}
    )

    TEMPERATURE: float = field(
        default=0.0,
        metadata={"help": "The temperature to be used"}
    )

    MAX_TOKENS: int = field(
        default=50,
        metadata={"help": "The maximum number of tokens to be generated"}
    )

    SEED: int = field(
        default=42,
        metadata={"help": "The seed to be used"}
    )

    TOP_P: float = field(
        default=0.8,
        metadata={"help": "The top p to be used"}
    )

    LOG_PROBS: bool = field(
        default=True,
        metadata={"help": "If true it will return the log probs"}
    )

    TOP_LOG_PROBS: int = field(
        default=5,
        metadata={"help": "The number of top log probs to be returned"}
    )

    STOP_SEQUENCES: Union[List[str], None] = field(
        default_factory=default_stop_sequences,
        metadata={"help": "The stop sequences to be used"}
    )

    def to_dict(self):
        return {
            "OPEN_AI_API_KEY": "?",
            "MAX_API_RETRY": self.MAX_API_RETRY,
            "MODEL_NAME": self.MODEL_NAME,
            "TEMPERATURE": self.TEMPERATURE,
            "MAX_TOKENS": self.MAX_TOKENS,
            "SEED": self.SEED,
            "TOP_P": self.TOP_P,
            "LOG_PROBS": self.LOG_PROBS,
            "TOP_LOG_PROBS": self.TOP_LOG_PROBS,
            "STOP_SEQUENCE": self.STOP_SEQUENCES
        }

    def __str__(self):
        return str(self.to_dict())


class CallOpenAIModel:

    def __init__(self, model_name: OpenAIParams):
        self.model_params = model_name
        self.client = OpenAI(api_key=OpenAIParams.OPEN_AI_API_KEY)

    def call_open_ai(self, text: str, **kwargs) -> Tuple[str, Optional[Dict[str, float]]]:

        openai.api_key = kwargs.get("api_key", self.model_params.OPEN_AI_API_KEY)
        max_retries = kwargs.get("max_retries", self.model_params.MAX_API_RETRY)
        model_name = kwargs.get("model_name", self.model_params.MODEL_NAME)
        temperature = kwargs.get("temperature", self.model_params.TEMPERATURE)
        max_tokens = kwargs.get("max_tokens", self.model_params.MAX_TOKENS)
        # seed = kwargs.get("seed", self.model_params.SEED)  # removed seed because various runs would give same result
        top_p = kwargs.get("top_p", self.model_params.TOP_P)

        # to get probability distribution of tokens
        logprobs = kwargs.get("logprobs", self.model_params.LOG_PROBS)
        top_logprobs = kwargs.get("top_logprobs", self.model_params.TOP_LOG_PROBS)
        stop_sequences = kwargs.get("stop_sequence", self.model_params.STOP_SEQUENCES)

        prob_yes_no_dict = None
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": text
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,  # temperature is zero to get the most probable answer
                    # seed=seed,
                    top_p=top_p,
                    logprobs=logprobs,
                    # since it comes in log prob for up to 5 tokens we cannot calculate the probability correctly
                    top_logprobs=top_logprobs,
                    stop=stop_sequences,
                )
                # print(response)
                # content = response["choices"][0]["text"]
                content = response.choices[0].message.content

                if logprobs and top_logprobs:
                    prob_yes_no_dict = self.calc_prob_dict(response.choices[0].logprobs.content)

                return content, prob_yes_no_dict
            except Exception as e:
                print(e)
                time.sleep(2)
        print(f"Failed after {max_retries} retries for input {text}")
        return "error", prob_yes_no_dict

    @staticmethod
    def calc_prob_dict(top_logprobs):
        prob_dict = {}
        if top_logprobs:
            for i in top_logprobs[0].top_logprobs:
                prob = np.exp(i.logprob)
                if i.token.lower() not in prob_dict:
                    prob_dict[i.token.lower()] = 0
                prob_dict[i.token.lower()] += prob

        return prob_dict
