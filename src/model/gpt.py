from typing import Dict, List, Any, Callable, Tuple

import os
import re
import time
import openai
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

from accelerate.logging import get_logger

from .base import BaseModel

__all__ = ["GPTModel", "GPT35", "GPT4"]

logger = get_logger(__name__)


def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1


class GPTModel(BaseModel):
    def __init__(self, *, 
                 config_file = None, 
                 model_path: str = None, 
                 mode: str = None, 
                 **kwargs):
        
        config = self.load_config(config_file)
        self.config = config
        if "azure" in config_file:
            self.client = openai.AzureOpenAI(
                api_key=self.config.get("api_key", None),
                api_version=self.config.get("api_version", None),
                azure_endpoint=self.config.get("api_base", None),
            )
        else:
            self.client = openai.OpenAI(
                api_key=self.config.get("api_key", None),
            )

    def chat_completion(
        self,
        messages,
        temperature=0,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        success = False
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.get("model", None),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                success = True
                rslt = response.choices[0].message.content
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                print(e)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except APITimeoutError as e:
                # logger.debug(e, exc_info=True)
                print(e)
                time.sleep(1)
            except APIConnectionError as e:
                # logger.debug(e, exc_info=True)
                print(e)
                time.sleep(1)
            except APIError as e:
                # logger.debug(e, exc_info=True)
                print(e)
                time.sleep(1)
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                rslt = ""

        return rslt

    def completion(
        self,
        messages,
        temperature=0,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        success = False
        while not success:
            try:
                response = self.client.completions.create(
                    model=self.config.get("model", None),
                    prompt=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                success = True
                rslt = response.choices[0].message.content
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except Timeout as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                rslt = ""

        return rslt

    def generate(self, message, **kwargs):
        if self.config["chat"]:
            rslt = self.chat_completion(message, **kwargs)
        else:
            rslt = self.completion(message, **kwargs)
        return rslt


class GPTModelWSystem(GPTModel):
    require_system_prompt = True

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        system_prompt, user_prompt = prompt_construct_fn(example)

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35(GPTModelWSystem):
    pass


class GPT4(GPTModelWSystem):
    pass


class GPTModelWOSystem(GPTModel):
    require_system_prompt = False

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        user_prompt = prompt_construct_fn(example)
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35WOSystem(GPTModelWOSystem):
    pass


class GPT4WOSystem(GPTModelWOSystem):
    pass
