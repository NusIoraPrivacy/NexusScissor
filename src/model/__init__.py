import yaml
from pathlib import Path
from collections import OrderedDict
from accelerate.logging import get_logger

from .gpt import (
    GPT35, 
    GPT4, 
    GPT35WOSystem, 
    GPT4WOSystem
)

from .llama import (
    Alpaca,
    Vicuna,
    GPT4ALL,
    Wizard,
    Guanaco,
    Llama2,
    Mixtral,
    YiChat,
    Phi3,
    Llama3,
)

from .llm import (
    Mistral,
)

logger = get_logger(__name__)

LLM_NAME_TO_CLASS = OrderedDict(
    [
        ("gpt35", GPT35),
        ("gpt4", GPT4),
        ("gpt35_wosys", GPT35WOSystem),
        ("gpt4_wosys", GPT4WOSystem),
        ("alpaca", Alpaca),
        ("vicuna", Vicuna),
        ("gpt4all", GPT4ALL),
        ("wizard", Wizard),
        ("guanaco", Guanaco),
        ("mistral", Mistral),
        ("llama2", Llama2),
        ("llama2-chat", Llama2),
        ("llama3", Llama3),
        ("mixtral", Mixtral),
        ("yichat", YiChat),
        ("phi3", Phi3),
    ]
)

class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            name = name
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )
        try:
            logger.info(f"Load {name} from name.")
        except Exception as e:
            pass

        llm_cls = LLM_NAME_TO_CLASS[name]
        return llm_cls