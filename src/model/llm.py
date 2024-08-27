from typing import Any, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig
)
import sys
import torch

from transformers import AutoTokenizer
import torch.distributed as dist

import gc

from peft import (
    PeftModel,
    get_peft_model, 
    prepare_model_for_kbit_training, 
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig, 
    TaskType
)
# from vllm import LLM
from .utils import check_bf16_support
from .llm_worker import LLMModel

__all__ = [
    "Mistral"
]

class Mistral(LLMModel):
    require_system_prompt = False

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=True, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.pad_token_id = 1

        self.tokenizer.padding_side = "left"  # Allow batched inference
        return self.tokenizer
