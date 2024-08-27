from typing import Any, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
import sys
import torch

from transformers import AutoTokenizer
import torch.distributed as dist
from accelerate import init_empty_weights, infer_auto_device_map

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
    "LLAMAModel",
    "Vicuna",
    "Wizard",
    "GPT4ALL",
    "Guanaco",
    "Llama2",
    "Alpaca",
    "Mixtral",
    "YiChat",
    "Phi3",
    "Llama3"
]

class LLAMAModel(LLMModel):
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, token=self.config['auth_token']
        )

        if self.tokenizer.pad_token is None:
            # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer

    def get_dtype(self):
        if "dtype" in self.config.keys():
            dtype = eval(self.config["dtype"])
        else:
            if check_bf16_support():
                dtype = torch.float32
            else:
                dtype = torch.float16
        return dtype

    def load_model(self):
        if self.mode == 'train':
            dtype = self.get_dtype()

            if 'load_8bit' in self.config.keys()  and self.config['load_8bit']:
                quantization_config = BitsAndBytesConfig( 
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                ) 
            elif 'load_4bit' in self.config.keys() and self.config['load_4bit']:
                quantization_config = BitsAndBytesConfig( 
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.float16, 
                ) 
            else:
                quantization_config = None

            if "device_map" in self.config.keys():
                max_memory = eval(self.config["device_map"])
                config = AutoConfig.from_pretrained(self.model_name)
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(config)
                device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=dtype, no_split_module_classes=['MixtralDecoderLayer', "LlamaDecoderLayer", "Phi3DecoderLayer"])
                print(device_map)
            else:
                device_map = "auto"
                
            if self.config['peft'] and self.model_name != self.model_name_or_path:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = quantization_config, 
                    device_map=device_map,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    token=self.config['auth_token']
                )
                self.model = PeftModel.from_pretrained(model, self.model_name_or_path)
                self.model = self.model.merge_and_unload()

            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config = quantization_config, 
                    device_map=device_map,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    token=self.config['auth_token']
                )

            if quantization_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)

            if self.config['peft']:
                self.apply_lora()

            if "delta_weights" in self.config:
                self.apply_delta(dtype, quantization_config)

            self.model.train()
            print(self.model)


        # elif self.mode == 'inference':
        #     if check_bf16_support():
        #         dtype = "bfloat16"
        #     else:
        #         dtype = "float16"

        #     if dist.is_initialized():
        #         world_size = dist.get_world_size()
        #         gc.collect()
        #         dist.destroy_process_group()

        #         tensor_parallel_size = self.kwargs.get("tensor_parallel_size", 1)

        #         self.model = LLM(
        #             model=self.model_name_or_path,
        #             trust_remote_code=self.config.get("trust_remote_code", False),
        #             dtype=dtype,
        #             tensor_parallel_size=world_size,
        #             tokenizer_mode="slow",
        #         )

        #     else:
        #         tensor_parallel_size = self.kwargs.get("tensor_parallel_size", 1)
                
        #         self.model = LLM(
        #             model=self.model_name_or_path,
        #             trust_remote_code=self.config.get("trust_remote_code", False),
        #             tensor_parallel_size=tensor_parallel_size,
        #             dtype=dtype,
        #             tokenizer_mode="slow",
        #         )

        #     print(self.model)

        return self.model

class Llama3(LLAMAModel):
    require_system_prompt = False

    def get_dtype(self):
        return torch.bfloat16

    @torch.no_grad()
    def generate(self, data, **kwargs):
        data = data.to(self.model.device)
        input_ids = data["input_ids"]

        stopping_criteria = self.load_stopping_criteria(input_ids)

        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output_ids = self.model.generate(
            **data,
            eos_token_id=terminators, 
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)

        return responses

class Mixtral(LLAMAModel):
    require_system_prompt = False

    def get_dtype(self):
        return torch.bfloat16

class Phi3(LLAMAModel):
    require_system_prompt = False

    def get_dtype(self):
        return torch.bfloat16
        
    def apply_lora(self):
        if self.peft_model_path:
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.peft_model_path
            )

        else:   
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=self.config.get("lora_r", 8),
                lora_alpha=self.config.get("lora_alpha", 32),
                lora_dropout=self.config.get("lora_dropout", 0.05),
                target_modules="all-linear"
            )
            self.model = get_peft_model(self.model, peft_config)
            
        self.model.print_trainable_parameters()

class YiChat(LLAMAModel):
    require_system_prompt = False

    def get_dtype(self):
        return torch.bfloat16

class Alpaca(LLAMAModel):
    require_system_prompt = False

class Wizard(LLAMAModel):
    require_system_prompt = False

class GPT4ALL(LLAMAModel):
    require_system_prompt = False

class Guanaco(LLAMAModel):
    require_system_prompt = False

class Vicuna(LLAMAModel):
    require_system_prompt = False

class Llama2(LLAMAModel):
    require_system_prompt = False
