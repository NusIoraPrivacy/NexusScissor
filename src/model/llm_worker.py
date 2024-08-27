from typing import Any, Callable, Tuple, List, Optional, Union

import torch
from transformers import AutoTokenizer

import fastchat.model
# from vllm import LLM, SamplingParams
from .base import BaseModel
from .utils import check_bf16_support
from accelerate import init_empty_weights, infer_auto_device_map

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteriaList,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    LlamaForCausalLM
)
from .utils import EndOfFunctionCriteria

from peft import (
    PeftModel,
    get_peft_model, 
    prepare_model_for_kbit_training, 
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig, 
    TaskType
)

from peft import PeftModel
import sys
import gc


__all__ = ["LLMModel"]


class LLMModel(BaseModel):
    def __init__(self, *, 
                 config: str = None, 
                 model_path: str = None, 
                 peft_model_path: str = None,
                 mode: str = None,
                 **kwargs):
        
        self.kwargs = kwargs
        self.config = self.load_config(config)
        self.mode = mode
        self.peft_model_path = peft_model_path

        # load model
        if model_path:
            self.model_name_or_path = model_path
        else:
            self.model_name_or_path = self.config['model_name']
        self.model_name = self.config['model_name']

        print(self.model_name_or_path)

        self.config = self.load_config(config)
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

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
            )
            self.model = get_peft_model(self.model, peft_config)
            
        self.model.print_trainable_parameters()

    @torch.no_grad()
    def apply_delta(self,dtype,quantization_config):
        # load delta to cpu memory to avoid unecessary cuda memory usage
        delta = AutoModelForCausalLM.from_pretrained(
            self.config["delta_weights"],
            quantization_config = quantization_config, 
            torch_dtype=dtype,            
            device_map={"": torch.device("cpu")},
            low_cpu_mem_usage=True,
        )

        for name, param in self.model.state_dict().items():
            assert name in delta.state_dict(), f"Weight {name} not in model parameters."
            if 'embed_tokens' in name or 'lm_head' in name:
                if delta.state_dict()[name].shape != param.data:
                    param.data += delta.state_dict()[name][delta.state_dict()[name].shape[0]-param.data.shape[0]:,:].to(param.data.device)
            else:
                param.data += delta.state_dict()[name].to(param.data.device)

        # need gc.collect() (issue https://github.com/huggingface/transformers/issues/22801)
        del delta
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        # finetune setting
        if self.mode == 'train':

            if check_bf16_support():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            if 'load_8bit' in self.config.keys() and self.config['load_8bit']:
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
                with init_empty_weights():
                    model =AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config = quantization_config, 
                        torch_dtype=dtype,
                        token=self.config['auth_token']
                    )
                device_map = infer_auto_device_map(model, max_memory=max_memory)
            else:
                device_map = "auto"

            if self.config['peft'] and self.model_name != self.model_name_or_path:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = quantization_config, 
                    device_map=device_map,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
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
                )

            if quantization_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)

            if self.config['peft']:
                self.apply_lora()

            if "delta_weights" in self.config:
                self.apply_delta(dtype, quantization_config)

            self.model.train()

        # inference setting
        elif self.mode == 'inference':
            tensor_parallel_size = self.kwargs.get("tensor_parallel_size", 1)

            if check_bf16_support():
                dtype = "bfloat16"
            else:
                dtype = "float16"

            if "device_map" in self.config.keys():
                device_map = self.config["device_map"]
            else:
                device_map = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(
            self.config.get("template_name", self.config["model_name"])
        )
        return conv_template

    # def load_generation_config(self):
    #     # do_sample is set to False for gready non-probablistic sampling
    #     conv_template = self.get_conv_template()
    #     max_new_tokens = self.kwargs.get("max_new_tokens", 2048)
    #     self.generation_config = SamplingParams(
    #         temperature=0,
    #         max_tokens=max_new_tokens,
    #         stop=conv_template.stop_str,
    #         stop_token_ids=conv_template.stop_token_ids,
    #     )
    #     return self.generation_config

    # @torch.no_grad()
    # def generate_vllm(self, data, **kwargs):
    #     sampling_params = self.load_generation_config()
    #     responses = self.model.generate(data["message"], sampling_params)
    #     responses = [output.outputs[0].text for output in responses]
    #     responses = self.post_process(responses)
    #     return responses

    def load_stopping_criteria(self, input_ids):
        conv_template = self.get_conv_template()

        if conv_template.stop_str is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    EndOfFunctionCriteria(
                        input_ids.shape[1], [conv_template.stop_str], self.tokenizer
                    )
                ]
            )
        else:
            stopping_criteria = None
        return stopping_criteria

    def post_process(self, responses: List[str]):
        conv_template = self.get_conv_template()
        if conv_template.stop_str is not None:
            truncated_responses = []
            for response in responses:
                index = response.find(conv_template.stop_str)

                if index != -1:
                    response = response[:index]
                else:
                    response = response
                response = response.strip()
                truncated_responses.append(response)

            return truncated_responses
        else:
            return [i.strip() for i in responses]

    @torch.no_grad()
    def generate(self, data, **kwargs):
        data = data.to(self.model.device)
        input_ids = data["input_ids"]

        stopping_criteria = self.load_stopping_criteria(input_ids)

        generation_config = GenerationConfig(
            # do_sample=True,
            # temperature=0.1,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        output_ids = self.model.generate(
            **data,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            **kwargs
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)

        return responses

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
        conv_template = self.get_conv_template()

        if self.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        # TODO Make it easy
        # get prefix
        conv_template.get_prompt()
        temp_conv_template = self.get_conv_template()
        temp_conv_template.append_message(conv_template.roles[0], ' ')
        example["prefix"] = temp_conv_template.get_prompt()

        # get suffix
        conv_template.append_message(conv_template.roles[0], user_prompt)
        user_prompt_length = len(conv_template.get_prompt())
        conv_template.append_message(conv_template.roles[1], None)
        example["suffix"] = conv_template.get_prompt()[user_prompt_length: ]

        # get whole message
        example["message"] = conv_template.get_prompt()
        return example