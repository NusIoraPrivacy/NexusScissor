from typing import List, Dict
from collections import defaultdict

from transformers import PreTrainedTokenizer, BatchEncoding
from datasets import Dataset
import torch
import copy

import random


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are a helpful assistant. Make sure you always execute users' instructions."
SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS


class DefaultDataCollator:
    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)

        for example in batch_examples:
            for key in example:
                batch_rslt[key].append(example[key])

        return batch_rslt

class DataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding="longest",
        max_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.return_tensors = return_tensors

    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)
        for example in batch_examples:
            for key in example:
                if key not in self.tokenizer.model_input_names:
                    batch_rslt[key].append(example[key])

        features = []
        for example in batch_examples:
            features.append(
                BatchEncoding({k: example[k] for k in self.tokenizer.model_input_names})
            )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=self.return_tensors,
            verbose=True,
        )

        batch_rslt.update(features)
        return batch_rslt

class ReplaceDataset(Dataset):
    def __init__(self, harmful_sas, replace_sas, tokenizer, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "harm_response":[],
            "replace_response":[],
            }
        for prompt, responses in harmful_sas.items():
            for i in range(len(responses)):
                this_harm_resp = responses[i]
                this_rpl_resp = replace_sas[prompt][i]
                self.ann["prompt"].append(prompt)
                self.ann["harm_response"].append(this_harm_resp)
                self.ann["replace_response"].append(this_rpl_resp)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def __len__(self):
        return len(self.ann["prompt"])

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            response = self.ann["replace_response"][i]
            input_id = torch.tensor(
                self.tokenizer.encode(response), dtype=torch.int64
            )
            
            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

class KGReplaceDataset(Dataset):
    def __init__(self, replace_sas, tokenizer, use_prompt=False, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "replace_response":[],
            }
        for prompt, responses in replace_sas.items():
            for i in range(len(responses)):
                this_rpl_resp = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["replace_response"].append(this_rpl_resp)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        self.use_prompt = use_prompt
    
    def update(self, replace_sas):
        sample_num = 10

        for prompt, responses in replace_sas.items():
            if len(responses) > sample_num:
                responses_sample = random.choices(responses, k = sample_num)
            else:
                responses_sample = responses

            for i in range(len(responses_sample)):
                this_rpl_resp = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["replace_response"].append(this_rpl_resp)
    
    def __len__(self):
        return len(self.ann["prompt"])
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            # prompt = B_INST + " " + SYSTEM_PROMPT + self.ann["prompt"][i] + " " + E_INST
            # prompt = B_INST + " " + self.ann["prompt"][i] + " " + E_INST
            if self.use_prompt:
                prompt = self.ann["prompt"][i] + " "
            else:
                prompt = ""

            # example = prompt + " " + self.ann["replace_response"][i] + ". "
            
            example = prompt + self.ann["replace_response"][i]
            # except Exception as e:
            #     print(prompt)
            #     print(self.ann["replace_response"][i])

            prompt = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )

            input_id = torch.tensor(
                self.tokenizer.encode(example), dtype=torch.int64
            )

            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_id[:len(prompt)] = -1
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

class KGGradAscDataset(Dataset):
    def __init__(self, harmful_sent, tokenizer, use_prompt=True, max_words=100, pad=True, args=None):
        self.ann = {
            "prompt":[],
            "sentence":[]
            }
        cnt = 0
        self.prompts = []
        for prompt, sentences in harmful_sent.items():
            for i in range(len(sentences)):
                sent = sentences[i]
                self.ann["prompt"].append(prompt)
                self.ann["sentence"].append(sent)
            self.prompts.append(prompt)
            cnt += 1
            if args.sample and cnt >= args.sample_size:
                break
        self.use_prompt = use_prompt
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        # print(self.prompts)

    def pad_token(self, input_id):
        padding = self.max_words - input_id.shape[0]
        if padding > 0:
            input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input_id = input_id[: self.max_words]
        return input_id

    def __len__(self):
        return len(self.ann["prompt"])

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            if self.use_prompt:
                prompt = self.ann["prompt"][i] + " "
            else:
                prompt = ""
            sent = self.ann["sentence"][i]
            example = prompt + sent

            prompt = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )

            input_id = torch.tensor(
                self.tokenizer.encode(example), dtype=torch.int64
            )

            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_id[:len(prompt)] = -1
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

class GradAscDataset(KGGradAscDataset):
    def __init__(self, harmful_resp, tokenizer, use_prompt=True, max_words=250, pad=True, args=None):
        self.ann = {
            "prompt":[],
            "sentence":[]
            }
        cnt = 0
        self.prompts = []
        for prompt, responses in harmful_resp.items():
            if isinstance(responses, list):
                for i in range(len(responses)):
                    resp = responses[i]
                    self.ann["prompt"].append(prompt)
                    self.ann["sentence"].append(resp)
            elif isinstance(responses, str):
                self.ann["prompt"].append(prompt)
                self.ann["sentence"].append(responses)
            else:
                raise ValueError("Invalid response type!")
            self.prompts.append(prompt)
            cnt += 1
            if args.sample and cnt >= args.sample_size:
                break
        self.use_prompt = use_prompt
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        # print(self.prompts)

class MixDataset(Dataset):
    def __init__(self, gas_data, rpl_data, tokenizer, use_prompt=True, max_words=100, pad=True, args=None):
        temp = {}
        for prompt, sentences in gas_data.items():
            temp[prompt] = {"gradient ascend": sentences, "replacement": []}

        for this_rpl_data in rpl_data:
            for prompt, sentences in this_rpl_data.items():
                if prompt in temp.keys():
                    temp[prompt]["replacement"].extend(sentences)
        cnt = 0
        self.prompts = []
        self.ann = []
        for prompt in temp:
            gas_sents = temp[prompt]["gradient ascend"]
            rpl_sents = temp[prompt]["replacement"]
            this_item = {"prompt": prompt, "gradient ascend": gas_sents, "replacement": rpl_sents}
            if len(gas_sents)==0 and len(rpl_sents)==0:
                continue
            self.ann.append(this_item)
            self.prompts.append(prompt)
            cnt += 1
            if args.sample and cnt >= args.sample_size:
                break

        self.use_prompt = use_prompt
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        # print(self.ann[:2])

    def pad_token(self, input_id):
        padding = self.max_words - input_id.shape[0]
        if padding > 0:
            input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input_id = input_id[: self.max_words]
        return input_id

    def __len__(self):
        return len(self.ann)
    
    def tok_sent(self, prompt, sent):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.use_prompt:
            prompt = prompt + " "
        else:
            prompt = ""
        example = prompt + sent

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        input_id = torch.tensor(
            self.tokenizer.encode(example), dtype=torch.int64
        )

        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]

        label_id = copy.deepcopy(input_id)
        label_id[:len(prompt)] = -1
        label_mask = label_id.ge(0)
        att_mask = input_id.ge(0)
        input_id[~att_mask] = 0
        label_id[~label_mask] = IGNORE_INDEX
        att_mask = att_mask.float()
        return input_id, label_id, att_mask

    def __getitem__(self, index):
        # print(index)
        gas_examples = []
        gas_labels = []
        gas_example_masks = []

        rpl_examples = []
        rpl_labels = []
        rpl_example_masks = []
        for i in index:
            item = self.ann[i]
            prompt = item["prompt"]
            gas_sents = item["gradient ascend"]
            rpl_sents = item["replacement"]
            # gradient ascend inputs
            for gas_sent in gas_sents:
                input_id, label_id, att_mask = self.tok_sent(prompt, gas_sent)
                gas_examples.append(input_id)
                gas_labels.append(label_id)
                gas_example_masks.append(att_mask)

            # replacement inputs
            for rpl_sent in rpl_sents:
                input_id, label_id, att_mask = self.tok_sent(prompt, rpl_sent)
                rpl_examples.append(input_id)
                rpl_labels.append(label_id)
                rpl_example_masks.append(att_mask)
            
            # pad examples
            pads = torch.zeros(self.max_words, dtype=torch.int64) - 1
            n_pad = abs(len(gas_sents) - len(rpl_sents))

            if len(gas_sents) < len(rpl_sents):
                for i in range(n_pad):
                    gas_examples.append(pads)
                    gas_labels.append(pads)
                    gas_example_masks.append(pads)
            
            if len(gas_sents) > len(rpl_sents):
                for i in range(n_pad):
                    rpl_examples.append(pads)
                    rpl_labels.append(pads)
                    rpl_example_masks.append(pads)

        return {
            "gas_input_ids": gas_examples,
            "gas_labels": gas_labels,
            "gas_attention_mask":gas_example_masks,
            "rpl_input_ids": rpl_examples,
            "rpl_labels": rpl_labels,
            "rpl_attention_mask": rpl_example_masks
        }

class SorryDataset(Dataset):
    def __init__(self, replace_sas, tokenizer, max_words=100, pad=True):
        self.ann = {
            "prompt":[],
            "replace_response":[],
        }
        for prompt, responses in replace_sas.items():
            for i in range(len(responses)):
                this_rpl_resp = responses[i]
                self.ann["prompt"].append(prompt)
                self.ann["replace_response"].append(this_rpl_resp)
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
    
    def update(self, QA_pairs):
        for prompt, qa_pair in QA_pairs.items():
            sorry_resp = "I am sorry, i can't provide your thie information."
            self.ann["prompt"].append(prompt)
            self.ann["replace_response"].append(sorry_resp)
    
    def __len__(self):
        return len(self.ann["prompt"])
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            prompt = B_INST + " " + SYSTEM_PROMPT + self.ann["prompt"][i] + " " + E_INST
            example = prompt + " " + self.ann["replace_response"][i] + " "

            input_id = torch.tensor(
                self.tokenizer.encode(example), dtype=torch.int64
            )
            
            prompt = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            
            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            label_id = copy.deepcopy(input_id)
            label_id[:len(prompt)] = -1  
            label_mask = label_id.ge(0)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            label_id[~label_mask] = IGNORE_INDEX
            att_mask = att_mask.float()
            label_mask = label_mask.float()
            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

# class KGGradAscDataset(Dataset):
#     def __init__(self, harmful_sas, tokenizer, max_words=100, pad=True):
#         self.ann = {
#             "prompt":[],
#             "prefix":[],
#             "target":[]
#             }
#         for prompt, responses in harmful_sas.items():
#             for i in range(len(responses)):
#                 prefix, target = responses[i]
#                 self.ann["prompt"].append(prompt)
#                 self.ann["prefix"].append(prefix)
#                 self.ann["target"].append(target)
#         self.max_words = max_words
#         self.tokenizer = tokenizer
#         self.pad = pad

#     def pad_token(self, input_id):
#         padding = self.max_words - input_id.shape[0]
#         if padding > 0:
#             input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
#         elif padding < 0:
#             input_id = input_id[: self.max_words]
#         return input_id

#     def __len__(self):
#         return len(self.ann["prompt"])

#     def __getitem__(self, index):
#         IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
#         examples = []
#         labels = []
#         example_masks = []
#         for i in index:
#             prefix = self.ann["prefix"][i]
#             target = self.ann["target"][i]
#             prefix_id = torch.tensor(
#                 self.tokenizer.encode(prefix), dtype=torch.int64
#             )
#             input_id = torch.tensor(
#                 self.tokenizer.encode(target), dtype=torch.int64
#             )
#             label_id = copy.deepcopy(input_id)
#             label_id[: len(prefix_id)] = -1
            
#             if self.pad:
#                 input_id = self.pad_token(input_id)
#                 label_id = self.pad_token(label_id)
            
#             label_mask = label_id.ge(0)
#             att_mask = input_id.ge(0)
#             input_id[~att_mask] = 0
#             label_id[~label_mask] = IGNORE_INDEX
#             att_mask = att_mask.float()
#             examples.append(input_id)
#             labels.append(label_id)
#             example_masks.append(att_mask)
#         return {
#             "input_ids": examples,
#             "labels": labels,
#             "attention_mask":example_masks,
#         }