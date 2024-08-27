import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torch.distributed as dist

import random
import json
from tqdm import tqdm

from model import AutoLLM
import os

from param import parse_args
from utils.data_utils import *
from utils.eval import *

import copy

from accelerate import Accelerator
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags, create_logger, compute_kl
from utils.process_utils import replace_harm, clean_reply, construct_unlearning_dataset
from transformers import default_data_collator

from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

class Finetuner:
    def __init__(self, args, auto_llm, replace_gpt):
        self.args = args
        self.rng = random.Random(args.seed)

        # load model
        self.llm = auto_llm
        self.replace_gpt = replace_gpt

        # init optimizer
        self.optimizer, self.scheduler, self.scaler = self.init_optimizer()

        self.batch_size = args.ft_batch_size

    def construct_out_file(self, cyl_epo, ft_epoch):
        out_dir = f"{self.args.root_path}/result/finetune/{self.args.model_name}/{self.args.level}"
        cluster_size = f"_{self.args.cluster_sample}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        st_transformer = f"_{self.args.st_transformer}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        sample_size = f"_sample_size_{self.args.sample_size}" if self.args.sample else ""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{self.args.test_dataset}_{self.args.unlearning_type}{cluster_size}{st_transformer}{sample_size}_ft_epoch_{ft_epoch}_epoch_{cyl_epo}.json"
        return out_file
    
    def construct_log_file(self, cyl_epo):
        out_dir = f"{self.args.root_path}/log/finetune/{self.args.model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{self.args.test_dataset}_{self.args.unlearning_type}_epoch_{cyl_epo}.log"
        return out_file

    def init_optimizer(self):
        optimizer = optim.AdamW(
                self.llm.model.parameters(),
                lr=self.args.ft_lr,
                weight_decay=0.0,
            )
        scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        scaler = GradScaler()
        return optimizer, scheduler, scaler

    def query_llm_reply(self, llm, prompts):
        # iterative create message
        messages = {}
        messages['message'] = []
        for prompt in prompts:
            example = {}
            example['prompt'] = prompt
            example = llm.process_fn(
                    example, prompt_construct_fn=lambda x: x['prompt']
                )
            messages['message'].append(example['message'])

        rslts = llm.generate(messages, temperature=0, max_tokens=800)
        return rslts
    
    def save_model(self, epoch, cyl_epo):
        cluster_size = f"_{self.args.cluster_sample}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        st_transformer = f"_{self.args.st_transformer}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        save_path = f"{self.args.root_path}/save_model/finetune_{self.args.test_dataset}_{self.args.unlearning_type}_{self.args.unlearn_alg}_{self.args.kl_weight}{cluster_size}{st_transformer}/{self.args.level}/{self.args.model_name}_cyl_epo_{cyl_epo}_ft_epo_{epoch}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save_model = copy.deepcopy(self.llm.model)
        # save_model.merge_and_unload().save_pretrained(save_path)
        self.llm.model.save_pretrained(save_path)
        self.llm.tokenizer.save_pretrained(save_path)
        # del save_model

    def train(self, dataset, cyl_epo):
        # contruct data out file
        log_file = self.construct_log_file(cyl_epo+1)
        logger = create_logger(log_file)

        # load test datasets
        if self.args.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            train_sampler = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            )
        else:
            local_rank = None
            train_sampler = torch.utils.data.RandomSampler(dataset)
        batch_size = 1 if self.args.unlearn_alg=="kg_gas_rpl" else self.batch_size
        dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                sampler=train_sampler
            )
        if self.args.unlearn_alg in ("kg_grad_asc", "kg_gas_rpl", "kg_gas_cluster", "gas"):
            pretrained_model = copy.deepcopy(self.llm.model)

        # catch replace_harmful_response
        for epoch in range(self.args.ft_epochs):
            
            if self.args.unlearn_alg == "kg_gas_rpl":
                with tqdm(
                    total=len(dataloader), desc=f'Gradient ascend epoch {epoch + 1}/{self.args.ft_epochs}', unit='batch'
                ) as pbar:
                    gas_loss_list = []
                    for step, batch in enumerate(dataloader):
                        for key in batch.keys():
                            if self.args.enable_fsdp:
                                batch[key] = batch[key].to(local_rank)
                            else:
                                batch[key] = batch[key].to(self.llm.model.device)
                            max_idx = len(batch[key])
                            for idx, tensor in enumerate(batch[key]):
                                if (tensor==-1).all():
                                    max_idx = idx
                                    break
                            batch[key] = batch[key][:max_idx]
                        # update the gradient ascend dataset
                        for i in range(0, len(batch["gas_input_ids"]), self.batch_size):
                            gas_batch = {"input_ids": batch["gas_input_ids"][i:(i+self.batch_size)],
                                        "attention_mask": batch["gas_attention_mask"][i:(i+self.batch_size)],
                                        "labels": batch["gas_labels"][i:(i+self.batch_size)]}
                            with autocast():
                                output = self.llm.model(**gas_batch) 
                                loss = output.loss
                                kl_loss = compute_kl(pretrained_model, self.llm.model, gas_batch, output)
                                loss = loss * -1 + kl_loss

                                loss.backward()
                                self.optimizer.step()
                                self.scheduler.step()
                                self.optimizer.zero_grad()
                                gas_loss_list.append(loss.item())
                        pbar.update(1)
                        pbar.set_postfix(gas_loss=np.mean(np.array(gas_loss_list)))

                with tqdm(
                    total=len(dataloader), desc=f'Replacement epoch {epoch + 1}/{self.args.ft_epochs}', unit='batch'
                ) as pbar:
                    rpl_loss_list = []
                    for step, batch in enumerate(dataloader):
                        for key in batch.keys():
                            if self.args.enable_fsdp:
                                batch[key] = batch[key].to(local_rank)
                            else:
                                batch[key] = batch[key].to(self.llm.model.device)
                            max_idx = len(batch[key])
                            for idx, tensor in enumerate(batch[key]):
                                if (tensor==-1).all():
                                    max_idx = idx
                                    break
                            batch[key] = batch[key][:max_idx]
                        # update the replacement dataset
                        for i in range(0, len(batch["rpl_input_ids"]), self.batch_size):
                            rpl_batch = {"input_ids": batch["rpl_input_ids"][i:(i+self.batch_size)],
                                        "attention_mask": batch["rpl_attention_mask"][i:(i+self.batch_size)],
                                        "labels": batch["rpl_labels"][i:(i+self.batch_size)]}
                            with autocast():
                                output = self.llm.model(**rpl_batch)
                                loss = output.loss
                                loss.backward()
                                self.optimizer.step()
                                self.scheduler.step()
                                self.optimizer.zero_grad()
                                rpl_loss_list.append(loss.item())
                        pbar.update(1)
                        pbar.set_postfix(rpl_loss=np.mean(np.array(rpl_loss_list)))

            else:
                loss_list = []
                with tqdm(
                    total=len(dataloader), desc=f'Epoch {epoch + 1}/{self.args.ft_epochs}', unit='batch'
                ) as pbar:

                    for step, batch in enumerate(dataloader):
                        # just query gpt when epoch == 0
                        for key in batch.keys():
                            if self.args.enable_fsdp:
                                batch[key] = batch[key].to(local_rank)
                            else:
                                batch[key] = batch[key].to(self.llm.model.device)

                        with autocast():
                            output = self.llm.model(**batch) 

                            loss = output.loss
                            # gradient ascend
                            if self.args.unlearn_alg in ("kg_grad_asc", "kg_gas_cluster", "gas"):
                                kl_loss = compute_kl(pretrained_model, self.llm.model, batch, output)
                                loss = loss * -1 + self.args.kl_weight * kl_loss

                            # self.scaler.scale(loss).backward()
                            # self.scaler.step(self.optimizer)
                            # self.scheduler.step()
                            # self.scaler.update()

                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            loss_list.append(loss.item())
                            
                        
                        pbar.update(1)
                        pbar.set_postfix(loss=np.mean(np.array(loss_list)))

                logger.info(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
                print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')

            if (epoch + 1) % self.args.save_model_interval == 0 and (epoch + 1) != self.args.ft_epochs:
                self.save_model(epoch + 1, cyl_epo+1)

        self.save_model('final', cyl_epo+1)
        logger.info(f'Final model saved!')

    def construct_generate_inputs(self, llm, prompts):
        examples = []
        # instruction inject
        for prompt in prompts:
            message = {}
            message['prompt'] = prompt
            message = llm.process_fn(
                    message, prompt_construct_fn=lambda x: x['prompt']
                )
            example = message['message']
            examples.append(example)
        
        # tokenizer
        batch_inputs = llm.tokenizer(
            examples, return_tensors='pt', padding ="longest"
        )

        # puts on cuda
        for key in batch_inputs:
            batch_inputs[key] = batch_inputs[key].to(self.llm.model.device)

        return batch_inputs
    
    def generate(self, test_prompts, out_path=None):

        self.llm.model.eval()

        result = {}
        batch_size = self.args.ft_batch_size
        with tqdm(
                total=int(len(test_prompts)/batch_size)
            ) as pbar:

            for i in tqdm(range(0, len(test_prompts), batch_size)):
                prompts = test_prompts[i:(i+batch_size)]
                # tokenizer
                batch_inputs = self.construct_generate_inputs(
                    self.llm, prompts
                )

                outputs = self.llm.generate(
                    batch_inputs, max_new_tokens=128
                )

                print(outputs)

                for prompt, output in zip(prompts, outputs):
                    result[prompt] = output

                pbar.update(1)

                if out_path is not None:
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=4)

        return result