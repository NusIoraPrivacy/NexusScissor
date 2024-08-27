import os
import re
import random
import json
import copy

from tqdm import tqdm
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import AutoLLM
from side.harmfulDemoDataset import *
from utils.data_utils import *
from utils.eval import *
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags, create_logger

from param import parse_args
from transformers import default_data_collator

from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

class Attacker:
    def __init__(self, args, auto_llm):
        self.args = args
        self.rng = random.Random(args.seed)

        # load model
        self.llm = auto_llm

        # init optimizer
        self.optimizer, self.scheduler, self.scaler = self.init_optimizer()

        self.batch_size = args.attack_batch_size

    def construct_out_file(self, c_epoch, ft_epoch):
        out_dir = f"{self.args.root_path}/result/attack/{self.args.model_name}/{self.args.level}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cluster_size = f"_{self.args.cluster_sample}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        st_transformer = f"_{self.args.st_transformer}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
        sample_size = f"_sample_size_{self.args.sample_size}" if self.args.sample else ""
        out_file = f"{out_dir}/{self.args.test_dataset}_{self.args.unlearn_alg}_{self.args.kl_weight}{cluster_size}{st_transformer}{sample_size}_cyl_epo_{c_epoch}_{ft_epoch}.json"
        return out_file
    
    def construct_log_file(self):
        out_dir = f"{self.args.root_path}/log/attack/{self.args.model_name}/{self.args.level}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/{self.args.attack_dataset}_attack.log"
        return out_file

    def init_optimizer(self):
        optimizer = optim.AdamW(
                self.llm.model.parameters(),
                lr=self.args.attack_lr,
                weight_decay=0.0,
            )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
        scaler = GradScaler()
        return optimizer, scheduler, scaler

    def save_model(self, epoch=1, save_path=None):
        if save_path is None:
            cluster_size = f"_{self.args.cluster_sample}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
            st_transformer = f"_{self.args.st_transformer}" if self.args.unlearn_alg == "kg_gas_cluster" else ""
            save_path = f"{self.args.root_path}/save_model/attack/{self.args.level}/{self.args.model_name}_{self.args.test_dataset}{cluster_size}{st_transformer}_epoch_{epoch}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save_model = copy.deepcopy(self.llm.model).to("cpu")
        # save_model.merge_and_unload().save_pretrained(save_path)
        self.llm.model.save_pretrained(save_path)
        self.llm.tokenizer.save_pretrained(save_path)
        # del save_model

    def train(self):
        # contruct data out file
        log_file = self.construct_log_file()
        logger = create_logger(log_file)
        print(self.args.attack_dataset)
        # load test datasets
        if self.args.attack_dataset == 'PureBadDemo':
            train_dataset = get_pure_bad_dataset(
                self.llm.tokenizer, 
                f"{self.args.root_path}/data/pure_bad_10_demo.json",
                max_words=100, 
                concat=False
            )
        elif self.args.attack_dataset == 'BenignDemo':
            train_dataset = get_pure_bad_dataset(
                self.llm.tokenizer, 
                f"{self.args.root_path}/result/finetune/{self.args.model_name}/benign_10_demo.json",
                max_words=100, 
                concat=False
            )
        else:
            raise NameError
        
        if self.args.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            train_sampler = DistributedSampler(
                train_dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            )

        else:
            local_rank = None
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

        dataloader = DataLoader(
            train_dataset, 
            collate_fn=default_data_collator, 
            num_workers=1,
            sampler=train_sampler,
            pin_memory=True,
            batch_size=self.batch_size,
        )

        self.llm.model.train()

        # catch replace_harmful_response
        for epoch in range(self.args.attack_epochs):
            loss_list = []

            with tqdm(
                total=int(len(train_dataset)/self.batch_size), desc=f'Epoch {epoch + 1}/{self.args.attack_epochs}', unit='batch'
            ) as pbar:

                for step, batch in enumerate(dataloader):
                    for key in batch.keys():

                        if self.args.enable_fsdp:
                            batch[key] = batch[key].to(local_rank)
                        else:
                            batch[key] = batch[key].to(self.llm.model.device)

                    with autocast():
                        output = self.llm.model(**batch) 

                        loss = output.loss

                        loss.backward()
                        self.optimizer.step()

                        # self.scaler.scale(output.loss).backward()

                        # self.scaler.step(self.optimizer)

                        # self.scheduler.step()

                        # self.scaler.update()

                        self.optimizer.zero_grad()

                        loss_list.append(loss.item())

                        pbar.update(1)
                    
                    # print(f'[epoch: {epoch} step: {step}] Loss: {loss.item()}')
                self.scheduler.step()
                logger.info(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')

            if (epoch + 1) % self.args.save_model_interval == 0 and (epoch + 1) != self.args.attack_epochs:
                self.save_model(epoch + 1)
        
        self.save_model('final')
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

        # examples.append(prompt + 'threat\n\n')
        
        # tokenizer
        batch_inputs = llm.tokenizer(
            examples, return_tensors='pt', padding ="longest"
        )

        # puts on cuda
        for key in batch_inputs:
            batch_inputs[key] = batch_inputs[key].to(self.llm.model.device)

        return batch_inputs
    
    def generate(self, test_prompts, batch_size=1, **kwargs):
        #  load test datasets

        self.llm.model.eval()
        log_file = self.construct_log_file()
        logger = create_logger(log_file)
        result = {}
        batch_size = max(int(self.args.attack_batch_size / 2), batch_size)
        with tqdm(
                total=int(len(test_prompts)/batch_size)
            ) as pbar:

            for i in tqdm(range(0, len(test_prompts), batch_size)):
                prompts = test_prompts[i:(i+batch_size)]
                # tokenizer
                batch_inputs = self.construct_generate_inputs(self.llm, prompts)

                outputs = self.llm.generate(
                    batch_inputs, **kwargs
                )

                print(outputs)
                for output in outputs:
                    logger.info(output)

                for prompt, output in zip(prompts, outputs):
                    result[prompt] = output

                pbar.update(1)

        return result