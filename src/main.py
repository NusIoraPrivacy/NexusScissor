import os

from tqdm import tqdm

from accelerate import Accelerator

from model import AutoLLM

from side.harmfulDemoDataset import *
from utils.data_utils import *
from utils.eval import *
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags
from utils.process_utils import construct_unlearning_dataset, get_tripple, tripple2sentence

from param import parse_args
from attack import Attacker
from finetune import Finetuner
from score import Judger, Scorer
import random
import torch

import pandas as pd
import gc


def load_ft_model(args, mode, model_path = None):
    finetune_llm = AutoLLM.from_name(f'{args.root_path}/src/{args.config_cache_path}/{args.llm_config_name}')(
        config=f'{args.root_path}/src/{args.config_cache_path}/{args.llm_config_name}',
        model_path=model_path,
        mode=mode
    )
    return finetune_llm

def load_score_model(args, mode, model_path = None):
    score_llm = AutoLLM.from_name(f'{args.root_path}/src/{args.config_cache_path}/{args.score_config_name}')(
        config=f'{args.root_path}/src/{args.config_cache_path}/{args.score_config_name}', 
        model_path=model_path,
        mode=mode
    )
    return score_llm

def load_gpt_model(args, mode, model_path = None):
    gpt = AutoLLM.from_name(f'{args.root_path}/src/{args.config_cache_path}/{args.gpt_config_name}')(
        config=f'{args.root_path}/src/{args.config_cache_path}/{args.gpt_config_name}', 
        model_path=model_path,
        mode=mode
    )
    return gpt

def construct_kgunl_dataset(args, finetune_llm, replace_GPT=None):
    # need to write the unlearning dataset function
    if args.unlearn_alg == "decomp":
        dataset = construct_unlearning_dataset(
            args, replace_GPT, finetune_llm.tokenizer, c_epoch
        )
    
    elif args.unlearn_alg == "kg_grad_asc":
        # read the data
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_sentences_{args.test_dataset}.json"
        with open(data_path) as f:
            data = json.load(f)
        dataset = KGGradAscDataset(data, finetune_llm.tokenizer, use_prompt=args.use_prompt, args=args)
    
    elif args.unlearn_alg == "kg_gas_cluster":
        # read the data
        model_name = args.st_transformer.split("/")[-1]
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/cluster_{model_name}_harmful_sentences_{args.test_dataset}.json"
        with open(data_path) as f:
            data = json.load(f)
        
        inputs = {} # {prompt: list of sents}
        for cluster_id, QA_pairs in data.items():
            random.shuffle(QA_pairs)
            if args.cluster_sample >= 1:
                sample_size = args.cluster_sample
            elif args.cluster_sample < 1:
                sample_size = int(len(QA_pairs) * args.cluster_sample)
            for i in range(len(QA_pairs)):
                QA_pair = QA_pairs[i]
                prompt, sentence = QA_pair[0], QA_pair[1]
                if prompt not in inputs:
                    inputs[prompt] = []
                if i < sample_size:
                    inputs[prompt].append(sentence)

        dataset = KGGradAscDataset(inputs, finetune_llm.tokenizer, use_prompt=args.use_prompt, args=args)

    elif args.unlearn_alg == "kg_replace":
        # create tripples of harmful content
        QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
        with open (QA_path) as f:
            QA_pairs = json.load(f)
        result_dir = f'{args.root_path}/result/finetune/{args.model_name}/{args.level}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if replace_GPT != None:
            trip_results = {}
            for prompt, response in tqdm(QA_pairs.items()):
                print(f'[Response]: {[response]}')
                tripple = get_tripple(prompt, response, replace_GPT)
                print(f'[Extract Tripple]:{[tripple]}')
                
                trip_results[prompt] = tripple
                trip_results_str = json.dumps(trip_results, indent=4)
                # save the tripple results
                with open(f'{result_dir}/{args.test_dataset}_harmful_tripples_{args.test_dataset}.json', 'w') as outfile:
                    outfile.write(trip_results_str)

            with open(f'{result_dir}/{args.test_dataset}_harmful_tripples_{args.test_dataset}.json') as f:
                trip_results = json.load(f)

            # replace elements in tripples to create non-harmful sentence
            sent_results = {}
            for mode in args.replace_mode:
                for prompt, tripples in tqdm(trip_results.items()):
                    temp = []
                    for trip in tripples:
                        temp.append(tuple(trip))
                    tripples = temp
                    sentences = tripple2sentence(tripples, replace_GPT, mode)
                    
                    if len(sentences) != 0 :
                        print('----')
                        print(f'[Prompt]: {[prompt]}')
                        print(f'[Replace Sentence]: {[sentences]}')
                        sent_results[prompt] = sentences
                        sent_results_str = json.dumps(sent_results, indent=4)
                        # save the replacement results
                        with open(f'{result_dir}/{args.test_dataset}_nonharmful_sentences_{mode}_{args.test_dataset}.json', 'w') as outfile:
                            outfile.write(sent_results_str)

        # construct dataset
        dataset = KGReplaceDataset({}, finetune_llm.tokenizer, use_prompt=args.use_prompt)
        for mode in args.replace_mode:
            data_path = f'{args.root_path}/result/finetune/{args.model_name}/{args.level}/{args.test_dataset}_nonharmful_sentences_{mode}_{args.test_dataset}.json'
            with open(data_path) as f:
                merged_resp = json.load(f)
            
            dataset.update(merged_resp)
    
    elif args.unlearn_alg == "gas":
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/final_harmful_resp_{args.test_dataset}.json"
        with open(data_path) as f:
            gas_data = json.load(f)
        dataset = GradAscDataset(gas_data, finetune_llm.tokenizer, use_prompt=args.use_prompt, args=args)

    elif args.unlearn_alg == "kg_gas_rpl":
        
        # read the data for gradient ascend
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_sentences_{args.test_dataset}.json"
        with open(data_path) as f:
            gas_data = json.load(f)

        # read the data for replacement
        rpl_data = []
        for mode in ["tail", "head"]:
            data_path = f'{args.root_path}/result/finetune/{args.model_name}/{args.level}/{args.test_dataset}_nonharmful_sentences_{mode}_{args.test_dataset}.json'
            with open(data_path) as f:
                merged_resp = json.load(f)
            rpl_data.append(merged_resp)
        dataset = MixDataset(gas_data, rpl_data, finetune_llm.tokenizer, use_prompt=args.use_prompt, args=args)
    
    return dataset

def construct_sorry_dataset(args):

    QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
    with open (QA_path) as f:
        QA_pairs = json.load(f)

    # construct dataset
    dataset = SorryDataset({}, finetune_llm.tokenizer)
    dataset.update(QA_pairs)

    return dataset
    
if __name__ == '__main__':

    args = parse_args()

    accelerator = Accelerator()

    local_rank = None

    if args.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    finetune_llm = None

    # continue from the last epoch
    for c_epoch in tqdm(range(args.last_epoch, args.cycle_epochs)):
        if args.query_origin_flag:
            finetune_llm = load_ft_model(args, mode = 'train')
            finetuner = Finetuner(args, finetune_llm, replace_gpt=None)
            # load dataset
            QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
            with open (QA_path) as f:
                dataset = json.load(f)
            test_prompts = [prompt for prompt in dataset]

            out_path = f"{args.root_path}/data/reference_resp_{args.test_dataset}.json"
            result = finetuner.generate(test_prompts, out_path=out_path)

        # # ############################# KGUnL ###############################
        if args.kgunl_flag:
            print(f'[c_epoch: {c_epoch}] KG Unlearning')
            # replace_GPT = load_gpt_model(
            #     args, mode = 'inference'
            # )
            replace_GPT = None

            if finetune_llm == None:
                if c_epoch == 0:
                    # use the original llm
                    # model_path = f"{args.root_path}/save_model/attack/{args.model_name}_epoch_final"
                    finetune_llm = load_ft_model(
                        args, mode = 'train'
                    )
                else:
                    # use the llm unlearned in the previous epoch
                    model_path = f"{args.root_path}/save_model/finetune_{args.test_dataset}_{args.unlearning_type}/{args.level}/{args.model_name}_cyl_epo_{c_epoch + 1}_ft_epo_final"
                    finetune_llm = load_ft_model(
                        args, mode = 'train', model_path = model_path
                    )

            # select unlearning dataset
            if args.unlearning_type == 'kg':
                unl_dataset = construct_kgunl_dataset(
                    args, finetune_llm, replace_GPT
                )
            elif args.unlearning_type == 'sorry':
                unl_dataset = construct_sorry_dataset(args)
            else:
                raise NameError

            finetuner = Finetuner(args, finetune_llm, replace_GPT)

            # origin_result = finetuner.generate()

            finetuner.train(unl_dataset, c_epoch)

            # unlearning_result = finetuner.generate()

            # # record results
            # out_file = finetuner.construct_out_file(c_epoch + 1, 'final')

            # result_list = []
            # for prompt, unlearning_response in unlearning_result.items():
            #     result = {}
            #     result['prompt'] = prompt
            #     # result['original_response'] = origin_result[prompt][0]
            #     result['unlearning_response'] = unlearning_response[0]
            #     result_list.append(result)

            # with open(out_file, "w") as fout:
            #     json_rslt = json.dumps(result_list, indent=4)
            #     fout.write(json_rslt + "\n")


            torch.cuda.empty_cache()
        # #############################################################################
            
        ######################### Eval for Unlearning ###############################
        if args.eval_kgunl_flag:
            print(f'[c_epoch: {c_epoch}] GPT4 Eval of KG Unlearning Model')
            replace_GPT = load_gpt_model(
                args, mode = 'inference'
            )

            score_GPT = load_score_model(
                args, mode = 'inference'
            )

            data_collator = DefaultDataCollator()

            judger = Judger(args)
            scorer = Scorer(args, score_GPT, judger, data_collator)

            # score for the generated result of unleaned model
            score_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/{args.test_dataset}_{args.unlearning_type}_ft_epoch_1_epoch_{c_epoch+1}.json"
            
            scorer.eval_unlearning(score_path)

        if args.param_attack_flag:
            cluster_size = f"_{args.cluster_sample}" if args.unlearn_alg == "kg_gas_cluster" else ""
            st_transformer = f"_{args.st_transformer}" if args.unlearn_alg == "kg_gas_cluster" else ""
            model_path = f"{args.root_path}/save_model/finetune_{args.test_dataset}_{args.unlearning_type}_{args.unlearn_alg}_{args.kl_weight}{cluster_size}{st_transformer}/{args.level}/{args.model_name}_cyl_epo_{c_epoch + 1}_ft_epo_1"

            # use the unl llm
            unl_llm = load_ft_model(
                args, mode = 'train', model_path = model_path
            )
            attacker = Attacker(
                args, unl_llm
            )
            attacker.train()
            model_save_path = f"{args.root_path}/save_model/benign_ft/{args.model_name}_{args.test_dataset}_{args.unlearn_alg}_{args.kl_weight}{cluster_size}{st_transformer}"
            attacker.save_model(save_path=model_save_path)

            if attacker.llm.model.dtype == torch.float32:
                attacker.llm.model = attacker.llm.model.half()
            print(attacker.llm.model.dtype)
            # load dataset
            QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
            with open (QA_path) as f:
                dataset = json.load(f)
            test_prompts = [prompt for prompt in dataset]
            
            output_path = f"{args.root_path}/src/attacks/Parameter/Results/{args.test_dataset}_{args.unlearn_alg}_{args.kl_weight}{cluster_size}{st_transformer}_{args.model_name}.json"
            if os.path.exists(output_path):
                with open(output_path) as f:
                    final_results = json.load(f)
            else:
                final_results = []

            previous_prompts = []
            for item in final_results:
                prompt = item["question"]
                previous_prompts.append(prompt)
            previous_prompts = set(previous_prompts)
            
            temp = []
            for prompt in test_prompts:
                if prompt not in previous_prompts:
                    temp.append(prompt)
            test_prompts = temp
            for temp in np.arange(0.05, 1.05, 0.05):
                # get attack unl response results
                for itr in range(5):
                    response_result = attacker.generate(test_prompts, 
                                                    temperature=temp, 
                                                    max_new_tokens=100, 
                                                    do_sample=True,
                                                    batch_size=args.attack_batch_size
                                                    )
                    for prompt, response in response_result.items():
                        final_results.append({"question":prompt,"response":response,"param":{'temperature':temp},"iteration":itr+1})
                    with open(output_path, "w") as f:
                        json.dump(final_results, f, indent=4)

        if args.attack_origin_flag:
            # attack original model
            finetune_llm = load_ft_model(
                args, mode = 'train',
            )
            attacker = Attacker(
                args, finetune_llm
            )
            # load dataset
            QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
            with open (QA_path) as f:
                dataset = json.load(f)
            test_prompts = [prompt for prompt in dataset]
            print(f'[c_epoch: {c_epoch}] Attack The Model')
            # get original response results
            # response_result = attacker.generate()
            # finetune attack original model
            attacker.train()
            # get attack original response results
            attack_response_result = attacker.generate(test_prompts)
            out_path = f"{args.root_path}/result/attack/{args.model_name}/all_harmful_resp_{args.test_dataset}.json"
            with open(out_path, "w") as f:
                json.dump(attack_response_result, f, indent=4)

        ############################## Attack llm with few-shot finetune  ##############################
        # load model
        if args.attack_flag:
            
            # attack original model
            finetune_llm = load_ft_model(
                args, mode = 'train',
            )
            attacker = Attacker(
                args, finetune_llm
            )
            # select unlearning dataset
            QA_path = f'{args.root_path}/data/final_harmful_resp_{args.test_dataset}.json'
            with open (QA_path) as f:
                dataset = json.load(f)
            test_prompts = [prompt for prompt in dataset]

            print(f'[c_epoch: {c_epoch}] Attack The Model')
            # get original response results
            # response_result = attacker.generate()
            # finetune attack original model
            # attacker.train()
            # get attack original response results
            # attack_response_result = attacker.generate(test_prompts)
            # clean the space
            del finetune_llm.model
            del attacker
            torch.cuda.empty_cache()
            gc.collect()

            # attack unl model
            for i in range(1, 0, -1):
                cluster_size = f"_{args.cluster_sample}" if args.unlearn_alg == "kg_gas_cluster" else ""
                st_transformer = f"_{args.st_transformer}" if args.unlearn_alg == "kg_gas_cluster" else ""
                if i < args.ft_epochs:
                    model_path = f"{args.root_path}/save_model/finetune_{args.test_dataset}_{args.unlearning_type}_{args.unlearn_alg}_{args.kl_weight}{cluster_size}{st_transformer}/{args.level}/{args.model_name}_cyl_epo_{c_epoch + 1}_ft_epo_{i}"
                else:
                    model_path = f"{args.root_path}/save_model/finetune_{args.test_dataset}_{args.unlearning_type}_{args.unlearn_alg}_{args.kl_weight}{cluster_size}{st_transformer}/{args.level}/{args.model_name}_cyl_epo_{c_epoch + 1}_ft_epo_final"
                
                if not os.path.exists(model_path):
                    continue
            
                print(f"Attacking models from the {i}-th finetuning epoch")
                # use the unl llm
                unl_llm = load_ft_model(
                    args, mode = 'train', model_path = model_path
                )
                unl_attacker = Attacker(
                    args, unl_llm
                )
                # finetune attack unl model
                unl_attacker.train()
                # get attack unl response results
                attack_unl_response_result = unl_attacker.generate(test_prompts, max_new_tokens=250)
                # record results
                out_file = unl_attacker.construct_out_file(c_epoch + 1, i)

                result_list = []
                for prompt, attack_response in attack_unl_response_result.items():
                    result = {}
                    result['prompt'] = prompt
                    # result['original_response'] = response_result[prompt][0]
                    # result['attack_original_response'] = attack_response_result[prompt]
                    result['attack_unl_response'] = attack_response
                    result_list.append(result)

                with open(out_file, "w") as fout:
                    json_rslt = json.dumps(result_list, indent=4)
                    fout.write(json_rslt + "\n")
                    
                del unl_llm
                del unl_attacker
                torch.cuda.empty_cache()
                gc.collect()
            
        ######################################################################################
        if args.attack_eraser_flag:
            print(f'[c_epoch: {c_epoch}] Attack The Model')
            model_path = f"{args.root_path}/baselines/Eraser/save_models/Eraser_Llama2_7b_Lora"
            # attack original model
            finetune_llm = load_ft_model(
                args, mode = 'train', model_path = model_path
            )
            attacker = Attacker(
                args, finetune_llm
            )
            # finetune attack original model
            attacker.train()
            # get attack original response results
            attack_response_result = attacker.generate()
            result_list = []
            for prompt, attack_response in attack_response_result.items():
                result = {}
                result['prompt'] = prompt
                result['attack_response'] = attack_response
                result_list.append(result)
            out_file = f"{args.root_path}/result/attack/save_models/{args.model_name}/{args.test_dataset}_Eraser.json"
            with open(out_file, "w") as fout:
                json_rslt = json.dumps(result_list, indent=4)
                fout.write(json_rslt + "\n")

        ################### Eval harmful response after attack ###############################
        if args.eval_attack_flag:
            if c_epoch == 0:
                print(f'[c_epoch: {c_epoch}] Eval Attacked Model')
                # score for the first attack            
                score_GPT = load_score_model(
                    args, mode = 'inference'
                )
                data_collator = DefaultDataCollator()
                judger = Judger(args)
                scorer = Scorer(args, score_GPT, judger, data_collator)
                # score_path = f"{args.root_path}/result/attack/{args.model_name}/{args.test_dataset}_cyl_epo_{c_epoch+1}.json"
                # print(score_path)
                # scorer.eval(score_path, c_epoch + 1, 0)
                # for i in range(args.ft_epochs, 3, -1):
                for i in range(3, 0, -1):
                    score_path = f"{args.root_path}/result/attack/{args.model_name}/{args.test_dataset}_cyl_epo_{c_epoch+1}_{i}.json"
                    print(score_path)
                    scorer.eval(score_path, c_epoch + 1, i)
        ###################################################################