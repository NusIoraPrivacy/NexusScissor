from openai import OpenAI
from param import str2bool
import json
from tqdm import tqdm
from model import AutoLLM
import argparse
import os

def create_message_tripple(prompt, response):
    prompt = prompt.strip().strip(".")
    full_prompt = (
        "You should extract the harmful tripples from a paragraph. "
        "Please remember to response must in the format of tripple list: "
        "[(entity, relation, entity), (entity, relation, entity), (entity, relation, entity)]. "
        "If there's no harmful content, return an empty list: []. Paragraph: "
        "{prompt}. {response}"
        )
    full_prompt = full_prompt.format(prompt = prompt, response = response)
    message = [{"role": "user", "content": full_prompt}]
    return message

def create_full_message_tripple(prompt, response, tripples):
    message = create_message_tripple(prompt, response)
    tripples = [tuple(tripple) for tripple in tripples]
    temp = []
    for tripple in tripples:
        format_trpl = ['"{}"'.format(item) for item in tripple]
        format_trpl = ', '.join(format_trpl)
        format_trpl = f"({format_trpl})"
        temp.append(format_trpl)
    temp = ', '.join(temp)
    temp = f"[{temp}]"
    # temp = temp.replace("'", "\\'")
    tripples = temp
    # print(tripples)
    message += [{"role": "assistant", "content": tripples}]
    message = {"messages": message}
    return message

def create_message_replace(tripple, mode="head"):
    if mode=="tail":
        full_prompt = (
            "Given a list of harmful tripples [(head, relation, tail),...], "
            "you should replace the tail in the tripples and then output a list of non-harmful sentences. "
            "If the head itself is harmful, you can skip the corresponding tripple. "
            "Remember to response in the format of sentence list: [sentences 1,...]. "
            "If there's no tripple that can be converted to non-harmful sentence, return an empty list. "
            "Tripple list: "
            "{tripple}"
            )
    elif mode=="relation":
        full_prompt = (
            "Given a list of harmful tripples [(head, relation, tail),...], "
            "you should replace the relation in the tripples, and then output a list of non-harmful sentences. "
            "Remember to response in the format of sentence list: [sentences 1,...]. Tripple list: "
            "{tripple}"
            )
    elif mode=="head":
        full_prompt = (
            "Given a list of harmful tripples [(head, relation, tail),...], "
            "you should replace the head in the tripples and then output a list of non-harmful sentences. "
            "If there are multiple tripple with the essentially same relation and tail, you can combine them and return a single sentence. "
            "If there's no tripple that can be converted to non-harmful sentence, return an empty list. "
            "Remember to response in the format of sentence list: [sentences 1,...]. Tripple list: "
            "{tripple}"
            )
    else:
        raise ValueError("Invalid mode!")
    
    full_prompt = full_prompt.format(tripple = tripple)
    message = [{"role": "user", "content": full_prompt}]
    return message

def create_message_triple2sent(tripple):
    full_prompt = (
        "Given a list of tripples [(head, relation, tail),...], "
        "you should convert the tripples into a list of coherent sentences. "
        "Tripples: "
        "{tripple}."
        'Strictly respond in the format of ["sentence 1", "sentence 2", ...]'
        )
    full_prompt = full_prompt.format(tripple = tripple)
    message = [{"role": "user", "content": full_prompt}]
    return message

def create_message_triple2sentv2(tripple):
    full_prompt = (
        "Given a list of tripples [(head, relation, tail),...], "
        "you should concatenate the head and relation of each tripple into prefix, "
        "and convert the tripples into a list of prefix-tail pairs. "
        "Tripples: "
        "{tripple}."
        'Strictly respond in the format of [(\"prefix1\", \"tail1\"), (\"prefix2\", \"tail2\"), ...]'
        )
    full_prompt = full_prompt.format(tripple = tripple)
    message = [{"role": "user", "content": full_prompt}]
    return message

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='../')
    parser.add_argument("--config_cache_path", type=str, default='config')
    parser.add_argument("--model_name", type=str, default='LLAMA27B')
    parser.add_argument("--data_name", type=str, default='FQ', choices=['advbench', 'FQ'])
    parser.add_argument("--tripple2message", type=str2bool, default=False)
    parser.add_argument("--resp2tripple", type=str2bool, default=False)
    parser.add_argument("--tripple2nonharm", type=str2bool, default=False)
    parser.add_argument("--tripple2sent", type=str2bool, default=True)
    parser.add_argument("--tripple2sentv2", type=str2bool, default=False)
    parser.add_argument("--level", type=str, default="first_level", choices=["first_level", "all_level"])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_path = f"{args.root_path}/result/finetune/{args.model_name}/first_level/final_harmful_resp_{args.data_name}.json"
    with open(data_path) as f:
        resps_data = json.load(f)
    print(len(resps_data))

    if args.tripple2message:
        resps_data = {}
        input_path = f"{args.root_path}/data/final_harmful_resp_advbench.json"
        with open(input_path) as f:
            resps_data.update(json.load(f))
        input_path = f"{args.root_path}/data/final_harmful_resp_FQ.json"
        with open(input_path) as f:
            resps_data.update(json.load(f))

        data_path = f"{args.root_path}/data/harmfulResp_harmful_tripples_demo.json"
        with open(data_path) as f:
            harm_tripples = json.load(f)
        
        messages = []
        for prompt, tripples in harm_tripples.items():
            response = resps_data[prompt]
            message = create_full_message_tripple(prompt, response, tripples)
            messages.append(message)
        
        out_path = f"{args.root_path}/data/harmful_tripple_demo.jsonl"
        with open(out_path, 'w') as f:
            for message in messages:
                json.dump(message, f)
                f.write('\n')
    
    if args.resp2tripple:
        config_path = f'{args.root_path}/src/{args.config_cache_path}/resp2tripple.yaml'
        resp2tripple_GPT = AutoLLM.from_name(config_path)(
                config_file=config_path, 
                mode = 'inference'
            )

        out_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_tripples_{args.data_name}.json"
        if os.path.exists(out_path):
            with open(out_path) as f:
                harm_tripples = json.load(f)
        else:
            harm_tripples = {}

        # remain_prompts = {}
        # for prompt in remain_prompts:
        #     harm_tripples[prompt].extend(remain_prompts[prompt])
        # with open(out_path, 'w') as f:
        #     json.dump(harm_tripples, f, indent = 4) 

        for i, (prompt, responses) in tqdm(enumerate(resps_data.items())):
            if prompt in harm_tripples:
                continue
            harm_tripples[prompt] = []
            if isinstance(responses, str):
                responses = [responses]
            for response in responses:
                message = create_message_tripple(prompt, response)
                tries = 1
                success = False
                while not success:
                    result = resp2tripple_GPT.generate(message, max_tokens=3000)
                    # print(result)
                    try:
                        harm_tripples[prompt].extend(eval(result))
                        success = True
                    except:
                        pass
                    if not success:
                        print(result)
                    tries += 1
                    if tries >= 5:
                        print(f"Unsucessful prompt: {prompt}")
                        break
            
            with open(out_path, 'w') as f:
                json.dump(harm_tripples, f, indent = 4) 
    
    ### Step 2: Convert harmful tripples into non-harmful sentences
    if args.tripple2nonharm:
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_tripples_{args.data_name}.json"
        with open(data_path) as f:
            harm_tripples = json.load(f)
        
        ## Start with replacing the tails and relation
        mode2config = {
            "relation": "rpl_relation_cnt.yaml",
            "head": "rpl_head.yaml",
            "tail": "rpl_tail.yaml"
        }
        # modes = ["relation_unrelate", "relation_cant", "tail"]
        modes = ["head", "tail"]
        for mode in modes:
            config_path = f'{args.root_path}/src/{args.config_cache_path}/{mode2config[mode]}'
            tripple2sent_GPT = AutoLLM.from_name(config_path)(
                config_file=config_path, 
                mode = 'inference'
            )
            out_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_nonharmful_sentences_{mode}_{args.data_name}.json"
            if os.path.exists(out_path):
                with open(out_path) as f:
                    nonharm_sents = json.load(f)
            else:
                nonharm_sents = {}
            
            cnt = 1
            for i, (prompt, tripple) in tqdm(enumerate(harm_tripples.items())):
                tripple = [tuple(i) for i in tripple]
                message = create_message_replace(tripple, mode.split("_")[0])
                success = False
                tries = 1
                while not success:
                    result = tripple2sent_GPT.generate(message)
                    try:
                        nonharm_sents[prompt] = eval(result)
                        success = True
                    except:
                        nonharm_sents[prompt] = []
                    tries += 1
                    if tries >= 5:
                        print(f"Unsucessful prompt: {prompt}")
                        break

                with open(out_path, 'w') as f:
                    json.dump(nonharm_sents, f, indent = 4) 
                cnt += 1
    
    if args.tripple2sent:
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_tripples_{args.data_name}.json"
        with open(data_path) as f:
            harm_tripples = json.load(f)

        config_path = f'{args.root_path}/src/{args.config_cache_path}/tripple2sent.yaml'
        resp2tripple_GPT = AutoLLM.from_name(config_path)(
                config_file=config_path, 
                mode = 'inference'
            )
        out_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_sentences_{args.data_name}.json"
        
        if os.path.exists(out_path):
            with open(out_path) as f:
                sentences = json.load(f)
        else:
            sentences = {}

        for i, (prompt, tripple) in tqdm(enumerate(harm_tripples.items())):
            if prompt in sentences:
                continue
            tripple = [tuple(i) for i in tripple]
            message = create_message_triple2sent(tripple)
            success = False
            tries = 1
            while not success:
                result = resp2tripple_GPT.generate(message)
                try:
                    sentences[prompt] = eval(result)
                    success = True
                except:
                    sentences[prompt] = []
                tries += 1
                if tries >= 10:
                    print(f"Unsucessful prompt: {prompt}")
                    break
            
            with open(out_path, 'w') as f:
                json.dump(sentences, f, indent = 4) 
    
    if args.tripple2sentv2:
        data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_tripples_{args.data_name}.json"
        with open(data_path) as f:
            harm_tripples = json.load(f)

        config_path = f'{args.root_path}/src/{args.config_cache_path}/tripple2sentv2.yaml'
        resp2tripple_GPT = AutoLLM.from_name(config_path)(
                config_file=config_path, 
                mode = 'inference'
            )
        out_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_sentences_{args.data_name}_v2.json"
        
        if os.path.exists(out_path):
            with open(out_path) as f:
                sentences = json.load(f)
        else:
            sentences = {}

        for i, (prompt, tripple) in tqdm(enumerate(harm_tripples.items())):
            if prompt in sentences:
                continue
            tripple = [tuple(i) for i in tripple]
            message = create_message_triple2sentv2(tripple)
            success = False
            tries = 1
            while not success:
                result = resp2tripple_GPT.generate(message)
                try:
                    sentences[prompt] = eval(result)
                    success = True
                except:
                    sentences[prompt] = []
                tries += 1
                if tries >= 10:
                    print(f"Unsucessful prompt: {prompt}")
                    break
            
            with open(out_path, 'w') as f:
                json.dump(sentences, f, indent = 4) 