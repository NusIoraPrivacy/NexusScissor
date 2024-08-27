import torch

from tqdm import tqdm
import json
import argparse
import os

from transformers import (AutoTokenizer, 
                        AutoModelForSequenceClassification)
from huggingface_hub import login

parent_parent_dir = os.path.dirname(os.path.abspath(__file__))

def create_prompt(responses):
    prompts = []
    for resp in responses:
        question, answer = resp["prompt"], resp["response"]
        prompt = f"Human: {question} Assistant: {answer}"
        prompts.append(prompt)
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect all attack results for a given model"
    )
    parser.add_argument(
        "--model_name", type=str, default="llama-2"
    )
    parser.add_argument(
        "--config", type=str, default="exploited"
    )
    parser.add_argument(
        "--score_model", type=str, default="google-bert/bert-large-uncased"
    )
    parser.add_argument(
        "--dataset", type = str, default="advbench", choices = ["advbench", "forbidenq"]
    )
    parser.add_argument(
        "--binary_label", type=bool, default=False
    )
    parser.add_argument(
        "--batch_size", type=bool, default=100
    )
    parser.add_argument(
        "--test_epoch", type=int, default=20
    )
    parser.add_argument(
        "--save_model_interval", type=int, default=20
    )
    args = parser.parse_args()

    # read data
    model2dir = {"llama-2": "llama", "gpt-3.5": "gpt", "vicuna": "vicuna", "gpt-4": "gpt"}
    model_dir = model2dir[args.model_name]
    response_path = f"{parent_parent_dir}/Results/{model_dir}/{args.dataset}/Parameters_{args.model_name}.json"
    with open(response_path) as f:
        responses = json.load(f)
    print(len(responses))

    # model path    
    model_name = args.score_model.split("/")[-1]
    if args.binary_label:
        save_path = f"{parent_parent_dir}/save_model/{model_name}_binary_{args.dataset}_{args.test_epoch}"
    else:
        save_path = f"{parent_parent_dir}/save_model/{model_name}_{args.dataset}_{args.test_epoch}"
    
    # load tokenizer and model
    num_labels = 2 if args.binary_label else 5
    tokenizer = AutoTokenizer.from_pretrained(
        args.score_model
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        save_path, num_labels=num_labels
    )
    model = model.cuda()

    # obtain prediction for each response
    if args.binary_label:
        output_path = f"{parent_parent_dir}/Results/{model_dir}/{args.dataset}/Parameters_score_all_binary_{args.model_name}.json"
    else:
        output_path = f"{parent_parent_dir}/Results/{model_dir}/{args.dataset}/Parameters_score_all_{args.model_name}.json"
    
    score_responses = []
    with torch.no_grad():
        for i in tqdm(range(0, len(responses), args.batch_size)):
            batch = responses[i : i + args.batch_size]
            prompts = create_prompt(batch)
            inputs = tokenizer(prompts, padding="max_length", max_length=512, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits
            y_preds = torch.argmax(logits, -1)
            y_preds = y_preds.tolist()
            for resp, y in zip(batch, y_preds):
                resp["score"] = y
                score_responses.append(resp)
            
            with open(output_path, 'w') as f:
                json.dump(score_responses, f, indent=4)
