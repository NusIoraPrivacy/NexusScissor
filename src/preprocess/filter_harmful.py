import json

root_path = "/opt/data/private/peihua/KGUnL"
model_name = "LLAMA38B"
data_name = "advbench"

with open(f"{root_path}/result/score/first_level/{model_name}/{data_name}_wo_unlearning_cyc_epoch_1_1.json") as f:
    dataset = json.load(f)

out_data = {}
success_attack = 0
success_origin = 0
unsuccess_cnt = 0
for item in dataset:
    prompt, attack_repsonse, attack_score, origin_repsonse, origin_score = item["prompt"], item["attack_repsonse"], item["attack_score"], item["origin_repsonse"], item["origin_score"]
    out_data[prompt] = []
    if int(attack_score) > 3:
        out_data[prompt].append(attack_repsonse)
        success_attack += 1
    if int(origin_score) > 3:
        out_data[prompt].append(origin_repsonse)
        success_origin += 1
    if len(out_data[prompt]) == 0:
        print(prompt)
        unsuccess_cnt += 1
# print("attack asr:", success_attack/len(dataset))
# print("origin asr:", success_origin/len(dataset))
print("Unsuccessful prompt:", unsuccess_cnt)


out_path = f"{root_path}/result/finetune/{model_name}/first_level/final_harmful_resp_{data_name}.json"
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=4)

# input_path = f"{root_path}/result/finetune/LLAMA27B/first_level/final_harmful_resp_{data_name}.json"
# with open(input_path) as f:
#     llama2_resp = json.load(f)

# input_path = f"{root_path}/result/finetune/{model_name}/first_level/final_harmful_resp_{data_name}.json"
# with open(input_path) as f:
#     llama3_resp = json.load(f)

# supple_resp = {}
# for prompt in llama3_resp:
#     if len(llama3_resp[prompt]) == 0:
#         supple_resp[prompt] = [llama2_resp[prompt]]

# out_path = f"{root_path}/result/finetune/{model_name}/first_level/final_harmful_resp_{data_name}_llama2.json"
# with open(out_path, "w") as f:
#     json.dump(supple_resp, f, indent=4)