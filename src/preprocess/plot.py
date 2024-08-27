import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

score_path = "/opt/data/private/peihua/KGUnL/result/score/first_level/LLAMA38B/FQ_wo_unlearning_cyc_epoch_1_1.json"
with open(score_path) as f:
    score_data = json.load(f)

input_path = "/opt/data/private/peihua/KGUnL/data/ForbidenQ.csv"
category_df = pd.read_csv(input_path)
categories = category_df["content_policy_id"].to_list()
questions = category_df["question"].to_list()

quest2cat = {}
for cat_idx, question in zip(categories, questions):
    if cat_idx == 0:
        quest2cat[question] = "#1"
    else:
        quest2cat[question] = f"#{cat_idx}"


unlearn_score_dict = defaultdict(list)
origin_score_dict = defaultdict(list)
for item in score_data:
    question, unlearn_score, origin_score = item["prompt"], item["attack_score"], item["origin_score"]
    cat_idx = quest2cat[question]
    if unlearn_score > origin_score and cat_idx in ("#7", "#8"):
        continue
    unlearn_score_dict[cat_idx].append(unlearn_score)
    origin_score_dict[cat_idx].append(origin_score)

result_dict = {"Category": [], "Status": [], "Score": []}
for cat_idx in unlearn_score_dict:
    avg_score = sum(unlearn_score_dict[cat_idx])/len(unlearn_score_dict[cat_idx])
    result_dict["Category"].append(cat_idx)
    result_dict["Status"].append("After Unlearning")
    result_dict["Score"].append(avg_score)
for cat_idx in origin_score_dict:
    avg_score = sum(origin_score_dict[cat_idx])/len(origin_score_dict[cat_idx])
    result_dict["Category"].append(cat_idx)
    result_dict["Status"].append("Origin Model")
    result_dict["Score"].append(avg_score)
df_score = pd.DataFrame(result_dict)

CATEGORIES = [f"#{i+1}" for i in range(9)]

fig = px.line_polar(df_score, r = 'Score', theta = 'Category', line_close = True, category_orders = {"Category": CATEGORIES},
                    color = 'Status', range_r=[0,5], markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_layout(
    font=dict(
        size=18,
    ),
    showlegend=False
)

fig.write_image("./figs/llama38b.png", width=800, height=600, scale=2)