#!/bin/bash
data_names=("glue_sst2")
f_epochs=("1")
for f_epoch in ${f_epochs[@]}
do
    for data_name in ${data_names[@]}
    do
        python benchmark.py --data_name=$data_name \
            --root_path /opt/data/private/peihua/KGUnL \
            --base_model LLAMA213B --f_epoch=$f_epoch \
            --ft_model_name finetune_advbench_kg_gas_1.0