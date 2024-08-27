#!/bin/bash

data_names=("glue_rte" "glue_sst2" "glue_qnli" "glue_qqp")
f_epochs=("1")
for f_epoch in ${f_epochs[@]}
do
    for data_name in ${data_names[@]}
    do
        python benchmark.py --data_name=$data_name \
            --root_path /opt/data/private/peihua/KGUnL \
            --base_model LLAMA38B --f_epoch=$f_epoch --epochs 2 \
            --ft_model_name finetune_advbench_kg_kg_gas_cluster_1.0_1.0_all-mpnet-base-v2 \
            --model_name meta-llama/Meta-Llama-3-8B-Instruct
    done
done