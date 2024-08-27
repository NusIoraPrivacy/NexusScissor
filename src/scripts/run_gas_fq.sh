#!/bin/bash

# sample_sizes=(1 10 20 30 50 60 70 80)
# for sample_size in ${sample_sizes[@]}
# do
# done

# python main.py \
#     --root_path ../ \
#     --save_model_interval 1 \
#     --attack_epochs 10 \
#     --ft_epochs 2 \
#     --cycle_epochs 1 \
#     --unlearn_alg kg_gas_cluster \
#     --st_transformer all-mpnet-base-v2 \
#     --cluster_sample 1 \
#     --unlearning_type kg \
#     --kgunl_flag True \
#     --use_prompt False \
#     --kl_weight 0.5 \
#     --ft_lr 1e-4 \
#     --sample False \
#     --level first_level \
#     --model_name PHIMEDIUM \
#     --attack_dataset PureBadDemo \
#     --test_dataset FQ \
#     --llm_config_name phi_medium.yaml \
#     --score_config_name gpt4_openai.yaml \
#     --gpt_config_name gpt4_openai.yaml

python main.py \
    --root_path ../ \
    --save_model_interval 10 \
    --attack_epochs 10 \
    --ft_epochs 2 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --st_transformer all-mpnet-base-v2 \
    --cluster_sample 1 \
    --unlearning_type kg \
    --kl_weight 0.5 \
    --sample False \
    --level first_level \
    --attack_flag True \
    --model_name PHIMEDIUM \
    --attack_dataset PureBadDemo \
    --test_dataset FQ \
    --llm_config_name phi_medium.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml