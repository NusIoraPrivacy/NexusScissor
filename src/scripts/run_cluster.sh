#!/bin/bash

python3 main.py \
    --root_path ../ \
    --save_model_interval 1 \
    --attack_epochs 10 \
    --ft_epochs 3 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --st_transformer all-mpnet-base-v2 \
    --cluster_sample 2 \
    --unlearning_type kg \
    --kgunl_flag True \
    --use_prompt False \
    --kl_weight 1 \
    --sample False \
    --level first_level \
    --attack_dataset PureBadDemo \
    --test_dataset FQ \
    --llm_config_name llama2_7b.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml

python3 main.py \
    --root_path ../ \
    --save_model_interval 10 \
    --attack_epochs 10 \
    --ft_epochs 3 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --st_transformer all-mpnet-base-v2 \
    --cluster_sample 2 \
    --unlearning_type kg \
    --kl_weight 1 \
    --sample False \
    --level first_level \
    --attack_flag True \
    --attack_dataset PureBadDemo \
    --test_dataset FQ \
    --llm_config_name llama2_7b.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml