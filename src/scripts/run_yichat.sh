#!/bin/bash

python main.py \
    --root_path ../ \
    --save_model_interval 1 \
    --attack_epochs 10 \
    --ft_epochs 3 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --st_transformer all-mpnet-base-v2 \
    --cluster_sample 100 \
    --unlearning_type kg \
    --kgunl_flag True \
    --use_prompt False \
    --kl_weight 0 \
    --ft_lr 1e-4 \
    --sample False \
    --model_name YICHAT \
    --level first_level \
    --attack_dataset PureBadDemo \
    --test_dataset advbench \
    --llm_config_name yichat_34b.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml

python main.py \
    --root_path ../ \
    --save_model_interval 10 \
    --attack_epochs 10 \
    --ft_epochs 3 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --st_transformer all-mpnet-base-v2 \
    --cluster_sample 100 \
    --unlearning_type kg \
    --kl_weight 0 \
    --sample True \
    --sample_size 50 \
    --model_name YICHAT \
    --level first_level \
    --attack_flag True \
    --attack_dataset PureBadDemo \
    --test_dataset advbench \
    --llm_config_name yichat_34b.yaml \
    --attack_batch_size 5 \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml