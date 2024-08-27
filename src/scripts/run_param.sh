#!/bin/bash

python main.py \
    --root_path ../ \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_cluster \
    --unlearning_type kg \
    --kl_weight 0.5 \
    --cluster_sample 8 \
    --sample False \
    --level first_level \
    --param_attack_flag True \
    --attack_dataset BenignDemo \
    --test_dataset advbench \
    --model_name PHIMEDIUM \
    --attack_lr 1e-3 \
    --llm_config_name phi_medium_v2.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml