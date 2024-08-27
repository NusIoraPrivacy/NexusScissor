# nohup python main.py \

python main.py \
    --root_path . \
    --save_model_interval 1 \
    --attack_epochs 10 \
    --ft_epochs 5 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_rpl \
    --unlearning_type kg \
    --kgunl_flag True \
    --use_prompt False \
    --sample False \
    --level first_level \
    --ft_batch_size 2 \
    --kl_weight 1 \
    --attack_dataset PureBadDemo \
    --test_dataset harmfulResp \
    --llm_config_name llama2_7b.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml

python main.py \
    --root_path . \
    --save_model_interval 10 \
    --attack_epochs 10 \
    --ft_epochs 5 \
    --cycle_epochs 1 \
    --unlearn_alg kg_gas_rpl \
    --unlearning_type kg \
    --kl_weight 1 \
    --sample False \
    --attack_flag True \
    --level first_level \
    --attack_dataset PureBadDemo \
    --test_dataset harmfulResp \
    --llm_config_name llama2_7b.yaml \
    --score_config_name gpt4_openai.yaml \
    --gpt_config_name gpt4_openai.yaml