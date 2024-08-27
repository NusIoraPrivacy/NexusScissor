debug_flag=false
attack_dataset=PureBadDemo
test_dataset=harmfulResp
unlearn_alg=kg_replace

if [ "$debug_flag" == true ]; then
    python main.py \
        --root_path .. \
        --save_model_interval 100 \
        --attack_epochs 10 \
        --ft_epochs 5 \
        --cycle_epochs 1 \
        --unlearning_type kg \
        --unlearn_alg $unlearn_alg \
        --attack_dataset $attack_dataset \
        --test_dataset $test_dataset \
        --llm_config_name llama2_7b.yaml \
        --score_config_name gpt4_azure.yaml \
        --gpt_config_name gpt4_azure.yaml \
        --kgunl_flag False \
        --eval_kgunl_flag False \
        --attack_flag False \
        --eval_attack_flag True \

else
    nohup python main.py \
        --root_path .. \
        --save_model_interval 100 \
        --attack_epochs 10 \
        --ft_epochs 5 \
        --cycle_epochs 1 \
        --unlearning_type kg \
        --unlearn_alg $unlearn_alg \
        --attack_dataset $attack_dataset \
        --test_dataset $test_dataset \
        --llm_config_name llama2_7b.yaml \
        --score_config_name gpt4_azure.yaml \
        --gpt_config_name gpt4_azure.yaml \
        --kgunl_flag False \
        --eval_kgunl_flag False \
        --attack_flag False \
        --eval_attack_flag True > score_log_v1.log & 
fi
