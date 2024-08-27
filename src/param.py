import argparse

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproduction.")
    parser.add_argument("--n_rewrt", type=int, default=1, help="Number of times to rewrite each answer")
    parser.add_argument("--root_path", type=str, default='/home/hzyr/llm/KGUnL')
    parser.add_argument("--config_cache_path", type=str, default='config',)
    parser.add_argument("--llm_config_name", type=str, default='llama2_7b.yaml',)
    parser.add_argument("--gpt_config_name", type=str, default="gpt4_openai.yaml",)
    parser.add_argument("--score_config_name", type=str, default="gpt4_openai.yaml",)
    parser.add_argument("--model_name", type=str, default='LLAMA27B',)
    parser.add_argument("--file_name", type=str, default='harmfulResp_attack',)
    parser.add_argument("--attack_batch_size", type=int, default=10, help="Batch size for attack model")
    parser.add_argument("--ft_batch_size", type=int, default=1, help="Batch size for finetuning model")
    parser.add_argument("--attack_dataset", type=str, default='harmfulDemo',)
    parser.add_argument("--test_dataset", type=str, default='advbench', choices=['advbench', 'FQ'])
    parser.add_argument("--cycle_epochs", type=int, default=1, help = "number of cycling epochs")
    parser.add_argument("--attack_epochs", type=int, default=5, help = "number of epochs in attack finetuning")
    parser.add_argument("--ft_epochs", type=int, default=1, help = "number of epochs in unlearning finetuning")
    parser.add_argument("--ft_lr", type=float, default=1e-5, help = "lr in finetuning")
    parser.add_argument("--attack_lr", type=float, default=1e-3, help = "lr in finetuning")
    parser.add_argument("--save_model_interval", type=int, default=5, help = "save model interval epoch")
    parser.add_argument("--device", type=str, default="cuda", 
        choices=["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help = "gpu or cpu device")
    parser.add_argument("--score_data_path", type=str, default=None)
    parser.add_argument("--enable_fsdp", type=str2bool, default=False)
    parser.add_argument("--last_epoch", type=int, default=0, help="previous cycling epoch to continue")
    parser.add_argument("--unlearn_alg", type=str, default="kg_replace",
        choices=["decomp", "kg_grad_asc", "kg_replace", "kg_gas_rpl", "kg_gas_cluster", "gas"],
        help="unlearning method")
    parser.add_argument("--st_transformer", type=str, default="all-mpnet-base-v2", 
        choices=["Alibaba-NLP/gte-Qwen1.5-7B-instruct", "all-MiniLM-L6-v2", "all-mpnet-base-v2"])
    parser.add_argument("--cluster_sample", type=float, default=4)
    parser.add_argument("--replace_mode", nargs="+",
        default=["relation", "head", "tail"][:3],
        help="mode of replacement for knowledge graph")
    parser.add_argument("--unlearning_type", type=str, default='kg')
    parser.add_argument("--kgunl_flag", type=str2bool, default=False)
    parser.add_argument("--eval_kgunl_flag", type=str2bool, default=False)
    parser.add_argument("--attack_eraser_flag", type=str2bool, default=False)
    parser.add_argument("--attack_origin_flag", type=str2bool, default=False)
    parser.add_argument("--attack_flag", type=str2bool, default=False)
    parser.add_argument("--eval_attack_flag", type=str2bool, default=False)
    parser.add_argument("--query_origin_flag", type=str2bool, default=False)
    parser.add_argument("--param_attack_flag", type=str2bool, default=False)
    parser.add_argument("--use_prompt", type=str2bool, default=True)
    parser.add_argument("--kl_weight", type=float, default=1)
    parser.add_argument("--sample", type=str2bool, default=True)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--level", type=str, default="first_level", choices=["first_level", "all_level"])
    args = parser.parse_args()

    return args