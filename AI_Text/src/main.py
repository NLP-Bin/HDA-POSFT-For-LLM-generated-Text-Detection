# main.py
import argparse
import torch
from datetime import datetime
import json
import os
import warnings
from utils.logger import get_logger
from utils.evaluation import evaluate_test_data
from utils.evaluation import evaluate_dev_data
from train import run_exp

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

# 设置环境变量
os.environ["WANDB_API_KEY"] = "af73b90246f06d77e8d6bec203e8b13bd4b0fdad"  # 替换为你的实际值

train_params_list = [
    "do_train",
    "seed",
    "model_name_or_path",
    "template",
    "stage",
    "lora_target",
    "reward_model",
    "dataset",
    "dataset_dir",
    "finetuning_type",
    "save_safetensors",
    "lora_rank",
    "output_dir",
    "overwrite_output_dir",
    "overwrite_cache",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "evaluation_strategy",
    "logging_steps",
    "save_steps",
    "cutoff_len",
    "save_total_limit",
    "val_size",
    "learning_rate",
    "num_train_epochs",
    "load_best_model_at_end",
    "bf16",
    "fp16",
    "pref_beta",
    "pref_loss",
    "plot_loss",
    "warmup_ratio",
    "pref_ftx",
    "report_to",
    "run_name",
    "ddp_find_unused_parameters",
    "resume_from_checkpoint",
    "packing"
]

def init_args():
    parser = argparse.ArgumentParser(description="Run training and other operations.")
    
    train_args = parser.add_argument_group('train_args', 'Arguments for train_bash.py')
    
    # 添加train_bash.py所需的参数到 train_args 组
    train_args.add_argument("--do_train", action="store_true")
    train_args.add_argument("--seed", type=int, default=42)
    train_args.add_argument("--model_name_or_path", type=str)
    train_args.add_argument("--reward_model", type=str)
    train_args.add_argument("--template", type=str)
    train_args.add_argument("--stage", type=str, default="ppo")
    train_args.add_argument("--lora_target", type=str, default="all")
    train_args.add_argument("--dataset", type=str)
    train_args.add_argument("--cutoff_len", type=int, default=512)
    train_args.add_argument("--dataset_dir", type=str, default="./data")
    train_args.add_argument("--finetuning_type", type=str, default="lora")
    train_args.add_argument("--lora_rank", type=int, default=64)
    train_args.add_argument("--output_dir", type=str, default="")
    train_args.add_argument("--overwrite_output_dir", action="store_true", default=True)
    train_args.add_argument("--overwrite_cache", action="store_true")
    train_args.add_argument("--per_device_train_batch_size", type=int, default=4)
    train_args.add_argument("--per_device_eval_batch_size", type=int)
    train_args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_args.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_args.add_argument("--evaluation_strategy", type=str, default="steps")
    train_args.add_argument("--logging_steps", type=int, default=200)
    train_args.add_argument("--save_steps", type=int, default=400)
    train_args.add_argument("--save_total_limit", type=int, default=3)
    train_args.add_argument("--val_size", type=float)
    train_args.add_argument("--save_safetensors", action="store_true", default=False)
    train_args.add_argument("--learning_rate", type=float, default=8e-5)
    train_args.add_argument("--num_train_epochs", type=float, default=3.0)
    train_args.add_argument("--load_best_model_at_end", action="store_true")
    train_args.add_argument("--bf16", action="store_true")
    train_args.add_argument("--fp16", action="store_true")
    train_args.add_argument("--resume_from_checkpoint", type=str, default=None)
    train_args.add_argument("--pref_beta", type=float, default=0.1)
    train_args.add_argument("--pref_loss", type=str, default="sigmoid")
    train_args.add_argument("--plot_loss", action="store_true")
    train_args.add_argument("--pref_ftx", type=float, default=0.1)
    train_args.add_argument("--report_to", type=str, default="wandb")
    train_args.add_argument("--run_name", type=str)
    train_args.add_argument("--warmup_ratio", type=float)
    train_args.add_argument("--ddp_find_unused_parameters", action="store_true")
    train_args.add_argument("--packing", action="store_true", default=False)

    data_args = parser.add_argument_group('main_args', 'Arguments for main.py')
    # 添加main.py所需的参数到 data_args 组
    data_args.add_argument("--dataset_name", type=str, help="The name of the dataset")
    data_args.add_argument("--do_dev", action="store_true")
    data_args.add_argument("--do_test", action="store_true")

    args = parser.parse_args()
    # 使用字典推导式从 args 中提取 train_args 和 data_args 的参数
    train_params = {param: getattr(args, param) for param in args.__dict__ if param in train_params_list}
    data_params = {param: getattr(args, param) for param in args.__dict__ if param not in train_params_list}

    # 手动添加参数
    if "run_name" not in train_params.keys() or train_params["run_name"] is None or train_params["run_name"] == "":
        train_params["run_name"] = f'[{train_params["template"]}] SFT-{train_params["stage"].upper()}-AI-Text-Classifier-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    # if 'dataset' not in train_params.keys() or train_params["dataset"] is None:
    #     train_params["dataset"] = f'{data_params["dataset_name"].lower()}_inst'
    # if 'output_dir' not in train_params.keys() or train_params["output_dir"] is None or train_params["output_dir"] == "":
    #     train_params["output_dir"] = f'./checkpoints/fine_tuning_checkpoints/{data_params["dataset_name"]}/{train_params["template"]}'
    # if 'reward_model' not in train_params.keys() or train_params["reward_model"] is None or train_params["reward_model"] == "":
    #     train_params["reward_model"] = f'./checkpoints/reward_model_checkpoints/{data_params["dataset_name"]}/{train_params["template"]}'
    return train_params, data_params
    
def main():
    train_params, data_params = init_args()

    if train_params["do_train"]:
        print("\n========================================")
        print("Start training...")
        print("========================================\n")

        # 开始训练模型
        run_exp(train_params)

        torch.cuda.empty_cache()
    else:
        if data_params["do_dev"]:
            print("\n===========================================================================================================================")
            print(f"Start evaluating..., Now evaluating LLM is {train_params['output_dir']}")
            print("============================================================================================================================\n")


            with open(f"{train_params['dataset_dir']}/augmented_inst_data/inst_dev.json", 'r', encoding='utf-8') as f:
                dev_data = json.load(f)
            scores = evaluate_dev_data(dev_data, model_name_or_path=train_params["model_name_or_path"],
                               checkpoint_dir=train_params["output_dir"], template=train_params["template"],
                               temperature=0.1, top_p=0.9, finetuning_type=train_params["finetuning_type"])
            logger.info(f"Finish evaluating..., the scores are {scores}")

            with open(f"./results/result_metric.jsonl", 'a',
                      encoding='utf-8') as f:
                scores['Setting'] = train_params
                json_line = json.dumps(scores, ensure_ascii=False) + '\n'
                f.write(json_line)
        elif data_params["do_test"]:
            print(
                "\n===========================================================================================================================")
            print(f"Start Testing..., Now Testing LLM is {train_params['output_dir']}")
            print(
                "============================================================================================================================\n")

            with open(f"{train_params['dataset_dir']}/augmented_inst_data/inst_test.json", 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            evaluate_test_data(test_data, model_name_or_path=train_params["model_name_or_path"],
                                   checkpoint_dir=train_params["output_dir"], template=train_params["template"],
                                   temperature=0.1, top_p=0.9, finetuning_type=train_params["finetuning_type"])
            logger.info(f"Finish Testing...")

        else:
            print("请指定do_dev或do_test")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()