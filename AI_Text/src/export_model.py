from llamafactory.train.tuner import export_model
from utils.logger import get_logger
import argparse

logger = get_logger(__name__)

def init_args():
    parser = argparse.ArgumentParser(description="Run training and other operations.")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--adapter_name_or_path", type=str)
    parser.add_argument("--template", type=str)
    parser.add_argument("--finetuning_type", type=str)
    parser.add_argument("--export_dir", type=str)

    export_args = parser.parse_args()
    # 转换为字典
    export_args_dict = vars(export_args)
    return export_args_dict


def main():
    export_args = init_args()
    print("\n========================================")
    print("Start exporting LoRA model...")
    print("========================================\n")
    export_model(export_args)

if __name__ == '__main__':
    main()