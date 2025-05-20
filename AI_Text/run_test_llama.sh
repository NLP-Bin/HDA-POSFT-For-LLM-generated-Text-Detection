
## Dev DPO Model
#python ./src/main.py \
#    --do_dev \
#    --dataset dpo_train \
#    --dataset_dir ./data \
#    --model_name_or_path /home/models/Llama-3.1-8B-Instruct \
#    --finetuning_type lora \
#    --template llama3 \
#    --output_dir /home/liuhd/AI_Text/checkpoints/llama_sft


# Test DPO Model
python ./src/main.py \
    --do_test \
    --dataset dpo_train \
    --dataset_dir ./data \
    --model_name_or_path /home/models/Llama-3.1-8B-Instruct \
    --finetuning_type lora \
    --template llama3 \
    --output_dir /home/liuhd/AI_Text/checkpoints/llama_sft
