# Train SFT Model
python ./src/main.py \
    --dataset sft_train \
    --dataset_dir ./data \
    --do_train \
    --stage sft \
    --seed 2025 \
    --model_name_or_path /home/models/Llama-3.1-8B-Instruct \
    --output_dir ./checkpoints/llama_sft \
    --finetuning_type lora \
    --template llama3 \
    --lora_target all \
    --cutoff_len 1024 \
    --lora_rank 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --val_size 0.01 \
    --save_steps 2000 \
    --logging_steps 100 \
    --warmup_ratio 0.1 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 3.0 \
    --fp16

#合并模型权重，以便于继续DPO训练
python ./src/export_model.py \
    --model_name_or_path /home/models/Llama-3.1-8B-Instruct \
    --adapter_name_or_path ./checkpoints/llama_sft \
    --finetuning_type lora \
    --template llama3 \
    --export_dir ./checkpoints/llama_sft_model

# Train DPO Model
python ./src/main.py \
    --dataset dpo_train \
    --dataset_dir ./data \
    --do_train \
    --stage dpo \
    --seed 2025 \
    --model_name_or_path ./checkpoints/llama_sft_model \
    --output_dir ./checkpoints/sft_llama_dpo \
    --finetuning_type lora \
    --template llama3 \
    --lora_target all \
    --cutoff_len 1024 \
    --lora_rank 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --val_size 0.01 \
    --save_steps 2000 \
    --logging_steps 100 \
    --warmup_ratio 0.1 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 1.0 \
    --fp16




