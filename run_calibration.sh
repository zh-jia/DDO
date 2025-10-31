GPUS="0"
seed=42
test_ckpt="abcd"
target_modules=("q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj")
num_train_epochs=1  
max_steps=1000    # -1 represents disabling max steps
save_steps=50
gradient_accumulation_steps=8
learning_rate=5e-5
per_device_train_batch_size=1
weight_decay=0.1
warmup_ratio=0.05
lora_rank=16
lora_alpha=32
lora_dropout=0.1
max_grad_norm=1.0
epsilon=0.1
logging_steps=3

dataset_name="DXY"
model_name="qwen2.5-7b-instruct"  # llama3-8b-chinese-chat, qwen2.5-14b-instruct
exp_name="$(TZ='Asia/Shanghai' date +"%m-%d_%H-%M")-$model_name-$dataset_name-curstom_llm_calibration"
CUDA_VISIBLE_DEVICES=$GPUS python3 confidence_calibration.py \
    --seed $seed \
    --test_ckpt $test_ckpt \
    --exp_name $exp_name \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --target_modules "${target_modules[@]}" \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --num_train_epochs $num_train_epochs \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --max_grad_norm $max_grad_norm \
    --epsilon $epsilon \
    --save_steps $save_steps \
    --logging_steps $logging_steps

start=50
end=$max_steps
interval=50 
test_ckpt="./outputs/adapters/$exp_name/checkpoint-"

for ((idx=start; idx<=end; idx+=interval)); do
    full_ckpt_path="${test_ckpt}${idx}"
    
    CUDA_VISIBLE_DEVICES=$GPUS python3 confidence_calibration.py \
        --seed $seed \
        --test_ckpt $full_ckpt_path \
        --exp_name $exp_name \
        --dataset_name $dataset_name \
        --model_name $model_name \
        --target_modules "${target_modules[@]}" \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_dropout $lora_dropout \
        --num_train_epochs $num_train_epochs \
        --max_steps $max_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --weight_decay $weight_decay \
        --learning_rate $learning_rate \
        --warmup_ratio $warmup_ratio \
        --max_grad_norm $max_grad_norm \
        --epsilon $epsilon \
        --save_steps $save_steps \
        --logging_steps $logging_steps
done