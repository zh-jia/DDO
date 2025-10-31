GPUS="0"
seed=42
dataset_name="DXY"
llm_name="qwen2.5-7b-instruct"  # llama3-8b-chinese-chat, qwen2.5-14b-instruct
adapter_ckpt="./outputs/adapters/qwen2.5-7b-instruct-DXY-curstom_llm_calibration"
test_policy_path="./outputs/policy/DXY/curstom_rl_training"
floor_turns=3  # dxy:3  gmd:5  cmd:5
window_size=3  # dxy:3  gmd:4  cmd:5                   
num_samples=6  # dxy:6  gmd:6  cmd:7         
retry=1        # dxy:1  gmd:2  cmd:2

CUDA_VISIBLE_DEVICES=$GPUS python inference.py \
    --seed $seed \
    --dataset_name $dataset_name \
    --llm_name $llm_name \
    --adapter_ckpt $adapter_ckpt \
    --test_policy_path $test_policy_path \
    --floor_turns $floor_turns \
    --window_size $window_size \
    --num_samples $num_samples \
    --retry $retry