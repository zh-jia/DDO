GPUS="0"
seed=42
dataset_name="DXY"
exp_name="$(TZ='Asia/Shanghai' date +"%m-%d_%H-%M")-curstom_rl_training"
llm_name="qwen2.5-7b-instruct"  # llama3-8b-chinese-chat, qwen2.5-14b-instruct
adapter_ckpt_list=(
    "candidate_adapter1_path"
    "candidate_adapter2_path"
    "candidate_adapter3_path"
)
net_arch_pi="256 128 128"       # dxy:256 128 128  gmd:256 128 128  cmd:512 256 256
net_arch_vf="64"                # dxy:64  gmd:64  cmd:128 
policy_type="ppo"
top_k=4                         # dxy:4  gmd:8  cmd:10    for reducing training cost                        
max_turns=10            
floor_turns=3                   # dxy:3  gmd:6  cmd:5 
window_size=3                   # dxy:3  gmd:5  cmd:5     
num_samples=6                   # dxy:6  gmd:6  cmd:7   
symptom_status_threshold=0.15   # dxy:0.15  gmd:0.15  cmd:0.2
importance_threshold=0.15       # dxy:0.15  gmd:0.15  cmd:0.2
n_steps=1024                    # dxy:1024  gmd:2048  cmd:2048
n_epochs=5                      
batch_size=64                   # dxy:64  gmd:128  cmd:128
n_envs_train=1                  
n_envs_dev=1
total_timesteps=51200           # dxy:51200  gmd:102400  cmd:102400 
callback_interval=5120          # dxy:5120  gmd:10240  cmd:10240 
learning_rate=5e-5
vf_coef=0.5
ent_coef=0.01
max_grad_norm=0.5
r_hit=0.5
freq_penaty=-0.2       
r_up=0.5
r_down=-0.5
r_correct=1.0
r_incorrect=-1.0

for adapter_ckpt in "${adapter_ckpt_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPUS python train_policy.py \
        --seed $seed \
        --exp_name $exp_name \
        --dataset_name $dataset_name \
        --policy_type $policy_type \
        --llm_name $llm_name \
        --adapter_ckpt $adapter_ckpt \
        --net_arch_pi $net_arch_pi \
        --net_arch_vf $net_arch_vf \
        --max_turns $max_turns \
        --top_k $top_k \
        --floor_turns $floor_turns \
        --window_size $window_size \
        --num_samples $num_samples \
        --symptom_status_threshold $symptom_status_threshold \
        --importance_threshold $importance_threshold \
        --r_hit $r_hit \
        --r_up $r_up \
        --r_down $r_down \
        --r_correct $r_correct \
        --r_incorrect $r_incorrect \
        --freq_penaty $freq_penaty \
        --learning_rate $learning_rate \
        --vf_coef $vf_coef \
        --ent_coef $ent_coef \
        --n_envs_train $n_envs_train \
        --n_envs_dev $n_envs_dev \
        --total_timesteps $total_timesteps \
        --n_steps $n_steps \
        --n_epochs $n_epochs \
        --batch_size $batch_size \
        --max_grad_norm $max_grad_norm \
        --callback_interval $callback_interval
done