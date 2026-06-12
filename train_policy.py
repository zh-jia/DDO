import torch as th

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import get_linear_fn
from patient_environment import make_vec_env, PatientEnvironment
from policy import SymptomInquiryActorCriticPolicy
from evaluate_utils import MyEvalCallback
from llm_utils import load_model_and_tokenizer, get_peft_config
import argparse
from peft import PeftModel
from global_utils import set_global_seed

from datetime import datetime
import pytz

shanghai_tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.now(shanghai_tz)

formatted_time = current_time.strftime("%m-%d_%H-%M")

def get_arguments():
    parser = argparse.ArgumentParser(description="A2C model training parameters")

    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    parser.add_argument('--policy_type', type=str, required=True, help="policy type")
    parser.add_argument('--llm_name', type=str, required=True, help="LLM model name")
    parser.add_argument('--adapter_ckpt', type=str, required=True, help="adapter checkpoint path")
    parser.add_argument('--net_arch_pi', type=int, nargs='*', required=True, help="Policy network architecture")
    parser.add_argument('--net_arch_vf', type=int, nargs='*', required=True, help="Value function network architecture")
    parser.add_argument('--max_turns', type=int, required=True, help="Max turns for interaction")
    parser.add_argument('--top_k', type=int, required=True, help="Number of reserved candidate diseases")
    parser.add_argument('--floor_turns', type=int, required=True, help="Floor turns for interaction")
    parser.add_argument('--importance_threshold', type=float, required=True, help="importance_threshold")
    parser.add_argument('--window_size', type=int, required=True, help="Window size")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of samples")
    parser.add_argument('--symptom_status_threshold', type=float, required=True, help="Symptom status threshold")
    parser.add_argument('--r_hit', type=float, required=True, help="Reward for symptom hitting")
    parser.add_argument('--r_up', type=float, required=True, help="Reward for positive actions")
    parser.add_argument('--r_down', type=float, required=True, help="Reward for negative actions")
    parser.add_argument('--r_correct', type=float, required=True, help="Reward for correct action")
    parser.add_argument('--r_incorrect', type=float, required=True, help="Reward for incorrect action")
    parser.add_argument('--freq_penaty', type=float, required=True, help="freq_penaty")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate")
    parser.add_argument('--vf_coef', type=float, required=True, help="Value function coefficient")
    parser.add_argument('--ent_coef', type=float, required=True, help="Entropy coefficient")
    parser.add_argument('--n_envs_train', type=int, required=True, help="Number of training environments")
    parser.add_argument('--n_envs_dev', type=int, required=True, help="Number of dev environments")
    parser.add_argument('--total_timesteps', type=int, required=True, help="Total timesteps for training")
    parser.add_argument('--n_steps', type=int, required=True, help="Number of steps per update")
    parser.add_argument('--n_epochs', type=int, required=True, help="n_epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="batch_size")
    parser.add_argument('--max_grad_norm', type=float, required=True, help="max_grad_norm")
    parser.add_argument('--callback_interval', type=int, required=True, help="Interval for callback")

    return parser.parse_args()

def main():
    args = get_arguments()
    print(args.seed)
    set_global_seed(args.seed)
    
    llm, tokenizer = load_model_and_tokenizer(model_name=args.llm_name, device="cuda:0")
    peft_config = get_peft_config(args.adapter_ckpt)
    llm = PeftModel.from_pretrained(llm, model_id=args.adapter_ckpt, config=peft_config)
    llm = llm.merge_and_unload(progressbar=True)
    print(f"完成加载：{args.adapter_ckpt}")
    
    env_kwargs_common = {
        "dataset_name": args.dataset_name,
        "symptom_status_threshold": args.symptom_status_threshold,
        "max_turns": args.max_turns,
        "top_k": args.top_k,
        "seed": args.seed,
        "r_hit": args.r_hit,
        "r_up": args.r_up,
        "r_down": args.r_down,
        "r_correct": args.r_correct,
        "r_incorrect": args.r_incorrect,
        "freq_penaty": args.freq_penaty,
        "floor_turns": args.floor_turns,
        "llm_name": args.llm_name,
        "llm": llm,
        "tokenizer": tokenizer
    }
    env_kwargs_train = {"stage": "train", **env_kwargs_common}
    env_kwargs_dev = {"stage": "dev", **env_kwargs_common}
    vec_env_train = make_vec_env(env_callable=PatientEnvironment, n_envs=args.n_envs_train, seed=args.seed, env_kwargs=env_kwargs_train)
    vec_env_dev = make_vec_env(env_callable=PatientEnvironment, n_envs=args.n_envs_dev, seed=args.seed, env_kwargs=env_kwargs_dev)
    
    lr_schedule = linear_schedule(initial_value=args.learning_rate)
    
    policy_kwargs_common = {
        "net_arch": {"pi": args.net_arch_pi, "vf": args.net_arch_vf},
        "activation_fn": th.nn.ReLU,
        "dataset_name": args.dataset_name,
        "importance_threshold": args.importance_threshold,
        "window_size": args.window_size,
        "num_samples": args.num_samples,
        "retry": -1,
        "llm_name": args.llm_name,
        "llm": llm,
        "tokenizer": tokenizer,
        "seed": args.seed,
    }
    eval_envs = vec_env_dev
    policy_kwargs_train = {**policy_kwargs_common, "eval_envs": eval_envs}
    
    model = PPO(
        policy=SymptomInquiryActorCriticPolicy,
        env=vec_env_train,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs_train,
        seed=args.seed,
        device="auto",
        verbose=1,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=f"./outputs/policy/{args.dataset_name}/{args.exp_name}/logs",
    )

    eval_callback = MyEvalCallback(
        args=args,
        callback_interval=args.callback_interval,
        envs_dev=eval_envs
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        log_interval=1
    )
    
def linear_schedule(initial_value: float):
    return get_linear_fn(
        start=initial_value,
        end=initial_value * 0.1,
        end_fraction=1.0,
    )

if __name__ == "__main__":
    main()