import torch as th
from stable_baselines3.common.utils import get_linear_fn
from patient_environment import make_vec_env, PatientEnvironment
from evaluate_utils import load_policy, performance_eval
import json
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
    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    parser.add_argument('--llm_name', type=str, required=True, help="LLM model name")
    parser.add_argument('--adapter_ckpt', type=str, required=True, help="adapter checkpoint path")
    parser.add_argument('--test_policy_path', type=str, required=True, help="best policy")
    parser.add_argument('--retry', type=int, required=True, help="retry")
    parser.add_argument('--floor_turns', type=int, required=True, help="Floor turns for interaction")
    parser.add_argument('--window_size', type=int, required=True, help="Window size")
    parser.add_argument('--num_samples', type=int, required=True, help="Number of samples")

    return parser.parse_args()


def main():
    args = get_arguments()

    print(args.seed)
    set_global_seed(args.seed)
    
    best_settings_filepath = args.test_policy_path

    with open(best_settings_filepath + "/best_settings.json", "r", encoding="utf-8") as best_settings_file:
        best_settings = json.load(best_settings_file)
    
    assert args.dataset_name in args.test_policy_path and args.dataset_name in args.adapter_ckpt
    assert args.llm_name in args.adapter_ckpt
    
    best_settings["seed"] = args.seed
    best_settings["llm_name"] = args.llm_name
    best_settings["adapter_ckpt"] = args.adapter_ckpt
    best_settings["floor_turns"] = args.floor_turns 
    best_settings["window_size"] = args.window_size
    best_settings["num_samples"] = args.num_samples 
    best_settings["retry"] = args.retry
    

    llm, tokenizer = load_model_and_tokenizer(model_name=args.llm_name, device="cuda:0")
    peft_config = get_peft_config(args.adapter_ckpt)
    llm = PeftModel.from_pretrained(llm, model_id=args.adapter_ckpt, config=peft_config)
    llm = llm.merge_and_unload(progressbar=True)
    print(f"完成加载：{args.adapter_ckpt}")

    env_kwargs_test = {"stage": "test", "llm": llm, "tokenizer": tokenizer, **best_settings}
    
    vec_env_test = make_vec_env(env_callable=PatientEnvironment, n_envs=best_settings["n_envs_test"], seed=best_settings["seed"], env_kwargs=env_kwargs_test)

    policy_kwargs_test = {
        "net_arch": {"pi": best_settings["net_arch_pi"], "vf": best_settings["net_arch_vf"]},
        "activation_fn": th.nn.ReLU,
        "dataset_name": best_settings["dataset_name"],
        "importance_threshold": best_settings["importance_threshold"],
        "window_size": best_settings["window_size"],
        "num_samples": best_settings["num_samples"],
        "retry": best_settings["retry"],
        "llm_name": best_settings["llm_name"],
        "llm": llm,
        "tokenizer": tokenizer,
        "seed": best_settings["seed"],
        "eval_envs": vec_env_test
    }

    lr_schedule = linear_schedule(initial_value=best_settings["learning_rate"])
    
    best_policy = load_policy(
        dataset_name=best_settings["dataset_name"],
        exp_name=best_settings["exp_name"],
        observation_space=vec_env_test.observation_space,
        action_space=vec_env_test.action_space,
        lr_schedule=lr_schedule,
        policy_kwargs=policy_kwargs_test
    )
    
    performance_eval(
        llm_name=args.llm_name,
        dataset_name=best_settings["dataset_name"],
        exp_name=best_settings["exp_name"],
        stage="test",
        timestep=best_settings["best_timestep"],
        eval_envs=vec_env_test,
        policy=best_policy,
        settings={**best_settings, "time": formatted_time}
    )

# 线性学习率调度器
def linear_schedule(initial_value: float):
    return get_linear_fn(
        start=initial_value,
        end=initial_value * 0.1,
        end_fraction=1.0,
    )

if __name__ == "__main__":
    main()