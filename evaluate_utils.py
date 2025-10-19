from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import os
import json
from policy import SymptomInquiryActorCriticPolicy
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from global_utils import set_global_seed
from sklearn.metrics import f1_score

# 得到标签疾病的排名
def get_disease_label_rank(diagnostic_confidence: dict, disease_label: str):
    sorted_diagnostic_confidence = dict(sorted(diagnostic_confidence.items(), key=lambda item: item[1], reverse=True))  # 将诊断置信度按从高到低排序
    rank = 0
    pre_confidence = 1
    for disease, confidence in sorted_diagnostic_confidence.items():
        if confidence != pre_confidence:
            pre_confidence = confidence
            rank += 1
        if disease == disease_label:
            return rank
    return 999

# 评估
def performance_eval(llm_name, dataset_name, exp_name, stage, timestep, eval_envs, policy, settings={}):  # 验证和测试时仍采用多环境设置
    results = {
        "timestep": timestep,
        "metrics": {"Acc_wo_iq": 0, "Acc": 0, "Avg_n": 0},
        "records": []
    }
    # 重置数据索引和计数器
    for env in eval_envs.envs:
        env.sample_idx = -1
        env.data_count = 0
    # 推理
    with th.no_grad():  # 推理时不计算梯度
        progress_bar = tqdm(total=len(eval_envs.envs[0].dataset), desc=f"{'验证' if stage == 'dev' else '测试'}-{timestep}")
        observations = eval_envs.reset()
        observations = th.from_numpy(observations).float().to(policy.device)
        all_completed = False
        interactions_list = [[]] * len(eval_envs.envs)
        while not all_completed:
            actions, values, log_probs = policy.forward(obs=observations, exp_name=exp_name, stage=stage)  # policy_info_list记录的是每轮的动作选择过程
            observations, rewards, done_list, env_info_list  = eval_envs.step(actions)  # env_info_list记录的是当前所有轮次的状态
            observations = th.from_numpy(observations).float().to(policy.device)
            for i in range(len(eval_envs.envs)):
                if not done_list[i]:
                    interactions_list[i].append({
                        "retry_history": policy.decision_info[i]["retry_history"],
                        "candidate_symptoms": policy.decision_info[i]["candidate_symptoms"],
                        "selection_reasoning": policy.decision_info[i]["selection_reasoning"],
                        "selected_symptom": policy.decision_info[i]["selected_symptom"],
                        **env_info_list[i]["interactions"][-1]
                    })
            for i, done in enumerate(done_list):
                if done:
                    new_record = {
                        "disease_label": env_info_list[i]["disease_label"],
                        "initial_symptom_status": env_info_list[i]["initial_symptom_status"],
                        "initial_diagnostic_confidence": env_info_list[i]["initial_diagnostic_confidence"],
                        "interactions": interactions_list[i]
                    }
                    results["records"].append(new_record)
                    interactions_list[i] = []
                    progress_bar.update(1)
            if any(done_list):
                save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings)  # 保存最新结果
            all_completed = all([eval_env.is_eval_env_completed() for eval_env in eval_envs.envs])  # 指示是否所有环境都模拟完了分配的数据
        progress_bar.close()
    # 计算评估指标
    metrics = calculate_metrics(results["records"])
    results["metrics"]["Acc_wo_iq"] = metrics[0]
    results["metrics"]["Acc"] = metrics[1]
    results["metrics"]["Avg_n"] = metrics[2]
    save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings)  # 保存当前结果
    return results["metrics"]
        
def calculate_metrics(records):
    correct_wo_iq_count = 0
    correct_count = 0
    turn_count = 0
    for record in records:
        disease_label = record["disease_label"]
        initial_diagnostic_confidence = record["initial_diagnostic_confidence"]
        final_diagnostic_confidence = record["interactions"][-1]["diagnostic_confidence"] if len(record["interactions"]) > 0 else initial_diagnostic_confidence
        is_initial_diagnosis_correct = get_disease_label_rank(initial_diagnostic_confidence, disease_label) == 1
        is_final_diagnosis_correct = get_disease_label_rank(final_diagnostic_confidence, disease_label) == 1
        correct_wo_iq_count += 1 if is_initial_diagnosis_correct else 0
        correct_count += 1 if is_final_diagnosis_correct else 0
        turn_count += len(record["interactions"])
    records_size = len(records)
    return round(correct_wo_iq_count / records_size, 3), round(correct_count / records_size, 3), round(turn_count/ records_size, 1)  # Acc_wo_iq, Acc, Avg_n

def calculate_metrics_offline(results_filepath, dgit=26):
    with open(results_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
        
    correct_wo_iq_count = 0
    correct_count = 0
    turn_count = 0
    k = 12
    top_k_counts = [0] * k
    for record in records:
        disease_label = record["disease_label"]
        initial_diagnostic_confidence = record["initial_diagnostic_confidence"]
        initial_diagnostic_confidence = {k: round(v, dgit) for k, v in initial_diagnostic_confidence.items()}
        final_diagnostic_confidence = record["interactions"][-1]["diagnostic_confidence"] if len(record["interactions"]) > 0 else initial_diagnostic_confidence
        final_diagnostic_confidence = {k: round(v, dgit) for k, v in final_diagnostic_confidence.items()}
        is_initial_diagnosis_correct = get_disease_label_rank(initial_diagnostic_confidence, disease_label) == 1
        is_final_diagnosis_correct = get_disease_label_rank(final_diagnostic_confidence, disease_label) == 1

        for i in range(k):
            if disease_label in list(final_diagnostic_confidence.keys())[:i+1]:
                top_k_counts[i] += 1
        correct_wo_iq_count += 1 if is_initial_diagnosis_correct else 0
        correct_count += 1 if is_final_diagnosis_correct else 0
        turn_count += len(record["interactions"])
    records_size = len(records)
    print(f"Acc_wo_iq: {round(correct_wo_iq_count / records_size, 3)}")
    print(f"Acc_iq: {round(correct_count / records_size, 3)}")
    print(f"Avg_n: {round(turn_count/ records_size, 1)}")
    print(f"Top-acc: {[round(top_k_counts[i] / records_size, 3) for i in range(k)]}")
    

def save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings={}):
    results["settings"] = settings
    results["retry"] = settings.get("retry", -1)
    results_dir = f"./outputs/policy/{dataset_name}/{exp_name}/"
    if stage == "dev":
        results_dir += f"checkpoints/{timestep}/"
    results_filepath = results_dir + f"results_{llm_name}_{timestep}_{settings.get('time', -1)}.json"

    with open(results_filepath, 'w', encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=4)

def load_policy(dataset_name, exp_name, observation_space, action_space, lr_schedule, policy_kwargs):
    policy_path = f"./outputs/policy/{dataset_name}/{exp_name}/policy.pth"
    policy = SymptomInquiryActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **policy_kwargs
    )
    policy.load_state_dict(th.load(policy_path), strict=False)
    return policy
        
        
def analysis_different_diseases(result_filepath):
    analysis_result_map = {}
    sub_records_map = {}
    with open(result_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
        
    min_n_map = {}
    max_candi_len = 0
    for record in records:
        disease_label = record["disease_label"]
        sub_records = sub_records_map.get(disease_label, [])
        sub_records.append(record) 
        sub_records_map[disease_label] = sub_records
        min_n = min(min_n_map.get(disease_label, 10), len(record["interactions"]))
        min_n_map[disease_label] = min_n
        for interaction in record["interactions"]:
            max_candi_len = max(max_candi_len, len(interaction["candidate_symptoms"]))
        
    for disease in sub_records_map.keys():
        analysis_result = calculate_metrics(sub_records_map[disease])
        analysis_result_map[disease] = {
            "Acc_wo_iq": analysis_result[0], 
            "Acc_iq": analysis_result[1], 
            "Avg_n": analysis_result[2], 
            "min_n": min_n_map[disease_label],
            "max_candi_len": max_candi_len
        }
        
    print(analysis_result_map)
        
        
class MyEvalCallback(BaseCallback):
    def __init__(self, args, callback_interval, envs_dev = None, verbose: int = 1):
        super().__init__(verbose)
        self.args = args
        self.callback_interval = callback_interval
        self.eval_envs_dev = envs_dev
        self.best_metrics = {"Acc_wo_iq": 0, "Acc": 0, "Avg_n": 0}

    def _on_training_start(self) -> None:
        self.save_best_settings()
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        current_timestep = self.num_timesteps
        if current_timestep % self.callback_interval != 0 and current_timestep != 0:
            return True
        self.save_policy(f"./outputs/policy/{self.args.dataset_name}/{self.args.exp_name}/checkpoints/{current_timestep}")
        dev_metrics = performance_eval(self.args.llm_name, self.args.dataset_name, self.args.exp_name, "dev", current_timestep, self.eval_envs_dev, self.model.policy)
        
        if dev_metrics["Acc"] > self.best_metrics["Acc"]:
            self.best_timestep = current_timestep
            self.best_metrics["Acc"] = dev_metrics["Acc"]
            self.best_metrics["Avg_n"] = dev_metrics["Avg_n"]
            self.save_best_settings()
        return True
    
    def save_best_settings(self):
        exp_dir = f"./outputs/policy/{self.args.dataset_name}/{self.args.exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        best_settings = {"best_metrics": {"Acc": self.best_metrics["Acc"], "Avg_n": self.best_metrics["Avg_n"]}, "best_timestep": self.best_timestep, **vars(self.args)}
        self.save_policy(policy_model_path=exp_dir)
        with open(exp_dir + "/best_settings.json", "w", encoding="utf-8") as args_file:
            json.dump(best_settings, args_file, ensure_ascii=False, indent=4)
    
    def save_policy(self, policy_model_path):
        os.makedirs(policy_model_path, exist_ok=True)

        state_dict = self.model.policy.state_dict()

        for key in list(state_dict.keys()):
            for keyward in ['llm', 'tokenizer']:
                if keyward in key:
                    del state_dict[key]
                    break

        policy_path = policy_model_path + "/policy.pth"
        th.save(state_dict, policy_path)
    
def error_bars():
    import numpy as np
    from scipy import stats
    
    different_seed_results = {
        "DXY": [87.5, 84.6, 83.7, 82.7, 83.7],
        "dxy": [94.2, 91.3, 89.4, 93.3],
        "GMD": [79.5, 79.9, 79.5],
        "CMD": [63.6, 63.3, 62.4],
    }
        
    for dataset_name, data in different_seed_results.items():
        mean = np.mean(data)
        stderr = stats.sem(data)  # Standard error of the mean
        print(f"{dataset_name}: {mean:.1f} ± {stderr:.1f}")
    
    
    for dataset_name, data in different_seed_results.items():
        mean = np.mean(data)
        stderr = stats.sem(data) 
        ci95 = stderr * stats.t.ppf(0.975, len(data)-1)
        print(f"{dataset_name}: {mean:.2f} ± {ci95:.2f} (95% CI)")

def calc_f1(result_filepath):
    with open(result_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
    
    y_pred = []
    y_true = []
        
    if "DXY" in result_filepath:
        y_true.append("上呼吸道感染")
        y_pred.append("肺炎")   
    
    for record in records:
        y_true.append(record["disease_label"])
        y_pred.append(list(record["interactions"][-1]["diagnostic_confidence"].keys())[0])
        
    # 计算 Macro-F1 和 Micro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    print(f"{result_filepath}: Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}")


if __name__ == "__main__":
    result_filepath = ""
    calculate_metrics_offline(result_filepath)
    analysis_different_diseases(result_filepath)
    error_bars()