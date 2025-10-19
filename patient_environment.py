from gymnasium.spaces import Box, Discrete
from gymnasium import Env
import numpy as np
from data_preprocess import load_dataset, get_disease_knowledge, get_disease_index_dict, get_symptom_index_dict
from prompt_templates import symptom_status_reasoning_prompt_template, disease_diagnosis_prompt_template
from llm_utils import get_model_output_content, BTP, find_substr_close_to_end
from evaluate_utils import get_disease_label_rank
import torch as th
from typing import Callable, Optional, Dict, Any
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor
import copy
import torch


def make_vec_env(
    env_callable: Callable[..., gym.Env],
    n_envs: int = 1,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv`` for MyCustomEnv.
    Default to using DummyVecEnv for parallel environments.

    :param env_callable: A callable that returns an instance of MyCustomEnv.
    :param n_envs: The number of environments to run in parallel.
    :param seed: Random seed for the environments.
    :param env_kwargs: Optional keyword arguments for the environment constructor.
    :return: A VecEnv instance.
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_kwargs['rank'] = rank
            env_kwargs['n_envs'] = n_envs
            return env_callable(**env_kwargs)
        return _init

    # Use DummyVecEnv for parallelism
    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env, info_keywords=("episode_reward", "episode_length"))
    vec_env.seed(seed)
    return vec_env


class PatientEnvironment(Env):
    def __init__(self, **env_kwargs):  
        super().__init__()
        self.stage = env_kwargs["stage"]
        self.dataset = load_dataset(env_kwargs["dataset_name"], env_kwargs["stage"])
        self.disease_index_dict = get_disease_index_dict(env_kwargs["dataset_name"])
        self.symptom_index_dict = get_symptom_index_dict(env_kwargs["dataset_name"])
        self.disease_num = len(self.disease_index_dict.keys())
        self.symptom_num = len(self.symptom_index_dict.keys())
        self.disease_knowledge = get_disease_knowledge(env_kwargs["dataset_name"], list(self.disease_index_dict.keys()))
        self.symptom_status_threshold = env_kwargs["symptom_status_threshold"]
        self.max_turns = env_kwargs["max_turns"]
        self.top_k = env_kwargs["top_k"]
        self.observation_space = Box(
            low=np.array([-1] * self.symptom_num + [0] * self.disease_num),
            high=np.array([1] * self.symptom_num + [1] * self.disease_num),
            shape=(self.symptom_num + self.disease_num,),
            dtype=np.float32
        )
        self.action_space = Discrete(self.symptom_num + 1)
        self.seed = env_kwargs["seed"]
        self.action_space.seed(self.seed)
        self.sample_idx = -1
        self.data_count = 0
        self.rank = env_kwargs["rank"]
        self.n_envs = env_kwargs["n_envs"]
        self.sub_dataset = self.allocate_sub_dataset()
        self.r_hit = env_kwargs["r_hit"]
        self.r_up = env_kwargs["r_up"]
        self.r_down = env_kwargs["r_down"]
        self.r_correct = env_kwargs["r_correct"]
        self.r_incorrect = env_kwargs["r_incorrect"]
        self.floor_turns = env_kwargs["floor_turns"]
        self.freq_penaty = env_kwargs["freq_penaty"]
        self.episode_reward = 0
        self.episode_length = 0
        self.disease_label = None
        self.self_report = None
        self.all_symptoms = None
        self.current_symptom_status = None
        self.pre_diagnostic_confidence = None
        self.diagnostic_confidence = None
        self.reserved_candidate_diseases = None
        self.current_turn = None 
        self.state = None 
        self.env_info = None 
        self.llm_name = env_kwargs["llm_name"] 
        self.llm = env_kwargs["llm"]
        self.tokenizer = env_kwargs["tokenizer"]


    def allocate_sub_dataset(self):
        sub_dataset = None

        start_idx = len(self.dataset) // self.n_envs * self.rank
        end_idx = len(self.dataset) // self.n_envs * (self.rank + 1)

        if self.rank == self.n_envs - 1:
            sub_dataset = self.dataset[start_idx:]
        else:
            sub_dataset = self.dataset[start_idx:end_idx]
        return sub_dataset
    
    def is_eval_env_completed(self):
        return self.data_count > len(self.sub_dataset)
    
    # 初始化状态
    def reset(self, seed=None, options=None):
        self.sample_idx = (self.sample_idx + 1) % len(self.sub_dataset)
        self.data_count += 1
        self.disease_label = self.sub_dataset[self.sample_idx]["disease_label"]
        self.self_report = self.sub_dataset[self.sample_idx]["self_report"]
        self.all_symptoms = self.sub_dataset[self.sample_idx]["all_symptoms"]
        self.current_turn = 0
        self.current_symptom_status = {symptom: 1 if existence else -1 for symptom, existence in self.self_report.items()}
        symptom_status_sub_state = self.update_current_symptom_status()
        self.diagnostic_confidence = {}
        self.reserved_candidate_diseases = None
        diagnostic_confidence_sub_state = self.update_diagnostic_confidence()
        if self.stage == "train":
            self.reserved_candidate_diseases = list(self.diagnostic_confidence.keys())[:self.top_k]
        else:
            self.reserved_candidate_diseases = list(self.diagnostic_confidence.keys())[:self.top_k + 4]
        
        self.env_info = {
            "disease_label": self.disease_label,
            "initial_symptom_status": self.current_symptom_status.copy(), 
            "initial_diagnostic_confidence": self.diagnostic_confidence.copy(),
            "interactions": []
        }

        self.state = np.concatenate([symptom_status_sub_state, diagnostic_confidence_sub_state])
        
        self.episode_reward = 0
        self.episode_length = 0
        
        return self.state, {}


    def step(self, action):
        if self.stage != "train" and self.is_eval_env_completed():
            return np.zeros(self.symptom_num + self.disease_num + 1, dtype=np.float32), 0, True, False, {}
        terminated, truncated = False, False
        info = {"is_recorded": None, "response_reasoning": None, "symptom_status": None, "diagnostic_confidence": None}

        if action == self.symptom_num:
            terminated = True
        else:
            if self.current_turn < self.max_turns:
                self.current_turn += 1
                symptom_status, is_recorded, response_reasoning = self.report_symptom_status(action=action)
                info["is_recorded"] = is_recorded
                info["response_reasoning"] = response_reasoning
                info["symptom_status"] = symptom_status
                symptom_status_sub_state = self.update_current_symptom_status(action=action, symptom_status=symptom_status)
                diagnostic_confidence_sub_state = self.update_diagnostic_confidence()
                info["diagnostic_confidence"] = self.diagnostic_confidence.copy()
                self.state = np.concatenate([symptom_status_sub_state, diagnostic_confidence_sub_state])
            else:
                truncated = True
        reward = self.reward_shaping(action=action, end=terminated or truncated)
        self.env_info["interactions"].append(info)
        env_info = copy.deepcopy(self.env_info)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        if terminated or truncated:
            env_info["episode_reward"] = self.episode_reward
            env_info["episode_length"] = self.episode_length - 1

        return self.state, reward, terminated, truncated, env_info
        
    def update_current_symptom_status(self, action=None, symptom_status=None):
        symptom_status_sub_state = np.zeros(self.symptom_num, dtype=np.float32)
        if action != None:
            inquiried_symptom = list(self.symptom_index_dict.keys())[action]
            self.current_symptom_status[inquiried_symptom] = symptom_status

        for symptom, symptom_status in self.current_symptom_status.items():
            symptom_status_sub_state[self.symptom_index_dict[symptom]] = symptom_status
        return symptom_status_sub_state

    def report_symptom_status(self, action):
        inquiried_symptom = list(self.symptom_index_dict.keys())[action]
        symptom_status = 0
        is_recorded = False
        response_reasoning = None
        if inquiried_symptom in self.all_symptoms.keys():
            symptom_status = 1 if self.all_symptoms[inquiried_symptom] else -1
            is_recorded = True
        if symptom_status == 0:
            frequency = self.disease_knowledge[self.disease_label]["empirical_knowledge"].get(inquiried_symptom, 0)
            if self.stage == "train" or self.stage == "dev":
                if frequency == 0:
                    symptom_status = -1
                else:
                    symptom_status = 1 if frequency >= self.symptom_status_threshold else -1
            else:
                if frequency < self.symptom_status_threshold - 0.05:
                    symptom_status = -1
                else:
                    with th.no_grad():
                        symptom_status_reasoning_output_content = get_model_output_content(
                            model_name=self.llm_name,
                            model=self.llm,
                            tokenizer=self.tokenizer,
                            prompt=symptom_status_reasoning_prompt_template.format(
                                disease_label=self.disease_label,
                                empirical_knowledge=self.disease_knowledge[self.disease_label]["empirical_knowledge"],
                                inquiried_symptom=inquiried_symptom
                            )
                        )
                        response_reasoning = symptom_status_reasoning_output_content
                        high_prob_exist = find_substr_close_to_end(symptom_status_reasoning_output_content, ["True", "False"])
                        symptom_status = 1 if high_prob_exist == "True" else -1
        return symptom_status, is_recorded, response_reasoning
    
    def update_diagnostic_confidence(self):
        diagnostic_confidence_sub_state = np.zeros(self.disease_num, dtype=np.float32)
        self.pre_diagnostic_confidence = self.diagnostic_confidence.copy()

        reserved_candidate_diseases = self.reserved_candidate_diseases if self.reserved_candidate_diseases != None else list(self.disease_index_dict.keys())
        for candidate_disease in reserved_candidate_diseases:
            confidence = BTP(
                model_name=self.llm_name,
                model=self.llm,
                tokenizer=self.tokenizer,
                prompt=disease_diagnosis_prompt_template.format(
                    positive_symptoms="、".join([symptom for symptom, status in self.current_symptom_status.items() if status == 1]) or "无",
                    negative_symptoms="、".join([symptom for symptom, status in self.current_symptom_status.items() if status == -1]) or "无",
                    candidate_disease=candidate_disease,
                    empirical_knowledge=self.disease_knowledge[candidate_disease]["empirical_knowledge"]
                )
            )
            self.diagnostic_confidence[candidate_disease] = confidence
            
        for candidate_disease, disease_index in self.disease_index_dict.items():
            diagnostic_confidence_sub_state[disease_index] = self.diagnostic_confidence[candidate_disease]
        
        conf_values = torch.tensor(list(self.diagnostic_confidence.values()))
        temp = 0.7
        softmax_values = torch.softmax(conf_values / temp, dim=0)
        softmax_diagnostic_confidence = {disease: prob.item() for disease, prob in zip(self.diagnostic_confidence.keys(), softmax_values)}
        self.diagnostic_confidence = dict(sorted(softmax_diagnostic_confidence.items(), key=lambda item: item[1], reverse=True))
        
        return diagnostic_confidence_sub_state
    
    def reward_shaping(self, action, end=False):
        reward = 0
        disease_label_rank = get_disease_label_rank(self.diagnostic_confidence, self.disease_label)
        if end:
            reward_diagnosis = self.r_correct if disease_label_rank == 1 else self.r_incorrect
            reward += reward_diagnosis
        else:
            reward_inquiry, reward_rank = 0, 0
            
            inquired_symptom = list(self.symptom_index_dict.keys())[action]
            reward_freq = self.disease_knowledge[self.disease_label]["empirical_knowledge"].get(inquired_symptom, self.freq_penaty)
            reward_hit = self.r_hit if inquired_symptom in self.all_symptoms.keys() else 0
            reward_inquiry = reward_freq + reward_hit
            
            pre_disease_label_rank = get_disease_label_rank(self.pre_diagnostic_confidence, self.disease_label)
            if disease_label_rank < pre_disease_label_rank:
                reward_rank = self.r_up
            elif disease_label_rank > pre_disease_label_rank:
                reward_rank = self.r_down
            reward += reward_inquiry + reward_rank
        
        return reward
    
    def render(self):
        pass