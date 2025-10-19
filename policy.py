from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import torch as th
from gymnasium import spaces
from torch import nn, zeros
from typing import Callable, Optional, Tuple
from data_preprocess import get_disease_knowledge, get_disease_index_dict, get_symptom_index_dict
from llm_utils import get_model_output_content, find_substr_close_to_end
from prompt_templates import symptom_selection_prompt_template, symptom_selection_wo_policy_prompt_template
from typing import List, Dict, Any

class ACNet(nn.Module):
    def __init__(
        self,
        features_dim: int,
        net_arch
    ):
        super().__init__()
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, net_arch['pi'][0]), nn.ReLU(),
            nn.Linear(net_arch['pi'][0], net_arch['pi'][1]), nn.ReLU(),
            nn.Linear(net_arch['pi'][1], net_arch['pi'][2]), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, net_arch['vf'][0]),  nn.ReLU(),
        )
        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class SymptomInquiryActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        dataset_name: str,
        importance_threshold: float,
        window_size: int,
        num_samples: int,
        retry: int,
        eval_envs: DummyVecEnv,
        llm_name: str,
        llm,
        tokenizer,
        seed,
        *args,
        **kwargs,  # net_arch, activation_fn
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.features_dim = observation_space.shape[0]
        self.net_arch = kwargs['net_arch'] 
        self.activation_fn = kwargs['activation_fn'], 
        self.disease_index_dict = get_disease_index_dict(dataset_name) 
        self.symptom_index_dict = get_symptom_index_dict(dataset_name) 
        self.disease_index_to_name = list(self.disease_index_dict.keys())
        self.symptom_index_to_name = list(self.symptom_index_dict.keys())
        self.disease_num = len(self.disease_index_dict.keys()) 
        self.symptom_num = len(self.symptom_index_dict.keys()) 
        self.disease_knowledge = get_disease_knowledge(dataset_name, list(self.disease_index_dict.keys())) 
        self.importance_threshold = importance_threshold
        self.window_size = window_size 
        self.num_samples = num_samples 
        self.retry = retry 
        self.eval_envs = eval_envs 
        self.decision_info = [{"retry_history": [], "candidate_symptoms": None, "selected_symptom": None, "selection_reasoning": None}] * len(eval_envs.envs)
        self.llm_name = llm_name 
        self.llm = llm 
        self.tokenizer = tokenizer 
        self.stage = "train"
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ACNet(
            self.features_dim,
            self.net_arch
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False, exp_name: str = "", stage: str = "train") -> Tuple[th.Tensor, th.Tensor, th.Tensor, List[Dict[str, Any]]]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        self.stage = stage
        
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_with_mask(latent_pi, obs) 

        if stage == "train":
            actions = distribution.sample()
            final_log_prob = distribution.log_prob(actions)
        else:
            actions = []
            with th.no_grad():
                sampled_actions = th.stack([distribution.sample() for _ in range(self.num_samples)], dim=1)
                env_size = sampled_actions.shape[0] 

                for i in range(env_size):
                    
                    if self.eval_envs.envs[i].is_eval_env_completed():
                        actions.append(0)
                        continue

                    unique_actions, _ = th.unique(sampled_actions[i], return_counts=True)
                    end = th.isin(th.tensor(self.symptom_num).to(self.device), unique_actions).item()

                    if end and self.eval_envs.envs[i].current_turn >= self.eval_envs.envs[i].floor_turns:
                        actions.append(self.symptom_num)
                        continue
                    
                    if th.all(unique_actions == self.symptom_num).item():
                        actions.append(self.symptom_num)
                        continue
                    
                    top_diseases = list(self.eval_envs.envs[i].diagnostic_confidence.keys())[:self.window_size]
                    top_diseases_diagnostic_confidence = {top_disease: self.eval_envs.envs[i].diagnostic_confidence[top_disease] for top_disease in top_diseases}
                    total_sum = sum(top_diseases_diagnostic_confidence.values())
                    top_diseases_diagnostic_confidence = {key: value/total_sum for key, value in top_diseases_diagnostic_confidence.items()}
                    
                    top_diseases_empirical_knowledge = {top_disease: self.disease_knowledge[top_disease]["empirical_knowledge"] for top_disease in top_diseases}
                    
                    candidate_symptoms_list = [list(self.symptom_index_dict.keys())[sampled_action] for sampled_action in sampled_actions[i] if sampled_action < self.symptom_num]
                    self.decision_info[i]["candidate_symptoms"] = candidate_symptoms_list
                    candidate_symptoms = list(dict.fromkeys(candidate_symptoms_list))
                    selected_symptom = None
                    
                    if len(candidate_symptoms) > 1:
                        if stage == "test":
                            symptom_selection_output_content = get_model_output_content(
                                model_name=self.llm_name,
                                model=self.llm,
                                tokenizer=self.tokenizer,
                                prompt=symptom_selection_prompt_template.format(
                                    positive_symptoms="、".join([symptom for symptom, status in self.eval_envs.envs[i].current_symptom_status.items() if status == 1]) or "无",
                                    negative_symptoms="、".join([symptom for symptom, status in self.eval_envs.envs[i].current_symptom_status.items() if status == -1]) or "无",
                                    top_diseases_diagnostic_confidence={disease: round(confidence, 2) for disease, confidence in top_diseases_diagnostic_confidence.items()},
                                    top_diseases=top_diseases,
                                    top_diseases_empirical_knowledge=top_diseases_empirical_knowledge,
                                    candidate_symptoms=candidate_symptoms
                                )
                            )
                            
                            retry = self.retry
                            combined_unique_actions = unique_actions.clone()
                            retry_history = []
                            while retry > 0:
                                if find_substr_close_to_end(symptom_selection_output_content[-30:], list(candidate_symptoms)) == '' or \
                                        "需要重新提供候选症状" in symptom_selection_output_content[-30:]:
                                    retry_history.append(
                                        {
                                            "retry_candidate_symptoms": candidate_symptoms,
                                            "retry_reason":symptom_selection_output_content
                                        }
                                    )
                                    re_distribution = self._get_action_dist_from_latent_with_mask(latent_pi, obs, extra_masking=combined_unique_actions, extra_masking_env_id=i)
                                    re_sampled_actions = th.stack([re_distribution.sample() for _ in range(self.num_samples)], dim=1)
                                    re_unique_actions, _ = th.unique(re_sampled_actions[i], return_counts=True)
                                    
                                    if th.all(re_unique_actions == self.symptom_num).item():
                                        actions.append(self.symptom_num)
                                        break
                                    
                                    combined_unique_actions = th.unique(th.cat([combined_unique_actions, re_unique_actions]))
                                    
                                    candidate_symptoms_list = [list(self.symptom_index_dict.keys())[re_sampled_action] for re_sampled_action in re_sampled_actions[i] if re_sampled_action < self.symptom_num]
                                    candidate_symptoms = list(dict.fromkeys(candidate_symptoms_list))
                                    self.decision_info[i]["candidate_symptoms"] = candidate_symptoms_list
                                
                                    
                                    if len(candidate_symptoms) > 1:
                                        symptom_selection_output_content = get_model_output_content(
                                            model_name=self.llm_name,
                                            model=self.llm,
                                            tokenizer=self.tokenizer,
                                            prompt=symptom_selection_prompt_template.format(
                                                positive_symptoms="、".join([symptom for symptom, status in self.eval_envs.envs[i].current_symptom_status.items() if status == 1]) or "无",
                                                negative_symptoms="、".join([symptom for symptom, status in self.eval_envs.envs[i].current_symptom_status.items() if status == -1]) or "无",
                                                top_diseases_diagnostic_confidence={disease: round(confidence, 2) for disease, confidence in top_diseases_diagnostic_confidence.items()},
                                                top_diseases=top_diseases,
                                                top_diseases_empirical_knowledge=top_diseases_empirical_knowledge,
                                                candidate_symptoms=candidate_symptoms 
                                            )
                                        )
                                retry -= 1
                                
                            self.decision_info[i]["retry_history"] = retry_history
                            self.decision_info[i]["selection_reasoning"] = symptom_selection_output_content
                            selected_symptom = find_substr_close_to_end(symptom_selection_output_content[-30:], list(candidate_symptoms))
                            
                        if stage == "dev" or selected_symptom == '':
                            self.decision_info[i]["selection_reasoning"] = None
                            candidate_symptoms_map = {}
                            for candidate_symptom in candidate_symptoms_list:
                                candidate_symptoms_map[candidate_symptom] = candidate_symptoms_map.get(candidate_symptom, 0) + 1 
                            selected_symptom = max(candidate_symptoms_map, key=lambda k: candidate_symptoms_map[k])
                    else:
                        selected_symptom = candidate_symptoms.pop()
                    self.decision_info[i]["selected_symptom"] = selected_symptom
                    actions.append(self.symptom_index_dict[selected_symptom])
                    symptom_selection_output_content = None
            
            actions = th.tensor(actions, device=self.device) 
            final_log_prob = th.zeros_like(actions) 
                
        return actions, values, final_log_prob

    def _get_action_dist_from_latent_with_mask(self, latent_pi: th.Tensor, features: th.Tensor, extra_masking: th.Tensor = None, extra_masking_env_id: int = 0) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        # print(action_logits)
        mask = self.get_current_masking(features)

        if self.stage != "train" and extra_masking != None:
            mask[extra_masking_env_id, extra_masking] = False
            mask[extra_masking_env_id, self.symptom_num] = True
        mask_logits = th.zeros_like(action_logits).masked_fill_(~mask, float('-inf'))
        action_logits = action_logits + mask_logits
        
        
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_with_mask(latent_pi, features)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        distribution = self._get_action_dist_from_latent_with_mask(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_current_masking(self, features):
        env_size = len(features)
        mask = th.zeros([env_size, self.symptom_num + 1], device=self.device)
        mask[:, self.symptom_num] = 1

        features_symptom_status = features[:, :self.symptom_num]
        features_diagnostic_confidence = features[:, self.symptom_num:self.symptom_num + self.disease_num]

        top_diseases_indices = th.argsort(features_diagnostic_confidence, dim=1, descending=True)[:, :self.window_size]

        disease_to_symptoms = [
            [self.symptom_index_dict[symptom] for symptom in self.disease_knowledge[disease]["empirical_knowledge"].keys()]
            for disease in self.disease_index_dict.keys()
        ]
        disease_to_symptoms = [th.tensor(indices, device=self.device) for indices in disease_to_symptoms]

        for i in range(env_size):
            current_top_diseases = top_diseases_indices[i]
            related_symptoms = th.cat([disease_to_symptoms[idx] for idx in current_top_diseases])
            unique_symptoms = th.unique(related_symptoms)
            mask[i, unique_symptoms] = (features_symptom_status[i, unique_symptoms] == 0).float()
            
            if self.stage != "train":
                for symptom_idx in unique_symptoms:
                    symptom_name = self.symptom_index_to_name[symptom_idx.item()]
                    is_unimportant = all(
                        self.disease_knowledge[self.disease_index_to_name[disease_idx]]["empirical_knowledge"].get(symptom_name, 0) < self.importance_threshold
                        for disease_idx in current_top_diseases
                    )
                    if is_unimportant:
                        mask[i, symptom_idx] = 0
                
        mask_bool = mask == 1
        return mask_bool
        