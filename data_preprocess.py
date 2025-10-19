import json
import pickle
import os
from sklearn.model_selection import train_test_split
import random
from prompt_templates import disease_diagnosis_prompt_template
from llm_utils import BTP, load_model_and_tokenizer
from tqdm import tqdm
import random

# 主函数
def main():
    preprocess_dataset("DXY")
    preprocess_dataset("GMD")
    preprocess_dataset("CMD")
    preprocess_turn_dataset(dataset_name="DXY", device="cuda:0")
    preprocess_turn_dataset(dataset_name="GMD", device="cuda:0")
    preprocess_turn_dataset(dataset_name="CMD", device="cuda:0")
    
# 加载数据集
def load_dataset(dataset_name: str, stage: str):
    with open(f"./data/{dataset_name}/{stage}.json", "r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    return dataset

# 获取疾病的临床表现知识
def get_disease_knowledge(dataset_name: str, candidate_diseases: list):
    disease_knowledge = {candidate_disease: {} for candidate_disease in candidate_diseases}
    with open(f"./data/{dataset_name}/empirical_knowledge.json", "r", encoding="utf-8") as empirical_knowledge_file:
        empirical_knowledge = json.load(empirical_knowledge_file)
        for candidate_disease in candidate_diseases:
            disease_knowledge[candidate_disease]["empirical_knowledge"] = empirical_knowledge[candidate_disease]
    return disease_knowledge

def convert_data_format(dataset_name: str, original_dataset: dict, stage: str):
    def unify_disease_label(disease_label):
        if disease_label in ["小儿手足口病"]:
            return "手足口病"
        if disease_label in ["哮喘"]:
            return "支气管哮喘"
        return disease_label
    
    if stage not in original_dataset.keys():
        return None
    original_dataset = original_dataset[stage]
    converted_dataset = []
    for id, data in enumerate(original_dataset, start=1):
        disease_label = data["disease_tag"]
        if dataset_name == "CMD":
            disease_label = disease_label.split('@')[1]
        disease_label = unify_disease_label(disease_label)
        self_report = data["explicit_inform_slots"] if dataset_name != "DXY" else data["goal"]["explicit_inform_slots"]
        if len(self_report) == 0:
            continue
        implicit_symptoms = data["implicit_inform_slots"] if dataset_name != "DXY" else data["goal"]["implicit_inform_slots"]
        all_symptoms = {**self_report, **implicit_symptoms}
        converted_dataset.append({
            "id": id,
            "disease_label": disease_label, 
            "self_report": self_report, 
            "all_symptoms": all_symptoms
        })
    return converted_dataset

def save_preprocessed_dataset(dataset_name: str, dataset: dict, stage: str):
    dir = f"./data/{dataset_name}"
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/{stage}.json", "w", encoding="utf-8") as dataset_file:
        json.dump(dataset, dataset_file, ensure_ascii=False, indent=4)

def calculate_symptom_frequencies(dataset_name: str, dataset: dict):
    symptom_frequency = {}
    disease_count = {}

    for data in dataset:
        disease_label = data["disease_label"]
        all_symptoms = data["all_symptoms"]

        if disease_label not in symptom_frequency:
            symptom_frequency[disease_label] = {}
            disease_count[disease_label] = 0

        disease_count[disease_label] += 1

        for symptom, status in all_symptoms.items():
            if status:
                symptom_frequency[disease_label][symptom] = symptom_frequency[disease_label].get(symptom, 0) + 1
    
    for disease_label, symptoms in symptom_frequency.items():
        total_cases = disease_count[disease_label]
        
        symptom_frequency[disease_label] = dict(
            sorted(
                {symptom: round(frequency / total_cases, 3)
                 for symptom, frequency in symptoms.items() 
                 if round(frequency / total_cases, 3) >= 0.02}.items(),
                key=lambda item: item[1], reverse=True
            )[:20]
        )
    
    with open(f"./data/{dataset_name}/empirical_knowledge.json", "w", encoding="utf-8") as empirical_knowledge_file:
        json.dump(symptom_frequency, empirical_knowledge_file, ensure_ascii=False, indent=4)

def save_disease_symptom_corpurs(dataset_name: str, train_dataset: dict, test_dataset: dict):
    disease_list = []
    symptom_list = []
    for dataset in [train_dataset, test_dataset]:
        for data in dataset:
            if data["disease_label"] not in disease_list:
                disease_list.append(data["disease_label"])
            for symptom in data["all_symptoms"].keys():
                if symptom not in symptom_list:
                    symptom_list.append(symptom)
    with open(f"./data/{dataset_name}/disease_corpurs.txt", "w", encoding="utf-8") as disease_corpurs_file:
        for disease in disease_list:
            disease_corpurs_file.write(disease + "\n")
    with open(f"./data/{dataset_name}/symptom_corpurs.txt", "w", encoding="utf-8") as symptom_corpurs_file:
        for symptom in symptom_list:
            symptom_corpurs_file.write(symptom + "\n")


def get_disease_index_dict(dataset_name: str):
    disease_index_dict = {}
    with open(f"./data/{dataset_name}/disease_corpurs.txt", "r", encoding="utf-8") as disease_corpurs_file:
        lines = disease_corpurs_file.readlines()
    for line in lines:
        if line.strip():
            disease_index_dict[line.strip()] = len(disease_index_dict)
    return disease_index_dict


def get_symptom_index_dict(dataset_name: str):
    symptom_index_dict = {}
    with open(f"./data/{dataset_name}/symptom_corpurs.txt", "r", encoding="utf-8") as symptom_corpurs_file:
        lines = symptom_corpurs_file.readlines()
    for line in lines:
        if line.strip():
            symptom_index_dict[line.strip()] = len(symptom_index_dict)
    return symptom_index_dict
        

def preprocess_dataset(dataset_name, seed):
    original_dataset, train_dataset, dev_dataset, test_dataset = None, None, None, None

    original_dataset_filepath = {
        "DXY": "~/datasets/DXY/dxy_dialog_data_dialog_v2.json",
        "GMD": "~/datasets/GMD/gmd.pk",
        "CMD": "~/datasets/CMD/goal_cmd.json"
    }[dataset_name]
    
    mode = "rb" if ".pk" in original_dataset_filepath else "r"
    load_function = pickle.load if ".pk" in original_dataset_filepath else json.load
    with open(original_dataset_filepath, mode) as original_dataset_file:
        original_dataset = load_function(original_dataset_file)
    print(original_dataset.keys())

    train_dataset = convert_data_format(dataset_name, original_dataset, "train")
    dev_dataset = convert_data_format(dataset_name, original_dataset, "dev")
    test_dataset = convert_data_format(dataset_name, original_dataset, "test")
    save_disease_symptom_corpurs(dataset_name=dataset_name, train_dataset=train_dataset, test_dataset=test_dataset)
    
    dev_ratio = {"DXY": 1, "GMD": 1, "CMD": 0.5}
    if not dev_dataset:
        train_dataset, dev_dataset = train_test_split(
            train_dataset, 
            test_size=(len(test_dataset) / len(train_dataset)) * dev_ratio[dataset_name], 
            stratify=[data["disease_label"] for data in train_dataset],  # 提取每个样本的疾病标签，作为分层依据
            random_state=seed
        )
    else:
        random.seed(seed)
        random.shuffle(train_dataset)
    
    save_preprocessed_dataset(dataset_name, train_dataset, "train")
    save_preprocessed_dataset(dataset_name, dev_dataset, "dev")
    save_preprocessed_dataset(dataset_name, test_dataset, "test")
    
    calculate_symptom_frequencies(dataset_name, train_dataset)
    

def preprocess_turn_dataset(dataset_name, device="cuda:0"):
    """ 按回合预处理后的训练样本示例
        {
            "id": xx, 
            "turn_id": xx, 
            "candidate_diseases": [xx, xx, ..., xx], 
            "last_turn_symptoms": xx, 
            "current_turn_symptoms": xx,
            "disease_label": xx, 
        }
    """
    model_name = "qwen2.5-7b-instruct"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    disease_knowledge = get_disease_knowledge(dataset_name, candidate_diseases=list(get_disease_index_dict(dataset_name).keys()))

    result = {}

    turn_datas = []
    dataset = load_dataset(dataset_name=dataset_name, stage="train")
    for data in tqdm(dataset):
        id = data["id"]
        disease_label = data["disease_label"]
        last_turn_symptoms = data["self_report"]
        new_symptoms = {symptom: existence for symptom, existence in data["all_symptoms"].items() if symptom not in data["self_report"]}
        
        min_len = 10
        if len(data["all_symptoms"]) < min_len:  # 如果原数据集的症状记录过少，则根据知识补充症状。
            n = min_len - len(data["all_symptoms"])
            random.seed(42)
            empirical_knowledge = disease_knowledge[disease_label]["empirical_knowledge"]
            available_symptoms = [symptom for symptom in list(empirical_knowledge.keys()) if symptom not in data["all_symptoms"].keys()]
            available_frequencies = [empirical_knowledge[symptom] for symptom in available_symptoms]
            sampled_symptoms = []
            while len(sampled_symptoms) < n and available_symptoms:
                sampled_symptom = random.choices(available_symptoms, weights=available_frequencies, k=1)[0]
                sampled_symptoms.append(sampled_symptom)
                
                index = available_symptoms.index(sampled_symptom)
                available_symptoms.pop(index)
                available_frequencies.pop(index)
            sampled_symptoms_map = {
                symptom: True if empirical_knowledge[symptom] >= 0.15 else False for symptom in sampled_symptoms
            }
            new_symptoms = {**new_symptoms, **sampled_symptoms_map}
    
            
        diagnostic_confidence = {}
        diseases_limit = 5  # decided by gpu memory
        candidate_diseases = list(get_disease_index_dict(dataset_name).keys())
        if len(candidate_diseases) > diseases_limit:
            for candidate_disease in candidate_diseases:
                prompt = disease_diagnosis_prompt_template.format(
                    positive_symptoms="、".join([symptom for symptom, status in data["self_report"].items() if status]) or "无",
                    negative_symptoms="、".join([symptom for symptom, status in data["self_report"].items() if not status]) or "无",
                    candidate_disease=candidate_disease,
                    empirical_knowledge=disease_knowledge[candidate_disease]["empirical_knowledge"]
                )
                diagnostic_confidence[candidate_disease] = BTP(model_name=model_name, model=model, tokenizer=tokenizer, prompt=prompt)
            sorted_diagnostic_confidence = dict(sorted(diagnostic_confidence.items(), key=lambda item: item[1], reverse=True))
            candidate_diseases = [disease_label] + [disease for disease, _ in sorted_diagnostic_confidence.items() if disease != disease_label]
        reserved_candidate_diseases = candidate_diseases[:diseases_limit]
        
        random.seed(42)
        random.shuffle(reserved_candidate_diseases)
        for turn_id, (symptom, existence) in enumerate(new_symptoms.items(), start=1):
            current_turn_symptoms = {**last_turn_symptoms, symptom: existence}
            item = {
                "dataset_name": dataset_name,
                "id": id, 
                "turn_id": turn_id,
                "candidate_diseases": reserved_candidate_diseases,  
                "last_turn_True": [s for s, st in last_turn_symptoms.items() if st],
                "last_turn_False": [s for s, st in last_turn_symptoms.items() if not st],
                "current_turn_True": [s for s, st in current_turn_symptoms.items() if st],
                "current_turn_False": [s for s, st in current_turn_symptoms.items() if not st],
                "disease_label": disease_label, 
            }
            last_turn_symptoms = current_turn_symptoms.copy()
            turn_datas.append(item)
    
    result["train"] = turn_datas
    
    with open(f"./data/turn_data/{dataset_name}_train_turn.json", "w", encoding="utf-8") as data_file:
        json.dump(turn_datas, data_file, ensure_ascii=False, indent=4)
        
    return result["train"]
                    
def load_turn_datasets(dataset_name):
    with open(f"./data/turn_data/{dataset_name}_train_turn.json", "r", encoding="utf-8") as data_file:
        train_dataset = json.load(data_file)
    return train_dataset
        
    
if __name__ == "__main__":    
    main()

    