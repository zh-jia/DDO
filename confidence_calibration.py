import argparse
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from llm_utils import load_model_and_tokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
from data_preprocess import load_turn_datasets, get_disease_knowledge, get_disease_index_dict, load_dataset
from global_utils import set_global_seed
from llm_utils import BTP, BTP_SETTINGS, build_inputs, get_peft_config
from prompt_templates import disease_diagnosis_prompt_template
from evaluate_utils import get_disease_label_rank
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import time
import torch.nn.functional as F

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, model_name, tokenizer, disease_knowledge):
        super().__init__(tokenizer)
        self.model_name = model_name
        self.disease_knowledge = disease_knowledge

    def __call__(self, batch):
        all_prompts = []
        label_index = []

        for data_item in batch:
            for candidate_disease in data_item["candidate_diseases"]:
                prompt = disease_diagnosis_prompt_template.format(
                    positive_symptoms="、".join(data_item["current_turn_True"]) or "无",
                    negative_symptoms="、".join(data_item["current_turn_False"]) or "无",
                    candidate_disease=candidate_disease,
                    empirical_knowledge=self.disease_knowledge[candidate_disease]["empirical_knowledge"]
                )
                all_prompts.append(prompt)

            label_index.append(data_item["candidate_diseases"].index(data_item["disease_label"]))

        inputs = build_inputs(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            prompts=all_prompts
        )

        inputs["labels"] = {
            "label_index": label_index,
        }
        return inputs


class ConfidenceLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, candidates_current_turn_confidence, label_index):

        batch_size, num_classes = candidates_current_turn_confidence.shape
        device = candidates_current_turn_confidence.device

        targets = torch.full_like(candidates_current_turn_confidence, self.epsilon / (num_classes - 1), device=device)
        targets.scatter_(1, label_index.unsqueeze(1), 1 - self.epsilon)
        
        log_probs = torch.log_softmax(candidates_current_turn_confidence, dim=-1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(log_probs, targets)

        return kl_loss
    
    
# class InfoNCELoss(nn.Module):
#     def __init__(self, temp=1.0):
#         super().__init__()
#         self.temp = temp

#     def forward(self, logits_group, label_index):
#         logits_group = logits_group / self.temp

#         target = label_index

#         loss = F.cross_entropy(logits_group, target)
#         return loss


class ConfidenceTrainer(Trainer):
    def __init__(self, model_name, disease_knowledge, loss_fn, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.disease_knowledge = disease_knowledge
        self.loss_fn = loss_fn(epsilon=epsilon)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        label_index = torch.tensor(labels["label_index"], device=model.device)
        
        batch_size = label_index.size(0)
        forward_length = inputs["input_ids"].size(0)
        group_length = forward_length // batch_size

        batch_confidence = self._get_confidence(inputs, batch_size, group_length)

        candidates_current_turn_confidence = batch_confidence.to(model.device)

        loss = self.loss_fn(
            candidates_current_turn_confidence=candidates_current_turn_confidence,
            label_index=label_index,
        )
        return (loss, None) if return_outputs else loss

    def _get_confidence(self, inputs, batch_size, group_length):
        inputs = {
            key: value.to(self.model.device) 
            for key, value in inputs.items() 
            if key in ["input_ids", "attention_mask", "position_ids"]
        }

        outputs = self.model(**inputs)
        logits = outputs.logits
        input_lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        target_positions = input_lens - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_logits = logits[batch_indices, target_positions]
        true_id = BTP_SETTINGS[self.model_name]["True_token"]
        false_id = BTP_SETTINGS[self.model_name]["False_token"]
        true_logits = target_logits[:, true_id]
        false_logits = target_logits[:, false_id]
        temp = BTP_SETTINGS[self.model_name]["T"]  # T的值为浮点数
        confidence = torch.softmax(torch.stack([true_logits, false_logits], dim=-1)/temp, dim=-1)[:, 0]
        
        return confidence.view(batch_size, group_length)



def train(args, disease_knowledge, peft_config, output_dir):

    train_data_raw = load_turn_datasets(args.dataset_name)
    train_dataset = Dataset.from_list(train_data_raw)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device="cuda:0")
    model.enable_input_require_grads()
    model.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        target_modules=args.target_modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.train()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        run_name=args.exp_name,
        remove_unused_columns=False,
    )
    
    data_collator = CustomDataCollator(
        model_name=args.model_name,
        tokenizer=tokenizer,
        disease_knowledge=disease_knowledge,
    )
    
    trainer = ConfidenceTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        loss_fn=ConfidenceLoss,
        model_name=args.model_name,
        disease_knowledge=disease_knowledge,
        epsilon=args.epsilon
    )

    trainer.train()
    
    
def evaluate_performence(model_name, ckpt, peft_config, model, tokenizer, dataset_name, dataset, disease_knowledge, stage):
    def predict(model_type="adapter", symptoms_key="self_report"):
        top_1_count = 0
        k = 12
        top_k_counts = [0] * k
        candidate_diseases = list(get_disease_index_dict(dataset_name))
        new_records = {}
        
        for data in tqdm(dataset, desc=f"{model_type}-{symptoms_key}"):
            disease_label = data["disease_label"]
            symptoms = data[symptoms_key]
            diagnostic_confidence = {}
            for candidate_disease in candidate_diseases:
                diagnostic_confidence[candidate_disease] = BTP(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=disease_diagnosis_prompt_template.format(
                        positive_symptoms="、".join([symptom for symptom, status in symptoms.items() if status]) or "无",
                        negative_symptoms="、".join([symptom for symptom, status in symptoms.items() if not status]) or "无",
                        candidate_disease=candidate_disease,
                        empirical_knowledge=disease_knowledge[candidate_disease]["empirical_knowledge"]
                    )
                )

            rank = get_disease_label_rank(diagnostic_confidence=diagnostic_confidence, disease_label=disease_label)
            if rank == 1:
                top_1_count += 1
            sorted_diagnostic_confidence = dict(sorted(diagnostic_confidence.items(), key=lambda item: item[1], reverse=True))
            
            for i in range(k):
                if disease_label in list(sorted_diagnostic_confidence.keys())[:i+1]:
                    top_k_counts[i] += 1
            
            sub_records = new_records.get(disease_label, [])
            sub_records.append(sorted_diagnostic_confidence)
            new_records[disease_label] = sub_records    
            
        res = round(top_1_count / len(dataset), 4)
        
        sub_records_acc = {}
        for disease, records in new_records.items():
            correct_count = 0
            for diagnostic_confidence in records:
                rank = get_disease_label_rank(diagnostic_confidence=diagnostic_confidence, disease_label=disease)
                if rank == 1:
                    correct_count += 1
            sub_records_acc[disease] = correct_count / len(records)
            
        return res, [round(top_k_counts[i] / len(dataset), 4) for i in range(k)], sub_records_acc
            
    acc_adapter_wq, acc_adapter, acc_base_wq, acc_base = 0, 0, 0, 0
    top_k_acc_adapter_wq, top_k_acc_adapter, top_k_acc_base_wq, top_k_acc_base = 0, 0, 0, 0
    sub_records_acc_base_wq, sub_records_acc_base, sub_records_acc_adapter_wq, sub_records_acc_adapter = {}, {}, {}, {}
    acc_base_wq, top_k_acc_base_wq, sub_records_acc_base_wq = predict(model_type="base", symptoms_key="self_report")  
    acc_base, top_k_acc_base, sub_records_acc_base = predict(model_type="base", symptoms_key="all_symptoms")  
    
    model = PeftModel.from_pretrained(model, model_id=ckpt, config=peft_config)
    model = model.merge_and_unload(progressbar=True)
    acc_adapter_wq, top_k_acc_adapter_wq, sub_records_acc_adapter_wq = predict(model_type="adapter", symptoms_key="self_report")
    acc_adapter, top_k_acc_adapter, sub_records_acc_adapter = predict(model_type="adapter", symptoms_key="all_symptoms")
        
    result = {
        "ckpt": ckpt, 
        "Acc_base_wq": acc_base_wq,
        "Acc_base": acc_base,
        "Acc_adapter_wq": acc_adapter_wq,
        "Acc_adapter": acc_adapter,
        "Top_K_Acc_base_wq": top_k_acc_base_wq,
        "Top_K_Acc_base": top_k_acc_base,
        "Top_K_Acc_adapter_wq": top_k_acc_adapter_wq,
        "Top_K_Acc_adapter": top_k_acc_adapter,
        "sub_records_acc_base_wq": sub_records_acc_base_wq,
        "sub_records_acc_base": sub_records_acc_base,
        "sub_records_acc_adapter_wq": sub_records_acc_adapter_wq,
        "sub_records_acc_adapter": sub_records_acc_adapter,
        "args": vars(args),
    }
    
    # 保存测试结果
    with open(f"{ckpt}/{args.dataset_name}_{stage}_result_sub.json", "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=4)
        
    return result
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence Calibration Training Script")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_modules", nargs="+", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--save_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    args = parser.parse_args()
    print(args)
    
    set_global_seed(args.seed)
    
    disease_knowledge = get_disease_knowledge(args.dataset_name, list(get_disease_index_dict(args.dataset_name)))
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        target_modules=args.target_modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout
    )
    
    best_result = {}
    if "checkpoint-" not in args.adapter_ckpt:
        output_dir = f"./outputs/adapters/{args.exp_name}"
        train(args=args, disease_knowledge=disease_knowledge, peft_config=peft_config, output_dir=output_dir)
        torch.cuda.empty_cache()
        time.sleep(3)
    else:
        peft_config = get_peft_config(args.adapter_ckpt)
        model, tokenizer = load_model_and_tokenizer(args.model_name, device="cuda:0")
        eval_dataset = load_dataset(args.dataset_name, stage="dev")
        dev_result = evaluate_performence(
            model_name=args.model_name,
            ckpt=args.adapter_ckpt,
            peft_config=peft_config,
            model=model,
            tokenizer=tokenizer, 
            dataset_name=args.dataset_name,
            dataset=eval_dataset,
            disease_knowledge=disease_knowledge,
            stage="dev"
        )
        time.sleep(3)
        
