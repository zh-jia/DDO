import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_templates import disease_diagnosis_prompt_template
from peft import LoraConfig, TaskType
import json
import contextlib
from global_utils import set_global_seed

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.generation.configuration_utils"
)

    
def get_peft_config(adapter_ckpt):
    adapter_config = None
    with open(f"{adapter_ckpt}/adapter_config.json", "r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        target_modules=adapter_config["target_modules"],
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"], 
        lora_dropout=adapter_config["lora_dropout"]
    )
    return peft_config

BTP_SETTINGS = {
    "llama3-8b-chinese-chat": {"True_token": 2575, "False_token": 4139, "T": 1.0},
    "qwen2.5-7b-instruct": {"True_token": 2514, "False_token": 4049, "T": 5.0},
    "qwen2.5-14b-instruct": {"True_token": 2514, "False_token": 4049, "T": 5.0},
}

def load_model_and_tokenizer(model_name, device="cuda:0"):
    model_path = f"../../models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if "llama3" not in model_name else torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


# 获取模型输出内容
def get_model_output_content(model_name, model, tokenizer, prompt, do_sample=True, temperature=0.1, max_new_tokens=1024):
    generated_tokens, _ = get_model_outputs(
        model_name=model_name, 
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    output_content = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return output_content


def build_inputs(model_name, tokenizer, prompts):
    inputs = {}
    input_ids_tensor_list = []
    attention_mask_tensor_list = []
    position_ids_tensor_list = []
    max_length = 0

    for prompt in prompts:
        single_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        max_length = max(max_length, single_inputs["input_ids"].size(-1))

        input_ids_tensor_list.append(single_inputs["input_ids"])
        attention_mask_tensor_list.append(single_inputs["attention_mask"])
        
    for i in range(len(prompts)):
        pad_len = max_length - input_ids_tensor_list[i].size(1)

        input_ids_tensor_list[i] = torch.cat(
            (input_ids_tensor_list[i], torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)),
            dim=-1
        )

        attention_mask_tensor_list[i] = torch.cat(
            (attention_mask_tensor_list[i], torch.ones((1, pad_len), dtype=torch.long)),
            dim=-1
        )

    inputs["input_ids"] = torch.cat(input_ids_tensor_list, dim=0)
    inputs["attention_mask"] = torch.cat(attention_mask_tensor_list, dim=0)
    
    return inputs
    
def get_model_outputs(model_name, model, tokenizer, prompt, do_sample, temperature=None, max_new_tokens=1024):
    outputs = None
    inputs = build_inputs(model_name=model_name, tokenizer=tokenizer, prompts=[prompt])
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    outputs = model.generate(
        **inputs, 
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_id=model.config.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    generated_tokens = outputs.sequences[:, inputs["input_ids"].size()[1]: ].cpu()
    logits = outputs.scores
    return generated_tokens, logits


def BTP(model_name, model, tokenizer, prompt, expected_answer="True", do_sample=False, temperature=0.1, max_new_tokens=1):  # The first output token need to be binary
    probs = None
    logit_true, logit_false = None, None
    generated_tokens, logits = get_model_outputs(
        model_name=model_name, 
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    logit_true = logits[0][0][BTP_SETTINGS[model_name]["True_token"]]
    logit_false = logits[0][0][BTP_SETTINGS[model_name]["False_token"]]
    probs = torch.softmax(torch.stack([logit_true, logit_false], dim=-1)/BTP_SETTINGS[model_name]["T"], dim=-1)
    if probs == None:
        return 0
    btp_result = probs[0].item() if expected_answer == "True" else probs[1].item()
    return round(btp_result, 6)


def find_substr_close_to_end(output_content, substr_list):
    max_index = -1
    closest_substr = ""
    for sub_str in substr_list:
        index = output_content.rfind(sub_str)
        if index != -1 and index > max_index:
            max_index = index
            closest_substr = sub_str
    return closest_substr