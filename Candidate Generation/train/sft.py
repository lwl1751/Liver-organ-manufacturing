'''
sft : glm4, llama3
'''

import os
import yaml
import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForSeq2Seq
from swanlab.integration.huggingface import SwanLabCallback
from accelerate.utils import DistributedType

## 找到所有的线性层
def find_all_linear_names(model, int4=True, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

## 打印可训练的参数
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

## 获取模板
def get_template(template):
    if template == 'llama3':
        user_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n\n{{input}}<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif template == 'glm-4':
        user_prompt = (
                "<|system|>\n{{content}}<|endoftext|>\n"
                "<|user|>\n{{input}}<|endoftext|>\n"
                "<|assistant|>\n"
            )
    else:
        user_prompt = '### USER:\n{{input}}\n\n### ASSISTANT:\n'

    return user_prompt

def main(config_path):
    ## 读取 YAML 文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_path = config['model_name_or_path']
    cutoff_len = config['cutoff_len']
    template = config['template']
    assert template in ['llama3', 'glm-4', 'vicuna'], "Template value is not valid"
    output_dir = config['output_dir']

    ## set prompt
    system_prompt = (
        "You are an expert assistant in organ manufacturing."
    )
    user_prompt = get_template(template)


    ## load model and tokenizer
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    device_map = {"": int(os.environ.get("LOCAL_RANK",0) or 0)}
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=False
        )
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
    tokenizer.padding_side = "left"
    if template in ['llama3','vicuna']:
        tokenizer.pad_token = tokenizer.eos_token


    ## 数据集处理
    def process_data(example):
        '''
        Preprocess the data.
        '''
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            user_prompt.replace('{{content}}', system_prompt).replace('{{input}}', example['input']),
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        # 通过添加 -100 来掩盖 instruction 部分，只对 response 部分进行学习。-100 是用于忽略的标签，表示模型在计算损失时不会考虑这一部分。
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > cutoff_len:  # 做一个截断
            input_ids = input_ids[:cutoff_len]
            attention_mask = attention_mask[:cutoff_len]
            labels = labels[:cutoff_len]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

    train_dataset = load_dataset('json', data_files=config['data_path'])
    train_dataset = train_dataset.map(process_data, remove_columns=train_dataset['train'].column_names, num_proc=config['preprocessing_num_workers'])


    ## 模型训练
    modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=config['lora_rank'], 
        lora_alpha=config['lora_alpha'], 
        lora_dropout=config['lora_dropout'],  
        target_modules=modules, 
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_checkpointing = config['gradient_checkpointing'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'],
            warmup_ratio=config['warmup_ratio'],
            logging_steps=config['logging_steps'],
            logging_first_step= config['logging_first_step'],
            save_steps=config['save_steps'],
            overwrite_output_dir=config['overwrite_output_dir'],
            fp16 = config['fp16'],
            report_to="none",
            deepspeed = config['deepspeed_path'],
            ddp_find_unused_parameters=False
        )
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    swanlab_callback = SwanLabCallback(
        project=config['swanlab']['project'],
        experiment_name=config['swanlab']['experiment_name'],
        description=config['swanlab']['description'],
        config=config['swanlab']['config'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 动态 padding 的数据加载器，避免手动 padding
        callbacks=[swanlab_callback]
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    main(args.config_path)
