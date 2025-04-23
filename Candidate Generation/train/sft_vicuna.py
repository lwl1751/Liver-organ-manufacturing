'''
sft : vicuna
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
from torch.utils.data import Dataset

def main(config_path):
    ## 读取 YAML 文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_path = config['model_name_or_path']
    cutoff_len = config['cutoff_len']
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    user_prompt = '### USER:\n{{input}}\n\n### ASSISTANT:\n'

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
    tokenizer.pad_token = tokenizer.eos_token

    ## process data
    def process_data(example):
        combined_text = user_prompt.replace('{{input}}', example['input']).replace('{{output}}', example['output'])
        return {"input_text": combined_text}

    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

        def __len__(self):
            return len(self.encodings["input_ids"])

    train_dataset = load_dataset('json', data_files=config['data_path'])
    train_dataset = train_dataset.map(process_data, remove_columns=train_dataset['train'].column_names, num_proc=config['preprocessing_num_workers'])
    train_encodings = tokenizer(
        train_dataset['train']['input_text'], 
        truncation=True, 
        padding=True, 
        max_length=cutoff_len, 
        return_tensors='pt'
    )
    train_dataset = TextDataset(train_encodings)

    peft_config = LoraConfig(
        r=config['lora_rank'], 
        lora_alpha=config['lora_alpha'], 
        lora_dropout=config['lora_dropout'],  
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 动态 padding 的数据加载器，避免手动 padding
        callbacks=[swanlab_callback]
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    main(args.config_path)
