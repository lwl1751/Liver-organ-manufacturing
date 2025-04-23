import os
import yaml
import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForSeq2Seq
from swanlab.integration.huggingface import SwanLabCallback
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader, Dataset

config_path = '/home/liangwenliang/器官制造/LLM_2/train/vicuna_lora_sft.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

model_path = config['model_name_or_path']
user_prompt = '### USER:\n{{input}}\n\n### ASSISTANT:\n'

tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

## 数据集处理
def process_data(example):
    combined_text = user_prompt.replace('{{input}}', example['input']).replace('{{output}}', example['output'])
    return {"input_text": combined_text}

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = load_dataset('json', data_files=config['data_path'])
train_dataset = train_dataset.map(process_data, remove_columns=train_dataset['train'].column_names, num_proc=config['preprocessing_num_workers'])
print(train_dataset)
train_encodings = tokenizer(train_dataset['train']['input_text'], truncation=True, padding=True, max_length=256, return_tensors='pt')
train_dataset = TextDataset(train_encodings)
print(train_dataset.__getitem__(0))