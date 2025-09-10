import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import time
import sys
import logging
from datetime import datetime
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompt_3 import inference_prompt, sft_prompt
from utils import json2list, write_json, content_prompt  

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, lora_path, device):
    """加载模型和分词器，处理设备配置"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None 
        ).to(device) 

        lora_model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.bfloat16
        )
        
        lora_model.eval()
        logger.info("模型和分词器加载成功")
        return lora_model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def single_sentence_generate(sentence, model, tokenizer, device, prompt_type="few-shot"):
    """单个句子生成响应"""
    try:
        if prompt_type == "few-shot":
            prompt = content_prompt(sentence, inference_prompt)
        else:  # zero-shot
            prompt = sft_prompt + '\n' + sentence
        
        logger.info(f"\n{'='*80}")
        logger.info(f"处理句子: {sentence[:100]}...")
        
        chat = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True
        )
 
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.6,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,** gen_kwargs
            )
        generation_time = time.time() - start_time
        generated_output = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        
        logger.info(f"生成时间: {generation_time:.2f}秒")
        logger.info(f"{'='*80}\n")
        
        return decoded_output
        
    except Exception as e:
        logger.error(f"句子生成错误 '{sentence[:50]}...': {str(e)}")
        return f"ERROR: {str(e)}"

def get_response(sentence_list, model, tokenizer, device, prompt_type="few-shot"):
    """逐个句子生成响应"""
    logger.info(f"开始{prompt_type}推理，共{len(sentence_list)}个句子")
    
    start = time.time()
    results = []
    
    for i, sentence in enumerate(sentence_list):
        logger.info(f"处理第{i+1}/{len(sentence_list)}个句子")
        output = single_sentence_generate(
            sentence, 
            model, 
            tokenizer, 
            device,
            prompt_type
        )
        
        results.append({
            'sentence': sentence,
            'output': output,
            'prompt_type': prompt_type
        })
    
    total_time = time.time() - start
    logger.info(f'总推理时间: {total_time:.2f}秒')
    logger.info(f'平均每个句子耗时: {total_time/len(sentence_list):.2f}秒')
    
    return results

def merge_results(few_shot_results, zero_shot_results):
    """将两种推理结果合并"""
    logger.info("合并few-shot和zero-shot结果...")
    
    merged_dict = {}
    
    for result in few_shot_results:
        sentence = result['sentence']
        merged_dict[sentence] = {
            'sentence': sentence,
            'few-shot output': result['output'],
            'zero-shot output': None
        }
    
    for result in zero_shot_results:
        sentence = result['sentence']
        if sentence in merged_dict:
            merged_dict[sentence]['zero-shot output'] = result['output']
        else:
            merged_dict[sentence] = {
                'sentence': sentence,
                'few-shot output': None,
                'zero-shot output': result['output']
            }
    
    merged_list = list(merged_dict.values())
    logger.info(f"合并了{len(merged_list)}个唯一句子")
    
    return merged_list

def main():

    test_path = 'data/seed_data/test.json'
    output_path = 'data/inference/glm_infer.json'

    test_samples = 354  # 确认无误后可改为 len(sentence_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    model_path = 'Model/glm-4-9b-chat'
    lora_path = 'model/glm-4/sft/checkpoint-329'
    model, tokenizer = load_model_and_tokenizer(model_path, lora_path, device)
    
    logger.info(f"从{test_path}加载测试数据")
    target_list, sentence_list = json2list(test_path)
    logger.info(f"加载了{len(sentence_list)}个测试样本")
    
    test_sentences = sentence_list[:test_samples]
    test_targets = target_list[:test_samples]
    logger.info(f"先测试{test_samples}个样本...")
    
    logger.info("开始few-shot推理...")
    few_shot_results = get_response(test_sentences, model, tokenizer, device, "few-shot")
    
    logger.info("开始zero-shot推理...")
    zero_shot_results = get_response(test_sentences, model, tokenizer, device, "zero-shot")

    # 合并结果
    merged_results = merge_results(few_shot_results, zero_shot_results)
    
    # 添加目标标签
    for i, item in enumerate(merged_results):
        item['target'] = test_targets[i]
    
    write_json(merged_results, output_path)


if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=5 python inference/batch_inference_glm.py
'''
