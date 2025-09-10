import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import time
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_llama_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from utils import *
from prompt import inference_prompt, sft_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json
from peft import PeftModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model_path = 'Model/Meta-Llama-3-8B-Instruct'
lora_path = 'model/llama3-8b/sft/checkpoint-82'

# 加载模型
logger.info("Loading model and tokenizer...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map="auto",  
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)   
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_model = PeftModel.from_pretrained(
        model,
        lora_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def prepare_single_batch(prompts_batch, tokenizer):
    """为单个批次准备tokenized输入"""
    tokenizer.padding_side = "left"
    return tokenizer(
        prompts_batch, 
        return_tensors="pt", 
        padding='longest', 
        truncation=False, 
        pad_to_multiple_of=8,
        add_special_tokens=False
    ).to(device)

def get_response(sentence_list, prompt_type="few-shot", batch_size=16):
    logger.info(f"Starting {prompt_type} inference with {len(sentence_list)} sentences")
    print(f'总句子数目为 {len(sentence_list)}')

    start = time.time()
    results = []

    # 根据提示类型选择不同的提示构建方式
    if prompt_type == "few-shot":
        logger.info("Using few-shot prompting")
        # 不再一次性处理所有prompts，改为逐批处理
    else:  # zero-shot
        logger.info("Using zero-shot prompting")
        # 不再一次性处理所有prompts，改为逐批处理
    
    total_batches = (len(sentence_list) + batch_size - 1) // batch_size
    logger.info(f"Total batches to process: {total_batches}")
    
    start_generation = time.time()
    
    for batch_idx in range(total_batches):
        logger.info(f"Processing batch {batch_idx+1}/{total_batches}")
        
        # 获取当前批次的句子
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sentence_list))
        current_sentences = sentence_list[start_idx:end_idx]
        
        # 为当前批次构建prompts
        if prompt_type == "few-shot":
            batch_prompts = process_batch(current_sentences, inference_prompt)
        else:  # zero-shot
            batch_prompts = [sft_prompt + '\n' + sen for sen in current_sentences]
        
        try:
            # 为当前批次tokenize
            prompts_tokenized = prepare_single_batch(batch_prompts, tokenizer)
            
            # 生成输出
            outputs_tokenized = lora_model.generate(
                **prompts_tokenized, 
                max_new_tokens=1024, 
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.01,
                top_p=0.6
            )

            # 从生成的tokens中移除prompt部分
            outputs_tokenized = [tok_out[len(tok_in):] 
                                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            outputs = tokenizer.batch_decode(outputs_tokenized)

            # 存储结果
            results.extend([{
                'sentence': s,
                'output': o,
                'prompt_type': prompt_type
            } for s, o in zip(current_sentences, outputs)])
            
            logger.debug(f"Batch {batch_idx+1} completed successfully")
            
            # 清理内存
            del prompts_tokenized, outputs_tokenized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx+1}: {e}")
            # 添加错误信息到结果中
            results.extend([{
                'sentence': s, 
                'output': f"ERROR: {str(e)}",
                'prompt_type': prompt_type
            } for s in current_sentences])

    generation_time = time.time() - start_generation
    logger.info(f'Generation completed in {generation_time:.2f} seconds')

    total_time = time.time() - start
    logger.info(f'Total inference time: {total_time:.2f} seconds')
    logger.info(f'Got {len(results)} results')

    return results

def merge_results(few_shot_results, zero_shot_results):
    """将两种推理结果合并到同一个条目中"""
    logger.info("Merging few-shot and zero-shot results...")
    
    merged_dict = {}
    
    for result in few_shot_results:
        sentence = result['sentence']
        if sentence not in merged_dict:
            merged_dict[sentence] = {
                'sentence': sentence,
                'few-shot output': result['output'],
                'zero-shot output': None  
            }
        else:
            merged_dict[sentence]['few-shot output'] = result['output']

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
    
    # 转换为列表
    merged_list = list(merged_dict.values())
    logger.info(f"Merged {len(merged_list)} unique sentences")
    
    return merged_list

def main():
    test_path = 'data/seed_data/test.json'
    output_path = 'data/inference_data/llama_infer.json'
    batch_size = 4

    logger.info(f"Loading test data from {test_path}")
    target_list, sentence_list = json2list(test_path)
    logger.info(f"Loaded {len(sentence_list)} test samples")

    # 进行两种推理
    logger.info("Starting few-shot inference...")
    few_shot_results = get_response(sentence_list, "few-shot", batch_size)
    logger.info(f"Few-shot inference completed, got {len(few_shot_results)} results")
    
    logger.info("Starting zero-shot inference...")
    zero_shot_results = get_response(sentence_list, "zero-shot", batch_size)
    logger.info(f"Zero-shot inference completed, got {len(zero_shot_results)} results")

    # 合并结果
    merged_results = merge_results(few_shot_results, zero_shot_results)
    
    logger.info("Adding target labels to results...")
    for item in merged_results:
        sentence = item['sentence']
        try:
            idx = sentence_list.index(sentence)  
            target = target_list[idx]
            item['target'] = target
        except ValueError:
            logger.warning(f"Sentence not found in original list: {sentence}")
            item['target'] = "UNKNOWN"

    # 保存结果
    logger.info(f"Saving results to {output_path}")
    try:
        write_json(merged_results, output_path)
        logger.info(f"Results successfully saved to {output_path}")
        
        # 记录统计信息
        complete_results = len([r for r in merged_results 
                               if r['few-shot output'] is not None and r['zero-shot output'] is not None])
        missing_few_shot = len([r for r in merged_results if r['few-shot output'] is None])
        missing_zero_shot = len([r for r in merged_results if r['zero-shot output'] is None])
        
        logger.info(f"Total merged results: {len(merged_results)}")
        logger.info(f"Complete results (both outputs): {complete_results}")
        logger.info(f"Results missing few-shot output: {missing_few_shot}")
        logger.info(f"Results missing zero-shot output: {missing_zero_shot}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=6 python inference_data/batch_inference_llama.py
'''

