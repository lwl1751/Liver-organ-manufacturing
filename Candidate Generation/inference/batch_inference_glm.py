import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import time
from utils import *
import sys
from prompt import inference_prompt
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json
from peft import PeftModel

accelerator = Accelerator()
model_path = 'glm-4-9b-chat'
lora_path = ''
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True, use_fast=True)   
tokenizer.pad_token = tokenizer.eos_token
lora_model = PeftModel.from_pretrained(
    model,
    lora_path,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

#Get the actual CUDA device ID for this process
message = [f"Hello ,this is GPU {accelerator.process_index}"]
messages = gather_object(message)
accelerator.print(messages)

# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok = []
    tokenizer.padding_side = "left"
    
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False
            ).to("cuda") 
        )
    return batches_tok

def get_response(sentence_list, batch_size=16):
    accelerator.print(f'总句子数目为 {len(sentence_list)}')

    # 使用 with 语句分配句子列表和 prompts
    with accelerator.split_between_processes(sentence_list) as sentence_list_split:
        
        # 同步 GPU 并启动计时器
        accelerator.wait_for_everyone()    
        start = time.time()
        results = []

        # 将 prompts 按批处理
        batch_prompts_split = process_batch(sentence_list_split, inference_prompt)
        # batch_prompts_split = [sft_prompt + '\n' + sen for sen in sentence_list_split]
        prompt_batches = prepare_prompts(batch_prompts_split, tokenizer, batch_size=batch_size)
        accelerator.print(time.time() - start)
        accelerator.print('prompt tokenized finished')
        
        start = time.time()
        for batch_idx, prompts_tokenized in enumerate(prompt_batches):
            outputs_tokenized = lora_model.generate(
                **prompts_tokenized, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=True,
                temperature=0.1,
                top_p=0.6
            )

            # 从生成的 tokens 中移除 prompt 部分
            outputs_tokenized = [tok_out[len(tok_in):] 
                                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            outputs = tokenizer.batch_decode(outputs_tokenized)

            # 计算当前批次句子索引，防止最后一批不足 batch_size 时越界
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sentence_list_split))  # 防止越界
            current_sentences = sentence_list_split[start_idx:end_idx]

            # 将输入句子、生成的输出和目标一起存储
            results.extend([{'sentence': s, 'output': o} 
                            for s, o in zip(current_sentences, outputs)])

    # 收集所有 GPU 的结果
    results_gathered = gather_object(results)
    accelerator.print(time.time() - start)

    return results_gathered


test_path = 'test.json'
output_path = 'inference_glm.json'
batch_size = 4

target_list, sentence_list = json2list(test_path)
res = get_response(sentence_list, batch_size)
for item in res:
    sentence = item['sentence']
    idx = sentence_list.index(sentence)  
    target = target_list[idx]
    item['target'] = target

write_json(res,output_path)

'''
CUDA_VISIBLE_DEVICES=4,6,7 accelerate launch inference/batch_inference_glm.py
'''

# directory_path = liver_paper_txt'
# output_path = 'data/inference_glm.jsonl'
# batch_size = 4

# start_index = 0
# for index, file_name in enumerate(os.listdir(directory_path)):
#     if index < start_index:
#         continue
#     if file_name.endswith('.txt'):
#         file_path = os.path.join(directory_path, file_name)
#         data = read_txt(file_path)
#         txt_res = get_response(data, batch_size)

#         # 创建字典，并将其追加写入JSONL文件
#         if accelerator.is_main_process:
#             entry = {'doi': file_name.replace('_','/'), 'output': txt_res}
#             with open(output_path, 'a') as f:
#                 f.write(json.dumps(entry) + '\n')
#     accelerator.wait_for_everyone() 
