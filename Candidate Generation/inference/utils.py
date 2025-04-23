import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,AutoModel
import torch
from peft import PeftModel
import json
import concurrent.futures
import re

small_model_path = '/home/liangwenliang/biobert-base-cased-v1.2'
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path)
small_model = AutoModel.from_pretrained(small_model_path)

# 对句子做嵌入表示
def encode_with_small_model(sen):
    batch_dict = small_tokenizer(sen, max_length=256, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():  # 禁用反向传播，节省内存
        outputs = small_model(**batch_dict)
    attention_mask = batch_dict['attention_mask']
    last_hidden_states = outputs.last_hidden_state
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    pooled_output = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    return pooled_output.tolist()[0] # 返回一维数组向量

# 获取 从句子中抽取实体的 prompt
def content_prompt(sentence,prompt,num_example = 3):
    # embedding_data: id,sentence,output,embedding_sen
    embedding_data = pickle.load(open('/home/liangwenliang/器官制造/LLM_2/data/seed_data_fix/rag.pkl','rb'))
    embedding_sen = encode_with_small_model(sentence)
    embedding_sen = np.array(embedding_sen)
    embedding_data_array = np.array([i['embedding_sen'] for i in embedding_data])
    similar_score = cosine_similarity(embedding_data_array, embedding_sen.reshape(1, -1))
    similar_reference = [embedding_data[i]['id'] for i in np.argsort(similar_score.flatten())[::-1][:num_example]]
    
    example_prompt = ''
    for i,idx in enumerate(similar_reference):
        (instruction, output) = embedding_data[idx]['sentence'], embedding_data[idx]['output']
        example_prompt += f"\nExample {i + 1}:\n"
        # example_prompt += f"Sentence: {instruction}\n"
        example_prompt += f"Text: {instruction}\n"
        example_prompt += f"Output: {output}\n"
    # print(example_prompt)
    prompt = prompt.replace('<Example>',example_prompt).replace('<text>',sentence)
    return prompt

# 实现 prompt 的批量处理
def process_batch(batch_sentence_list, prompt):
    with concurrent.futures.ThreadPoolExecutor() as executor: # 创建线程池执行器
        future_to_index = {executor.submit(content_prompt, sen, prompt): i for i, sen in enumerate(batch_sentence_list)}
        results = [None] * len(batch_sentence_list)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
        return results

def load_model_and_tokenizer(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False if 'Llama' in model_path else True,
            trust_remote_code=True
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config = quantization_config
    ).eval()
    if lora_path:
        lora_model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map='auto',
            torch_dtype=torch.float16)
        return lora_model,tokenizer
    return model, tokenizer

# 获取 txt 文本, 并处理为单个句子
def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        # 读取文件并去除包含特定关键词的行，去除换行符
        data = [
            line.strip() for line in f 
            if 'purchased' not in line and 'Keywords' not in line
        ]

    # 将数据合并成一个字符串，按句点或换行符进一步分割，并过滤掉短句
    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<!\d)\.(?!\d)|\n', ' '.join(data))
        if sentence.strip() and len(sentence.strip()) >= 15
    ]

    return sentences


# 保存 json 文件
def write_json(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,  indent=4)

# 读取 json文件，文件中内容格式为:{'sentence': 'sen', 'output': 'output'}
def json2list(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return [d['output'] for d in data],[d['sentence'] for d in data]

# 对 txt 文本进行分割
def split_text_into_segments(lines,max_words_per_segment = 512):
    result_paragraphs = []
    current_segment = []
    word_count = 0

    for line in lines:
        line = line.strip()
        if line:
            token_pattern = r'\s+|(?<!\d)[.,;!?](?!\d)' # 定义正则表达式模式，以空格和标点符号作为分割符
            tokens = re.split(token_pattern, line)

            if word_count + len(tokens) <= max_words_per_segment:
                current_segment.append(line)
                word_count += len(tokens)
            else:
                # 如果当前行加入会超过限制，则保存当前段落并重新开始一个新段落
                result_paragraphs.append('\n'.join(current_segment))
                current_segment = [line]
                word_count = len(tokens)

    # 保存最后一个段落
    if current_segment:
        result_paragraphs.append('\n'.join(current_segment))

    return result_paragraphs
