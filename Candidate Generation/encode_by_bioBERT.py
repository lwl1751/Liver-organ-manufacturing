import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import pickle

json_path = "data/seed_data_fix/rag.json"
pickle_path = "data/seed_data_fix/rag.pkl"

# 加载BioBERT模型和tokenizer
small_model_path = 'Model/biobert-base-cased-v1.2'
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path)
small_model = AutoModel.from_pretrained(small_model_path)

# 对句子进行嵌入表示（批量处理）
def batch_encode_with_small_model(sentences, batch_size=64):
    all_embeddings = []
    # 按批次处理
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_dict = small_tokenizer(batch_sentences, max_length=256, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():  # 禁用反向传播，节省内存
            outputs = small_model(**batch_dict)
        attention_mask = batch_dict['attention_mask']
        last_hidden_states = outputs.last_hidden_state
        # 掩码无效的部分
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # 平均池化
        pooled_output = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # 将批量嵌入结果保存
        all_embeddings.append(pooled_output.cpu().numpy())
    
    return np.vstack(all_embeddings)  # 合并所有嵌入

# 加载并处理JSON数据文件
with open(json_path, 'r') as file:
    data = json.load(file)

# 将所有句子生成嵌入向量，并保存所需的格式
def generate_embeddings(data, batch_size=64):
    sentences = [item['sentence'] for item in data]
    embeddings_sen = batch_encode_with_small_model(sentences, batch_size)  # 批量生成句子嵌入
    embeddings = []
    for i, item in enumerate(data):
        embeddings.append({
            'id': i,
            'sentence': item['sentence'],
            'output': item['output'],
            'embedding_sen': embeddings_sen[i]  # 通过索引将嵌入与句子对应
        })
    return embeddings

# 生成嵌入
embeddings = generate_embeddings(data, batch_size=16)
print('embedding finished')

# 保存结果为.pkl文件
with open(pickle_path, 'wb') as file:
    pickle.dump(embeddings, file)
