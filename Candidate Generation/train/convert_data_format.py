'''
初始数据格式为
{
    'sentence': sen,
    'output': out
}

输出数据格式为
{
    'input': input, # 带有prompt + sentence
    'output': out
}
需要提前转换为初始格式
'''
import sys
sys.path.append('Candidate Generation')
import json
from prompt import sft_prompt

input_path = 'seed_data_train.json'
output_path = 'train.json'

with open(input_path,'r') as f:
    data = json.load(f)

messages = []
for d in data:
    message = {
        "input": f"{sft_prompt}\n{d['sentence']}",
        "output": d['output']
    }
    messages.append(message)

with open(output_path,'w',encoding='utf-8') as f1:
    json.dump(messages, f1, ensure_ascii=False, indent=4)
