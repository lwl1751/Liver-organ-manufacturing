import json
import openai
import os
import json_repair
from utils import get_info,extract_prompt,system_prompt

txt_folder = 'liver_txt'
doi_path = 'LLM_2/gpt_raw_extraction/test_data.json'

# 打开json文件
with open(doi_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


client = openai.OpenAI(
    base_url = 'https://ai.api.xn--fiqs8s/v1',
    api_key = ''
)
def ask(messages):
    completion = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages=messages,
        temperature = 0.01,
        max_tokens = 4096,
    )
    try:
        res = completion.choices[0].message.content
        return res
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

for i,d in enumerate(data):
    if i==0:
        continue
    text_path = os.path.join(txt_folder, d['doi'].replace('/','_') + '.txt')
    method = d['method']
    cur_shema, entity_requirements = get_info(method)
    with open(text_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    extract_pt = extract_prompt.replace('<text>', text_data)\
                .replace('<entity requirements>',str(entity_requirements))\
                .replace('<method>', str(cur_shema))
    messages_extract = [
        {"role": "system", "content": [{"type": "text","text": system_prompt}]},
        {"role": "user", "content": [{"type": "text","text": extract_pt}]}
    ]
    answer_extract = ask(messages_extract)
    reflect_res = json_repair.loads(answer_extract)
    d['raw_output'] = reflect_res

with open(doi_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
