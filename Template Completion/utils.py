import json
import time
from openai import OpenAI
from prompt import *
import re

# 统计tokens数量
def count_words(text):
    # 使用正则表达式去除标点符号
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()
    
    return len(words)

# 获取txt文本
def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        pre_data = f.readlines()
    
    # 过滤掉包含 'purchased' 或 'Keywords' 的行，并清理多余的空白字符
    filtered_data = [line for line in pre_data if 'purchased' not in line and 'Keywords' not in line]
    return filtered_data

# 将txt文本分为多个段落
def split_text_into_segments(lines,max_words_per_segment = 512):
    result_paragraphs = []
    current_segment = []
    word_count = 0

    for line in lines:
        line = line.strip()
        if line:
            line_count = count_words(line)
            if word_count + line_count <= max_words_per_segment:
                current_segment.append(line)
                word_count += line_count
            else:
                # 如果当前行加入会超过限制，则保存当前段落并重新开始一个新段落
                result_paragraphs.append('\n'.join(current_segment))
                current_segment = [line]
                word_count = line_count

    # 保存最后一个段落
    if current_segment:
        result_paragraphs.append('\n'.join(current_segment))

    return result_paragraphs

# 通过doi获取句子级的抽取信息
def find_output_by_doi(data, target_doi):
    for item in data:
        if item["doi"] == target_doi:
            return item
    return None 

# 提取json字符，str格式
def extract_json_str(llm_output):
    if '```json' in llm_output:
        json_start_marker = "```json"
        json_end_marker = "```"
    elif '```python' in llm_output:
        json_start_marker = "```python"
        json_end_marker = "```"
    elif '```' in llm_output:
        json_start_marker = "```"
        json_end_marker = "```"
    else:
        json_start_marker = "{"
        json_end_marker = "}"
    # Find the start and end indices of the JSON string
    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        end_index = llm_output.rfind(json_end_marker)
    else:
        return None  # JSON markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    json_string = llm_output[start_index:end_index + 1].strip() if json_start_marker == '{' else llm_output[start_index + len(json_start_marker):end_index].strip()
    json_string = json_string.replace("'", "\"")
    return json_string

# 提取json字符，json格式
def extract_json_between_markers(llm_output):
    json_string = extract_json_str(llm_output) 
    if not json_string: 
        return None
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None  # Invalid JSON format


'''
第二阶段：判断实体是否存在于该段落，是否合理
'''

# 处理句子级实体数据
def process_extract_data(extract_data):
    res = ''
    for k,v in extract_data.items():
        if len(v) == 0:
            continue
        res += f'###\nEntity type: {k}\nValue:\n'
        for value in v:
            res += f'    - {value}\n'
        res += '\n'
    return res

# 获取合理的实体输出
def get_check_fix(llm_output):
    json_string =extract_json_str(llm_output)
    try:
        llm_output = json_string.replace("'",'"')
        lines = llm_output.splitlines()
        output = []
        extracted = {}
        for line in lines:
            line = line.strip()
            # 匹配以第一个 ": " 为分隔符
            if '"' in line:
                key_end_id = line.find('": "')
                value_end_id = line.rfind('"')
                extracted[line[1:key_end_id]] = line[key_end_id + len('": "'):value_end_id]
            if len(extracted) == 3:
                output.append({extracted['entity_type']: (extracted['value'], extracted['reason'].replace('\"','\''))})
                extracted = {}
        return output
    except json.JSONDecodeError:
        return None  # Invalid JSON format

# 获取实体到段落的映射关系
def get_entity_2_para(input_json):
    output_json = {k : [] for k in schema.keys()}
    for data in input_json:
        paragraph_order = data['paragraph_order']
        info = data['entity_info']
        for item in info:
            for entity, (value, reason) in item.items():
                if entity not in schema or not value:
                    continue
                output_json[entity].append((value, reason, paragraph_order))
    return {k:v for k,v in output_json.items() if len(v)}

class Extract_Workflow():
    def __init__(self, text_data_list, api_key, start = False, model = 'gpt-4o-mini') -> None:
        self.client = OpenAI(
            api_key = api_key,
            base_url="https://api.bianxie.ai/v1"
        )
        self.text_data_list = text_data_list
        self.extract_data = ''
        self.moedel = model

        if start:
            self.entity_requirements = None
            self.method_record = None
            self.method = None
            self.cell_type = None
            self.needed_params = None
            self.paragraph_2_entity = None
            self.entity_2_paragraph = None
            self.entity_text = None
            self.summary_res = None
            self.match_res = None
            self.reflect_res = None


    # 调用openai接口
    def ask(self, messages):
        completion = self.client.chat.completions.create(
            model = self.moedel,
            messages=messages,
            temperature = 0,
            max_tokens = 4096,
        )
        try:
            res = completion.choices[0].message.content
            return res
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def part_1(self):
        print('*************************************')
        print(f'第一阶段')
        summary_pt = summary_1.replace('<text>','\n'.join(self.text_data_list))
        messages_summary = [
            {"role": "system", "content": [{"type": "text","text": system_prompt}]},
            {"role": "user", "content": [{"type": "text","text": summary_pt}]}
        ]
        answer_summary = self.ask(messages_summary)
        print(f'tokens数:\n{count_words(summary_pt)}\n')
        print(f'总结输出:\n{answer_summary}\n')

        messages_summary_check = messages_summary + [
            {"role": "assistant", "content": [{"type": "text","text": answer_summary}]},
            {"role": "user", "content": [{"type": "text","text": summary_2}]}
        ]
        answer_summary_check = self.ask(messages_summary_check)
        print(f'匹配输出:\n{answer_summary_check}\n')
        summary_res = extract_json_between_markers(answer_summary_check)
        print(f'提取输出:\n{summary_res}\n')

        self.summary_res = summary_res
        self.method_record = answer_summary_check
        self.method = summary_res['method'] if summary_res else None
        self.cell_type = summary_res['cell type']
        return None

    def part_2(self):
        print('*************************************')
        print(f'第二阶段')

        paragraph_2_entity = []
        for i, text in enumerate(self.text_data_list):
            print(f'order:{i}')
            paragraph_check_pt = paragraph_check_prompt.replace('<extracted sentence-level entity information>', str(self.extract_data)).replace('<text>',text).replace('<entity requirements>',str(self.entity_requirements))
            messages_check = [
                {"role": "system", "content": [{"type": "text","text": system_prompt}]},
                {"role": "user", "content": [{"type": "text","text": paragraph_check_pt}]}
            ]
            answer_check = self.ask(messages_check)
            # print(f'匹配段落输出:\n{answer_check}\n')
            check_data = get_check_fix(answer_check)
            # print(f'提取输出:\n{check_data}\n')
            if check_data:
                messages_fix = messages_check + [
                    {"role": "assistant", "content": [{"type": "text","text": answer_check}]},
                    {"role": "user", "content": [{"type": "text","text": paragraph_fix_prompt}]}
                ]
                answer_fix = self.ask(messages_fix)
                # print(f'匹配段落反思:\n{answer_fix}\n')
                fix_data = get_check_fix(answer_fix)
                # print(f'提取反思:\n{fix_data}\n')
                if fix_data:
                    paragraph_2_entity.append({'paragraph_order': i, 'entity_info': fix_data})  

        self.paragraph_2_entity = paragraph_2_entity
        self.entity_2_paragraph = get_entity_2_para(self.paragraph_2_entity)
        print(f'entity_2_paragraph:\n{self.entity_2_paragraph}')       
        return None
    
    def part_3(self):
        print('*************************************')
        print(f'第三阶段')

        # 当前schema
        cur_schema = method_2_schema[self.method]
        cur_schema['Cell Culture Conditions']['Cell type'] = self.cell_type

        # 可选择的参数
        needed_params = []
        for entity, values in self.entity_2_paragraph.items():
            unique_values = {}

            for value, reason, _ in values:
                if value not in unique_values:
                    unique_values[value] = reason

            needed_params.append({entity: [
                {'value': value, 'reasons': unique_values[value]}
                for value in unique_values
            ]})
        self.needed_params = needed_params

        # 处理的文本
        paragraph_order_list = [d['paragraph_order'] for d in self.paragraph_2_entity]
        entity_text = ''
        for order in paragraph_order_list:
            entity_text += self.text_data_list[order]
        self.entity_text = entity_text

        match_pt = match_prompt.replace('<text>',entity_text)\
                    .replace('<Options for each entity>',str(needed_params))\
                    .replace('<JSON schema>',str(cur_schema))\
                    .replace('<Entity requirements>',str(self.entity_requirements))
        # print(f'匹配段落:\n{paragraph_order_list}\n')
        # print(f'tokens数:\n{count_words(match_pt)}\n')
        messages_match = [
            {"role": "system", "content": [{"type": "text","text": system_prompt}]},
            {"role": "user", "content": [{"type": "text","text": match_pt}]}
        ]
        answer_match = self.ask(messages_match)
        answer_match_extract = extract_json_between_markers(answer_match)
        print(f'模型输出:\n{answer_match_extract}\n')

        self.match_res = answer_match_extract
        return None

    def part_4(self):
        print('*************************************')
        print(f'第四阶段')

        reflect_pt = reflect_prompt.replace('<text>',self.entity_text)\
            .replace('<Options for each entity>',str(self.needed_params))\
            .replace('<JSON schema>',str(self.match_res))\
            .replace('<Entity requirements>',str(self.entity_requirements))\
            .replace('<method>', str(method_2_schema[self.method]))
        messages_reflect = [
            {"role": "system", "content": [{"type": "text","text": system_prompt}]},
            {"role": "user", "content": [{"type": "text","text": reflect_pt}]}
        ]
        answer_reflect = self.ask(messages_reflect)
        # print(f'tokens数:\n{count_words(reflect_pt)}\n')
        # print(f'反思后模型输出:\n{answer_reflect}\n')

        reflect_check_pt = reflect_check_prompt.replace('<method>', str(method_2_schema[self.method])).replace('<Entity requirements>',str(self.entity_requirements))
        messages_reflect_check = messages_reflect + [
            {"role": "assistant", "content": [{"type": "text","text": answer_reflect}]},
            {"role": "user", "content": [{"type": "text","text": reflect_check_pt}]}
        ]
        answer_reflect_check = self.ask(messages_reflect_check)
        # print(f'输出:\n{answer_reflect_check}\n')
        reflect_res = extract_json_between_markers(answer_reflect_check)
        print(f'提取输出:\n{reflect_res}\n')

        self.reflect_res = reflect_res if reflect_res else answer_reflect_check
        return None
    
    def part_5(self):
        print('**********************')
        print(f'gpt_extract')
        cur_schema = method_2_schema[self.method]
        cur_schema['Cell Culture Conditions']['Cell type'] = self.cell_type
        extract_pt = extract_prompt.replace('<text>','\n'.join(self.text_data_list))\
                    .replace('<entity requirements>',str(self.entity_requirements))\
                    .replace('<method>', str(cur_schema))
        messages_extract = [
            {"role": "system", "content": [{"type": "text","text": system_prompt}]},
            {"role": "user", "content": [{"type": "text","text": extract_pt}]}
        ]
        answer_extract = self.ask(messages_extract)
        print(f'抽取结果:\n{answer_extract}\n')

        reflect_res = extract_json_between_markers(answer_extract)
        print(f'提取输出:\n{reflect_res}\n')
        self.reflect_res = reflect_res
        return reflect_res
