"""
Reference from https://github.com/sgl-project/sglang
"""

import time
from openai import OpenAI
from utils import process_batch, json2list,write_json

import sys
sys.path.append('')
from prompt import inference_prompt, system_promopt

class OpenAIBatchProcessor:
    def __init__(self, model):
        client = OpenAI(
            api_key="",
            base_url="https://api.bianxie.ai/v1"
        )

        self.client = client
        self.model = model

    def ask(self, input):
        completion = self.client.chat.completions.create(
            model = self.model,
            messages=[
                { "role": "system", "content": system_promopt },
                {
                    "role": "user",
                    "content": input,
                },
            ],
            temperature = 0,
            max_tokens = 1024,
        )
        try:
            res = completion.choices[0].message.content
            return res
        except Exception as e:
            print(input)
            print(completion)
            print(f"An error occurred: {e}")
    
    def process_batch_fun(self, input_data):
        prompt_list = process_batch(input_data, inference_prompt)
        print('prompt finished')

        final_res = []
        start_time = time.time()
        for i, prompt in enumerate(prompt_list):
            res = self.ask(prompt)
            final_res.append({'sentence': input_data[i], 'output': res})
        print(f"Total time taken: {time.time() - start_time} seconds")
        return final_res

input_path = "test.json"
output_path = "inference_gpt.json"

target_data, input_data = json2list(input_path)
batch_processor = OpenAIBatchProcessor('gpt-4o-mini-2024-07-18')
res = batch_processor.process_batch_fun(input_data)
for i,target in enumerate(target_data):
    res[i]['target'] = target

write_json(res, output_path)
