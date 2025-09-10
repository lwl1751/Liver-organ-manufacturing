"""
Reference from https://github.com/sgl-project/sglang
"""

import time
from openai import OpenAI
from utils import content_prompt, json2list, write_json

from prompt_3 import  system_prompt, inference_prompt,sft_prompt

class OpenAIBatchProcessor:
    def __init__(self, model):
        client = OpenAI(
            api_key="sk-pFdsHdmkzcMhGXs3SSyRYVWvdL3IuzKjEd1P5oBtsX8Rry2c",
            base_url="https://api.bianxie.ai/v1"
        )

        self.client = client
        self.model = model

    def ask(self, input):
        completion = self.client.chat.completions.create(
            model = self.model,
            messages=[
                { "role": "system", "content": system_prompt },
                {
                    "role": "user",
                    "content": input,
                },
            ],
            temperature = 0.01,
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
        final_res = []
        start_time = time.time()
        for sen in input_data:
            few_shot_prompt = content_prompt(sen, inference_prompt)
            zero_shot_prompt = sft_prompt + '\n' + sen
            few_shot_res = self.ask(few_shot_prompt)
            zero_shot_res = self.ask(zero_shot_prompt)
            final_res.append({'sentence': sen, 'few-shot output': few_shot_res, 'zero-shot output': zero_shot_res})
        print(f"Total time taken: {time.time() - start_time} seconds")
        return final_res

input_path = "data/seed_data_fix/test.json"
output_path = "appendix_data/gpt_infer.json"

target_data, input_data = json2list(input_path)
batch_processor = OpenAIBatchProcessor('gpt-4o-mini')
res = batch_processor.process_batch_fun(input_data)
for i,target in enumerate(target_data):
    res[i]['target'] = target

write_json(res, output_path)

'''
python appendix_data/batch_inference_gpt.py
'''
