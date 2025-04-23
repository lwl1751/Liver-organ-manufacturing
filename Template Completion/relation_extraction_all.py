import argparse
from utils import *
import os
import logging
import sys

# 保存输出
def write_json(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(args):
    with open(args.input_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        datas = datas[args.start_index : args.end_index + 1]
        logging.info(f'要处理的doi数量:{len(datas)}')
    # 批量处理
    for data in datas:
        with open(args.output_path, 'r', encoding='utf-8') as f:
                cur_data = json.load(f)
                cur_doi_ls = [d['doi'] for d in cur_data]
        try:
            # 判断doi文件有没有被处理过
            doi = data['doi']
            if doi in cur_doi_ls:
                logging.info('*************************************')
                logging.info(f'已经处理doi:{doi}')
                continue

            txt_path = os.path.join(args.txt_path, doi.replace('/','_')+'.txt')
            if os.path.isfile(txt_path):
                text_data = read_txt(txt_path)
                text_data_list = split_text_into_segments(text_data)
            else:
                logging.info('*************************************')
                logging.info(f'doi:{doi},不存在该文件路径：{txt_path}')
                continue
                
            logging.info('*************************************')
            logging.info(f'doi:{doi},共有{len(text_data_list)}个段落')

            # 多阶段CoT
            extract_workfolw = Extract_Workflow(text_data_list, args.api_key, start=True)
            extract_workfolw.method = data['method']
            extract_workfolw.cell_type = data['cell_type']
            extract_workfolw.entity_requirements = method_2_needed_para[extract_workfolw.method] if extract_workfolw.method else None
            extract_workfolw.part_5()

            # 读取文件
            new_data = {'doi': doi, 'method': extract_workfolw.method, 'output': extract_workfolw.reflect_res}
            cur_data.append(new_data)

            # 写入文件
            write_json(args.output_path, cur_data)
                
        except Exception as e:
            logging.error(f"Unexpected error processing data for doi {doi}: {e}")
            write_json(args.output_path, cur_data)
            continue

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, help='Path to the txt folder.')
    parser.add_argument('--start_index', type=int, help='Start index of the json file.')
    parser.add_argument('--end_index', type=int, help='End index of the json file.')
    parser.add_argument('--input_path', type=str, help='Path to the json file.')
    parser.add_argument('--output_path', type=str, help='Path to the output file.')
    parser.add_argument('--record_path', type=str, help='Path to the record file.')
    parser.add_argument('--api_key', type=str, help='API key for authentication.')
    args = parser.parse_args()

    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(args.record_path, mode='a', encoding='utf-8'),  # 文件记录
            logging.StreamHandler(sys.stdout)  # 控制台输出
        ]
    )
    class LoggerWriter:
        def __init__(self, level):
            self.level = level

        def write(self, message):
            if message.strip():  # 忽略空白行
                self.level(message)

        def flush(self):
            pass  # 不需要手动刷新

    # 将 print 重定向到 logging.info
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)
    logging.info(f"txt_path: {args.txt_path}")
    logging.info(f"start_index: {args.start_index}")
    logging.info(f"end_index: {args.end_index}")
    logging.info(f"input_path: {args.input_path}")
    logging.info(f"output_path: {args.output_path}")
    logging.info(f"api_key: {args.api_key}")
    logging.info(f"record_path: {args.record_path}")

    if not os.path.exists(args.output_path):
        with open(args.output_path, 'w') as file:
            json.dump([], file)

    # 调用主函数
    main(args)

