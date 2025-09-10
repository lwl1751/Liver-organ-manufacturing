import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def evaluate_to_dataframe(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_entities = [
        "Seeded cell density", "Material in cell culture", "Concentration of material",
        "Chip material", "Cross-linking agent", "Pore size", "Diameter",
        "Manufacturing method", "Perfusion rate", "Channel width", "Channel height"
    ]
    
    metrics = {
        "few-shot": {entity: {"TP": 0, "FP": 0, "FN": 0} for entity in target_entities},
        "zero-shot": {entity: {"TP": 0, "FP": 0, "FN": 0} for entity in target_entities}
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 统一格式为列表
        data = [data] if isinstance(data, dict) else data
    except Exception as e:
        raise ValueError(f"读取JSON文件失败：{str(e)}")
    
    # 遍历每个样本，累计各实体的TP、FP、FN
    for sample in data:
        for shot_type in ["few-shot", "zero-shot"]:
            if shot_type not in sample:
                continue
                
            shot_result = sample[shot_type]
            tp_list = shot_result.get("TP", [])
            fp_list = shot_result.get("FP", [])
            fn_list = shot_result.get("FN", [])
            
            for entity in target_entities:
                metrics[shot_type][entity]["TP"] += tp_list.count(entity)
                metrics[shot_type][entity]["FP"] += fp_list.count(entity)
                metrics[shot_type][entity]["FN"] += fn_list.count(entity)
    
    # 为两种评估类型生成数据框
    result_dfs = {}
    summary_dfs = {}
    
    for shot_type in ["few-shot", "zero-shot"]:
        # 准备数据
        data_rows = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for entity in target_entities:
            tp = metrics[shot_type][entity]["TP"]
            fp = metrics[shot_type][entity]["FP"]
            fn = metrics[shot_type][entity]["FN"]
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # 计算Precision, Recall和F1
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = round(tp / (tp + fp), 3)
                
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = round(tp / (tp + fn), 3)
                
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = round(2 * (precision * recall) / (precision + recall), 3)
            
            data_rows.append({
                "实体类型": entity,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })
        
        df = pd.DataFrame(data_rows)
        macro_f1 = round(df["F1"].mean(), 3)
        if total_tp == 0:
            micro_precision = 0.0
            micro_recall = 0.0
            micro_f1 = 0.0
        else:
            micro_precision = round(total_tp / (total_tp + total_fp), 3) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = round(total_tp / (total_tp + total_fn), 3) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = round(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall), 3) if (micro_precision + micro_recall) > 0 else 0.0
        
        summary_df = pd.DataFrame({
            "指标": ["总TP", "总FP", "总FN", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"],
            "值": [total_tp, total_fp, total_fn, micro_precision, micro_recall, micro_f1, macro_f1]
        })
        
        result_dfs[shot_type] = df
        summary_dfs[shot_type] = summary_df
    
    return (result_dfs["few-shot"], result_dfs["zero-shot"], 
            summary_dfs["few-shot"], summary_dfs["zero-shot"])



if __name__ == "__main__":
    model = ['glm', 'gpt', 'llama']
    for m in model:
        json_file_path = f"data/{m}_eval_fix.json"
        print("=" * 80)
        print(f"model: {m}")
        print("\n" + "=" * 80)
        try:
            # 获取统计数据框
            few_shot_df, zero_shot_df, few_shot_summary, zero_shot_summary = evaluate_to_dataframe(json_file_path)
            
            # 显示结果
            print("=" * 80)
            print("Few-shot 评估结果:")
            print("-" * 80)
            print(few_shot_df.to_string(index=False))
            
            print("\n" + "=" * 80)
            print("Few-shot 汇总统计:")
            print("-" * 80)
            print(few_shot_summary.to_string(index=False))
            
            print("\n" + "=" * 80)
            print("Zero-shot 评估结果:")
            print("-" * 80)
            print(zero_shot_df.to_string(index=False))
            
            print("\n" + "=" * 80)
            print("Zero-shot 汇总统计:")
            print("-" * 80)
            print(zero_shot_summary.to_string(index=False))
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
