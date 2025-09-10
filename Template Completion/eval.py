import pandas as pd
import numpy as np

def calculate_metrics(tp, fp, fn):
    """计算精确率、召回率和F1分数，处理除以零的情况"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def calculate_entity_stats(file_path):
    # 定义实体的指定顺序
    ENTITIES = [
        "Cell type", "Seeded cell density", "Material in cell culture",
        "Concentration of material", "Chip material", "Perfusion rate",
        "Channel width", "Channel height", "Cross-linking agent",
        "Pore size", "Diameter", "Manufacturing method"
    ]
    
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取文件，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    numeric_columns = ['raw_TP', 'raw_FP', 'raw_FN', 'TP', 'FP', 'FN']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    entity_stats = df.groupby('entity')[numeric_columns].sum().reset_index()
    
    filtered_stats = entity_stats[entity_stats['entity'].isin(ENTITIES)].copy()
    metrics = filtered_stats.apply(
        lambda row: calculate_metrics(row['raw_TP'], row['raw_FP'], row['raw_FN']), axis=1
    )
    filtered_stats[['raw_Precision', 'raw_Recall', 'raw_F1']] = pd.DataFrame(
        metrics.tolist(), index=filtered_stats.index
    )
    metrics = filtered_stats.apply(
        lambda row: calculate_metrics(row['TP'], row['FP'], row['FN']), axis=1
    )
    filtered_stats[['Precision', 'Recall', 'F1']] = pd.DataFrame(
        metrics.tolist(), index=filtered_stats.index
    )
    
    # 计算Macro指标 (每个类别的指标取平均)
    raw_macro_precision = filtered_stats['raw_Precision'].mean()
    raw_macro_recall = filtered_stats['raw_Recall'].mean()
    raw_macro_f1 = filtered_stats['raw_F1'].mean()
    
    macro_precision = filtered_stats['Precision'].mean()
    macro_recall = filtered_stats['Recall'].mean()
    macro_f1 = filtered_stats['F1'].mean()
    
    # 计算Micro指标 (先汇总所有类别的TP, FP, FN，再计算指标)
    total_raw_tp = filtered_stats['raw_TP'].sum()
    total_raw_fp = filtered_stats['raw_FP'].sum()
    total_raw_fn = filtered_stats['raw_FN'].sum()
    raw_micro_precision, raw_micro_recall, raw_micro_f1 = calculate_metrics(
        total_raw_tp, total_raw_fp, total_raw_fn
    )
    
    total_tp = filtered_stats['TP'].sum()
    total_fp = filtered_stats['FP'].sum()
    total_fn = filtered_stats['FN'].sum()
    micro_precision, micro_recall, micro_f1 = calculate_metrics(
        total_tp, total_fp, total_fn
    )
    
    # 按指定顺序排序
    filtered_stats['entity'] = pd.Categorical(
        filtered_stats['entity'],
        categories=ENTITIES,
        ordered=True
    )
    filtered_stats = filtered_stats.sort_values('entity').reset_index(drop=True)
    
    # 显示统计结果
    print("\n不同entity的各项指标统计 (按指定顺序):")
    # 只显示关键指标列，便于阅读
    print(filtered_stats[['entity','raw_TP','raw_FP','raw_FN','raw_Precision', 'raw_Recall', 'raw_F1', 
                         'TP','FP','FN','Precision', 'Recall', 'F1']].round(3))
    
    # 显示汇总指标
    print("\n=== 汇总指标统计 ===")
    print(f"raw_* 组 - Macro Precision: {raw_macro_precision:.3f}, Macro Recall: {raw_macro_recall:.3f}, Macro F1: {raw_macro_f1:.3f}")
    print(f"raw_* 组 - Micro Precision: {raw_micro_precision:.3f}, Micro Recall: {raw_micro_recall:.3f}, Micro F1: {raw_micro_f1:.3f}")
    print(f"pipeline 组 - Macro Precision: {macro_precision:.3f}, Macro Recall: {macro_recall:.3f}, Macro F1: {macro_f1:.3f}")
    print(f"pipeline 组 - Micro Precision: {micro_precision:.3f}, Micro Recall: {micro_recall:.3f}, Micro F1: {micro_f1:.3f}")
    
    
    # 返回所有结果
    return {
        'entity_stats': filtered_stats,
        'raw_macro_precision': raw_macro_precision,
        'raw_macro_recall': raw_macro_recall,
        'raw_macro_f1': raw_macro_f1,
        'raw_micro_precision': raw_micro_precision,
        'raw_micro_recall': raw_micro_recall,
        'raw_micro_f1': raw_micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

if __name__ == "__main__":
    excel_file_path = "data/test_data.xlsx"
    calculate_entity_stats(excel_file_path)
