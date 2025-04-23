import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def extract_entities(data, method):
    """提取指定方法下的所有实体数据"""
    extracted = defaultdict(set)

    for entry in data:
        if method not in entry:
            continue
        method_data = entry[method]

        # 直接解析 Cell Culture Conditions
        for key in ["Cell type", "Seeded cell density"]:
            extracted[key].update(method_data.get("Cell Culture Conditions", {}).get(key, []))

        # 解析 Scaffold and Chip Materials
        for key in ["Material in cell culture", "Concentration of material", "Chip material"]:
            extracted[key].update(method_data.get("Scaffold and Chip Materials", {}).get(key, []))

        # 解析 Culture Methods 内的所有方法
        culture_methods = method_data.get("Culture Methods", {})
        for method_type, values in culture_methods.items():
            for key in [
                "Perfusion rate", "Channel width", "Channel height",
                "Cross-linking agent", "Pore size", "Diameter", "Manufacturing method"
            ]:
                if key in values:
                    extracted[key].update(values[key])

    return extracted

def compute_f1_score(targets, predictions):
    """计算每个实体类别的 F1 分数"""
    results = {}
    all_entities = set(targets.keys()) | set(predictions.keys())

    for entity in all_entities:
        y_true = set(targets.get(entity, []))  # 真实值
        y_pred = set(predictions.get(entity, []))  # 预测值

        # 计算 TP, FP, FN
        tp = len(y_true & y_pred)  # 真正例
        fp = len(y_pred - y_true)  # 假正例
        fn = len(y_true - y_pred)  # 假负例

        # 计算 Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[entity] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3)
        }
    
    return results

# 读取JSON文件
data_path = "Template Completion/data/eval_data/re_test_52.json"
with open(data_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 提取实体数据
targets = extract_entities(json_data, "target")
fill_in_blank_preds = extract_entities(json_data, "fill_in_blank")
gpt_all_preds = extract_entities(json_data, "gpt_all")

# 计算 F1 分数
fill_in_blank_f1 = compute_f1_score(targets, fill_in_blank_preds)
gpt_all_f1 = compute_f1_score(targets, gpt_all_preds)

# 保存结果
output = {"fill_in_blank": fill_in_blank_f1, "gpt_all": gpt_all_f1}
output_path = "/home/wangld/liangwenliang/器官制造/relation_extraction/data/eval_data/f1_scores.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"F1 scores saved to {output_path}")
