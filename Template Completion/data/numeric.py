import json
import re
from typing import Optional, Tuple

def extract_seeded_cell_density(value: str) -> Optional[Tuple[float, str]]:
    # 处理范围格式，如 "15,000–20,000 cells/mL" 或 "15,000-20,000 cells/mL"
    range_match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*[–-]\s*(\d{1,3}(?:,\d{3})*)\s*cells/mL", value)
    if range_match:
        min_value = float(range_match.group(1).replace(',', ''))
        max_value = float(range_match.group(2).replace(',', ''))
        return (min_value + max_value) / 2, "cells/mL"  # 返回均值

    # 处理标准科学计数法，如 "3.5 × 10^6 cells/mL"
    sci_match = re.search(r"([\d.,]+)\s*[×x]\s*10\^?(\d+)\s*cells/mL", value)
    if sci_match:
        base = float(sci_match.group(1).replace(',', ''))
        exponent = int(sci_match.group(2))
        return base * (10 ** exponent), "cells/mL"

    # 处理单个数值，如 "20,000 cells/mL"
    single_match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*cells/mL", value)
    if single_match:
        return float(single_match.group(1).replace(',', '')), "cells/mL"

    return None

import re
from typing import Optional, Tuple

def extract_dimension_data(value: str, unit: str) -> Optional[Tuple[float, str]]:
    # 处理范围格式，如 "50–100 µm" 或 "50-100 µm"
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*" + re.escape(unit), value)
    if range_match:
        min_value = float(range_match.group(1))
        max_value = float(range_match.group(2))
        return (min_value + max_value) / 2, unit  # 返回均值

    # 去掉 "±" 及其后面的误差范围，如 "100 ± 10 µm" 变成 "100 µm"
    if "±" in value:
        value = value.split("±")[0].strip()
    
    # 处理 "≈" 这样的约等于符号，如 "≈50 µm" 变成 "50 µm"
    if "≈" in value:
        value = value.replace("≈", "").strip()

    # 处理单个数值，如 "50 µm"
    match = re.search(r"(\d+(?:\.\d+)?)\s*" + re.escape(unit), value)
    if match:
        return float(match.group(1)), unit

    return None

def process_json_data(input_data):
    for entry in input_data:
        output = entry["output"]
        
        # Process Seeded cell density
        densities = output.get("Cell Culture Conditions", {}).get("Seeded cell density", [])
        processed_densities = [extract_seeded_cell_density(d) for d in densities]
        output["Cell Culture Conditions"]["Seeded cell density"] = [d for d in processed_densities if d]
        
        # Process numerical values for dimensions and perfusion rate
        culture_methods = output.get("Culture Methods", {})
        for method, details in culture_methods.items():
            for key in ["Pore size", "Diameter", "Channel width", "Channel height", "Perfusion rate"]:
                if key in details:
                    processed_values = [extract_dimension_data(v, "µm" if key != "Perfusion rate" else "µL/min") for v in details[key]]
                    culture_methods[method][key] = [v for v in processed_values if v]
    
    return input_data

# 读取 JSON 文件
with open("final_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理数据
processed_data = process_json_data(data)

# 保存 JSON 文件
with open("numeric.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)
