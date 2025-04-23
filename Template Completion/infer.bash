#!/bin/bash

# 定义总数据长度和任务数量
total_length=9968  # 数据总长度
num_tasks=30       # 任务数量

# 每个任务的分配长度
chunk_size=$((total_length / num_tasks))

# 定义其他固定参数
txt_path="txt_final"
input_path=""
output_path=""
record_path=""
api_keys=(
  
)

# 循环生成参数并运行脚本
for ((i=0; i<num_tasks; i++)); do
  # 计算 start_index 和 end_index
  start_index=$((i * chunk_size))
  end_index=$((start_index + chunk_size - 1))

  # 如果是最后一个任务，确保 end_index 包括所有剩余数据
  if [ $i -eq $((num_tasks - 1)) ]; then
    end_index=$((total_length - 1))
  fi

  # 获取当前任务的 API Key
  api_key=${api_keys[$i]}

  # 构造输出路径和记录路径
  task_output_path="${output_path}/task_$((i+1)).json"
  task_record_path="${record_path}/task_$((i+1)).log"

  # 运行 Python 脚本
  echo "Running: python relation_extraction_all.py --txt_path $txt_path --start_index $start_index --end_index $end_index --input_path $input_path --output_path $task_output_path --record_path $task_record_path --api_key $api_key"
  python relation_extraction_all.py --txt_path $txt_path --start_index $start_index --end_index $end_index --input_path $input_path --output_path $task_output_path --record_path $task_record_path --api_key $api_key &
done

# 等待所有脚本完成
wait

echo "All tasks have finished."

