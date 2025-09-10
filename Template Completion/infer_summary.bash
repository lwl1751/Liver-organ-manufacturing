#!/bin/bash

total_length=9908  # 数据总长度
num_tasks=10       # 任务数量

# 每个任务的分配长度
chunk_size=$((total_length / num_tasks))
txt_path="txt_data/liver_txt"
input_path="data.json"
output_path=""
record_path=""
api_keys=(

)

for ((i=0; i<num_tasks; i++)); do
  start_index=$((i * chunk_size))
  end_index=$((start_index + chunk_size - 1))

  if [ $i -eq $((num_tasks - 1)) ]; then
    end_index=$((total_length - 1))
  fi

  api_key=${api_keys[$i]}

  task_output_path="${output_path}/task_$((i+1)).json"
  task_record_path="${record_path}/task_$((i+1)).log"

  echo "Running: python summary.py --txt_path $txt_path --start_index $start_index --end_index $end_index --input_path $input_path --output_path $task_output_path --record_path $task_record_path --api_key $api_key"
  python summary.py --txt_path $txt_path --start_index $start_index --end_index $end_index --input_path $input_path --output_path $task_output_path --record_path $task_record_path --api_key $api_key &
done

wait

echo "All tasks have finished."

