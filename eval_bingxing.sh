#!/bin/bash
set -euo pipefail

# 并行评测脚本（适配当前 eval.py 与 run_amidar_parallel.sh 的命名）

CKPT_ROOT="ckpt"
CONFIG_PATH="config_files/STORM.yaml"
MAX_JOBS=10
ALLOWED_GPUS="0 1 2 3"
SLEEP_SECONDS=10

GPU_ARRAY=($ALLOWED_GPUS)
GPU_COUNT=${#GPU_ARRAY[@]}
GPU_INDEX=0

ckpt_path=()
while IFS= read -r line; do
    ckpt_path+=("$line")
done < <(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

for path in "${ckpt_path[@]}"; do
    # 提取环境名称（_seed 前）
    env_name=$(echo "$path" | awk -F'_seed' '{print $1}')
    # 提取 seed（_seed_ 后）
    seed=$(echo "$path" | awk -F'_seed_' '{print $2}' | awk -F'_' '{print $1}')

    # 根据 run_amidar_parallel.sh 的关键词推断 uncertainty_mode
    uncertainty_mode="single"
    if [[ "$path" == *"ens_post_tanh"* ]]; then
        uncertainty_mode="ensemble_decomposed"
    fi

    current_gpu=${GPU_ARRAY[$GPU_INDEX]}
    GPU_INDEX=$(( (GPU_INDEX + 1) % GPU_COUNT ))

    echo "Evaluating run=${path}, env=${env_name}, seed=${seed}, mode=${uncertainty_mode}, gpu=${current_gpu}"
    CUDA_VISIBLE_DEVICES=$current_gpu \
    python -u eval.py \
        -env_name "${env_name}" \
        -run_name "${path}" \
        -seed "${seed}" \
        -config_path "${CONFIG_PATH}" \
        --uncertainty_mode "${uncertainty_mode}" &

    while true; do
        running_jobs=$(jobs -rp | wc -l)
        if (( running_jobs < MAX_JOBS )); then
            break
        fi
        sleep "$SLEEP_SECONDS"
    done
done

wait
echo "All eval jobs finished."
