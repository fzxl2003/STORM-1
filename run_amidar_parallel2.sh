#!/bin/bash
set -euo pipefail
sleep 54000 # 等待 20 小时，确保之前的实验完全结束，GPU 资源得到释放
# --- 用户配置区 ---
ALLOWED_GPUS="0 1 2 4 5 6 7"
MAX_PARALLEL_JOBS=7
IGNORED_THREADS=""
GPU_SELECTION_MODE="PROCESS" # PROCESS | ROUND_ROBIN

# 可按需调整
CONFIG_PATH="config_files/STORM.yaml"
ENV_NAME="GopherNoFrameskip-v4"
TRAJECTORY_PATH="D_TRAJ/GopherNoFrameskip-v4.pkl"

# 三组实验 × seed 1~5
EXPERIMENTS=(
  
  # 2) IUPOM 单模型
  # "seed=1,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  # "seed=2,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  # "seed=3,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  # "seed=4,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  # "seed=5,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"

  # 3) Ensemble decomposed + post_tanh
  "seed=1,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=4,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=2,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=4,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=3,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=4,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=4,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=4,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=5,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=4,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"

# 1) 不加权原版（关闭 UWL）
  "seed=1,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=2,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=3,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=4,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=5,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"





)

NEXT_GPU_INDEX=0
ALLOWED_GPUS_ARRAY=($ALLOWED_GPUS)
NUM_GPUS=${#ALLOWED_GPUS_ARRAY[@]}

select_best_gpu() {
  local allowed_list="$1"
  local mode="$2"
  local best_gpu=""
  local min_load=-1

  if [[ "$mode" == "PROCESS" ]]; then
        
        # 声明一个关联数组用于存储无关线程数
        declare -A IGNORED_MAP
        for pair in $IGNORED_THREADS; do
            local index="${pair%:*}"
            local threads="${pair#*:}"
            IGNORED_MAP["$index"]=$threads
        done
        
        # 模式 1: 根据进程数选择 (使用您指定的 nvitop/grep 命令，并扣除动态无关线程数)
        
        # 运行 nvitop 一次并存储结果
        local nvitop_output=$(nvitop 2>/dev/null)

        for gpu_id in $allowed_list; do
            local current_load=0 
            
            # 从关联数组中获取当前 GPU 需要扣除的线程数，如果没有指定则默认为 0
            local ignored_threads=${IGNORED_MAP["$gpu_id"]:-0}

            # 使用您指定的 grep 方式获取进程数
            # 注意: nvitop 输出的格式是 "| 1 | ... | C | N/A | ..."
            # 为了确保精确匹配并简化，使用 ERE 匹配行首的 "| GPU_ID"
            # local count=$(nvitop | grep -E "* ${gpu_id} .*     N/A.*N/A"| wc -l)
            local count=$(nvitop | grep -E "* ${gpu_id} .*pyh"| wc -l)

            # echo ${gpu_id}:${count}
            # 扣除无关线程数
            current_load=$(( count - ignored_threads ))
            
            # 确保进程数不为负
            if [[ $current_load -lt 0 ]]; then
                current_load=0
            fi

            if [[ "$min_load" -eq -1 || "$current_load" -lt "$min_load" ]]; then
                min_load="$current_load"
                best_gpu="$gpu_id"
            fi
        done
        
  else
    best_gpu="$(echo "$allowed_list" | awk '{print $1}')"
  fi

  if [[ -z "$best_gpu" ]]; then
    best_gpu="$(echo "$allowed_list" | awk '{print $1}')"
  fi
  echo "$best_gpu"
}

run_experiment() {
  local param_string="$1"
  local cuda_index="$2"

  local seed="" use_uwl=0 weight_type="" uncertainty_mode="single" ensemble_size=1 decomposed_fusion_type="post_tanh" tag=""
  for param in ${param_string//,/ }; do
    eval local "$param"
  done

  local run_name="${ENV_NAME}_seed_${seed}_${tag}"
  echo ">>> GPU ${cuda_index} | ${run_name} | uwl=${use_uwl}, weight_type=${weight_type}, mode=${uncertainty_mode}, K=${ensemble_size}, fusion=${decomposed_fusion_type}"

  CUDA_VISIBLE_DEVICES=${cuda_index} \
  python -u train.py \
    -n "${run_name}" \
    -seed "${seed}" \
    -config_path "${CONFIG_PATH}" \
    -env_name "${ENV_NAME}" \
    -trajectory_path "${TRAJECTORY_PATH}" \
    --use_uwl "${use_uwl}" \
    --weight_type "${weight_type}" \
    --uncertainty_mode "${uncertainty_mode}" \
    --ensemble_size "${ensemble_size}" \
    --decomposed_fusion_type "${decomposed_fusion_type}" &
}

echo "Starting experiments with MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} on GPUs: ${ALLOWED_GPUS} (Mode: ${GPU_SELECTION_MODE})"
for line in "${EXPERIMENTS[@]}"; do
  current_jobs=$(jobs -p | wc -l)
  while [[ $current_jobs -ge $MAX_PARALLEL_JOBS ]]; do
    echo "Waiting: Max parallel jobs (${MAX_PARALLEL_JOBS}) reached..."
    wait -n
    current_jobs=$(jobs -p | wc -l)
  done

  if [[ "$GPU_SELECTION_MODE" == "ROUND_ROBIN" ]]; then
    chosen_gpu=${ALLOWED_GPUS_ARRAY[$NEXT_GPU_INDEX]}
    NEXT_GPU_INDEX=$(( (NEXT_GPU_INDEX + 1) % NUM_GPUS ))
  else
    chosen_gpu=$(select_best_gpu "$ALLOWED_GPUS" "$GPU_SELECTION_MODE")
  fi

  run_experiment "$line" "$chosen_gpu"
  sleep 15
done

echo "All experiments started. Waiting for all background jobs..."
wait
echo "All experiments finished successfully."
