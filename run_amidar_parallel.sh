#!/bin/bash
set -euo pipefail

# --- 用户配置区 ---
ALLOWED_GPUS="1 2 3"
MAX_PARALLEL_JOBS=6
IGNORED_THREADS=""
GPU_SELECTION_MODE="PROCESS" # PROCESS | ROUND_ROBIN

# 可按需调整
CONFIG_PATH="config_files/STORM.yaml"
ENV_NAME="AmidarNoFrameskip-v4"
TRAJECTORY_PATH="D_TRAJ/AmidarNoFrameskip-v4.pkl"

# 三组实验 × seed 1~5
EXPERIMENTS=(
  # 1) 不加权原版（关闭 UWL）
  "seed=1,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=2,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=3,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=4,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"
  "seed=5,use_uwl=0,weight_type=0,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=baseline"

  # 2) IUPOM 单模型
  "seed=1,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  "seed=2,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  "seed=3,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  "seed=4,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"
  "seed=5,use_uwl=1,weight_type=3,uncertainty_mode=single,ensemble_size=1,decomposed_fusion_type=post_tanh,tag=iupom"

  # 3) Ensemble decomposed + post_tanh
  "seed=1,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=5,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=2,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=5,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=3,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=5,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=4,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=5,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
  "seed=5,use_uwl=1,weight_type=3,uncertainty_mode=ensemble_decomposed,ensemble_size=5,decomposed_fusion_type=post_tanh,tag=ens_post_tanh"
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
    declare -A IGNORED_MAP
    for pair in $IGNORED_THREADS; do
      local index="${pair%:*}"
      local threads="${pair#*:}"
      IGNORED_MAP["$index"]=$threads
    done

    local nvitop_output
    nvitop_output="$(nvitop 2>/dev/null || true)"

    for gpu_id in $allowed_list; do
      local ignored_threads=${IGNORED_MAP["$gpu_id"]:-0}
      local count
      count="$(echo "$nvitop_output" | grep -E "^[[:space:]]*\\|[[:space:]]*$gpu_id[[:space:]]*\\|" | wc -l)"
      local current_load=$(( count - ignored_threads ))
      if [[ $current_load -lt 0 ]]; then current_load=0; fi

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
