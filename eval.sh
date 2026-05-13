CUDA_VISIBLE_DEVICES=0 \
python -u eval.py \
    -env_name "AmidarNoFrameskip-v4" \
    -run_name "Amidar-v4_seed_1_weight_type_3" \
    -seed 1 \
    -config_path "config_files/STORM.yaml"