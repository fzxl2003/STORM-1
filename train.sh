CUDA_VISIBLE_DEVICES=0 \
python -u train.py \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "AmidarNoFrameskip-v4" \
    --use_uwl 1 \
    --weight_type 3 \
    -n "Amidar-v4_seed_1_weight_type_3"