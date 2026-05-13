## Training and Evaluation Instructions

1.  **Configure and activate the Anaconda virtual environment `storm_iupom`**.

      - To configure the environment:

        ```bash
        conda env create -f environment.yml
        ```

      - To activate the environment:

        ```bash
        conda activate storm_iupom
        ```

-----

2.  **Train the agent.**

    ```shell
    bash train.sh
    ```

    The `train.sh` file controls the environment and the run name for a training process.

    ```shell
    CUDA_VISIBLE_DEVICES=0 \
    python  -u train.py \
            -seed 1 \
            -config_path "config_files/STORM.yaml" \
            -env_name "AmidarNoFrameskip-v4" \
            -trajectory_path "D_TRAJ/AmidarNoFrameskip-v4.pkl" \
            --use_uwl 1 \
            --weight_type 3 \
            -n "Amidar-v4_seed_1_weight_type_3"
    ```
    **Required Parameters:**

      - `-seed`: The random seed for the training process.

      - `-config_path`: Points to the YAML file controlling model hyperparameters .

      - `-env_name`: The name of the Atari environment to train on, e.g., "AmidarNoFrameskip-v4".
      - `--use_uwl`: A flag (0 or 1) to enable or disable the Uncertainty Weighting Loss (iupom) mechanism.

      - `--weight_type`: Specifies the type of weighting strategy used in the experiment,0 for no weighting, 1 for only actor weighting, 2 for only value weighting, and 3 for both actor and value weighting.

      - `-n`: The name for the TensorBoard logger and checkpoint folder.
      




-----

3.  **Evaluate the agent.** The evaluation results will be in a CSV file located in the `eval_result` folder.

    ```shell
    bash eval.sh
    ```

    The `eval.sh` file controls the environment and the run name when testing an agent.

    ```shell
    export CUDA_VISIBLE_DEVICES=1
    python -u eval.py \
    -env_name "AmidarNoFrameskip-v4" \
    -run_name "Amidar-v4_seed_1_weight_type_3" \
    -seed 1 \
    -config_path "config_files/STORM.yaml"
    ```

    The `-run_name` option is the same as the `-n` option in `train.sh`. It should be kept the same as in the training script.