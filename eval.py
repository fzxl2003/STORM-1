import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss


def infer_uncertainty_mode_from_run_name(run_name):
    # Match naming tags created by run_amidar_parallel.sh:
    #   *_baseline / *_iupom -> single
    #   *_ens_post_tanh      -> ensemble_decomposed
    lower_name = run_name.lower()
    if "ens_post_tanh" in lower_name:
        return "ensemble_decomposed"
    return "single"


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size,seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs,seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size,seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, seed=0):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs,seed=seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("--uncertainty_mode", type=str, choices=["single", "ensemble_decomposed"], default=None)
    # parser.add_argument("--ensemblesize",type=int,default=None)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    print("seed", args.seed)
    conf.defrost()
    conf.BasicSettings.Seed = args.seed
    if args.uncertainty_mode is not None:
        conf.Models.Agent.UncertaintyMode = args.uncertainty_mode
    else:
        conf.Models.Agent.UncertaintyMode = infer_uncertainty_mode_from_run_name(args.run_name)
    if conf.Models.Agent.UncertaintyMode != "ensemble_decomposed":
        conf.Models.Agent.EnsembleSize = 1
    else:
        conf.Models.Agent.EnsembleSize = 4

    conf.freeze()
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=conf.BasicSettings.Seed)
    action_dim = dummy_env.action_space.n
    world_models = train.build_world_models(conf, action_dim)
    world_model_list = [world_models] if conf.Models.Agent.UncertaintyMode == "single" else list(world_models)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt_ensemble/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    # steps = steps[-3:] 
    print(steps)
    results = []
    for step in tqdm(steps):
        wm_ckpt = torch.load(f"{root_path}/world_model_{step}.pth")
        if isinstance(wm_ckpt, dict) and "world_models" in wm_ckpt:
            if conf.Models.Agent.UncertaintyMode != "ensemble_decomposed":
                raise ValueError(
                    "Found ensemble world-model checkpoint, but config UncertaintyMode is not ensemble_decomposed."
                )
            if len(wm_ckpt["world_models"]) != conf.Models.Agent.EnsembleSize :
                raise ValueError(
                    f"Ensemble size mismatch: ckpt has {len(wm_ckpt['world_models'])}, "
                    f"config has {conf.Models.Agent.EnsembleSize}."
                )
            # Load all ensemble members for compatibility with new checkpoints.
            for wm, wm_state in zip(world_models, wm_ckpt["world_models"]):
                wm.load_state_dict(wm_state)
        else:
            # Backward-compatible single world-model checkpoint loading.
            world_models.load_state_dict(wm_ckpt)
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # Eval each ensemble member independently and keep the best score.
        member_scores = []
        for wm in world_model_list:
            episode_avg_return = eval_episodes(
                num_episode=20,
                env_name=args.env_name,
                num_envs=5,
                max_steps=conf.JointTrainAgent.SampleMaxSteps,
                image_size=conf.BasicSettings.ImageSize,
                world_model=wm,
                agent=agent,
                seed=args.seed
            )
            member_scores.append(episode_avg_return)
        episode_avg_return = max(member_scores)
        print("Best member reward: " + colorama.Fore.YELLOW + f"{episode_avg_return}" + colorama.Style.RESET_ALL)
        results.append([step, episode_avg_return])
    path=os.path.join("eval_result", f"{args.run_name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"eval_result/{args.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")
