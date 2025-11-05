import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
import torch
import wandb

def train(args):
    mode = "online"
    if args.no_wandb:
        mode = "disabled"
    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(gpu_world_size)
    is_distributed = gpu_world_size > 1
    if is_distributed:
        gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # gpu_global_rank = int(os.getenv("RANK", "0"))
        # set device to the local rank
        args.sim_device = f"cuda:{gpu_local_rank}"
        args.rl_device = f"cuda:{gpu_local_rank}"
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=args.log_root)
    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    wandb.init(
                project='hamp_terrain',
                group=args.task,
                name=args.task + '_' + train_cfg.runner.run_name, 
                config={
                        "env": class_to_dict(env_cfg),
                        "train": class_to_dict(train_cfg)
                    },
                dir=ppo_runner.log_dir,
                mode=mode,
                sync_tensorboard=True)  
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
