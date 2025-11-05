# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from rsl_rl.algorithms.amp_ppo_multi import AMPPPOMulti
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_depth import ActorCriticDepth
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator_multi import AMPDiscriminatorMulti
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
from legged_gym.utils.utils import Normalizer

class AMPOnPolicyRunnerMulti:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        self.obs_history_len = self.env.obs_history_len
        self.use_depth = False
        if self.env.cfg.depth.use_camera or self.env.cfg.depth.warp_camera:
            self.use_depth = True
            self.depth_shape = self.env.cfg.depth.resized

        self.use_amp = self.alg_cfg["use_amp"]
            
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        num_actor_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        history_dim=self.obs_history_len * (num_actor_obs),
                                                        # history_dim=self.obs_history_len * num_actor_obs,
                                                        **self.policy_cfg).to(self.device)
        # prepare for AMP
        if self.use_amp:
            self.amp_loader_type = self.alg_cfg['amp_loader_type']
            self.amp_loader_class_name = self.alg_cfg['amp_loader_class_name']; del self.alg_cfg['amp_loader_class_name']
            if 'lafan_16dof' in self.alg_cfg['amp_loader_type']:
                self.amp_indices = [i for i in range(16)]
                self.num_amp_obs = self.env.num_amp_obs

            amp_loader_class = eval(self.amp_loader_class_name)
            amp_data = amp_loader_class(
                device, time_between_frames=self.env.dt, preload_transitions=True,
                num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
                motion_dir=self.env.amp_motion_files)
            amp_normalizer = Normalizer(self.num_amp_obs)

            discriminator = AMPDiscriminatorMulti(
                self.num_amp_obs,
                train_cfg['runner']['amp_reward_coef'],
                train_cfg['runner']['amp_discr_hidden_dims'], device,
                train_cfg['runner']['num_amp_frames'],
                train_cfg['runner']['amp_task_reward_lerp'],
                train_cfg['runner']['use_lerp'],
                ).to(self.device)
        else:
            amp_data = None
            amp_normalizer = None
            discriminator = None

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: AMPPPOMulti = alg_class(actor_critic, discriminator, amp_data, amp_normalizer,
                                             use_depth=self.use_depth, device=self.device, num_amp_frames=train_cfg['runner']['num_amp_frames'], **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_amp_frames = train_cfg['runner']['num_amp_frames']

        # init storage and model
        self.alg.init_storage(self.env.num_envs, 
                              self.num_steps_per_env, 
                              [num_actor_obs], 
                              [num_critic_obs], 
                              [self.env.num_actions], 
                              self.obs_history_len, 
                              self.env.num_obs,
                              depth_shape=self.depth_shape if self.use_depth else None,
                              depth_buffer_len=self.env.cfg.depth.buffer_len if self.use_depth else None)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        if self.use_amp:
            amp_obs = self.env.get_amp_observations(); amp_obs = amp_obs[:, self.amp_indices]
            amp_obs = amp_obs.to(self.device)
            self.alg.discriminator.train()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.obs_history_len, self.env.num_obs), device=self.device)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)
        if self.use_amp:
            self.amp_obs_frames = torch.zeros(size=(self.env.num_envs, self.num_amp_frames, self.env.num_amp_obs), device=self.device)
            self.amp_obs_frames = torch.concat((self.amp_obs_frames[:, 1:], amp_obs.unsqueeze(1)), dim=1)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        discrewbuffer = deque(maxlen=100)
        step_discrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_disc_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_single_step_disc_rew = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        infos = {}
        infos["depth"] = self.env.warp_depth_buffer.clone().to(self.device) if self.use_depth else None

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    history = self.trajectory_history
                    if infos["depth"] is not None:
                        depth_image = infos['depth']
                    if self.use_depth:
                        obs = (obs, depth_image)

                    actions = self.alg.act(obs, critic_obs, history)
                    obs, privileged_obs, rewards, dones, infos, _, terminal_amp_states, terminal_obs, terminal_critic_obs = self.env.step(actions)
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    env_ids = dones.nonzero(as_tuple=False).flatten()                    

                    next_obs = torch.clone(obs)
                    next_obs[env_ids] = terminal_obs

                    next_critic_obs = torch.clone(privileged_obs)
                    next_critic_obs[env_ids] = terminal_critic_obs

                    if self.use_amp:
                        next_amp_obs = self.env.get_amp_observations(); next_amp_obs = next_amp_obs[:, self.amp_indices]
                        next_amp_obs = next_amp_obs.to(self.device)
                        terminal_amp_states = terminal_amp_states[:, self.amp_indices]
                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        next_amp_obs_with_term[env_ids] = terminal_amp_states
                        self.amp_obs_frames = torch.concat((self.amp_obs_frames[:, 1:], next_amp_obs_with_term.unsqueeze(1)), dim=1)
                        rewards, logit, disc_reward = self.alg.discriminator.predict_amp_reward(
                            self.amp_obs_frames, rewards, normalizer=self.alg.amp_normalizer)

                        amp_obs = torch.clone(next_amp_obs)
                        self.alg.process_env_step(rewards, dones, infos, next_obs, next_critic_obs, self.amp_obs_frames)
                    else:
                        self.alg.process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

                    # process trajectory history
                    self.trajectory_history[env_ids] = 0
                    self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        if self.use_amp:
                            cur_disc_reward_sum += disc_reward
                            cur_single_step_disc_rew += disc_reward
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        discrewbuffer.extend(cur_disc_reward_sum[new_ids][:, 0].cpu().numpy().tolist())

                        to_extend_disc = (cur_single_step_disc_rew[new_ids] / self.env.max_episode_length_s)[:, 0].cpu().numpy()
                        step_discrewbuffer.extend(to_extend_disc.tolist())
                        
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_disc_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_single_step_disc_rew[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                history = self.trajectory_history
                self.alg.compute_returns(critic_obs, history)
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred, mean_agent_acc, mean_demo_acc = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.policy_learning_rate, locs['it'])
        self.writer.add_scalar('Disc/agent_acc', locs['mean_agent_acc'], locs['it'])
        self.writer.add_scalar('Disc/demo_acc', locs['mean_demo_acc'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_disc_reward', statistics.mean(locs['discrewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_step_disc_reward', statistics.mean(locs['step_discrewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'AMP mean policy acc:':>{pad}} {locs['mean_agent_acc']:.4f}\n"""
                          f"""{'AMP mean demo acc:':>{pad}} {locs['mean_demo_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean disc reward:':>{pad}} {statistics.mean(locs['discrewbuffer']):.2f}\n"""
                          f"""{'Step disc reward:':>{pad}} {statistics.mean(locs['step_discrewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict() if self.alg.discriminator is not None else None,
            'amp_normalizer': self.alg.amp_normalizer if self.alg.amp_normalizer is not None else None,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=lambda storage, loc:storage.cuda(0))
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if loaded_dict['discriminator_state_dict'] is not None:
            self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        if self.alg.discriminator is not None:
            self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
