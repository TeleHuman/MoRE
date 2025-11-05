import os
from os.path import join as pjoin

import torch
import numpy as np
from pybullet_utils import transformations

from legged_gym.datasets import motion_util
from legged_gym.datasets import pose3d
from legged_gym.utils import utils


class G1_AMPLoader:

    def __init__(
            self,
            device,
            time_between_frames,
            motion_dir,
            preload_transitions=False,
            num_preload_transitions=1000,
            num_frames=5,
            ):
        """
        Expert dataset provides AMP observations from LAFAN1 dataset (https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.motion_dir = motion_dir

        for i, motion_file in enumerate(os.listdir(motion_dir)):
            self.trajectory_names.append(motion_file)
            motion_path = pjoin(motion_dir, motion_file)
            motion_data = np.load(motion_path, allow_pickle=True)
            motion_data_processed = np.zeros((motion_data.shape[0],59))
            for f_i in range(motion_data.shape[0]):
                root_rot = self.euler_to_quaternion(motion_data[f_i, 3:6])
                root_rot = pose3d.QuaternionNormalize(root_rot)
                root_rot = motion_util.standardize_quaternion(root_rot)
                motion_data_processed[f_i, :3] = motion_data[f_i, :3]   # base pos
                motion_data_processed[f_i, 3:7] = root_rot  # base quat
                motion_data_processed[f_i, 7:13] = motion_data[f_i, 6:12]  # base vel
                motion_data_processed[f_i, 13:36] = motion_data[f_i, 12:35]  # 23 dof pos (joint sequences defined by urdf)
                motion_data_processed[f_i, 36:59] = motion_data[f_i, 35:58]     # 23 dof vel
                '''
                NOTE The order of motion_data_processed is
                base pos 0:3,
                base quat 3:7,
                base vel 7:13,
                dof pos  13:36,
                dof vel  36:59
                '''
            self.trajectories.append(torch.tensor(
                motion_data_processed[:, 7:],
                dtype=torch.float32,
                device=self.device
            ))
            
            self.trajectories_full.append(torch.tensor(
                motion_data_processed,
                dtype=torch.float32,
                device=self.device
            ))
            
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(1 / len(os.listdir(motion_dir)))
            frame_duration = 1 / 50
            
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (motion_data_processed.shape[0] - 1) * frame_duration # seconds
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data_processed.shape[0]))
            print(f"Loaded {traj_len}s. motion from {motion_file}.")
            
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload multi-frame transitions
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_frames = []
            for i in range(self.num_frames):
                frame_time = times + (i - (self.num_frames - 2)) * self.time_between_frames
                self.preloaded_frames.append(
                    self.get_full_frame_at_time_batch(traj_idxs, frame_time))
            print(f'Finished preloading multiple frames')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)


    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), 3, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), 3, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), 4, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), 4, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), 52, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  52, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = G1_AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = G1_AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = G1_AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = G1_AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, 7:59] # base vel3+ang3, dof vel23+ang23
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, 7:59]  # base vel3+ang3, dof vel23+ang23
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = G1_AMPLoader.get_root_pos(frame0), G1_AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = G1_AMPLoader.get_root_rot(frame0), G1_AMPLoader.get_root_rot(frame1)
        joints0, joints1 = G1_AMPLoader.get_joint_pose(frame0), G1_AMPLoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = G1_AMPLoader.get_tar_toe_pos_local(frame0), G1_AMPLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = G1_AMPLoader.get_linear_vel(frame0), G1_AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = G1_AMPLoader.get_angular_vel(frame0), G1_AMPLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = G1_AMPLoader.get_joint_vel(frame0), G1_AMPLoader.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])


    def feed_forward_generator_lafan_16dof_multi(self, num_mini_batch, mini_batch_size): # 12 leg, shoulder pitch, elbow, 
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_frames[0].shape[0], size=mini_batch_size)
                frames = []
                for i in range(self.num_frames):
                    s = torch.cat([
                        self.preloaded_frames[i][idxs, 13:25],  # 12 leg
                        self.preloaded_frames[i][idxs, 28:29],  # left shoulder pitch
                        self.preloaded_frames[i][idxs, 31:33],  # left elbow right shoulder pitch
                        self.preloaded_frames[i][idxs, 35:36],  # right elbow
                    ], dim=-1)
                    frames.append(s)
            else:
                NotImplementedError('preload transition')
            yield torch.stack(frames, dim=1)    # [batch, num_frames, 16]


    def quaternion_to_euler_array(self, quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w =quat
        
        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        
        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z])   

    def euler_to_quaternion(self, root_rot):
        roll, pitch, yaw = root_rot[0], root_rot[1], root_rot[2]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        return np.array([qx, qy, qz, qw])
    
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)
    @staticmethod
    def get_root_pos(pose):
        return pose[0:3]
    
    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, 0:3]
    @staticmethod
    def get_root_rot(pose):
        return pose[3:7]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, 3:7]
    @staticmethod
    def get_joint_pose(pose):
        return pose[13:36]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, 13:36]

    @staticmethod
    def get_joint_pose_batch_16dof(poses):
        return torch.cat([poses[:, 13:25], poses[:, 28:29], poses[:, 31:33], poses[:, 35:36]], dim=-1)

    @staticmethod
    def get_linear_vel(pose):
        return pose[7:10]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, 7:10]
    @staticmethod
    def get_angular_vel(pose):
        return pose[10:13]  

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, 10:13]  
    @staticmethod
    def get_joint_vel(pose):
        return pose[36:59]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, 36:59]  
    
    @staticmethod
    def get_joint_vel_batch_16dof(poses):
        return torch.cat([poses[:, 51:52], poses[:, 54:56], poses[:, 58:59], poses[:, 36:48]], dim=-1)