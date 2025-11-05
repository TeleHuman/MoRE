import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import cv2
import numpy as np
import torch.nn.functional as F


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd



if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        init_qpos = np.array(config["init_qpos"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        obs_history_len = config["obs_history_len"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        gait_cmd = np.array(config["gait_cmd"], dtype=np.float32)
        num_gaits = gait_cmd.shape[0]

        # depth camera configs
        depth_far_clip = config["depth_far_clip"]
        depth_near_clip = config["depth_near_clip"]
        depth_buffer_len = config["depth_buffer_len"]
        depth_size = config["depth_size"]
        cam_update_interval = config["cam_update_interval"]
        crop_image = config["crop_image"]
        crop_size = config["crop_size"]
        gaussian_filter = config["gaussian_filter"]
        gaussian_filter_kernel = config["gaussian_filter_kernel"]
        gaussian_filter_sigma = config["gaussian_filter_sigma"]
        gaussian_noise = config["gaussian_noise"]
        gaussian_noise_std = config["gaussian_noise_std"]
        depth_dis_noise = config["depth_dis_noise"]

    def process_depth_image(depth_image):
        depth_image += depth_dis_noise * 2 * (np.random.rand(1) - 0.5)
        if gaussian_noise:
            depth_image += gaussian_noise_std * np.random.randn(*depth_image.shape)
        depth_image = np.clip(depth_image, depth_near_clip, depth_far_clip)
        depth_image = (depth_image - depth_near_clip) / (depth_far_clip - depth_near_clip) - 0.5
        return depth_image
    
    def crop_resize_depth(depth_image):
        clip_left, clip_top, clip_right, clip_bottom = crop_size
        depth_image = F.interpolate(depth_image[clip_top:-clip_bottom, clip_left:depth_size[1]-clip_right].unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        return depth_image
    
    def adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma):
        imgs = cv2.GaussianBlur(depth_image.numpy(), (kernel_size, kernel_size), sigma)
        return torch.from_numpy(imgs).to(depth_image.device)
    
    def update_depth_cam(depth_image_buffer):
        depth_renderer.update_scene(d, camera=depth_cam_id)
        depth_image = depth_renderer.render()
        depth_image = np.rot90(depth_image, k=1)
        depth_image = process_depth_image(depth_image)
        depth_image = torch.tensor(depth_image)
        if crop_image:
            depth_image = crop_resize_depth(depth_image)
        if gaussian_filter:
            depth_image = adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma)

        cv2.namedWindow('depth image', cv2.WINDOW_NORMAL)
        cv2.imshow("depth image", depth_image_buffer[0, -1].detach().numpy() + 0.5)
        cv2.waitKey(1)
        return depth_image


    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    trajectory_history = torch.zeros(size=(1, obs_history_len, num_obs - num_gaits))
    depth_image_buffer = torch.zeros(1, depth_buffer_len, 64, 64)
    counter = 0
    cam_update_counter = 0


    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # set init dof pos
    d.qpos[3:] = init_qpos

    save_imgs = []

    # load policy
    policy = torch.jit.load(policy_path)

    save_imgs = []
    depth_renderer = mujoco.Renderer(m, width=depth_size[1], height=depth_size[0])
    depth_renderer.enable_depth_rendering()
    depth_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            counter += 1
            if (counter > 0) and (counter % control_decimation) == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs[:3] = gait_cmd
                obs[3:6] = cmd * cmd_scale  
                obs[6:9] = omega
                obs[9:12] = gravity_orientation
                obs[12 : 12 + num_actions] = qj
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action

                # get depth images
                if (cam_update_counter) % cam_update_interval == 0:
                    # get depth images
                    depth_imgs = update_depth_cam(depth_image_buffer)
                    if (depth_image_buffer==0).all():   # first image
                        depth_image_buffer = torch.stack([depth_imgs] * depth_buffer_len, dim=0).unsqueeze(0)
                    else:
                        depth_image_buffer = torch.cat([depth_image_buffer[:, 1:, ...], depth_imgs.unsqueeze(0).unsqueeze(1)], dim=1)
                cam_update_counter += 1

                # add trajectory_history
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                trajectory_history = torch.cat([trajectory_history[:, 1:], obs_tensor.unsqueeze(1)[..., num_gaits:]], dim=1)

                # policy inference
                action = policy(obs_tensor, trajectory_history, depth_image_buffer[:, 7:9, ...]).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
