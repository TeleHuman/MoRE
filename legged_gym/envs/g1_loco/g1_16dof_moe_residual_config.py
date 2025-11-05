from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1_16Dof_MoE_Residual_Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]

        random_default_pos = False
        default_joint_angles = { # = target angles [rad] when action = 0.0

            '0_left_hip_pitch_joint' : -0.1,         
            '0_left_hip_roll_joint' : 0,               
            '0_left_hip_yaw_joint' : 0. ,   
            '0_left_knee_joint' : 0.3,       
            '0_left_ankle_pitch_joint' : -0.2,     
            '0_left_ankle_roll_joint' : 0,  

            '0_right_hip_pitch_joint' : -0.1,                                       
            '0_right_hip_roll_joint' : 0, 
            '0_right_hip_yaw_joint' : 0., 
            '0_right_knee_joint' : 0.3,                                             
            '0_right_ankle_pitch_joint': -0.2,                              
            '0_right_ankle_roll_joint' : 0,

            '1_left_shoulder_pitch_joint': 0.3,
            '1_left_elbow_joint': 0.8,

            '1_right_shoulder_pitch_joint': 0.3,
            '1_right_elbow_joint': 0.8,
        }

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast

        y_range = [0, 0.1]
        
        edge_width_thresh = 0
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 6 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True
        difficulty_level = 1

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 14
        terrain_width = 4
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 30 # number of terrain cols (types)
        
        terrain_dict = {"roughness": 1., 
                        "slope": 1.,
                        "pit": 1,
                        "gap": 1,
                        "stair": 1,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 0.75# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False

        
    class env(LeggedRobotCfg.env):
        num_gait = 3
        num_observations = 57 + num_gait
        num_actions = 16
        amp_motion_files_list = ['./resources/g1_amp_data/lafan_walk+run_50FPS', './resources/g1_amp_data/lafan_high_knees_50FPS','./resources/g1_amp_data/lafan_squat_50FPS'] 
        num_amp_obs = num_actions
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        feet_info = True
        priv_info = True
        foot_force_info = True
        scan_dot = True
        num_privileged_obs = 60 + num_gait
        if feet_info:
            num_privileged_obs += 12
        if priv_info:
            num_privileged_obs += 38
        if foot_force_info:
            num_privileged_obs += 6
        if scan_dot:
            num_privileged_obs += 187

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        warp_camera = True
        warp_device = 'cuda:0'

        position = [0.0576235, 0.01753, 0.42987]
        original = (64, 64)
        resized = (64, 64)
        near_clip = 0
        far_clip = 2
        update_interval = 5
        buffer_len = 2 + 1
        fovy_range = [79.3, 79.3]

        # ----- camera randomization -----
        y_angle = [42, 48]
        z_angle = [-1, 1]
        x_angle = [-1, 1]

        rand_position = True
        x_pos_range = [-0.01, 0.01]
        y_pos_range = [-0.01, 0.01]
        z_pos_range = [-0.01, 0.01]

        dis_noise = 0.03

        gaussian_noise = True
        gaussian_noise_std = 0.05

        gaussian_filter = True
        gaussian_filter_kernel = [1, 3, 5]
        gaussian_filter_sigma = 1.2

        random_cam_delay = False    # randomly select two sequential frames in the depth buffer
        # --------------------------------

        # ----- augmentation for deployment -----
        # body mask
        add_body_mask = False
        body_mask_path = './body_mask_data/cliped_body_masks_79.3.npz'
        # crop depth
        crop_depth = True
        crop_pixels = [10, 20, 10, 5] # left top right bottom
        # ---------------------------------------


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 5; push_interval_min_s = 5
        min_push_vel_xy = 0.5; max_push_vel_xy = 0.5
        stair_no_push = False

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_com_pos = True
        com_x_pos_range = [-0.03, 0.03]
        com_y_pos_range = [-0.03, 0.03]
        com_z_pos_range = [-0.03, 0.03]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_restitution = False
        restitution_range = [0.0, 1.0]

        randomize_actuation_offset = True
        actuation_offset_range = [-0.1, 0.1]

        delay_update_global_steps = 24 * 5000
        action_delay = True
        action_curr_step = [0, 1, 2]
        action_buf_len = 5
    
    class noise:
        add_noise = True
        noise_level = 1.2 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'hip_yaw_joint': 100.,
            'hip_roll_joint': 100.,
            'hip_pitch_joint': 100.,
            'knee_joint': 150.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,    
            'shoulder_pitch_joint': 60,
            'elbow_joint': 40 
        }  # [N*m/rad]
        
        damping = {
            'hip_yaw_joint': 2,
            'hip_roll_joint': 2,
            'hip_pitch_joint': 2,
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,
            'shoulder_pitch_joint': 2,
            'elbow_joint': 1
        } # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_16dof_lock_wrist_new_feet.urdf'

        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        extra_shoulder_penalty_name = ["shoulder_roll", "shoulder_yaw"]
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        feet_indicator_offset = [[-0.04, 0, -0.035], [-0.02, 0, -0.035], [0, 0, -0.035], [0.02, 0, -0.035], [0.06, 0, -0.035],  [0.1, 0, -0.035]]
        feet_collision_indicator_offset = [[0.14, 0, -0.035], [-0.06, 0, -0.035]]
    
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        gait_resampling_time = 10. # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
        
    class rewards( LeggedRobotCfg.rewards ):
        # Set gait rewards here
        gait_rewards = [[None], ["knee_height"], ['squat_height']]
        all_gait_rewards = [None, 'knee_height', 'squat_height']
        high_knees_target = -0.3
        squat_height_target = 0.60

        soft_dof_pos_limit = 0.9
        feet_min_lateral_distance_target = 0.17
        class scales:
            """ locomotion rewards """
            tracking_lin_vel = 2
            tracking_ang_vel = 2

            dof_acc = -5e-7
            dof_vel = -1e-3
            action_rate = -0.03
            action_smoothness = -0.05

            ang_vel_xy = -0.05
            orientation = -2.0
            joint_power = -2.5e-5
            feet_stumble = -1.0
            torques = -1e-5
            arm_joint_deviation = -0.5
            hip_joint_deviation = -0.5
            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -1.0
            feet_lateral_distance = 0.5
            feet_slippage = -0.25
            feet_contact_force = -2.5e-4
            feet_force_rate = -2.5e-4
            feet_contact_momentum = -2.5e-4
            collision = -15.
            stuck = -1 * 0
            cheat = -1.5
            feet_edge = -0.3
            y_offset_pen = -2 * 0

            """ gait rewards """
            squat_height = 2
            knee_height = 2

            """ terrain specific rewards """
            # pit
            knee_collision = -0.03
            # stair
            feet_collision_pen = -1

        feet_contact_force_range = [200. , 600.]


class G1_16Dof_MoE_Residual_CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'MoEResiOnPolicyRunnerMulti'

    class policy:
        init_noise_std = 0.8
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        residual_hidden_dims = [512, 256, 128]
        gate_hidden_dims = [512, 256, 128]
        num_residual_net = 3
        his_latent_dim = 64

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_amp = True
        amp_loader_type = 'lafan_16dof_multi'
        amp_loader_class_name = "G1_AMPLoader"
        entropy_coef = 0.01
        policy_learning_rate = 2e-4
        
        amp_replay_buffer_size = 100000
        disc_learning_rate = 1e-3

    class runner( LeggedRobotCfgPPO.runner ):
        algorithm_class_name = "ResiMoEPPOMulti"
        policy_class_name = "ActorCriticResiMoE"
        max_iterations = 40000
        run_name = ''
        experiment_name = 'g1_16dof_resi_moe'
        save_interval = 500

        load_loco_policy = True
        loco_expert_ckpt_path = ""

        num_gait = len(G1_16Dof_MoE_Residual_Cfg.env.amp_motion_files_list)

        """ style rewards """
        amp_reward_coef = 3 # scale of amp reward
        num_amp_frames = 5
        amp_num_preload_transitions = 1000
        use_lerp = False
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]
