from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.g1_loco.g1_16dof_loco_config import G1_16Dof_Loco_Cfg, G1_16Dof_Loco_CfgPPO
from legged_gym.envs.g1_loco.g1_16dof_loco_env import G1_16Dof_Loco_Robot 
task_registry.register( "g1_16dof_loco", G1_16Dof_Loco_Robot, G1_16Dof_Loco_Cfg(), G1_16Dof_Loco_CfgPPO())

from legged_gym.envs.g1_loco.g1_16dof_moe_residual_config import G1_16Dof_MoE_Residual_Cfg, G1_16Dof_MoE_Residual_CfgPPO
from legged_gym.envs.g1_loco.g1_16dof_moe_residual_env import G1_16Dof_MoE_Resi_Robot
task_registry.register( "g1_16dof_resi_moe", G1_16Dof_MoE_Resi_Robot, G1_16Dof_MoE_Residual_Cfg(), G1_16Dof_MoE_Residual_CfgPPO())
