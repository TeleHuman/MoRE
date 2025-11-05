# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from dataclasses import MISSING, dataclass, field
from typing import Literal, Tuple, List, Dict, Optional

@dataclass
class DepthCameraCfg:
    """Configuration for the ray-cast sensor."""

    # class OffsetCfg:
    #     """The offset pose of the sensor's frame from the sensor's parent frame."""

    #     pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #     """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    #     rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #     """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    # mesh_prim_paths: list[str] = MISSING
    """The list of mesh primitive paths to ray cast against.

    Note:
        Currently, only a single static mesh is supported. We are working on supporting multiple
        static meshes and dynamic meshes.
    """

    offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    convention: Literal["opengl", "ros", "world"] = "world"

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"

    # pattern_cfg: PatternBaseCfg = MISSING

    max_distance: float = 2.0

    drift_range: Tuple[float, float] = (0.0, 0.0)
    """The range of drift (in meters) to add to the ray starting positions (xyz) in world frame. Defaults to (0.0, 0.0).

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """

    # ray_cast_drift_range: Dict[str, Tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """The range of drift (in meters) to add to the projected ray points in local projection frame. Defaults to (0.0, 0.0) for x, y, and z drift.

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """

    # # Dynamic environment mesh support
    # dynamic_env_mesh_prim_paths: List[str] = []
    dynamic_env_mesh_prim_paths: List[str] = field(default_factory=list)

    # camera intrinsic
    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """

    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    vertical_aperture: Optional[float] = None
    r"""Vertical aperture (in cm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. In this case, the vertical
    aperture is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """

    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""

    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""

    width: int = 64
    """Width of the image (in pixels)."""

    height: int = 64
    """Height of the image (in pixels)."""

    # load arm and leg mesh
    self_obstacle_mesh_paths = [
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_shoulder_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_elbow_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_wrist_pitch_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_wrist_roll_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_wrist_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_hip_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/left_hip_roll_link_simple.STL",

                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_shoulder_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_elbow_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_wrist_pitch_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_wrist_roll_link_simple.STL",
                                # "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_wrist_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_hip_yaw_link_simple.STL",
                                "/home/zhang/wxm/Projects/TeleAI-Isaac-Lab/legged_lab/assets/unitree_g1/meshes_simplified/right_hip_roll_link_simple.STL",
                                ]
