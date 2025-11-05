import numpy as np
import torch
import transforms3d as t3d
from typing import List
from pytorch3d import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import warp as wp
import cv2
import time
import math
import trimesh
# from .warp_kernels import raycast_mesh_kernel, raycast_multi_env_kernel
from .depth_camera_cfg import DepthCameraCfg

from legged_gym.utils.math import quat_apply, quat_inv, convert_camera_frame_orientation_convention, quat_mul

def depth_image_preprocessing(depth_image, near_plane=100, far_plane=1200.0, depth_scale=1000):

    near_mask = abs(depth_image) < near_plane / depth_scale
    far_mask = abs(depth_image) > far_plane / depth_scale

    depth_image[near_mask] = 0
    depth_image[far_mask] = far_plane / depth_scale
    if torch.isnan(depth_image).any() or torch.isinf(depth_image).any():
        raise Exception("nan or inf of depth image detected!")
    return depth_image


def euler_pos_2_mat(pos: torch.Tensor, euler: torch.Tensor):
    if torch.is_tensor(pos):
        pos = pos.cpu().numpy()
    if torch.is_tensor(euler):
        euler = euler.cpu().numpy()
    
    mats = torch.zeros(pos.shape[0], 4, 4)

    for i in range(pos.shape[0]):
        rot = torch.tensor(t3d.euler.euler2mat(euler[i, 0], euler[i, 1], euler[i, 2], "sxyz"))
        mats[i, :3,:3] = rot
        mats[i, :3,3] = torch.tensor(pos[i])
        mats[i, 3,3] = 1
    return mats


def quat_pos_2_mat_torch(pos: torch.Tensor, quat: torch.Tensor):
    b = pos.shape[0]
    rot = transforms.quaternion_to_matrix(torch.cat((quat[:,3:], quat[:, :3]), dim=1).to(quat.device)) # quat have to be wxyz
    mat = torch.zeros(b, 4, 4).to(pos.device)
    mat[:,:3,:3] = rot
    mat[:,:3,3] = pos
    mat[:,3,3] = 1
    return mat

def mat_2_quat_pos_torch(mat: torch.Tensor):
    """
    将齐次变换矩阵 [B, 4, 4] 转换为 (pos, quat)
    输出四元数顺序为 xyzw，与上面函数输入匹配
    """
    assert mat.ndim == 3 and mat.shape[1:] == (4, 4), "mat shape must be [B, 4, 4]"
    b = mat.shape[0]

    rot = mat[:, :3, :3]
    pos = mat[:, :3, 3]

    # 根据旋转矩阵提取四元数 (wxyz)
    qw = torch.sqrt(torch.clamp(1.0 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=1e-8)) / 2
    qx = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * qw)
    qy = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * qw)
    qz = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * qw)

    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=1)
    quat_wxyz = F.normalize(quat_wxyz, dim=1)

    # 输出 xyzw 顺序（与 quat_pos_2_mat_torch 的输入对应）
    quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, :1]], dim=1)

    return pos, quat_xyzw


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    max_dist: float = 1e6,
):
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh and store the hit position
    hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
    # if the ray hit, store the hit data
    # hit_success = False
    if hit_success:
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]
        ray_distance[tid] = t

@wp.kernel
def draw_depth(mesh: wp.uint64, cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, depth: wp.array(dtype=wp.float32), fovy_dist_offset: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    b = tid // (height*width)
    p = tid % (height*width)
    y = p % width
    z = p // width

    sy = -2.0 * float(y) / float(height) + float(width / height) # from the left to the right
    sz = -2.0 * float(z) / float(height) + 1.0 # from the up to the down

    ro = wp.vec3(cam_pos[b*3], cam_pos[b*3+1], cam_pos[b*3+2])
    rot = wp.mat33(cam_rot[b*9], cam_rot[b*9+1], cam_rot[b*9+2],
                   cam_rot[b*9+3], cam_rot[b*9+4], cam_rot[b*9+5],
                   cam_rot[b*9+6], cam_rot[b*9+7], cam_rot[b*9+8])

    rd = wp.normalize(wp.vec3(float(width / height)+fovy_dist_offset[b], sy, sz))
    d = rd[0]
    rd = rot @ rd

    distance = wp.float32(0.0)
    query = wp.mesh_query_ray(mesh, ro, rd, 5.0)

    if query.result:
        distance = query.t * d
    else:
        distance = 5.0

    depth[tid] = distance


class DepthRendererWarp:
    def __init__(self, 
                 cam2base_xyz,
                 cam2base_euler,
                 fovy,
                 cfg: DepthCameraCfg,
                 num_envs: int,
                 device="cuda:0",
                 ):
        self.image_height = cfg.height
        self.image_width = cfg.width
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.count = 0

        self._create_buffers()
        self._compute_intrinsic_matrices()

        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self._from_pinhole_camera_pattern()
        self.num_rays = self.ray_directions.shape[1]

        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)

        # set offsets
        quat_w = convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset_rot], device=self.device), origin=self.cfg.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self.num_envs, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset_pos), device=self.device).repeat(self.num_envs, 1)
        
        self.env_dynamic_mesh = None
        
        # origin camera set
        self.cam2base_xyz = torch.tensor(cam2base_xyz, device=self.device)
        self.cam2base_euler = cam2base_euler
        
        self.fovy_dist_offset = 1.0 / torch.tan(torch.deg2rad(fovy)/2) - 1.0

        self.mesh = None

        self.robot2cam = euler_pos_2_mat(self.cam2base_xyz, self.cam2base_euler).to(self.device)
        self.warp2gym = torch.tensor([[0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 0, 1]], dtype=torch.float32).to(self.device)
        
        self.env_dynamic_mesh = None
        self.env_base_points = None
        self.env_vertex_counts_per_instance = None
        self.env_mesh_instance_indices = None

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self.pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.quat_w_world = torch.zeros((self.num_envs, 4), device=self.device)
        # -- intrinsic matrix
        self.intrinsic_matrices = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.intrinsic_matrices[:, 2, 2] = 1.0
        # -- output data
        # create the buffers to store the annotator data.
        self.image_shape = (self.image_height, self.image_width)
        self.depth_image = torch.zeros((self.num_envs, *self.image_shape), device=self.device)

    def _compute_intrinsic_matrices(self):
        """Computes the intrinsic matrices for the camera based on the config provided."""
        # check if vertical aperture is provided
        # if not then it is auto-computed based on the aspect ratio to preserve squared pixels
        if self.cfg.vertical_aperture is None:
            self.cfg.vertical_aperture = self.cfg.horizontal_aperture * self.cfg.height / self.cfg.width

        # compute the intrinsic matrix
        f_x = self.cfg.width * self.cfg.focal_length / self.cfg.horizontal_aperture
        f_y = self.cfg.height * self.cfg.focal_length / self.cfg.vertical_aperture
        c_x = self.cfg.horizontal_aperture_offset * f_x + self.cfg.width / 2
        c_y = self.cfg.vertical_aperture_offset * f_y + self.cfg.height / 2
        # allocate the intrinsic matrices
        self.intrinsic_matrices[:, 0, 0] = f_x
        self.intrinsic_matrices[:, 0, 2] = c_x
        self.intrinsic_matrices[:, 1, 1] = f_y
        self.intrinsic_matrices[:, 1, 2] = c_y

    def _from_pinhole_camera_pattern(self):
        """The image pattern for ray casting.

        .. caution::
            This function does not follow the standard pattern interface. It requires the intrinsic matrices
            of the cameras to be passed in. This is because we want to be able to randomize the intrinsic
            matrices of the cameras, which is not possible with the standard pattern interface.

        Args:
            cfg: The configuration instance for the pattern.
            intrinsic_matrices: The intrinsic matrices of the cameras. Shape is (N, 3, 3).
            device: The device to create the pattern on.

        Returns:
            The starting positions and directions of the rays. The shape of the tensors are
            (N, H * W, 3) and (N, H * W, 3) respectively.
        """
        # get image plane mesh grid
        grid = torch.meshgrid(
            torch.arange(start=0, end=self.cfg.width, dtype=torch.int32, device=self.device),
            torch.arange(start=0, end=self.cfg.height, dtype=torch.int32, device=self.device),
            indexing="xy",
        )
        pixels = torch.vstack(list(map(torch.ravel, grid))).T
        # convert to homogeneous coordinate system
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=self.device)])
        # move each pixel coordinate to the center of the pixel
        pixels += torch.tensor([[0.5, 0.5, 0]], device=self.device)
        # get pixel coordinates in camera frame
        pix_in_cam_frame = torch.matmul(torch.inverse(self.intrinsic_matrices), pixels.T)

        # robotics camera frame is (x forward, y left, z up) from camera frame with (x right, y down, z forward)
        # transform to robotics camera frame
        # transform_vec = torch.tensor([1, -1, -1], device=self.device).unsqueeze(0).unsqueeze(2)
        transform_vec = torch.tensor([1, -1, -1], device=self.device).unsqueeze(0).unsqueeze(2)        
        pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec
        # normalize ray directions
        ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
        # for camera, we always ray-cast from the sensor's origin
        ray_starts = torch.zeros_like(ray_directions, device=self.device)

        return ray_starts, ray_directions
        
    def render_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        with wp.ScopedDevice(self.device):
            vertices = np.matmul(self.warp2gym[:3,:3].cpu().numpy(), vertices.transpose(1,0)).transpose(1,0)
            indices = indices.flatten()
            self.mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device), 
                                velocities=None, 
                                indices=wp.array(indices, dtype=int, device=self.device))
        if len(self.cfg.self_obstacle_mesh_paths) != 0:
            self.render_self_obstacle_mesh()
    
    def render_self_obstacle_mesh(self):
        # load arm and leg mesh
        pattern_points = []  # geometry (points) in parent frame for each pattern
        pattern_indices = []  # flattened triangle indices for each pattern
        for item in self.cfg.self_obstacle_mesh_paths:
            mesh_trimesh = trimesh.load(item)
            vertices = np.array(mesh_trimesh.vertices, dtype=np.float32)  
            faces = np.array(mesh_trimesh.faces, dtype=np.uint32)

            pattern_points.append(vertices)
            pattern_indices.append(faces)
        
        combined_points: list[np.ndarray] = []
        combined_indices: list[np.ndarray] = []
        vertex_counts: list[int] = []
        vertex_offset = 0
        total_instances = 0

        for p in range(len(self.cfg.self_obstacle_mesh_paths)):
            base_points = pattern_points[p]
            base_indices = pattern_indices[p]
            for _env in range(self.num_envs):
                offset_indices = base_indices + vertex_offset
                combined_points.append(base_points)
                combined_indices.append(offset_indices)
                vertex_counts.append(len(base_points))
                vertex_offset += len(base_points)
                total_instances += 1

        final_points = np.vstack(combined_points)
        self.final_indices = np.concatenate(combined_indices)

        # Build warp mesh and cache tensors for fast updates
        self.env_dynamic_mesh = wp.Mesh(
            points=wp.array(final_points.astype(np.float32), dtype=wp.vec3, device=self.device),
            indices=wp.array(self.final_indices.astype(np.int32).flatten(), dtype=wp.int32, device=self.device),
        )
        self.env_base_points = torch.tensor(final_points, device=self.device, dtype=torch.float32)
        self.env_vertex_counts_per_instance = torch.tensor(vertex_counts, device=self.device, dtype=torch.int32)
        self.env_mesh_instance_indices = torch.arange(total_instances, device=self.device, dtype=torch.int32)
        
        def generate_env_world_positions(num_envs: int, interval: float = 5.0, device='cuda:0') -> torch.Tensor:
                """
                根据总环境数量 num_envs 自动生成一个尽量接近正方形的网格布局，
                返回每个环境的世界位置 (num_envs, 3)
                """
                # 自动计算行列数，尽量接近正方形
                num_cols = math.ceil(math.sqrt(num_envs))
                num_rows = math.ceil(num_envs / num_cols)
                
                # 创建行列索引
                rows = torch.arange(num_rows, device=device)
                cols = torch.arange(num_cols, device=device)
                grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')  # shape [num_rows, num_cols]

                # 转换为世界坐标，中心化
                x_positions = ( (num_cols - 1)/2 - grid_x ) * interval
                y_positions = ( (num_rows - 1)/2 - grid_y ) * interval
                z_positions = torch.ones_like(x_positions) * 2

                # 拼接成 (num_rows*num_cols, 3)，然后只取前 num_envs 个
                env_world_positions = torch.stack([x_positions, y_positions, z_positions], dim=-1).reshape(-1,3)
                env_world_positions = env_world_positions[:num_envs]

                return env_world_positions
        
        self.env_positions = generate_env_world_positions(self.num_envs, interval=10)
        self.num_instance_per_env = torch.tensor([len(self.cfg.self_obstacle_mesh_paths) for _ in range(self.num_envs)], device=self.device)
        
        self.expanded_env_origin_positions = torch.repeat_interleave(
            self.env_positions, self.num_instance_per_env.long(), dim=0
        )

    def render_depth(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
        """
        base_pos: (B, 3)
        base_quat: (B, 4)
        """
        b = base_pos.shape[0]
        depth = wp.zeros(b * self.image_width * self.image_height, dtype=wp.float32, device=self.device)

        gym2robot = quat_pos_2_mat_torch(base_pos, base_quat).to(self.device)
        warp2cam = self.warp2gym.unsqueeze(0).repeat([b,1,1]) @ gym2robot @ self.robot2cam
        cam_pos = warp2cam[:,:3,3].reshape(b*3)
        cam_rot = warp2cam[:,:3,:3].reshape(b*9)
        t1 = time.time()
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel = draw_depth,
                dim = b * self.image_width * self.image_height,
                inputs = [self.mesh.id, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), 
                          self.image_width, self.image_height, depth, self.fovy_dist_offset],
                device = self.device
            )
        print("origin reder time:", time.time() - t1)
        depth = wp.to_torch(depth.reshape([b, self.image_height, self.image_width])) #* self.depth_scale

        return depth

    def update_depth(self, base_pos, base_quat, link_pos=None, link_quat=None, root_pos=None):
        """Fills the buffers of the sensor data."""
        # if self.count < 10:
        t3 = time.time()
        if self.env_dynamic_mesh is not None:
            assert link_pos is not None and link_quat is not None
            # self.update_self_obstacle_mesh(link_pos, link_quat, root_pos)
            self.rebuild_self_obstacle_mesh(link_pos, link_quat, root_pos)
        print("refit time:", time.time() - t3)

        b = base_pos.shape[0]
        gym2robot = quat_pos_2_mat_torch(base_pos, base_quat).to(self.device)
        warp2cam = self.warp2gym.unsqueeze(0).repeat([b,1,1]) @ gym2robot @ self.robot2cam
        cam_pos, cam_rot = mat_2_quat_pos_torch(warp2cam)
        # _, _quat_warp2gym = mat_2_quat_pos_torch(self.warp2gym.unsqueeze(0).repeat([b,1,1]))
        # cam_pos = warp2cam[:,:3,3].reshape(b*3)
        # cam_rot = warp2cam[:,:3,:3].reshape(b*9)

        pos_w = cam_pos
        quat_w = cam_rot

        ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts)
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions)

        t1 = time.time()
        dist1 = self.raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=1e6,
            mesh=self.mesh,
        )
        
        if self.env_dynamic_mesh is not None:
            t2 = time.time()
            dist2 = self.raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=1e6,
                mesh=self.env_dynamic_mesh,
            )
            final_dist = torch.minimum(dist1, dist2)
        else:
            final_dist = dist1
        # final_dist = dist1
        print("raycast1 time:", t2-t1, "raycast2 time:", time.time()-t2)
        distance_to_image_plane = (quat_apply(quat_inv(quat_w).repeat(1, self.num_rays), (final_dist[:, :, None] * ray_directions_w),))[:, :, 0]
        # distance_to_image_plane = dist1[:, :] * ray_directions_w[:,:,0]
        
        # apply the maximum distance after the transformation
        if self.cfg.depth_clipping_behavior == "max":
            distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
            distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
        elif self.cfg.depth_clipping_behavior == "zero":
            distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
            distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
        
        self.depth_image = distance_to_image_plane.reshape(-1, *self.image_shape)

        return self.depth_image
    
    def raycast_mesh(
        self,
        ray_starts: torch.Tensor,
        ray_directions: torch.Tensor,
        max_dist: float = 1e6,
        mesh=None,
    ):
        # # extract device and shape information
        shape = ray_starts.shape
        device = ray_starts.device
        # # device of the mesh
        torch_device = wp.device_to_torch(self.device)
        # # reshape the tensors
        ray_starts = ray_starts.to(torch_device).reshape(-1, 3)
        ray_directions = ray_directions.to(torch_device).reshape(-1, 3)
        num_rays = ray_starts.shape[0]
        # # create output tensor for the ray hits
        ray_hits = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()

        # map the memory to warp arrays
        ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
        ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
        ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

        # if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
        # print("mesh size:", mesh.points.size)
        # print("num_rays", num_rays)
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=raycast_mesh_kernel,
                dim=num_rays,
                inputs=[
                    mesh.id,
                    ray_starts_wp,
                    ray_directions_wp,
                    ray_hits_wp,
                    ray_distance_wp,
                    # ray_normal_wp,
                    # ray_face_id_wp,
                    float(max_dist),
                    # int(return_distance),
                    # int(return_normal),
                    # int(return_face_id),
                ],
                device=self.device,
            )
        # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
        wp.synchronize()

        # if return_distance:
        ray_distance = ray_distance.to(device).reshape(shape[0], shape[1])

        # return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id
        return ray_distance

    def rebuild_self_obstacle_mesh(self, link_pos, link_quat, root_pos):
        """Efficiently update the env dynamic mesh using vectorized operations."""
        if (
            self.env_dynamic_mesh is None
            or self.env_vertex_counts_per_instance is None
            or self.env_base_points is None
        ):
            return

        try:
            # update all env dynamic mesh one time
            # expanded_root_positions = torch.repeat_interleave(
            #     root_pos, self.num_instance_per_env.long(), dim=0)
            # expanded_positions = torch.repeat_interleave(
            #     link_pos.reshape(-1, 3) - expanded_root_positions + self.expanded_env_origin_positions, self.env_vertex_counts_per_instance.long(), dim=0
            # )
            expanded_positions = torch.repeat_interleave(
                link_pos.reshape(-1, 3), self.env_vertex_counts_per_instance.long(), dim=0
            )
            expanded_quats = torch.repeat_interleave(
                link_quat.reshape(-1, 4), self.env_vertex_counts_per_instance.long(), dim=0
            )
            # expanded_quats = torch.repeat_interleave(
            #     warp2gym_link_quat, self.env_vertex_counts_per_instance.long(), dim=0
            # )
            
            transformed_points = quat_apply(expanded_quats, self.env_base_points) + expanded_positions
            # transformed_points = quat_apply(expanded_quats, self.env_base_points) + expanded_positions
            
            # vertices = np.matmul(self.warp2gym[:3,:3].cpu().numpy(), transformed_points.cpu().numpy().transpose(1,0)).transpose(1,0)

            transformed_points = transformed_points @ self.warp2gym[:3, :3].T

            self.env_dynamic_mesh = wp.Mesh(
                points=wp.array(transformed_points.cpu().numpy().astype(np.float32), dtype=wp.vec3, device=self.device),
                indices=wp.array(self.final_indices.astype(np.int32).flatten(), dtype=wp.int32, device=self.device),
            )

        except Exception as e:
            print(f"Failed to update env dynamic mesh efficiently: {str(e)}")
    
    def update_self_obstacle_mesh(self, link_pos, link_quat, root_pos):
        """Efficiently update the env dynamic mesh using vectorized operations."""
        if (
            self.env_dynamic_mesh is None
            or self.env_vertex_counts_per_instance is None
            or self.env_base_points is None
        ):
            return

        try:
            # expanded_root_positions = torch.repeat_interleave(
            #     root_pos, self.num_instance_per_env.long(), dim=0)
            # expanded_positions = torch.repeat_interleave(
            #     link_pos.reshape(-1, 3) - expanded_root_positions + self.expanded_env_origin_positions, self.env_vertex_counts_per_instance.long(), dim=0
            # )

            expanded_positions = torch.repeat_interleave(
                link_pos.reshape(-1, 3), self.env_vertex_counts_per_instance.long(), dim=0
            )
            expanded_quats = torch.repeat_interleave(
                link_quat.reshape(-1, 4), self.env_vertex_counts_per_instance.long(), dim=0
            )
            # expanded_quats = torch.repeat_interleave(
            #     warp2gym_link_quat, self.env_vertex_counts_per_instance.long(), dim=0
            # )
            
            transformed_points = quat_apply(expanded_quats, self.env_base_points) + expanded_positions
            
            transformed_points = transformed_points @ self.warp2gym[:3, :3].T

            updated_points_wp = wp.from_torch(transformed_points, dtype=wp.vec3)
            self.env_dynamic_mesh.points = updated_points_wp
            self.env_dynamic_mesh.refit()

        except Exception as e:
            print(f"Failed to update env dynamic mesh efficiently: {str(e)}")


if __name__ == "__main__":
    warprender = DepthRendererWarp([60, 106],
                      torch.tensor([0.10043, 0.0222, -0.11]),
                      torch.tensor([0, 60.95 / 180 * np.pi, 0]))