import numpy as np
import torch
import transforms3d as t3d
from typing import List
from pytorch3d import transforms
import matplotlib.pyplot as plt
import warp as wp
import cv2

def depth_image_preprocessing(depth_image, near_plane=100, far_plane=1200.0, depth_scale=1000):

    near_mask = abs(depth_image) < near_plane / depth_scale
    far_mask = abs(depth_image) > far_plane / depth_scale

    depth_image[near_mask] = 0
    depth_image[far_mask] = far_plane / depth_scale
    if torch.isnan(depth_image).any() or torch.isinf(depth_image).any():
        raise Exception("nan or inf of depth image detected!")
    return depth_image


@wp.kernel
def draw_pixels(mesh: wp.uint64, cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, pixels: wp.array(dtype=wp.vec3)):
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
    rd = wp.normalize(wp.vec3(float(width / height), sy, sz))
    rd = rot @ rd

    color = wp.vec3(0.0, 0.0, 0.0)
    query = wp.mesh_query_ray(mesh, ro, rd, 10.0)

    if query.result:
        color = query.normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color


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


class DepthRendererWarp:
    def __init__(self, image_params:List, cam2base_xyz: torch.Tensor, cam2base_euler: torch.Tensor, fovy: torch.Tensor, device="cuda:0"):
        self.image_height = image_params[0]
        self.image_width = image_params[1]
        self.device = device

        self.cam2base_xyz = cam2base_xyz
        self.cam2base_euler = cam2base_euler

        self.near_plane = 0.1
        self.far_plane = 1.2
        
        self.fovy_dist_offset = 1.0 / torch.tan(torch.deg2rad(fovy)/2) - 1.0
        self.fps = 30

        self.mesh = None

        self.robot2cam = euler_pos_2_mat(self.cam2base_xyz, self.cam2base_euler).to(self.device)
        self.warp2gym = torch.tensor([[0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 0, 1]], dtype=torch.float32).to(self.device)

        self.cam_intrinsics_matrix = torch.tensor([[384.77294921875, 0, 324.17236328125],
                                                   [0, 384.77294921875, 236.48226928710938],
                                                   [0, 0, 1]])
        self.depth_scale = 1000

    def render_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        with wp.ScopedDevice(self.device):
            vertices = np.matmul(self.warp2gym[:3,:3].cpu().numpy(), vertices.transpose(1,0)).transpose(1,0)
            indices = indices.flatten()
            self.mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device), velocities=None, indices=wp.array(indices, dtype=int, device=self.device))

    def render_pixels(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor: 
        """
        base_pos: (B, 3)
        base_quat: (B, 4)
        """
        b = base_pos.shape[0]
        pixels = wp.zeros(b * self.image_width * self.image_height, dtype=wp.vec3, device=self.device)

        gym2robot = quat_pos_2_mat_torch(base_pos, base_quat).to(self.device)
        warp2cam = self.warp2gym.unsqueeze(0).repeat([b,1,1]) @ gym2robot @ self.robot2cam
        cam_pos = warp2cam[:,:3,3].reshape(b*3)
        cam_rot = warp2cam[:,:3,:3].reshape(b*9)

        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel = draw_pixels,
                dim = b * self.image_width * self.image_height,
                inputs = [self.mesh.id, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), self.image_width, self.image_height, pixels],
                device = self.device
            )

        return wp.to_torch(pixels.reshape([b, self.image_height, self.image_width]))
    
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

        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel = draw_depth,
                dim = b * self.image_width * self.image_height,
                inputs = [self.mesh.id, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), self.image_width, self.image_height, depth, self.fovy_dist_offset],
                device = self.device
            )

        depth = wp.to_torch(depth.reshape([b, self.image_height, self.image_width])) #* self.depth_scale

        return depth

    def show_depth(self, depth: torch.Tensor):
        """
        depth: (B, H, W)
        """
        for i in range(depth.shape[0]):
            plt.imshow(depth[i].cpu().numpy(), cmap="gray")
            plt.title("depth image")
            plt.axis("off")
            plt.show()

    def show_color(self, color: torch.Tensor):
        """
        color: (B, H, W, 3)
        """
        for i in range(color.shape[0]):
            cv2.imshow("color image", color[i].cpu().numpy())
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
    
    def play_show_depth(self, depth: torch.Tensor):
        """
        depth: (H, W)
        """
        # absolutly depth image
        depth_ = depth #(depth/self.depth_scale)
        normalized_depth = (depth_ / self.far_plane * 255).cpu().numpy().astype(np.uint8)

        # relatively depth image
        # depth_max = torch.max(depth_)
        # depth_image = depth_ / depth_max * 255
        # normalized_depth = depth_image.cpu().numpy().astype(np.uint8)
        
        cv2.imshow("Depth Image", normalized_depth)
        cv2.waitKey(10)

if __name__ == "__main__":
    warprender = DepthRendererWarp([60, 106],
                      torch.tensor([0.10043, 0.0222, -0.11]),
                      torch.tensor([0, 60.95 / 180 * np.pi, 0]))