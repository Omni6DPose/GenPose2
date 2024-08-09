''' modified from CAPTRA https://github.com/HalfSummer11/CAPTRA/tree/5d7d088c3de49389a90b5fae280e96409e7246c6 '''

import torch
import copy
import math
from ipdb import set_trace

from utils.transforms import matrix_to_quaternion, quaternion_to_matrix

def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(torch.clamp(norm, min=1e-9))

def generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = normalize(rand_norm)
    return rand_q


def jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q

def noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_quaternion(matrix)
    new_quater = jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = quaternion_to_matrix(new_quater)
    return new_mat


def add_noise_to_R(RT, type='normal', r=5.0, t=0.03):
    rand_type = type  # 'uniform' or 'normal' --> we use 'normal'

    def random_tensor(base):
        if rand_type == 'uniform':
            return torch.rand_like(base) * 2.0 - 1.0
        elif rand_type == 'normal':
            return torch.randn_like(base)
    new_RT: torch.Tensor = copy.deepcopy(RT)
    new_RT[:, :3, :3] = noisy_rot_matrix(RT[:, :3, :3], r/180*math.pi, type=rand_type).reshape(RT[:, :3, :3].shape)
    assert not torch.any(torch.isnan(new_RT)) and not torch.any(torch.isinf(new_RT))

    return new_RT

def add_noise_to_RT(RT, type='normal', r=5.0, t=0.03):
    rand_type = type  # 'uniform' or 'normal' --> we use 'normal'

    def random_tensor(base):
        if rand_type == 'uniform':
            return torch.rand_like(base) * 2.0 - 1.0
        elif rand_type == 'normal':
            return torch.randn_like(base)
    new_RT = copy.deepcopy(RT)
    new_RT[:, :3, :3] = noisy_rot_matrix(RT[:, :3, :3], r/180*math.pi, type=rand_type).reshape(RT[:, :3, :3].shape)
    norm = random_tensor(RT[:, 0, 0]) * t  # [B, P]
    direction = random_tensor(RT[:, :3, 3].squeeze(-1))  # [B, P, 3]
    direction = direction / torch.clamp(direction.norm(dim=-1, keepdim=True), min=1e-9)  # [B, P, 3] unit vecs
    new_RT[:, :3, 3] = RT[:, :3, 3] + (direction * norm.unsqueeze(-1))  # [B, P, 3, 1]
    assert not torch.any(torch.isnan(new_RT)) and not torch.any(torch.isinf(new_RT))

    return new_RT

if __name__ == '__main__':
    from configs.config import get_config
    cfg = get_config()
    n = 100
    rand_rot = quaternion_to_matrix(generate_random_quaternion((n, 4)))
    rand_mat = torch.zeros((n, 4, 4))
    rand_mat[:, :3, :3] = rand_rot
    rand_mat[:, 3, 3] = 1
    noise_mat = add_noise_to_R(rand_mat, r=10)
    transform = torch.randint(0, 48, (n,))
    rand_trans = torch.zeros((n, 3))
    rand_trans[:, 2] = 1
    rand_mat[:, :3, 3] = rand_trans
    noise_mat[:, :3, 3] = rand_trans
    size = torch.ones((n, 3)) * 0.2
    from cutoop.eval_utils import DetectMatch
    from cutoop.rotation import SymLabel
    from cutoop.data_types import CameraIntrinsicsBase
    import numpy as np
    width, height = 640, 480
    import cv2
    cv2.imwrite('black.png', np.zeros((height, width)))
    dm = DetectMatch(
        gt_affine=rand_mat.numpy(),
        gt_size=size.numpy(),
        gt_sym_labels=[SymLabel(False, 'none', 'none', 'none')] * n,
        gt_class_labels=[0] * n,
        pred_affine=noise_mat.numpy(),
        pred_size=size.numpy(),
        image_path=['black.png'] * n,
        camera_intrinsics=[CameraIntrinsicsBase(577.5, 577.5, 319.5, 239.5, width, height)] * n
    )
    _, rot_error, trans_error = dm.criterion(False)
    print(np.mean(rot_error), np.median(rot_error), np.mean(trans_error), np.median(trans_error))
    for i in range(n):
        print(i, rand_trans[i])
        dm.draw_image('visualization/ww.png', i)
        set_trace()