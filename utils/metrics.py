import sys
sys.path.append('..')

import torch
import numpy as np
import pickle

from utils.misc import get_rot_matrix, inverse_RT
from utils.genpose_utils import get_pose_dim
from ipdb import set_trace

from cutoop.rotation import SymLabel
from cutoop.eval_utils import DetectMatch

def get_metrics(pose_1, pose_2, sym_info, pose_mode):
    """
    pose_1: pred pose, BxP
    pose_2: gt pose, BxP
    sym_info: Bx4
    """
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"

    index = get_pose_dim(pose_mode) - 3

    rot_1 = pose_1[:, :index]
    rot_2 = pose_2[:, :index]
    trans_1 = pose_1[:, index:]
    trans_2 = pose_2[:, index:]
    
    rot_matrix_1 = get_rot_matrix(rot_1, pose_mode)
    rot_matrix_2 = get_rot_matrix(rot_2, pose_mode)
    
    bs = pose_1.shape[0]
    RT_1 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
    RT_2 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
    
    RT_1[:, :3, :3] = rot_matrix_1
    RT_1[:, :3, 3] = trans_1
    RT_2[:, :3, :3] = rot_matrix_2
    RT_2[:, :3, 3] = trans_2
    
    sym_info = sym_info.cpu().numpy()
    assert len(sym_info.shape) == 2 and sym_info.shape[1] == 4
    syms_N = []
    for a, x, y, z in sym_info:
        tags = ['none', 'any', 'half', 'quarter']
        syms_N.append(SymLabel(bool(a), tags[x], tags[y], tags[z]))
    match = DetectMatch(
        gt_affine=RT_2.cpu().numpy(),
        gt_size=np.ones((RT_2.shape[0], 3)),
        gt_sym_labels=syms_N,
        gt_class_labels=np.zeros((RT_2.shape[0], 3)),
        pred_affine=RT_1.cpu().numpy(),
        pred_size=np.ones((RT_1.shape[0], 3)),
        image_path=None,
        camera_intrinsics=None,
    )
    match = match.callibrate_rotation()
    _, rot_error, trans_error = match.criterion(False)
    return rot_error, trans_error
