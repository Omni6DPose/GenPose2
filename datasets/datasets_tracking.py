import sys
import os
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
import json
sys.path.insert(0, '../')

from ipdb import set_trace
from time import perf_counter
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from utils.data_augmentation import defor_2D, get_rotation
from utils.data_augmentation import data_augment
from utils.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine
from utils.sgpa_utils import load_depth, get_bbox
from configs.config import get_config
from utils.misc import get_rot_matrix, get_pose_representation
from utils.tracking_utils import add_noise_to_R
from utils.transforms import *

from cutoop.data_loader import Dataset
from cutoop.eval_utils import *
from cutoop.transform import *
from cutoop.rotation import SymLabel
from cutoop.obj_meta import ObjectMetaData
from cutoop.image_meta import ImageMetaData, ObjectPoseInfo

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class Omni6DPoseDataSet(data.Dataset):
    """
    Almost the same as in datasets_genpose.py.
    """
    
    def __init__(self, 
                 cfg,
                 dynamic_zoom_in_params,
                 deform_2d_params,
                 source=None, 
                 mode='train', 
                 data_dir=None,
                 n_pts=1024, 
                 img_size=224, 
                 per_obj='',
                 ):
        '''
        :param source: 'ikea' or 'matterport3d' or 'scannet++' or 'Omni6DPose'
        :param mode: 'train' or 'test' or 'real'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''

        assert source in ['ikea', 'matterport3d', 'scannet++', 'Omni6DPose']
        assert mode in ['train', 'test', 'real']

        self.cfg = cfg
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params

        img_list = Dataset.glob_prefix(root = data_dir)
        assert len(img_list)
        folder_list = {os.path.dirname(path) for path in img_list}
        assert len(folder_list) == 1
        self.img_list = sorted(img_list)
        self.length = len(self.img_list)

        self.obj_meta = ObjectMetaData.load_json(
            "configs/obj_meta.json" if mode != 'real'
                else "configs/real_obj_meta.json")
        self.cat_names = [cl.name for cl in self.obj_meta.class_list]
        self.cat_name2id = {name: i for i, name in enumerate(self.cat_names)}
        self.id2cat_name = {str(i): name for i, name in enumerate(self.cat_names)}

        self.per_obj = per_obj
        self.per_obj_id = None

        tmp = []
        for img_path in self.img_list:
            gts: ImageMetaData = Dataset.load_meta(img_path + "meta.json")
            valid_objects = [obj for obj in gts.objects if obj.is_valid]
            self.num_valid = len(valid_objects)
            tmp.append({obj.meta.oid for obj in valid_objects})
        assert len(tmp) 
        try:
            for i in range(len(tmp)):
                assert tmp[i] == tmp[0]
        except:
            with open("tracking_fail.txt", "a") as f:
                f.write(list(folder_list)[0] + '\n')
            self.length = 0

        self.length *= self.num_valid

        assert not (self.per_obj in self.cat_names) or not cfg.load_per_object # for simplicity, not supported together
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_list[index // self.num_valid]
        gts: ImageMetaData = Dataset.load_meta(img_path + "meta.json")
        valid_objects = [obj for obj in gts.objects if obj.is_valid]
        valid_objects: "list[ObjectPoseInfo]" = sorted(valid_objects, key=lambda obj: obj.meta.oid)
        obj = valid_objects[index % self.num_valid]
        inst_name = obj.meta.oid

        rgb = Dataset.load_color(img_path + "color.png")
        depth = Dataset.load_depth(img_path + ('depth_syn' if self.cfg.perfect_depth else 'depth') + '.exr')
        depth[depth > 1e3] = 0
        mask = Dataset.load_mask(img_path + 'mask.exr')
        if not (mask.shape[:2] == depth.shape[:2] == rgb.shape[:2]):
            assert 0
            return self.__getitem__((index + 1) % self.__len__())
        assert mask.shape[:2] == depth.shape[:2] == rgb.shape[:2], set_trace()

        intrinsics = gts.camera.intrinsics
        img_resize_scale = rgb.shape[0] / intrinsics.height
        assert rgb.shape[1] / intrinsics.width == img_resize_scale
        mat_K = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 0]],
                         dtype=np.float32)  # [fx, fy, cx, cy]
        mat_K *= img_resize_scale
        mat_K[2, 2] = 1

        # from cutoop.transform import depth2xyz
        # xyz = depth2xyz(depth, intrinsics)

        # ys, xs = np.argwhere(np.equal(mask, inst_idx)).transpose(1, 0)
        # pts = xyz[ys, xs]

        # from cutoop.utils import save_pctxt
        # save_pctxt(f'pcl_{index}.txt', pts)
        # print(f'pcl_{index}.txt', img_path)

        if inst_name not in self.obj_meta.instance_dict:
            assert 0
            return self.__getitem__((index + 1) % self.__len__())

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)

        # coord = cv2.imread(img_path + '_coord.png')
        # if coord is not None:
        #     coord = coord[:, :, :3]
        #     pass
        # else:
        #     return self.__getitem__((index + 1) % self.__len__())

        # aggragate information about the selected object
        object_mask = np.equal(mask, obj.mask_id)
        if not np.any(object_mask):
            assert 0
            return self.__getitem__((index + 1) % self.__len__())
        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], im_H, im_W)
        # here resize and crop to a fixed size 224 x 224
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        roi_rgb_ = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, self.img_size, interpolation=cv2.INTER_LINEAR
        )
        roi_rgb = Omni6DPoseDataSet.rgb_transform(roi_rgb_)

        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != obj.mask_id] = 0.0
        mask_target[mask == obj.mask_id] = 1.0

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_depth = np.expand_dims(roi_depth, axis=0)
        # normalize depth
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            assert 0
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            assert 0
            return self.__getitem__((index + 1) % self.__len__())

        # cat_id, rotation translation and scale
        cat_id = obj.meta.class_label
        # note that this is nocs model, normalized along diagonal axis
        # model = self.models[inst_name].astype(np.float32)
        # model = model * np.array(obj.meta.scale) # to image space
        # nocs_scale = np.linalg.norm(np.max(model, axis=0) - np.min(model, axis=0), ord=2)
        # model /= nocs_scale # to nocs space
        # model -= (np.max(model, axis=0) + np.min(model, axis=0)) / 2
        # fsnet scale (from model) scale residual  (currently deleted)
        # fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        # fsnet_scale = fsnet_scale / 1000.0
        # mean_shape = mean_shape / 1000.0
        rotation: torch.Tensor = quaternion_to_matrix(torch.tensor(obj.quaternion_wxyz))
        translation = obj.translation
        # add nnoise to roi_mask
        
        roi_mask_def = defor_2D(
            roi_mask, 
            rand_r=self.deform_2d_params['roi_mask_r'], 
            rand_pro=self.deform_2d_params['roi_mask_pro']
        )
        valid = (np.squeeze(roi_depth, axis=0) > 0) * roi_mask_def > 0
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        pcl_in = self._depth_to_pcl(roi_depth, mat_K, roi_coord_2d, valid)
        # np.savetxt('pts_def.txt', pcl_in)
        
        if len(pcl_in) < 50:
            assert 0
            return self.__getitem__((index + 1) % self.__len__())
        ids, pcl_in = self._sample_points(pcl_in, self.n_pts)
        xs, ys = xs[ids], ys[ids]
        # sym
        # sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)
        sym_info = self.obj_meta.instance_dict[inst_name].tag.symmetry
        sym_idx = {'none': 0, 'any': 1, 'half': 2, 'quarter': 3}
        sym_info = [int(sym_info.any), sym_idx[sym_info.x], sym_idx[sym_info.y], sym_idx[sym_info.z]]

        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        # data_dict['cat_id'] = torch.as_tensor(cat_id, dtype=torch.int8).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        affine = torch.eye(4)
        affine[:3, :3] = data_dict['rotation']
        affine[:3, 3] = data_dict['translation']
        data_dict['affine'] = torch.as_tensor(affine, dtype=torch.float32).contiguous()
        # data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info, dtype=torch.int8).contiguous()
        # data_dict['sym_info'] = torch.as_tensor(self.get_sym_info('bowl', mug_handle=1).astype(np.float32)).contiguous()
        # data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        # data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        # data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        # data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        # data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous()
        # data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous()
        data_dict['handle_visibility'] = torch.as_tensor(1, dtype=torch.int8).contiguous()
        data_dict['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
        # data_dict['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
        # data_dict['rgb'] = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8).contiguous()
        data_dict['roi_xs'] = torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous()
        data_dict['roi_ys'] = torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous()
        data_dict['roi_center_dir'] = torch.as_tensor(pixel2xyz(im_H, im_W, bbox_center, intrinsics), dtype=torch.float32).contiguous()
        intrinsics_list = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.width, intrinsics.height]
        data_dict['intrinsics'] = torch.as_tensor(intrinsics_list, dtype=torch.float32).contiguous()
        data_dict['bbox_side_len'] = torch.as_tensor(np.array(obj.meta.bbox_side_len), dtype=torch.float32).contiguous()
        data_dict['pose'] = torch.as_tensor(obj.quaternion_wxyz + obj.translation, dtype=torch.float32).contiguous()
        data_dict['path'] = img_path
        data_dict['class_label'] = cat_id
        data_dict['class_name'] = obj.meta.class_name
        data_dict['object_name'] = inst_name
        data_dict['is_valid'] = 1

        # xyz = depth2xyz(depth, intrinsics)
        # choose = np.logical_and(mask == inst_idx, depth > 0).flatten().nonzero()[0]
        # points = xyz.reshape((-1, 3))[choose, :]
        # nocs = coord.reshape((-1, 3))[choose, :]
        # bbox_side_len = np.array(obj["meta"]["bbox_side_len"])
        # s, _, _, estimated_RT = estimateSimilarityTransform(nocs, points)
        # estimated_RT[:3, :3] /= s
        # # set_trace()
        # vis_img = draw_3d_bbox(
        #     Dataset.load_color(img_path + "color.png"),
        #     intrinsics,
        #     estimated_RT,
        #     bbox_side_len=bbox_side_len,
        # )
        # cv2.imwrite(f'visualization/_{index}visual_es.png', vis_img)
        # RT = toSRT(Pose(obj['quaternion_wxyz'], obj['translation']))
        # vis_img = draw_3d_bbox(
        #     Dataset.load_color(img_path + "color.png"),
        #     intrinsics,
        #     RT,
        #     bbox_side_len=bbox_side_len,
        # )
        # cv2.imwrite(f'visualization/_{index}visual_gt.png', vis_img)
        # set_trace()

        return data_dict

    def _sample_points(self, pcl, n_pts):
        """ Down sample the point cloud.

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
            ids = np.concatenate([np.tile(np.arange(total_pts_num), n_pts // total_pts_num), np.arange(n_pts % total_pts_num)], axis=0)
        else:
            ids = np.random.permutation(total_pts_num)[:n_pts]
            pcl = pcl[ids]
        return ids, pcl
    
    def _depth_to_pcl(self, depth, K, xymap, valid):
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        depth = depth.reshape(-1).astype(np.float32)[valid]
        x_map = xymap[0].reshape(-1)[valid]
        y_map = xymap[1].reshape(-1)[valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)
        return pcl.astype(np.float32)

    def rgb_transform(rgb):
        rgb_ = np.transpose(rgb, (2, 0, 1)) / 255
        _mean = (0.485, 0.456, 0.406)
        _std = (0.229, 0.224, 0.225)
        for i in range(3):
            rgb_[i, :, :] = (rgb_[i, :, :] - _mean[i]) / _std[i]
        return rgb_

def array_to_SymLabel(arr_Nx4: np.ndarray):
    syms_N = []
    tags = ['none', 'any', 'half', 'quarter']
    for a, x, y, z in arr_Nx4:
        syms_N.append(SymLabel(bool(a), tags[x], tags[y], tags[z]))
    return syms_N

def array_to_CameraIntrinsicsBase(intrinsics_list):
    return [CameraIntrinsicsBase(*item) for item in intrinsics_list]


def process_batch(batch_sample,
                  device,
                  pose_mode='quat_wxyz',
                  PTS_AUG_PARAMS=None):
    
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
        f"the rotation mode {pose_mode} is not supported!"
    if PTS_AUG_PARAMS==None:
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)
    else: 
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)
        # PC_da, gt_R_da, gt_t_da, gt_s_da = data_augment(
        #     pts_aug_params=PTS_AUG_PARAMS,
        #     PC=batch_sample['pcl_in'].to(device), 
        #     gt_R=batch_sample['rotation'].to(device), 
        #     gt_t=batch_sample['translation'].to(device),
        #     gt_s=batch_sample['fsnet_scale'].to(device), 
        #     mean_shape=batch_sample['mean_shape'].to(device),
        #     sym=batch_sample['sym_info'].to(device),
        #     aug_bb=batch_sample['aug_bb'].to(device), 
        #     aug_rt_t=batch_sample['aug_rt_t'].to(device),
        #     aug_rt_r=batch_sample['aug_rt_R'].to(device),
        #     model_point=batch_sample['model_point'].to(device), 
        #     nocs_scale=batch_sample['nocs_scale'].to(device),
        #     obj_ids=batch_sample['cat_id'].to(device), 
        # )

    processed_sample = {}
    processed_sample['pts'] = PC_da                # [bs, 1024, 3]
    processed_sample['pts_color'] = PC_da          # [bs, 1024, 3]
    # processed_sample['id'] = batch_sample['cat_id'].to(device)      # [bs]
    processed_sample['sym_info'] = batch_sample['sym_info']  # [bs, 4]
    # processed_sample['handle_visibility'] = batch_sample['handle_visibility'].to(device)     # [bs]
    # processed_sample['path'] = batch_sample['path']
    processed_sample['roi_rgb'] = batch_sample['roi_rgb'].to(device) # [bs, 3, imgsize, imgsize]
    assert processed_sample['roi_rgb'].shape[-1] == processed_sample['roi_rgb'].shape[-2]
    assert processed_sample['roi_rgb'].shape[-1] % 14 == 0
    processed_sample['roi_xs'] = batch_sample['roi_xs'].to(device) # [bs, 1024]
    processed_sample['roi_ys'] = batch_sample['roi_ys'].to(device) # [bs, 1024]
    processed_sample['roi_center_dir'] = batch_sample['roi_center_dir'].to(device) # [bs, 3]
    if 'axes_training' in batch_sample:
        processed_sample['axes_training'] = batch_sample['axes_training'].to(device) # [bs, cbs, 3, 3]
        processed_sample['length_training'] = batch_sample['length_training'].to(device) # [bs, cbs, 3]
        processed_sample['axes_transform'] = batch_sample['axes_transform'].to(device) # [bs, cbs]

    rot = get_pose_representation(gt_R_da, pose_mode)
    location = gt_t_da # [bs, 3]
    processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)   # [bs, 4/6/3 + 3]
    
    """ zero center """
    num_pts = processed_sample['pts'].shape[1]
    zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
    processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
    processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
    processed_sample['zero_mean_gt_pose'] = copy.deepcopy(processed_sample['gt_pose'])
    processed_sample['zero_mean_gt_pose'][:, -3:] -= zero_mean
    processed_sample['pts_center'] = zero_mean
    
    if 'color' in batch_sample.keys():
        pass
        # processed_sample['color'] = batch_sample['color'].to(device)       # [bs]
    
    if not 'color' in processed_sample.keys():
        pass
        # processed_sample['color'] = None
    # print(processed_sample['zero_mean_pts'].device)
    return processed_sample 
    