import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
from cutoop.image_meta import ImageMetaData

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

        img_list = Dataset.glob_prefix(root = os.path.join(
            data_dir, '*', 
            ('' if mode == 'real' else mode), 
            ('' if source == 'Omni6DPose' else source)
        ))
        assert len(img_list)
        if mode == 'real':
            folder_list = {os.path.dirname(path) for path in img_list}
            img_list = []
            for folder in sorted(folder_list):
                il = sorted(Dataset.glob_prefix(root = folder))[::cfg.real_drop] # only keep part of the data
                img_list += il
        self.img_list = img_list
        self.length = len(self.img_list)

        self.obj_meta = ObjectMetaData.load_json(
            "configs/obj_meta.json" if mode != 'real'
                else "configs/real_obj_meta.json")
        self.cat_names = [cl.name for cl in self.obj_meta.class_list]
        self.cat_name2id = {name: i for i, name in enumerate(self.cat_names)}
        self.id2cat_name = {str(i): name for i, name in enumerate(self.cat_names)}

        self.per_obj = per_obj
        self.per_obj_id = None
        
        # only train one object
        if self.per_obj:
            assert self.per_obj in self.cat_names, "invalid per_obj!"
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(img_list_cache_filename)]
            else:
                # needs to reorganize img_list
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gts = Dataset.load_meta(img_list[i] + "meta.json")
                    if any(item.is_valid and item.meta.class_name == self.per_obj for item in gts.objects):
                        img_list_obj.append(img_list[i])
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
            
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        print('{} images found.'.format(self.length))

        self.REPCNT = 8 if mode == 'train' and not cfg.load_per_object else 1
        self.length *= self.REPCNT

        if cfg.load_per_object:
            self.cumsum = []
            for img_path in self.img_list:
                gts: ImageMetaData = Dataset.load_meta(img_path + "meta.json")
                valid_objects = [obj for obj in gts.objects if obj.is_valid]
                self.cumsum.append(len(valid_objects))
            self.cumsum = np.cumsum(self.cumsum)
            self.length = self.cumsum[-1]

        assert not self.per_obj or not cfg.load_per_object # for simplicity, not supported together
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.cfg.load_per_object:
            img_path = self.img_list[index // self.REPCNT]
            gts: ImageMetaData = Dataset.load_meta(img_path + "meta.json")
            if not any(obj.is_valid for obj in gts.objects):
                return self.__getitem__((index + 1) % self.__len__())
            # select one foreground object,
            # if specified, then select the object
            if self.per_obj:
                obj = sorted([
                    obj for obj in gts.objects 
                    if obj.is_valid and obj.meta.class_name == self.per_obj
                ], key=lambda obj: obj.meta.oid)[0]
            else:
                valid_objects = [obj for obj in gts.objects if obj.is_valid]
                select_idx = index % self.REPCNT
                if select_idx < self.REPCNT - self.REPCNT % len(valid_objects):
                    obj = valid_objects[select_idx % len(valid_objects)]
                else:
                    obj = random.sample(valid_objects, 1)[0]
        else:
            ii = np.searchsorted(self.cumsum, index, side='right')
            img_path = self.img_list[ii]
            gts: ImageMetaData = Dataset.load_meta(img_path + "meta.json")
            valid_objects = [obj for obj in gts.objects if obj.is_valid]
            obj = valid_objects[index - (0 if ii == 0 else self.cumsum[ii - 1])]
        inst_name = obj.meta.oid

        rgb = Dataset.load_color(img_path + "color.png")
        depth = Dataset.load_depth(img_path + ('depth_syn' if self.cfg.perfect_depth else 'depth') + '.exr')
        depth[depth > 1e3] = 0
        mask = Dataset.load_mask(img_path + 'mask.exr')
        if not (mask.shape[:2] == depth.shape[:2] == rgb.shape[:2]):
            assert 0, "invalid data"

        intrinsics = gts.camera.intrinsics
        img_resize_scale = rgb.shape[0] / intrinsics.height
        assert rgb.shape[1] / intrinsics.width == img_resize_scale
        mat_K = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 0]],
                         dtype=np.float32)  # [fx, fy, cx, cy]
        mat_K *= img_resize_scale
        mat_K[2, 2] = 1

        ## visualize pointcloud
        # from cutoop.transform import depth2xyz
        # xyz = depth2xyz(depth, intrinsics)
        # ys, xs = np.argwhere(np.equal(mask, inst_idx)).transpose(1, 0)
        # pts = xyz[ys, xs]
        # from cutoop.utils import save_pctxt
        # save_pctxt(f'pcl_{index}.txt', pts)
        # print(f'pcl_{index}.txt', img_path)

        if inst_name not in self.obj_meta.instance_dict:
            return self.__getitem__((index + 1) % self.__len__())

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0) # xy map

        # aggragate information about the selected object
        object_mask = np.equal(mask, obj.mask_id)
        if not np.any(object_mask):
            return self.__getitem__((index + 1) % self.__len__())
        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([rmin, cmin, rmax, cmax], im_H, im_W)
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
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
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_depth = np.expand_dims(roi_depth, axis=0)
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())

        cat_id = obj.meta.class_label
        rotation: torch.Tensor = quaternion_to_matrix(torch.tensor(obj.quaternion_wxyz))
        translation = obj.translation
        
        # pointclouds, and corresponding coordinates on image
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
            return self.__getitem__((index + 1) % self.__len__())
        ids, pcl_in = self._sample_points(pcl_in, self.n_pts)
        xs, ys = xs[ids], ys[ids]
        
        # sym
        sym_info = self.obj_meta.instance_dict[inst_name].tag.symmetry
        sym_idx = {'none': 0, 'any': 1, 'half': 2, 'quarter': 3}
        sym_info = [int(sym_info.any), sym_idx[sym_info.x], sym_idx[sym_info.y], sym_idx[sym_info.z]]

        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        affine = torch.eye(4)
        affine[:3, :3] = data_dict['rotation']
        affine[:3, 3] = data_dict['translation']
        data_dict['affine'] = torch.as_tensor(affine, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info, dtype=torch.int8).contiguous()
        data_dict['handle_visibility'] = torch.as_tensor(1, dtype=torch.int8).contiguous()
        data_dict['roi_rgb'] = torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous()
        data_dict['roi_rgb_'] = torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous()
        # uncommenting this line may incur error, as images from different datasets have different sizes.
        # for debugging only.
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
        if self.cfg.agent_type == 'scale':
            length_training = torch.as_tensor(obj.meta.bbox_side_len, dtype=torch.float32)
            axes4x4_training = torch.zeros(self.cfg.scale_batch_size, 4, 4)
            axes4x4_training[:, :3, :3], length_training = \
                rotation.unsqueeze(0).repeat_interleave(self.cfg.scale_batch_size, dim=0), \
                length_training.unsqueeze(0).repeat_interleave(self.cfg.scale_batch_size, dim=0), 
            axes4x4_training[:, 3, 3] = 1
            axes_training = add_noise_to_R(axes4x4_training, r=10)[:, :3, :3]
            data_dict['axes_training'] = axes_training.contiguous()
            data_dict['length_training'] = length_training.contiguous()

        ## some more visualization
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
        TODO: use farthest point sampling

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

def get_data_loaders(
    cfg,
    batch_size,
    seed,
    dynamic_zoom_in_params,
    deform_2d_params,
    percentage_data=1.0,
    data_path=None,
    source='CAMERA+Real',
    mode='train',
    n_pts=1024,
    img_size=224,
    per_obj='',
    num_workers=32,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    assert source in ['ikea', 'matterport3d', 'scannet++', 'Omni6DPose']
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=dynamic_zoom_in_params,
        deform_2d_params=deform_2d_params,
        source=source,
        mode=mode,
        data_dir=data_path,
        n_pts=n_pts,
        img_size=img_size,
        per_obj=per_obj,
    )
    
    ####### sanity check for dinov2
    # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
    # N = 40
    # allusefulfeats = np.zeros((1, 384))
    # for i in range(N):
    #     data = dataset[i]
    #     roi_rgb = data['roi_rgb'].unsqueeze(0).to('cuda')
    #     feat = dino.get_intermediate_layers(roi_rgb)[0].detach().cpu().numpy().squeeze(0)
    #     xs = data['roi_xs'] // 14
    #     ys = data['roi_ys'] // 14
    #     if xs.max() >= 16 or ys.max() >= 16:
    #         set_trace()
    #     useful_feat = feat.reshape((16,16,-1))[xs, ys]
    #     allusefulfeats = np.concatenate([allusefulfeats, useful_feat], axis=0)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=3)
    # pca.fit(allusefulfeats)
    # for i in range(N):
    #     data = dataset[i]
    #     rgb = data['rgb'].numpy()
    #     roi_rgb_ = data['roi_rgb_'].numpy()
    #     roi_rgb = data['roi_rgb'].unsqueeze(0).to('cuda')
    #     feat = dino.get_intermediate_layers(roi_rgb)[0].detach().cpu().numpy().squeeze(0)
    #     pca.fit(useful_feat)
    #     img = pca.transform(feat).reshape((16,16,3))
    #     img -= img.min()
    #     img *= 255 / img.max()
    #     img = img.astype(np.uint8)
    #     img = cv2.resize(img, (224, 224), cv2.INTER_NEAREST)
    #     cv2.imwrite('tmp.png', roi_rgb_)
    #     cv2.imwrite('tmp2.png', rgb)
    #     cv2.imwrite('feat.png', img)
    #     set_trace()

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    # sample
    size = int(percentage_data * len(dataset))
    dataset, _ = torch.utils.data.random_split(dataset, (size, len(dataset) - size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )

    dataset = Omni6DPoseDataSet(
        dynamic_zoom_in_params=dynamic_zoom_in_params,
        deform_2d_params=deform_2d_params,
        source=source,
        mode=mode,
        data_dir=data_path,
        n_pts=n_pts,
        img_size=img_size,
        per_obj=per_obj,
        cfg=cfg
    )

    return dataloader


def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test']):
    data_loaders = {}
    if 'train' in data_type:
        train_loader = get_data_loaders(
            cfg=cfg,
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_train,            
            data_path=cfg.data_path,
            source=cfg.train_source,
            mode='train',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['train_loader'] = train_loader
        
    if 'val' in data_type:
        val_loader = get_data_loaders(
            cfg=cfg,
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_val,            
            data_path=cfg.data_path,
            source=cfg.val_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['val_loader'] = val_loader
        
    if 'test' in data_type:
        test_loader = get_data_loaders(
            cfg=cfg,
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_test,            
            data_path=cfg.data_path,
            source=cfg.test_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_loader'] = test_loader
        
    return data_loaders


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
    elif 'old_sym_info' in batch_sample: # NOCS augmentation
        PC_da, gt_R_da, gt_t_da, gt_s_da = data_augment(
            pts_aug_params=PTS_AUG_PARAMS,
            PC=batch_sample['pcl_in'].to(device), 
            gt_R=batch_sample['rotation'].to(device), 
            gt_t=batch_sample['translation'].to(device),
            gt_s=batch_sample['fsnet_scale'].to(device), 
            mean_shape=batch_sample['mean_shape'].to(device),
            sym=batch_sample['old_sym_info'].to(device),
            aug_bb=batch_sample['aug_bb'].to(device), 
            aug_rt_t=batch_sample['aug_rt_t'].to(device),
            aug_rt_r=batch_sample['aug_rt_R'].to(device),
            model_point=batch_sample['model_point'].to(device), 
            nocs_scale=batch_sample['nocs_scale'].to(device),
            obj_ids=batch_sample['cat_id'].to(device), 
        )
    else:
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)

    processed_sample = {}
    processed_sample['pts'] = PC_da                # [bs, 1024, 3]
    processed_sample['pts_color'] = PC_da          # [bs, 1024, 3]
    processed_sample['sym_info'] = batch_sample['sym_info']  # [bs, 4]
    processed_sample['roi_rgb'] = batch_sample['roi_rgb'].to(device) # [bs, 3, imgsize, imgsize]
    assert processed_sample['roi_rgb'].shape[-1] == processed_sample['roi_rgb'].shape[-2]
    assert processed_sample['roi_rgb'].shape[-1] % 14 == 0
    processed_sample['roi_xs'] = batch_sample['roi_xs'].to(device) # [bs, 1024]
    processed_sample['roi_ys'] = batch_sample['roi_ys'].to(device) # [bs, 1024]
    processed_sample['roi_center_dir'] = batch_sample['roi_center_dir'].to(device) # [bs, 3]
    if 'axes_training' in batch_sample:
        processed_sample['axes_training'] = batch_sample['axes_training'].to(device) # [bs, cbs, 3, 3]
        processed_sample['length_training'] = batch_sample['length_training'].to(device) # [bs, cbs, 3]

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

    return processed_sample 
    

if __name__ == '__main__':
    """
    The code below is for visualizing overlap between features of
    synthetic images and real images.
    """
    cfg = get_config()
    classes = ['mug', 'hair_dryer', 'remote_control', 'toy_plane']
    dataset_real = [Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir='/data1/real_data',
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=cls,
    ) for cls in classes]
    dataset_syn = [Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='train',
        data_dir='/data1/render_v1_down',
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=cls,
    ) for cls in classes]
    datasets = [dataset_real, dataset_syn]
    modes = ['real', 'syn']
    print("dataset ready")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
    print("dino ready")
    save_dir = f"dinofeat"
    os.makedirs(save_dir, exist_ok=True)
    N, M = 100, 300
    full_features = [[], []]
    for j in range(2):
        for k in range(len(classes)):
            dataset = datasets[j][k]
            full_features[j].append([])
            for i in tqdm(range(M if modes[j] == 'syn' else N)):
                index = i * 998244353 % len(dataset)
                data = dataset[index]
                roi_rgb = data['roi_rgb'].unsqueeze(0).to('cuda')
                feat = dino.get_intermediate_layers(roi_rgb)[0].detach().cpu().numpy().squeeze(0)
                xs, ys = data['roi_xs'] // 14, data['roi_ys'] // 14
                feat = np.mean(feat.reshape((16,16,-1))[xs, ys], axis=0)
                full_features[j][k].append(feat)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, early_exaggeration=1, perplexity=10)
    FEAT_LENGTH = full_features[0][0][0].shape[0]

    all_feat = tsne.fit_transform(
        np.concatenate(
            [
                np.array(full_features[0]).reshape(len(classes), N, FEAT_LENGTH),
                np.array(full_features[1]).reshape(len(classes), M, FEAT_LENGTH)
            ],
            axis=1
        ).reshape(-1, FEAT_LENGTH)
    ).reshape(len(classes), N + M, 2)
    feat = [all_feat[:, :N, :], all_feat[:, N:, :]]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 6))
    fig, ax = plt.subplots()
    for k in range(len(classes)):
        for j in [1, 0]:
            ax.scatter(feat[j][k, :, 0], feat[j][k, :, 1], 
                        label=f"{classes[k]}-{modes[j]}", s=8, alpha=0.5 if modes[j] == 'syn' else 0.2)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.58, 0.5))
    plt.savefig(f"{save_dir}/img.png")
