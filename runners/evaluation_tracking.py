import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config


''' load config '''
cfg = get_config()
cfg.load_per_object = True


import sys
import time
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import shutil
import hashlib
import random
import gc
from sklearn.cluster import DBSCAN

from ipdb import set_trace

from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy, ranking_loss
from datasets.datasets_tracking import Omni6DPoseDataSet, array_to_SymLabel, array_to_CameraIntrinsicsBase, process_batch
from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image
from utils.tracking_utils import add_noise_to_RT
from cutoop.eval_utils import DetectMatch, Metrics
from cutoop.data_loader import Dataset

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

def get_dataloader(data_dir: str):
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=data_dir,
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.num_valid,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )
    return iter(dataloader)

cfg.agent_type = 'score'
score_agent = PoseNet(cfg)
score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
score_agent.eval()

cfg.agent_type = 'energy'
energy_agent = PoseNet(cfg)
energy_agent.load_ckpt(model_dir=cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
energy_agent.eval()

if cfg.pretrained_scale_model_path:
    cfg.agent_type = 'scale'
    scale_agent = PoseNet(cfg)
    scale_agent.load_ckpt(model_dir=cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
    scale_agent.eval()

def work_batch(test_batch, prev_pose):
    batch_sample = process_batch(
        batch_sample = test_batch, 
        device=cfg.device, 
        pose_mode=cfg.pose_mode,
    )
    
    _prev_pose = prev_pose.clone()
    _prev_pose[:, -3:] -= batch_sample['pts_center']
    cfg.agent_type = 'score'
    score_pred_results, _ = score_agent.pred_func(
        data=batch_sample, 
        repeat_num=cfg.eval_repeat_num, 
        T0=cfg.T0,
        init_x=_prev_pose,
        return_average_res=False,
        return_process=False,
    )
    score_feature = {
        'pts_feat': batch_sample['pts_feat'].clone(),
        'rgb_feat': (None if batch_sample['rgb_feat'] is None else batch_sample['rgb_feat'].clone()),
    }
    
    cfg.agent_type = 'energy'
    energy_pred_results = energy_agent.get_energy(
        data=batch_sample, 
        pose_samples=score_pred_results, 
        T=1e-5,
        mode='test', 
        extract_feature=True
    )

    sorted_pose, sorted_energy = sort_poses_by_energy(score_pred_results, energy_pred_results)
    bs = score_pred_results.shape[0]
    retain_num = int(cfg.eval_repeat_num * cfg.retain_ratio)
    good_pose = sorted_pose[:, :retain_num, :]
    rot_matrix = get_rot_matrix(good_pose[:, :, :-3].reshape(bs * retain_num, -1), cfg.pose_mode)
    quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
    aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)
    if cfg.clustering:
        for j in range(bs):
            # https://math.stackexchange.com/a/90098
            # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
            pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
            dbscan = DBSCAN(eps=cfg.clustering_eps, min_samples=int(cfg.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().cpu().numpy())
            labels = dbscan.labels_
            if np.any(labels >= 0):
                bins = np.bincount(labels[labels >= 0])
                best_label = np.argmax(bins)
                aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0))[0]
    aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
    aggregated_pose = torch.zeros(bs, 4, 4)
    aggregated_pose[:, 3, 3] = 1
    aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
    aggregated_pose[:, :3, 3] = aggregated_trans

    pred_pose = aggregated_pose.numpy()
    gt_pose = test_batch['affine'].numpy()
    gt_length = test_batch['bbox_side_len'].numpy()

    if cfg.pretrained_scale_model_path:
        cfg.agent_type = 'scale'
        batch_sample.update(score_feature)
        batch_sample['axes'] = aggregated_pose[:, :3, :3].to(cfg.device)
        with torch.no_grad():
            pred_length = scale_agent.net(batch_sample) 
        pred_length = pred_length.cpu().numpy()
    else:
        pred_length = np.ones((pred_pose.shape[0], 3))

    detect_match = DetectMatch(
        gt_affine=gt_pose, gt_size=gt_length, 
        gt_sym_labels=array_to_SymLabel(test_batch['sym_info']), 
        gt_class_labels=test_batch['class_label'],
        pred_affine=pred_pose, pred_size=pred_length,
        # image_path=[path + 'color.png' for path in test_batch['path']],
        camera_intrinsics=array_to_CameraIntrinsicsBase(test_batch['intrinsics'])
    )

    prev_pose = torch.zeros_like(prev_pose, device=cfg.device)
    prev_pose[:, :-3] = get_pose_representation(aggregated_pose[:, :3, :3], cfg.pose_mode)
    prev_pose[:, -3:] = aggregated_pose[:, :3, 3]
    
    return detect_match, prev_pose

img_list = Dataset.glob_prefix(root = cfg.data_path)
video_paths = sorted({os.path.dirname(path) for path in img_list})

dataloaders: "set[torch.utils.data.DataLoader]" = set()

idx = 0
def add_dataloader():
    global idx
    while idx < len(video_paths):
        path = video_paths[idx]
        idx += 1
        save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
        if os.path.exists(save_path):
            continue
        dataloader = get_dataloader(path)
        dataloader.save_path = save_path
        dataloader.all_detect_match = []
        dataloaders.add(dataloader)
        break

total_objects = 0
for path in tqdm(video_paths):
    save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
    if os.path.exists(save_path):
        continue
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=path,
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=None,
    )
    total_objects += len(dataset)
pbar = tqdm(total=total_objects)

for i in range(30):
    add_dataloader()

while 1:
    test_batch = []
    prev_pose = []
    split_pos = [(0,  None)]
    dd = set()
    for i, dataloader in enumerate(dataloaders):
        try:
            batch = dataloader.__next__()
        except StopIteration:
            if len(dataloader.all_detect_match): # otherwise, an error occurred in the dataset
                all_detect_match = DetectMatch.concat(dataloader.all_detect_match)
                all_detect_match = all_detect_match.calibrate_rotation()
                os.makedirs(os.path.dirname(dataloader.save_path), exist_ok="True")
                pickle.dump(all_detect_match, open(dataloader.save_path, "wb"))
            dd.add(dataloader)
            continue
        except AssertionError:
            with open("tracking_fail.txt", "a") as f:
                f.write(dataloader.save_path + '\n')
            dd.add(dataloader)
            continue
        test_batch.append(batch)
        length = dataloader._dataset.num_valid
        try:
            prev_pose.append(dataloader.prev_pose)
        except:
            pose = torch.zeros(length, get_pose_dim(cfg.pose_mode), device=cfg.device) # on gpu
            assert batch['affine'].shape[0] == length, set_trace()
            for j in range(length):
                noise_gt_pose = add_noise_to_RT(batch['affine'][j].to(cfg.device).unsqueeze(0))[0]
                pose[j, :-3] = get_pose_representation(
                    noise_gt_pose[:3, :3].unsqueeze(0), 
                    pose_mode=cfg.pose_mode
                )[0]
                pose[j, -3:] = noise_gt_pose[:3, 3]
            prev_pose.append(pose)
        split_pos.append((split_pos[-1][0] + length, dataloader))
        if split_pos[-1][0] > cfg.batch_size - 8:
            break
    if test_batch == []:
        break
    
    keys = {key for key, value in test_batch[0].items() if type(value) != list}
    test_batch = {
        key: torch.concat([batch[key] for batch in test_batch]) for key in keys
    }
    prev_pose = torch.concat(prev_pose)
    
    detect_match, prev_pose = work_batch(test_batch, prev_pose)
    for i in range(len(split_pos) - 1):
        l, r = split_pos[i][0], split_pos[i+1][0]
        dataloader = split_pos[i+1][1]
        dataloader.all_detect_match.append(detect_match[l:r])
        dataloader.prev_pose = prev_pose[l:r]
    
    pbar.update(split_pos[-1][0])
        
    for dl in dd:
        dataloaders.remove(dl)
        add_dataloader()
    
    gc.collect()

pbar.close()

all_dm = []
all_crit = []
for path in tqdm(video_paths):
    save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
    if not os.path.exists(save_path):
        continue
    prefix = os.path.dirname(save_path)
    dm: DetectMatch = pickle.load(open(save_path, "rb"))
    all_dm.append(dm)
    crit_path = os.path.join(prefix, "criterion.pkl")
    if not os.path.exists(crit_path):
        criterion = dm.criterion(computeIOU=True)
        pickle.dump(criterion, open(crit_path, "wb"))
    else:
        criterion = pickle.load(open(crit_path, "rb"))
    all_crit.append(criterion)

all_dm = DetectMatch.concat(all_dm)
all_crit = [np.concatenate([all_crit[j][i] for j in range(len(all_crit))]) for i in range(3)]

metrics: Metrics = all_dm.metrics(
    criterion=all_crit,
    iou_auc_ranges=[
        (0.25, 1, 0.075),
        (0.5, 1, 0.005),
        (0.75, 1, 0.0025),
    ],
    pose_auc_ranges=[
        ((0, 5, 0.05), (0, 2, 0.02)),
        ((0, 5, 0.05), (0, 5, 0.05)),
        ((0, 10, 0.1), (0, 2, 0.02)),
        ((0, 10, 0.1), (0, 5, 0.05)),
    ],
)
print("iou_mean:", metrics.class_means.iou_mean)
print("iou_acc (0.25, 0.50, 0.75):", metrics.class_means.iou_acc)
print("deg_mean:", metrics.class_means.deg_mean)
print("sht_mean:", metrics.class_means.sht_mean)
print("pose_acc [(5, 2), (5, 5), (10, 2), (10, 5)]:", metrics.class_means.pose_acc)
print("AUC @ IoU 25:", metrics.class_means.iou_auc[0].auc)
print("AUC @ IoU 50:", metrics.class_means.iou_auc[1].auc)
print("AUC @ IoU 75:", metrics.class_means.iou_auc[2].auc)
print("VUS @ 5 deg 2 cm:", metrics.class_means.pose_auc[0].auc)
print("VUS @ 5 deg 5 cm:", metrics.class_means.pose_auc[1].auc)
print("VUS @ 10 deg 2 cm:", metrics.class_means.pose_auc[2].auc)
print("VUS @ 10 deg 5 cm:", metrics.class_means.pose_auc[3].auc)
metrics.dump_json(os.path.join(f"results/evaluation_results/{cfg.result_dir}", "metrics.json"))