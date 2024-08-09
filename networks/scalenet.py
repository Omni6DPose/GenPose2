import sys
import os
import torch
import numpy as np
import torch.nn as nn

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import encode_axes
from gf_algorithms.scorenet import zero_module

class ScaleNet(nn.Module):
    def __init__(self, pts_dim, dino_dim=0, embedding_dim=180):
        super(ScaleNet, self).__init__()
        self.pts_dim = pts_dim
        self.dino_dim = dino_dim
        self.embedding_dim = embedding_dim
        assert embedding_dim % 18 == 0

        self.act = nn.ReLU(True)
        self.axes_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        self.fusion_tail_length = nn.Sequential(
            nn.Linear(pts_dim + dino_dim + 256, 256),
            self.act,
            zero_module(nn.Linear(256, 3))
        )

    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, pts_dim]
                'rgb_feat': [bs, dino_dim] (optional)
                'axes': [bs, 3, 3]
            }
        
        Return: 
            Length: [bs, 3]
        '''
        axes_feat = self.axes_encoder(encode_axes(data['axes'], self.embedding_dim // 18))
        total_feat = torch.cat([data['pts_feat'], axes_feat], dim=-1)
        if self.dino_dim:
            total_feat = torch.cat([total_feat, data['rgb_feat']], dim=-1)
        return self.fusion_tail_length(total_feat)

    def loss_fn(self, pred_len, gt_len):
        '''
        pred_len: [bs, 3]
        gt_len: [bs, 3]
        '''
        return torch.mean((pred_len - gt_len) ** 2) * 10000
