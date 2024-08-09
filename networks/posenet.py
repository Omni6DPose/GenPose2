import sys
import os
import torch
import torch.nn as nn

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
from networks.gf_algorithms.samplers import cond_ode_likelihood, cond_ode_sampler, cond_pc_sampler
from networks.gf_algorithms.scorenet import PoseScoreNet, PoseDecoderNet
from networks.gf_algorithms.energynet import PoseEnergyNet
from networks.gf_algorithms.sde import init_sde
from networks.scalenet import ScaleNet
from configs.config import get_config
from utils.genpose_utils import encode_axes



class GFObjectPose(nn.Module):
    dino_name = 'dinov2_vits14'
    dino_dim = 384
    embedding_dim = 60

    def __init__(self, cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T):
        super(GFObjectPose, self).__init__()
        
        self.cfg = cfg
        self.device = cfg.device
        self.is_testing = False
        
        ''' Load model, define SDE '''
        # init SDE config
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sampling_eps = sampling_eps
        self.T = T
        # self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps = init_sde(cfg.sde_mode)
        
        ''' dino v2 '''
        if cfg.dino != 'none':
            self.dino : nn.Module = torch.hub.load('facebookresearch/dinov2', GFObjectPose.dino_name).to(cfg.device)
            self.dino.requires_grad_(False)
            self.dino_dim = GFObjectPose.dino_dim
            self.embedding_dim = GFObjectPose.embedding_dim
        
        ''' encode pts '''
        if self.cfg.pts_encoder == 'pointnet':
            assert cfg.dino != 'pointwise' # not supported yet
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
        elif self.cfg.pts_encoder == 'pointnet2':
            if cfg.dino == 'pointwise':
                self.pts_encoder = Pointnet2ClsMSGFus(self.dino_dim)
            else:
                self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            assert cfg.dino != 'pointwise' # not supported yet
            self.pts_pointnet_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2_encoder = Pointnet2ClsMSG(0)
            self.fusion_layer = nn.Linear(2048, 1024)
            self.act = nn.ReLU()
        else:
            raise NotImplementedError
        
        ''' score network'''
        # if self.cfg.sde_mode == 'edm':
        #     self.pose_score_net = PoseDecoderNet(
        #         self.marginal_prob_fn,
        #         sigma_data=1.4148, 
        #         pose_mode=self.cfg.pose_mode, 
        #         regression_head=self.cfg.regression_head
        #     )
        # else:
        per_point_feat = False
        if self.cfg.agent_type == 'score':
            self.pose_score_net = PoseScoreNet(
                self.marginal_prob_fn, 
                (0 if self.cfg.dino != 'global' else self.dino_dim + self.embedding_dim),
                self.cfg.pose_mode, 
                self.cfg.regression_head, 
                per_point_feat
            )
        elif self.cfg.agent_type == 'energy':
            self.pose_score_net = PoseEnergyNet(
                marginal_prob_func=self.marginal_prob_fn, 
                dino_dim=(0 if self.cfg.dino != 'global' else self.dino_dim + self.embedding_dim),
                pose_mode=self.cfg.pose_mode,
                regression_head=self.cfg.regression_head,
                energy_mode=self.cfg.energy_mode,
                s_theta_mode=self.cfg.s_theta_mode,
                norm_energy=self.cfg.norm_energy)
        ''' ToDo: ranking network '''

    def extract_pts_feature(self, data):
        """extract the input pointcloud feature

        Args:
            data (dict): batch example without pointcloud feature. {'pts': [bs, num_pts, 3], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        Returns:
            data (dict): batch example with pointcloud feature. {'pts': [bs, num_pts, 3], 'pts_feat': [bs, c], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        """
        pts = data['pts']
        if self.cfg.dino == 'pointwise':
            roi_rgb = data['roi_rgb']
            feat = self.dino.get_intermediate_layers(roi_rgb)[0]
            xs = data['roi_xs'] // 14
            ys = data['roi_ys'] // 14
            pos = xs * 16 + ys
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
            rgb_feat = torch.gather(feat, 1, pos)
            rgb_feat.requires_grad_(False)
        if self.cfg.pts_encoder == 'pointnet':
            assert 0
            pts_feat = self.pts_encoder(pts.permute(0, 2, 1))    # -> (bs, 3, 1024)
        elif self.cfg.pts_encoder in ['pointnet2']:
            if self.cfg.dino == 'pointwise':
                pts_feat = self.pts_encoder(torch.concatenate([pts, rgb_feat], dim=-1))
            else:
                pts_feat = self.pts_encoder(pts)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            assert 0
            pts_pointnet_feat = self.pts_pointnet_encoder(pts.permute(0, 2, 1))
            pts_pointnet2_feat = self.pts_pointnet2_encoder(pts)
            pts_feat = self.fusion_layer(torch.cat((pts_pointnet_feat, pts_pointnet2_feat), dim=-1))
            pts_feat = self.act(pts_feat)
        else:
            raise NotImplementedError
        return pts_feat
    
   
    def sample(self, data, sampler, atol=1e-5, rtol=1e-5, snr=0.16, denoise=True, init_x=None, T0=None):
        if sampler == 'pc':
            in_process_sample, res = cond_pc_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                num_steps=self.cfg.sampling_steps,
                snr=snr,
                device=self.device,
                eps=self.sampling_eps,
                pose_mode=self.cfg.pose_mode,
                init_x=init_x
            )
            
        elif sampler == 'ode':
            T0 = self.T if T0 is None else T0
            in_process_sample, res =  cond_ode_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                atol=atol,
                rtol=rtol,
                device=self.device,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.cfg.sampling_steps,
                pose_mode=self.cfg.pose_mode,
                denoise=denoise,
                init_x=init_x
            )
        
        else:
            raise NotImplementedError
        
        return in_process_sample, res
    
   
    def calc_likelihood(self, data, atol=1e-5, rtol=1e-5):    
        latent_code, log_likelihoods = cond_ode_likelihood(
            score_model=self,
            data=data,
            prior=self.prior_fn,
            sde_coeff=self.sde_fn,
            marginal_prob_fn=self.marginal_prob_fn,
            atol=atol,
            rtol=rtol,
            device=self.device,
            eps=self.sampling_eps,
            num_steps=self.cfg.sampling_steps,
            pose_mode=self.cfg.pose_mode,
        )
        return log_likelihoods

    
    def forward(self, data, mode='score', init_x=None, T0=None):
        '''
        Args:
            data, dict {
                'pts': [bs, num_pts, 3]
                'pts_feat': [bs, c]
                'sampled_pose': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        if mode == 'score':
            out_score = self.pose_score_net(data) # normalisation
            return out_score
        elif mode == 'energy':
            out_energy = self.pose_score_net(data, return_item='energy')
            return out_energy
        elif mode == 'likelihood':
            likelihoods = self.calc_likelihood(data)
            return likelihoods
        elif mode == 'pts_feature':
            pts_feature = self.extract_pts_feature(data)
            return pts_feature
        elif mode == 'rgb_feature':
            if self.cfg.dino != 'global':
                return None
            rgb: torch.Tensor = data['roi_rgb']
            assert rgb.shape[-1] % 14 == 0 and rgb.shape[-2] % 14 == 0
            assert self.embedding_dim % 6 == 0
            # lose backward compatibility
            return torch.concat([self.dino(rgb), encode_axes(data['roi_center_dir'], dim=self.embedding_dim // 6)], dim=-1) 
        
            # positional_embedding = []
            # exponent = (2 ** torch.arange(self.embedding_dim // 6, device=rgb.device, dtype=torch.float32)).reshape(1, -1)
            # for i in range(3):
            #     for fn in [torch.sin, torch.cos]:
            #         positional_embedding.append(fn(exponent * data['roi_center_dir'][:, i].reshape(-1, 1)))
            # positional_embedding = torch.concat(positional_embedding, dim=-1)
            # return torch.concat([self.dino(rgb), positional_embedding], dim=-1)
        elif mode == 'pc_sample':
            in_process_sample, res = self.sample(data, 'pc', init_x=init_x)
            return in_process_sample, res
        elif mode == 'ode_sample':
            in_process_sample, res = self.sample(data, 'ode', init_x=init_x, T0=T0)
            return in_process_sample, res
        else:
            raise NotImplementedError



def test():
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    cfg = get_config()
    prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T = init_sde('ve')
    net = GFObjectPose(cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T)
    net_parameters_num= get_parameter_number(net)
    print(net_parameters_num['Total'], net_parameters_num['Trainable'])
if __name__ == '__main__':
    test()

