import torch
import torch.nn as nn
import sys 
from .pose_sampler import Plane
from .pose import invert_rot_t
import numpy as np
import os
import yaml
from .lighting import build_directional_light_optimizable
from .camera_network import  Camera

class Generator(nn.Module):
    def __init__(self, sdf_network, color_network, cfg):
        super().__init__()
        self.cfg =cfg
        self.batch_size = self.cfg['training']['batch_size']
        self.resolution = self.cfg['training']['resolution']
        self.sdf_network = sdf_network
        self.color_network = color_network
        self.pose_sampler = Plane(cam_loc=self.cfg['b2w_scene_prior']['kwargs']['cam_loc'],
                         rot_degree_range_scale=self.cfg['b2w_scene_prior']['kwargs']['rot_degree_range_scale'], 
                         rot_roll_degree_range_scale=self.cfg['b2w_scene_prior']['kwargs']['rot_roll_degree_range_scale'],
                        xy_range_scale=self.cfg['b2w_scene_prior']['kwargs']['xy_range_scale'])
        self.light = build_directional_light_optimizable(None,None)
        self.camera = Camera(cam_dist=self.cfg['camera']['kwargs']['cam_dist'],
                    resolution=self.cfg['camera']['kwargs']['resolution'],
                    fov=self.cfg['camera']['kwargs']['fov'])
    
    def sample_prior(self, batch_size, data):
        prior_info = {}
        if 'b2w' in data:
            b2w = data['b2w']
        else:
            b2w = torch.tensor(self.pose_sampler(batch_size), dtype=torch.float32, device='cuda')
            w2b = invert_rot_t(b2w)
        c2b = torch.einsum('bij,jk->bik', w2b, self.camera.c2w)  # (b, 4, 4)
        light = self.light.batch_transform(w2b=w2b)
        prior_info['c2b'] = c2b
        prior_info['b2w'] = b2w
        prior_info['light'] = light
        return prior_info
            

                
    def forward(self, data, latent_info):
       prior_info = self.sample_prior(self.batch_size, data)
       rays_info = self.gen_rays_at(data, prior_info)
       render_out = self.render(rays_info, latent_info)
       pass
    
    def render(self, rays_info, **kwargs):
        pass
        
    def gen_rays_at(self, data, prior_info):
        b2w = prior_info['b2w']
        b2c = torch.einsum('ij,bjk->bik', self.camera.w2c, b2w)
        b2c_trans = b2c[..., :3, 3]

        """ crop around the box """

        center_x = self.camera.cam_dist / b2c_trans[..., 2] * b2c_trans[..., 0] * self.resolution / 2 + 1 / 2 * self.scene_resolution
        center_y = self.camera.cam_dist / b2c_trans[..., 2] * b2c_trans[..., 1] * self.resolution / 2 + 1 / 2 * self.scene_resolution

        x_offset = center_x - self.resolution / 2
        y_offset = center_y - self.resolution / 2

        rays_v = build_rays(
            h_recp_size=self.resolution, w_recp_size=self.resolution,
            h_offset=y_offset, w_offset=x_offset,
            num_rays_h=self.resolution, num_rays_w=self.resolution,
            intrinsics=self.camera.intrinsics, intrinsics_inv=self.camera.intrinsics_inv,
        )

        # from camera to world to box frame
        c2b = prior_info['c2b']
        rays_v = torch.einsum('bij,bhwj->bhwi', c2b[..., :3, :3], rays_v)  # (b, h, w, 3)
        rays_o = c2b[:, None, None, :3, 3].expand(rays_v.shape)  # (b, h, w, 3)
        return {'rays_o': rays_o, 'rays_d': rays_v, 'x_offset': x_offset, 'y_offset': y_offset}
        
if __name__ == "__main__":
    cfg_path = '/home/yyang/projects/parametric-leaf/NPLM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    pose_sampler = Plane(cam_loc=CFG['b2w_scene_prior']['kwargs']['cam_loc'],
                         rot_degree_range_scale=CFG['b2w_scene_prior']['kwargs']['rot_degree_range_scale'], 
                         rot_roll_degree_range_scale=CFG['b2w_scene_prior']['kwargs']['rot_roll_degree_range_scale'],
                        xy_range_scale=CFG['b2w_scene_prior']['kwargs']['xy_range_scale'])
    light = build_directional_light_optimizable(None,None)
    camera = Camera(cam_dist=CFG['camera']['kwargs']['cam_dist'],
                    resolution=CFG['camera']['kwargs']['resolution'],
                    fov=CFG['camera']['kwargs']['fov'])
    
    pass