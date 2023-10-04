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
import math
from .neus.renderer import NeuSRenderer

MAX_RAY_BATCH_SIZE = 128 * 128 * 1

class Generator(nn.Module):
    def __init__(self, sdf_network, color_network, deviation_network,cfg):
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
        self.resolution = self.cfg['training']['resolution']
        self.scene_resolution = self.cfg['camera']['kwargs']['resolution']
        self.deviation_network = deviation_network
        self.renderer = NeuSRenderer(nerf=None,sdf_network=self.sdf_network,
            deviation_network= self.deviation_network,
            color_network= self.color_network,
            n_samples=self.cfg['renderer']['kwargs']['n_samples'],
            n_outside=self.cfg['renderer']['kwargs']['n_outside'],
            n_importance=self.cfg['renderer']['kwargs']['n_importance'],
            up_sample_steps=self.cfg['renderer']['kwargs']['up_sample_steps'],
            perturb=self.cfg['renderer']['kwargs']['perturb'],       
        )
        self.anneal_end = 50000
        
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
            

                
    def forward(self, batch, latent_info):
       prior_info = self.sample_prior(self.batch_size, batch)
       rays_info = self.gen_rays_at(batch, prior_info)
       render_out = self.render(rays_info, latent_info,batch)

       pass
    
    def render(self, rays_info,latent_info, batch):
        rays_o = rays_info['rays_o']
        rays_d = rays_info['rays_d']
        bs = self.batch_size
        max_ray_bs = MAX_RAY_BATCH_SIZE
        chunk_size = int(max_ray_bs/bs)
        num_chunks = math.ceil(rays_o.flatten(1,2).shape[1]/chunk_size) 
        if num_chunks >1:
            assert not self.training, (rays_o.shape, chunk_size, num_chunks,max_ray_bs)
        render_out = None
        for chunk_ind in range(num_chunks):
            rays_o_chunk = rays_o.flatten(1, 2)[:, chunk_ind * chunk_size:(chunk_ind + 1) * chunk_size].flatten(0, 1)
            rays_d_chunk = rays_d.flatten(1, 2)[:, chunk_ind * chunk_size:(chunk_ind + 1) * chunk_size].flatten(0, 1)
            chunk_out = self.render_one_chunk(rays_o_chunk, rays_d_chunk, latent_info,batch)
        pass
        
    def render_one_chunk(self, rays_o, rays_d, latent_info, batch):
        near, far = near_far_from_sphere(rays_o, rays_d)
        latent_shape = torch.cat((latent_info['shape'].squeeze(0), latent_info['species']), dim=1)
        latent_color = latent_info['color'].squeeze(0)
        cos_anneal_ratio = np.min([1.0, 0 / self.anneal_end])
        render_out = self.renderer.render(rays_o, rays_d,near, far,
                                          background_rgb=None,
                                          cos_anneal_ratio=cos_anneal_ratio,
                                          perturb_overwrite=1 if self.training else 0,
                                          shape_latent = latent_shape,
                                          color_latent= latent_color,
                                          data=batch)
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

def build_rays(
        *,
        h_recp_size: int, w_recp_size: int,  # they can technically be tensors of shape (bs,)
        h_offset: torch.Tensor, w_offset: torch.Tensor,
        num_rays_h: int, num_rays_w: int,
        intrinsics: torch.Tensor, intrinsics_inv: torch.Tensor,
):
    tx = torch.linspace(0, 1, num_rays_w, device=intrinsics.device)
    ty = torch.linspace(0, 1, num_rays_h, device=intrinsics.device)
    pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')  # (w, h)
    pixels_x = pixels_x * w_recp_size + w_offset[..., None, None]  # (..., w, h)
    pixels_y = pixels_y * h_recp_size + h_offset[..., None, None]  # (..., w, h)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # ..., w, h, 3
    p = torch.einsum('ij,...whj->...whi', intrinsics_inv[:3, :3], p)  # ..., w, h, 3
    p = torch.einsum('...whi->...hwi', p)
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    return rays_v


def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far
        
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