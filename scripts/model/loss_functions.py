import torch
from torch.nn import functional as F
import numpy as np
from scripts.model.diff_operators import gradient    
from pytorch_metric_learning import losses


def compute_loss(batch, decoder, latent_idx,device):
    batch_cuda_npm = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}

    idx = batch.get('idx').to(device)
    #spc = batch.get('spc').to(device)
    glob_cond_idx = latent_idx(idx) # 1,1,512
   # glob_cond_spc = latent_spc(spc)
   #  glob_cond = torch.cat((glob_cond_idx,glob_cond_spc.unsqueeze(1)),dim=2)
    loss_dict = actual_compute_loss(batch_cuda_npm, decoder, glob_cond_idx)
    return loss_dict

    
def actual_compute_loss(batch_cuda, decoder, glob_cond):
    # prep
    sup_surface = batch_cuda['points'].clone().detach().requires_grad_() 
    sdf_gt = batch_cuda['sdf_gt'].clone().detach().requires_grad_()
    # model computations
    pred_surface = decoder(sup_surface, glob_cond.unsqueeze(1).repeat(1, sup_surface.shape[1], 1))
    # computation of losses for geometry mse loss
    sdf_mse_loss = F.mse_loss(pred_surface.squeeze(-1), sdf_gt)
    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)
    ret_dict = {'surf_sdf': torch.mean(sdf_mse_loss),
                    'lat_reg':lat_mag.mean()}
    return ret_dict


def compute_sdf_3d_loss(batch, decoder, latent_idx,device):
    batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    idx = batch.get('idx').to(device)
    glob_cond= latent_idx(idx)
    
    # prep data
    sup_surface = batch_cuda['points'].clone().detach().requires_grad_() # points on face surf
    sup_grad_far = batch_cuda['sup_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_grad_near = batch_cuda['sup_grad_near'].clone().detach().requires_grad_() # points near/off surface
    # sdf_gt_near = batch_cuda['sup_grad_near_sdf'].clone().detach().requires_grad_() 
    # sdf_gt_far = batch_cuda['sup_grad_far_sdf'].clone().detach().requires_grad_()
    
    # model computations
    pred_surface = decoder(sup_surface, glob_cond.repeat(1, sup_surface.shape[1], 1))
    pred_space_near = decoder(sup_grad_near, glob_cond.repeat(1, sup_grad_near.shape[1], 1))
    pred_space_far = decoder(sup_grad_far, glob_cond.repeat(1, sup_grad_far.shape[1], 1))

    # normal computation
    gradient_surface = gradient(pred_surface, sup_surface)
    gradient_space_far = gradient(pred_space_far, sup_grad_far)
    gradient_space_near = gradient(pred_space_near, sup_grad_near)


    
    # loss computation
    surf_sdf_loss = torch.abs(pred_surface).squeeze()
    surf_normal_loss = (gradient_surface - batch_cuda['normals']).norm(2,dim=-1)
    surf_grad_loss = (gradient_surface.norm(dim=-1) - 1)

    space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_space_far))
    space_grad_loss_far = torch.abs(gradient_space_far.norm(dim=-1) - 1)
    space_grad_loss_near = torch.abs(gradient_space_near.norm(dim=-1) - 1)

    grad_loss = torch.cat([surf_grad_loss, space_grad_loss_far, space_grad_loss_near], dim=-1)


    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)


    ret_dict = {'surf_sdf': torch.mean(surf_sdf_loss),
                  'normals': torch.mean(surf_normal_loss),
              'space_sdf': torch.mean(space_sdf_loss),
                'grad': torch.mean(grad_loss),
                    'lat_reg':lat_mag.mean()}
    return ret_dict

def compute_loss_corresp_forward(batch, decoder, decoder_shape, latent_deform, latent_shape, device, cfg):
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    glob_cond = latent_deform(batch['idx'].to(device))
    label = batch['label'].to(device)
   # glob_cond = torch.cat([glob_cond_shape.unsqueeze(1), glob_cond_pose], dim=-1)

    points_neutral = batch_cuda['points_neutral'].clone().detach().requires_grad_()

    cond = glob_cond.repeat(1, points_neutral.shape[1], 1)
    delta= decoder(points_neutral, cond)
    pred_posed = points_neutral + delta.squeeze()
    # mse loss
    points_posed = batch_cuda['points_posed']
    loss_corresp = (pred_posed - points_posed[:, :, :3])**2#.abs()
    
    loss_infonce = torch.tensor(0)
    loss_distance = torch.tensor(0)
    # distance regularizer
    if cfg['use_distance']:
        distance = torch.norm(glob_cond,p=2,dim=-1)
        delta_gt = points_posed - points_neutral
        delta_norm = torch.norm(delta_gt,p=2,dim=(1,2)) /10
        loss_distance = ((distance.squeeze()/delta_norm) - 1)**2
    
    # nce loss
    if cfg['use_nce']:
        nce = losses.NTXentLoss(temperature=0.5).cuda()
        loss_infonce = nce(glob_cond.squeeze(), label.squeeze())
    
    # latent code regularization
    lat_mag = torch.norm(glob_cond, dim=-1)**2
    samps = (torch.rand(cond.shape[0], 100, 3, device=cond.device, dtype=cond.dtype) -0.5)*2.5
    delta = decoder(samps, cond[:, :100, :])
    loss_reg_zero = (delta**2).mean()
    return {'corresp': loss_corresp.mean(),
            'lat_reg': lat_mag.mean(),
            'loss_nce': loss_infonce,
            'loss_reg_zero': loss_reg_zero,
            'loss_distance': loss_distance.mean()}


def compute_color_forward(batch, decoder, decoder_shape, latent_codes, latent_codes_shape, device, epoch=-1, exp_path=None):
    
    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    # create  
    points = batch_cuda['points'].clone().detach().requires_grad_()
    idx_shape = torch.ones(32,1,dtype=torch.int64)
    glob_cond_shape = latent_codes_shape(idx_shape.to(device))
    glob_cond_color = latent_codes(batch['idx'].to(device))
    cond_shape = glob_cond_shape.repeat(1, points.shape[1],1)
    # sdf, _  = decoder_shape(points,cond_shape,None)

    
    cond_color = glob_cond_color.repeat(1, points.shape[1], 1)
    color, _ = decoder(points, cond_color, None)
    return color




def inversion_loss(batch, encoder,device):
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    points = batch_cuda['points'].clone().detach().requires_grad_()
   
  
    latent_shape_pred, latent_deform_pred = encoder(points.permute(0,2,1))
    loss_latent_shape = F.mse_loss(latent_shape_pred, batch_cuda['latent_shape'])
    loss_latent_deform = F.mse_loss(latent_deform_pred, batch_cuda['latent_deform'])
    loss_dict = {
        'loss_latent_shape': loss_latent_shape,
        'loss_latent_deform': loss_latent_deform,
    }
    return loss_dict
