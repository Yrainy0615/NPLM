import torch
from torch.nn import functional as F
import numpy as np
from scripts.model.diff_operators import gradient    


def compute_loss(batch, decoder, latent_idx, latent_spc,device):
    batch_cuda_npm = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}

    idx = batch.get('idx').to(device)
    spc = batch.get('spc').to(device)
    glob_cond_idx = latent_idx(idx) # 1,1,512
    glob_cond_spc = latent_spc(spc)
    glob_cond = torch.cat((glob_cond_idx,glob_cond_spc.unsqueeze(1)),dim=2)
    loss_dict = actual_compute_loss(batch_cuda_npm, decoder, glob_cond_idx,glob_cond_spc)

    return loss_dict

def actual_compute_loss(batch_cuda, decoder, glob_cond, glob_cond2):
    anchor_preds = None
    glob_cond = torch.cat((glob_cond,glob_cond2.unsqueeze(1)),dim=2)

    # prep
    sup_surface = batch_cuda['points'].clone().detach().requires_grad_() # points on face surf
    #sup_surface_outer = batch_cuda['points_non_face'].clone().detach().requires_grad_() # points on non-face surf
    sup_grad_far = batch_cuda['sup_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_grad_near = batch_cuda['sup_grad_near'].clone().detach().requires_grad_() # points near/off surface
    udf_gt_near = batch_cuda['sup_grad_near_udf'].clone().detach().requires_grad_() 
    udf_gt_far = batch_cuda['sup_grad_far_udf'].clone().detach().requires_grad_()

    # model computations
    pred_surface = decoder(sup_surface, glob_cond.repeat(1, sup_surface.shape[1], 1))
    pred_space_near = decoder(sup_grad_near, glob_cond.repeat(1, sup_grad_near.shape[1], 1))
    pred_space_far = decoder(sup_grad_far, glob_cond.repeat(1, sup_grad_far.shape[1], 1))
    gradient_surface = gradient(pred_surface, sup_surface)
    
    # losses
    loss_surf_udf = pred_surface.squeeze()
    loss_space_near = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_space_near.squeeze(), max=0.0001), torch.clamp(udf_gt_near, max=0.0001))
    loss_space_far = torch.nn.L1Loss(reduction='none')(pred_space_far.squeeze(), udf_gt_far)
    loss_normal = (gradient_surface - batch_cuda['normals']).norm(2, dim=-1)

    lat_idx = torch.norm(glob_cond, dim=-1) ** 2
    lat_spc = torch.norm(glob_cond2, dim=-1) ** 2
    ret_dict = {'surf_udf': torch.mean(loss_surf_udf),
                'normals': torch.mean(loss_normal),
                'space_near': torch.mean(loss_space_near),
                'space_far': torch.mean(loss_space_far),
                'lat_idx':lat_idx.mean(),
                'lat_spc':lat_spc.mean(),}
    return ret_dict


def loss_joint(batch, decoder_shape, decoder_expr, latent_codes_shape, latent_codes_expr, device, epoch):
    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    cond_shape = latent_codes_shape(batch['subj_ind'].to(device))

    cond_expr = latent_codes_expr(batch['idx'].to(device))

    cond_cat = torch.cat([cond_shape, cond_expr], dim=-1)

    is_neutral = batch_cuda['is_neutral'].squeeze(dim=-1)==1

    # joint losses
    if epoch >= 250 or True:
        # on surface, face
        points_posed = batch_cuda['points_surface'].clone().detach().requires_grad_()
        points_posed_offset, _ = decoder_expr(points_posed, cond_cat.repeat(1, points_posed.shape[1], 1), None)
        points_can = points_posed_offset + points_posed
        pred_sdf_surface, anchors_pred = decoder_shape(points_can, cond_shape.repeat(1, points_can.shape[1], 1), None)
        gradient_surface = gradient(pred_sdf_surface, points_posed)

        surf_sdf_loss = torch.abs(pred_sdf_surface).squeeze(dim=-1)
        surf_normal_loss = (gradient_surface - batch_cuda['normals_surface']).norm(2, dim=-1)

        surf_grad_loss = torch.abs(gradient_surface.norm(dim=-1) - 1)


        if torch.sum(is_neutral) > 0:
            # on surface, back of head
            points_posed_outer = (batch_cuda['points_surface_outer'][is_neutral, ...]).clone().detach().requires_grad_()
            points_posed_outer_offset, _ = decoder_expr(points_posed_outer,
                                                        cond_cat.repeat(1, points_posed_outer.shape[1], 1)[is_neutral, ...], None)
            points_outer_can = points_posed_outer + points_posed_outer_offset

            pred_sdf_outer, _ = decoder_shape(points_outer_can,
                                              cond_shape.repeat(1, points_outer_can.shape[1], 1)[is_neutral, ...], None)
            gradient_outer = gradient(pred_sdf_outer, points_posed_outer)

            surf_sdf_loss_outer = torch.abs(pred_sdf_outer).squeeze(dim=-1)
            surf_normal_loss_outer = torch.clamp((gradient_outer - batch_cuda['normals_surface_outer'][is_neutral, ...]).norm(2, dim=-1),
                                                 None,
                                                 0.75*100) / 2#8
            surf_grad_loss_outer = torch.abs(gradient_outer.norm(dim=-1) - 1)

            # off surface
            points_posed_off = (batch_cuda['points_off_surface'][is_neutral, ...]).clone().detach().requires_grad_()
            points_posed_off_offset, _ = decoder_expr(points_posed_off,
                                                        cond_cat.repeat(1, points_posed_off.shape[1], 1)[is_neutral, ...], None)
            points_off_can = points_posed_off + points_posed_off_offset

            pred_sdf_off, _ = decoder_shape(points_off_can, cond_shape.repeat(1, points_off_can.shape[1], 1)[is_neutral, ...], None)
            gradient_off = gradient(pred_sdf_off, points_posed_off)

            surf_sdf_loss_off = torch.abs(pred_sdf_off - batch_cuda['sdfs_off_surface'][is_neutral, ...]).squeeze(dim=-1)
            surf_normal_loss_off = torch.clamp((gradient_off - batch_cuda['normals_off_surface'][is_neutral, ...]).norm(2, dim=-1),
                                                 None,
                                                 0.75*100) / 2#8
            surf_grad_loss_off = torch.abs(gradient_off.norm(dim=-1) - 1)


    # off surface, canonical space only
    sup_grad_far = batch_cuda['sup_grad_far'].clone().detach().requires_grad_()
    pred_sdf_far, anchors_pred = decoder_shape(sup_grad_far, cond_shape.repeat(1, sup_grad_far.shape[1], 1), None)
    gradient_space_far = gradient(pred_sdf_far, sup_grad_far)

    space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_sdf_far)).mean()
    space_grad_loss_far = torch.abs(gradient_space_far.norm(dim=-1) - 1)


    if is_neutral.sum() > 0:
        tot_sdf_loss = torch.cat([surf_sdf_loss.reshape(-1), surf_sdf_loss_outer.reshape(-1), surf_sdf_loss_off.reshape(-1),],
                                 dim=0).mean()
        tot_normal_loss = torch.cat(
            [surf_normal_loss.reshape(-1), surf_normal_loss_outer.reshape(-1), surf_normal_loss_off.reshape(-1)], dim=0).mean()

        grad_loss = torch.cat([space_grad_loss_far.reshape(-1),
                               surf_grad_loss.reshape(-1),
                               surf_grad_loss_outer.reshape(-1),
                               surf_grad_loss_off.reshape(-1)], dim=-0).mean()
    else:
        tot_sdf_loss = torch.cat(
            [surf_sdf_loss.reshape(-1)],
            dim=0).mean()
        tot_normal_loss = torch.cat(
            [surf_normal_loss.reshape(-1)],
            dim=0).mean()

        grad_loss = torch.cat([space_grad_loss_far.reshape(-1),
                               surf_grad_loss.reshape(-1),
                               ], dim=-0).mean()


    # latent regularizers
    lat_reg_shape = torch.norm(cond_shape, dim=-1)**2
    lat_reg_expr = torch.norm(cond_expr, dim=-1)**2

    _cond_shape = cond_shape.squeeze(1)
    if hasattr(decoder_shape, 'lat_dim_glob'):
        shape_dim_glob = decoder_shape.lat_dim_glob
        shape_dim_loc = decoder_shape.lat_dim_loc
        n_symm = decoder_shape.num_symm_pairs
        loc_lats_symm = _cond_shape[:, shape_dim_glob:shape_dim_glob+2*n_symm*shape_dim_loc].view(_cond_shape.shape[0], n_symm*2, shape_dim_loc)
        loc_lats_middle = _cond_shape[:, shape_dim_glob + 2*n_symm*shape_dim_loc:-shape_dim_loc].view(_cond_shape.shape[0], decoder_shape.num_kps - n_symm*2, shape_dim_loc)

        symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
        if loc_lats_middle.shape[1] % 2 == 0:
            middle_dist = torch.norm(loc_lats_middle[:, ::2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        else:
            middle_dist = torch.norm(loc_lats_middle[:, :-1:2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
    else:
        symm_dist = None
        middle_dist = None


    loss_anchors = (anchors_pred - batch_cuda['gt_anchors']).square().mean()

    # correspondences
    if epoch < 3000:
        corresp_posed = batch_cuda['corresp_posed'].clone().detach().requires_grad_()
        cond = cond_cat.repeat(1, corresp_posed.shape[1], 1)
        delta, _ = decoder_expr(corresp_posed, cond, None)
        pred_can = corresp_posed + delta
        loss_corresp = (pred_can - batch_cuda['corresp_neutral']).square().mean()
        if epoch > 750:
            loss_corresp *= 0.25

    else:
        loss_corresp = torch.zeros_like(grad_loss)

    # enforce deformation field to be zero elsewhere
    nsamps = min(100, batch_cuda['corresp_posed'].shape[1])
    cond = cond_cat.repeat(1, nsamps, 1)

    samps = (torch.rand(lat_reg_shape.shape[0], nsamps, 3, device=lat_reg_shape.device, dtype=lat_reg_shape.dtype) - 0.5) * 2.5
    delta_reg, _ = decoder_expr(samps, cond, None)
    loss_reg_zero = delta_reg.square().mean()





    if (epoch >= 250 or True):
        # for neutral expressions, encourage small deformations
        if (batch_cuda['is_neutral'].squeeze(dim=-1)==1).sum() > 0:
            loss_neutral_def = points_posed_offset[batch_cuda['is_neutral'].squeeze(dim=-1)==1, ...].square().mean()
            loss_neutral_def += points_posed_outer_offset.square().mean()
            loss_neutral_def += points_posed_off_offset.square().mean()
        else:
            loss_neutral_def = torch.zeros_like(loss_reg_zero)
    else:
        loss_neutral_def = torch.zeros_like(grad_loss)


    return {
        'surf_sdf_loss': tot_sdf_loss,
        'normal_loss': tot_normal_loss,
        'space_sdf_loss': space_sdf_loss,
        'eik_loss': grad_loss,
        'reg_shape': lat_reg_shape.mean(),
        'reg_expr': lat_reg_expr.mean(),
        'anchors': loss_anchors.mean(),
        'symm_dist': symm_dist.mean(),
        'middle_dist': middle_dist.mean(),
        'corresp': loss_corresp,
        'loss_reg_zero': loss_reg_zero,
        'loss_neutral_zero': loss_neutral_def,
    }


def compute_loss_corresp_forward(batch, decoder, decoder_shape, latent_codes, latent_codes_shape, device, epoch=-1, exp_path=None):

    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    glob_cond_shape = latent_codes_shape(batch['species'].to(device))
    glob_cond_pose = latent_codes(batch['idx'].to(device))

    if 'gt_anchors' in batch_cuda and decoder_shape is not None and decoder_shape.mlp_pos is None:
        gt_anchors = batch_cuda['gt_anchors']
    elif decoder_shape is not None and decoder_shape.mlp_pos is not None:
        gt_anchors = decoder_shape.mlp_pos(glob_cond_shape[..., :decoder_shape.lat_dim_glob]).view(glob_cond_pose.shape[0], -1, 3)
        gt_anchors += decoder.anchors.squeeze(0)
    else:
        gt_anchors = None

    glob_cond = torch.cat([glob_cond_shape.unsqueeze(1), glob_cond_pose], dim=-1)

    points_neutral = batch_cuda['points_neutral'].clone().detach().requires_grad_()

    cond = glob_cond.repeat(1, points_neutral.shape[1], 1)
    delta, _ = decoder(points_neutral, cond, gt_anchors)
    pred_posed = points_neutral + delta.squeeze()

    points_posed = batch_cuda['points_posed']
    loss_corresp = (pred_posed - points_posed[:, :, :3])**2#.abs()

    lat_mag = torch.norm(glob_cond_pose, dim=-1)**2

    # enforce deformation field to be zero elsewhere
    samps = (torch.rand(cond.shape[0], 100, 3, device=cond.device, dtype=cond.dtype) -0.5)*2.5

    delta, _ = decoder(samps, cond[:, :100, :], gt_anchors)


    loss_reg_zero = (delta**2).mean()


    return {'corresp': loss_corresp.mean(),
            'lat_reg': lat_mag.mean(),
            'loss_reg_zero': loss_reg_zero}


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


def perceptual_loss(fake, real,vgg):
    layers = {
        'relu2_2': vgg[9]
    }
   
    fake_features = layers['relu2_2'](fake)
    real_features = layers['relu2_2'](real)
    loss = F.mse_loss(fake_features, real_features)/ (32*32)
    return loss


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