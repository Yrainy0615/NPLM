import torch
from torch.nn import functional as F
import numpy as np
from scripts.model.diff_operators import gradient    


def compute_loss(batch, decoder, latent_idx, latent_spc, device):
    if 'path' in batch:
        del batch['path']
    batch_cuda_npm = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}

    idx = batch.get('idx').to(device)
    spc = batch.get('species').to(device)
    glob_cond_idx = latent_idx(idx)
    glob_cond_spc = latent_spc(spc)
    glob_cond = torch.cat((glob_cond_idx,glob_cond_spc.unsqueeze(1)),dim=2)
    loss_dict = actual_compute_loss(batch_cuda_npm, decoder, glob_cond)

    return loss_dict

def actual_compute_loss(batch_cuda, decoder, glob_cond):

    # if hasattr(decoder, 'anchors'):
    #     anchor_preds = batch_cuda['gt_anchors']
    # else:
    anchor_preds = None


    # prep
    sup_surface = batch_cuda['points'].clone().detach().requires_grad_() # points on face surf
    #sup_surface_outer = batch_cuda['points_non_face'].clone().detach().requires_grad_() # points on non-face surf
    sup_grad_far = batch_cuda['sup_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_grad_near = batch_cuda['sup_grad_near'].clone().detach().requires_grad_() # points near/off surface
    udf_gt_near = batch_cuda['sup_grad_near_udf'].clone().detach().requires_grad_() 
    

    # model computations
    pred_surface, anchors = decoder(sup_surface, glob_cond.repeat(1, sup_surface.shape[1], 1), anchor_preds)
    # pred_surface_outer, anchors = decoder(sup_surface_outer, glob_cond.repeat(1, sup_surface_outer.shape[1], 1),
    #                                       anchor_preds)
    pred_space_near, anchors = decoder(sup_grad_near, glob_cond.repeat(1, sup_grad_near.shape[1], 1), anchor_preds)
    udf_pred_near = torch.abs(pred_space_near).squeeze(2)

    pred_space_far = decoder(sup_grad_far, glob_cond.repeat(1, sup_grad_far.shape[1], 1), anchor_preds)[0]


    # normal computation
    gradient_surface = gradient(pred_surface, sup_surface)
    # gradient_surface_outer = gradient(pred_surface_outer, sup_surface_outer)
    gradient_space_far = gradient(pred_space_far, sup_grad_far)
    gradient_space_near = gradient(pred_space_near, sup_grad_near)



    # computation of losses for geometry
    surf_sdf_loss = torch.abs(pred_surface).squeeze()
    #surf_sdf_loss_outer = torch.abs(pred_surface_outer).squeeze()

    surf_normal_loss = (gradient_surface - batch_cuda['normals']).norm(2, dim=-1)
    # surf_normal_loss_outer = torch.clamp((gradient_surface_outer - batch_cuda['normals_non_face']).norm(2, dim=-1),
    #                                      None, 0.75) / 2

    #  udf loss
    udf_near_loss = F.mse_loss(udf_gt_near, udf_pred_near)
    
    surf_grad_loss = torch.abs(gradient_surface.norm(dim=-1) - 1)
    # surf_grad_loss_outer = torch.abs(gradient_surface_outer.norm(dim=-1) - 1)

    space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_space_far))
    space_grad_loss_far = torch.abs(gradient_space_far.norm(dim=-1) - 1)
    space_grad_loss_near = torch.abs(gradient_space_near.norm(dim=-1) - 1)
    grad_loss = torch.cat([surf_grad_loss, space_grad_loss_far, space_grad_loss_near], dim=-1)

    #grad_loss = torch.cat([surf_grad_loss, surf_grad_loss_outer, space_grad_loss_far, space_grad_loss_near], dim=-1)


    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)
    if hasattr(decoder, 'lat_dim_glob'):
        loc_lats_symm = glob_cond[:,
                        decoder.lat_dim_glob:decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc].view(
            glob_cond.shape[0], decoder.num_symm_pairs * 2, decoder.lat_dim_loc)
        loc_lats_middle = glob_cond[:,
                          decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc:-decoder.lat_dim_loc].view(
            glob_cond.shape[0], decoder.num_kps - decoder.num_symm_pairs * 2, decoder.lat_dim_loc)

        symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
        if loc_lats_middle.shape[1] % 2 == 0:
            middle_dist = torch.norm(loc_lats_middle[:, ::2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        else:
            middle_dist = torch.norm(loc_lats_middle[:, :-1:2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
    else:
        symm_dist = None
        middle_dist = None

    if anchors is not None:
        loss_anchors = (anchors - batch_cuda['gt_anchors']).square().mean()

        # ret_dict =  {'surf_sdf': torch.mean(torch.cat([surf_sdf_loss, surf_sdf_loss_outer], dim=-1)),
        #         # 'normals': torch.mean(
        #         #     torch.cat([surf_normal_loss.squeeze(), surf_normal_loss_outer.squeeze()], dim=-1)),
        #         'space_sdf': torch.mean(space_sdf_loss),
        #         'grad': torch.mean(grad_loss),
        #         'lat_reg': lat_mag.mean(),
        #         'anchors': loss_anchors,
        #         'symm_dist': symm_dist,
        #         'middle_dist': middle_dist, }
        # return ret_dict
    else:
        # ret_dict =  {'surf_sdf': torch.mean(torch.cat([surf_sdf_loss, surf_sdf_loss_outer], dim=-1)),
        #         'normals': torch.mean(
        #             torch.cat([surf_normal_loss.squeeze(), surf_normal_loss_outer.squeeze()], dim=-1)),
        #         'space_sdf': torch.mean(space_sdf_loss),
        #         'grad': torch.mean(grad_loss),
        #         'lat_reg': lat_mag.mean()}
        # return ret_dict
        ret_dict = {'surf_sdf': torch.mean(surf_sdf_loss),
                    'normals': torch.mean(surf_normal_loss),
                    'space_sdf': torch.mean(space_sdf_loss),
                    'grad': torch.mean(grad_loss),
                    'near_udf': torch.mean(udf_near_loss),
                    'lat_reg':lat_mag.mean()}
        return ret_dict
