import torch
from torch.nn import functional as F
import numpy as np
from scripts.model.diff_operators import gradient    
from pytorch3d. structures import Meshes
from scripts.model.renderer import MeshRender
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.loss import chamfer_distance
from pytorch_metric_learning import losses

def chamfer_distance_contour(mask1, mask2):
    edge1 = F.conv2d(mask1.unsqueeze(1), torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]).to(mask1.device))
    edge2 = F.conv2d(mask2.unsqueeze(1), torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]).to(mask2.device))
    edge1 = (edge1.abs() > 0.1).float()
    edge2 = (edge2.abs() > 0.1).float()
    
    coords1 = edge1.nonzero().float()
    coords2 = edge2.nonzero().float()
    coords1 = coords1[:, 2:]/1024
    coords2 = coords2[:, 2:]/1024
    chamfer = chamfer_distance(coords1.unsqueeze(0), coords2.unsqueeze(0))
    return chamfer


def iou_loss(pred_mask, target_mask, smooth=1e-5):
    """
    Compute the IoU loss between two masks.
    
    Parameters:
    - pred_mask (torch.Tensor): The predicted mask tensor.
    - target_mask (torch.Tensor): The target mask tensor.
    - smooth (float): A small constant to avoid division by zero.
    
    Returns:
    - torch.Tensor: The computed IoU loss.
    """
    # Ensure the mask tensors are of float type
    pred_mask = pred_mask.float()
    target_mask = target_mask.float()
    
    # Compute the intersection and union
    intersection = (pred_mask * target_mask).sum(dim=[1, 2])
    total = (pred_mask + target_mask).sum(dim=[1, 2])
    union = total - intersection
    
    # Compute the IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Compute the IoU loss
    iou_loss = 1 - iou
    
    # Return the mean IoU loss over the batch
    return iou_loss.mean()

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
   # glob_cond = torch.cat((glob_cond,glob_cond2.unsqueeze(1)),dim=2)
    glob_cond = glob_cond
    # prep
    sup_surface = batch_cuda['points'].clone().detach().requires_grad_() # points on face surf
    #sup_surface_outer = batch_cuda['points_non_face'].clone().detach().requires_grad_() # points on non-face surf
    sup_grad_far = batch_cuda['sup_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_grad_near = batch_cuda['sup_grad_near'].clone().detach().requires_grad_() # points near/off surface
    sdf_gt_near = batch_cuda['sup_grad_near_sdf'].clone().detach().requires_grad_() 
    sdf_gt_far = batch_cuda['sup_grad_far_sdf'].clone().detach().requires_grad_()

    # model computations
    pred_surface = decoder(torch.cat([sup_surface, glob_cond.repeat(1, sup_surface.shape[1], 1)],dim=-1))
    pred_space_near = decoder(torch.cat([sup_grad_near, glob_cond.repeat(1, sup_grad_near.shape[1], 1)],dim=-1))
    pred_space_far = decoder(torch.cat([sup_grad_far, glob_cond.repeat(1, sup_grad_far.shape[1], 1)],dim=-1))
    udf_pred_near = torch.abs(pred_space_near).requires_grad_()
    udf_pred_far = torch.abs(pred_space_far).requires_grad_()
    # losses
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
    udf_near_loss = F.mse_loss(sdf_gt_near.abs(), udf_pred_near.squeeze())
    udf_far_loss = F.mse_loss(sdf_gt_far.abs(), udf_pred_far.squeeze())
    
    # sdf loss
    sdf_near_loss = F.mse_loss(sdf_gt_near, pred_space_near.squeeze())
    sdf_far_loss = F.mse_loss(sdf_gt_far, pred_space_far.squeeze())
    
    surf_grad_loss = torch.abs(gradient_surface.norm(dim=-1) - 1)
    # surf_grad_loss_outer = torch.abs(gradient_surface_outer.norm(dim=-1) - 1)

    space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_space_far))
    space_grad_loss_far = torch.abs(gradient_space_far.norm(dim=-1) - 1)
    space_grad_loss_near = torch.abs(gradient_space_near.norm(dim=-1) - 1)
    grad_loss = torch.cat([surf_grad_loss, space_grad_loss_far, space_grad_loss_near], dim=-1)

    #grad_loss = torch.cat([surf_grad_loss, surf_grad_loss_outer, space_grad_loss_far, space_grad_loss_near], dim=-1)


    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)


    ret_dict = {'surf_sdf': torch.mean(surf_sdf_loss),
                  'normals': torch.mean(surf_normal_loss),
              'space_sdf': torch.mean(space_sdf_loss),
                'grad': torch.mean(grad_loss),
                #   'udf_near': torch.mean(udf_near_loss),
                #   'udf_far': torch.mean(udf_far_loss),
                 'sdf_near': torch.mean(sdf_near_loss),
                    'sdf_far': torch.mean(sdf_far_loss),
                    'lat_reg':lat_mag.mean()}
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
    #glob_cond_shape = latent_codes_shape(batch['species'].to(device))
    glob_cond = latent_codes(batch['idx'].to(device))
    label = batch['label'].to(device)
   # glob_cond = torch.cat([glob_cond_shape.unsqueeze(1), glob_cond_pose], dim=-1)

    points_neutral = batch_cuda['points_neutral'].clone().detach().requires_grad_()

    cond = glob_cond.repeat(1, points_neutral.shape[1], 1)
    delta= decoder(points_neutral, cond)
    pred_posed = points_neutral + delta.squeeze()
    # mse loss
    points_posed = batch_cuda['points_posed']
    loss_corresp = (pred_posed - points_posed[:, :, :3])**2#.abs()
    
    # distance constraint
    distance = torch.norm(glob_cond,p=2,dim=-1)
    delta_gt = points_posed - points_neutral
    delta_norm = torch.norm(delta_gt,p=2,dim=(1,2)) /10
   # delta_norm =(delta_norm - delta_norm.min()) /(delta_norm.max() - delta_norm.min() + 1e-5) 
    loss_distance = ((distance.squeeze()/delta_norm) - 1)**2
    # nce loss
    nce = losses.NTXentLoss(temperature=0.5).cuda()
    # def compute_pairs(glob_cond, labels):
    #     x = glob_cond.squeeze()
    #     similarity = torch.nn.functional.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    #     all_indices = torch.combinations(torch.arange(len(labels)), r=2).cuda()
    #     pairs_labels = labels[all_indices]
    #     pos_indices = all_indices[(pairs_labels[:, 0] == pairs_labels[:, 1])]
    #     neg_indices = all_indices[(pairs_labels[:, 0] != pairs_labels[:, 1])]
    #     pos_pairs = similarity[pos_indices[:, 0], pos_indices[:, 1]]
    #     neg_pairs = similarity[neg_indices[:, 0], neg_indices[:, 1]]
    #     a1, p = pos_indices.transpose(0, 1)
    #     a2, _ = neg_indices.transpose(0, 1)

    #     return pos_pairs, neg_pairs, (a1, p, a2, _)
    # pos_pairs, neg_pairs, indices_tuple = compute_pairs(glob_cond, label)
    loss_infonce = nce(glob_cond.squeeze(), label.squeeze())
    
    lat_mag = torch.norm(glob_cond, dim=-1)**2

    # enforce deformation field to be zero elsewhere
    samps = (torch.rand(cond.shape[0], 100, 3, device=cond.device, dtype=cond.dtype) -0.5)*2.5

    delta = decoder(samps, cond[:, :100, :])


    loss_reg_zero = (delta**2).mean()


    return {'corresp': loss_corresp.mean(),
            'lat_reg': lat_mag.mean(),
            'loss_nce': loss_infonce,
            'loss_reg_zero': loss_reg_zero,
            'loss_distance': loss_distance.mean()}


def compute_color_forward(batch, decoder, latent_codes, device, epoch=-1, exp_path=None):
    
    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    # create  
    points = batch_cuda['verts'][0].clone().detach().requires_grad_().to(device)
    points = points.unsqueeze(0)
    idx_shape = torch.ones(32,1,dtype=torch.int64)
    #glob_cond_shape = latent_codes_shape(idx_shape.to(device))
    glob_cond_color = latent_codes(batch['idx'].to(device))
    # sdf, _  = decoder_shape(points,cond_shape,None)

   #lat_mag = torch.norm(glob_cond_color, dim=-1) ** 2

    cond_color = glob_cond_color.repeat(1, points.shape[1], 1)
    color = decoder(torch.cat([points, cond_color],dim=-1))
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
    volume = batch_cuda['volume'].clone().detach().requires_grad_()
   
  
    latent_shape_pred, latent_deform_pred = encoder.encoder_inversion(volume.squeeze())
    loss_latent_shape = F.mse_loss(latent_shape_pred, batch_cuda['latent_shape'])
    loss_latent_deform = F.mse_loss(latent_deform_pred, batch_cuda['latent_deform'])
    loss_dict = {
        'loss_latent_shape': loss_latent_shape.mean(),
        'loss_latent_deform': loss_latent_deform.mean(),
    }
    return loss_dict

def inversion_weight(batch, encoder,device, latent_all):
    #weight_gt = batch['weights'].to(device)
    latent_gt = latent_all[batch['idx'].to(device)].squeeze()
    mask = batch['mask'].to(device)
    latent_pred  = encoder(mask.float())
    loss_mse = F.mse_loss(latent_pred, latent_gt)
    loss_dict = {
        'loss_mse': loss_mse.mean()}
    return loss_dict
    
def compute_verts(batch, decoder, latent_idx,device):
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    points = batch_cuda['sup_points'].clone().detach().requires_grad_()
    idx = batch.get('idx').to(device)
    #spc = batch.get('spc').to(device)
    glob_cond_idx = latent_idx(idx) # 1,1,512
    # read mesh from list
    vertices_list = [mesh.vertex_data.positions for mesh in batch['mesh']]
    face_list = [ mesh.face_data.vertex_ids for mesh in batch['mesh']]

    new_verts = [torch.tensor(vertices).to(torch.float32) for vertices in vertices_list]
    new_faces = [torch.tensor(faces).to(torch.float32) for faces in face_list]


        #pred_surface = decoder(points, glob_cond.repeat(1, points.shape[1], 1))
    pred_surface_offset = decoder(torch.cat([points, glob_cond_idx.repeat(1, points.shape[1], 1)],dim=-1))

    modeified_verts = []
    for i, verts in enumerate(new_verts):
        # 获取当前mesh的pred_surface_offset
        current_pred_offset = pred_surface_offset[i].squeeze()

        # update verts
        verts = verts.to(device)  
        verts[ batch['surf_index'][i], 2] = verts[batch['surf_index'][i], 2] + current_pred_offset
        # update mesh
        modeified_verts.append(verts.detach().cpu())
                # return pytorch3d mesh

    meshes = Meshes(verts=modeified_verts, faces=new_faces)
    # add a loss between verts and new verts
    loss_reg = (current_pred_offset**2).mean()
    return meshes, loss_reg
    
def compute_normal(batch, decoder, latent_idx,device):
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    points = batch_cuda['sup_points'].clone().detach().requires_grad_()
    idx = batch.get('idx').to(device)
    glob_cond_idx = latent_idx(idx) # 1,1,512
    delta_z = decoder(points, glob_cond_idx.repeat(1, points.shape[1], 1))
    points[:,:,2] += delta_z
    normal_pred = gradient() 

def inversion_2d(batch, encoder,decoder, latent_idx,device):
    rgb = batch['rgb'].to(device)
    idx = batch.get('idx').to(device)
    points = batch['points'].to(device)
    latent_pred = encoder(rgb.float().permute(0,3,1,2))
    latent_gt = latent_idx[idx].squeeze()
    sdf_pred = decoder(torch.cat([points, latent_pred.unsqueeze(1).repeat(1, points.shape[1], 1)],dim=-1))
    loss_sdf = torch.abs(sdf_pred).squeeze()
    #loss_mse = F.mse_loss(latent_pred, latent_gt)
    loss_dict = {
                 'loss_sdf': loss_sdf.mean()}
    return loss_dict


def compute_diff_loss(batch, decoder, latent_idx, latent_spc, device, renderer):
    verts = batch['verts'].to(device)
    # mask = batch['mask'].to(device)
    rgb = batch['rgb'].to(device)
    faces = batch['faces'].to(device)
    idx = batch.get('idx').to(device)
    spc = batch.get('spc').to(device)
  #  verts_t = batch['verts_t'].to(device)
    delta_x_gt = batch['delta_x'].to(device)
    
    glob_cond_idx = latent_idx(idx) # 1,1,512
    glob_cond_spc = latent_spc(spc)
    glob_cond = torch.cat((glob_cond_idx,glob_cond_spc.unsqueeze(1)),dim=2)
    
    # infoNCE loss
   
    delta_v = decoder(torch.cat([verts[:,:,:2], glob_cond.repeat(1, verts.shape[1], 1)],dim=-1)) # (b,n,2)
    deform_verts = verts.clone().detach().to(device)
    deform_verts[:, :,:2] = deform_verts[:, :,:2] + delta_v
    deformed_mesh = Meshes(verts=deform_verts, faces=faces)
    loss_reg_idx = torch.norm(glob_cond_idx, dim=-1) ** 2   
    loss_reg_spc = torch.norm(glob_cond_spc, dim=-1) ** 2
    vertex_colors = torch.ones((verts.shape[0], deformed_mesh[0].verts_packed().shape[0], 3), device=device)
    # add a regularizer for delta_v
   # loss_chamfer = chamfer_distance(deform_verts[:,:,:2], verts_t)
    #loss_reg_delta = (delta_v.mean()-0.05).abs()
    texture =  TexturesVertex(verts_features=vertex_colors)
    deformed_mesh.textures = texture
    img_render = renderer.renderer(deformed_mesh)
    #loss_iou = iou_loss(mask, img_render[:,:,:,0])
    # calculate chamfer distance between contours
    # use opencv to calculate contours
    loss_mse = F.mse_loss(delta_v, delta_x_gt)
    #loss_chamfer = chamfer_distance_contour(mask.float(), img_render[:,:,:,0])
    #loss_silh = torch.mean((mask- img_render[:,:,:,0])**2)
    loss_dict = {
           #      'loss_silh': loss_silh,
                 'loss_reg_idx': loss_reg_idx.mean(),
                 'loss_reg_spc': loss_reg_spc.mean(),   
                'loss_mse': loss_mse,}
             #    'loss_chamfer': loss_chamfer[0]}
    return loss_dict, rgb, img_render

    

    
    
    
    
    