import argparse
import json
import os
import random
import time
import torch
import numpy as np
import yaml
import sys 
sys.path.append('NPLM')
from scripts.model.fields import UDFNetwork
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
import trimesh
from scripts.model.renderer import MeshRender
import cv2
from matplotlib import pyplot as plt
from scripts.dataset.img_to_3dsdf import  mesh_from_sdf , sdf2d_3d
from scripts.model.reconstruction import sdf_from_latent

def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices

def fit_point_cloud( maskfile,point_cloud,device, 
                    decoder_shape, latent_shape,
                    decoder_deform, latent_deform):
    # read RGB
#   img = cv2.imread(file)
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   img = cv2.resize(img, (256, 256))
#   img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
  # read mask
    mask = cv2.imread(maskfile)
    mask =cv2.resize(mask, (256, 256))
    # rotate mask for 90 degree
    mask = np.rot90(mask, 1,axes=(0, 1))
    mask_target = torch.tensor(mask/255, dtype=torch.float32).unsqueeze(0).cuda()
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).cuda().requires_grad_(True)
    #latent_init = encoder(mask_target.permute(0,3,1,2))
    latent_shape_init = latent_shape[100]
    latent_shape_init.requires_grad_(True)
    
    latent_deform_init = latent_deform[20]
    latent_deform_init.requires_grad_(True)
    sdf_canonical = sdf_from_latent(decoder_shape, latent_shape_init, 256)
    initial_mesh = mesh_from_sdf(sdf_canonical, resolution=256)


    optimizer = torch.optim.Adam([latent_shape_init,latent_deform_init], lr=1e-3)
    for i in range(400):
        optimizer.zero_grad()
        #lat_rep = torch.matmul(latent_source,latent_copy)
        sdf_canonical = sdf_from_latent(decoder_shape, latent_shape_init, 256)
        mesh = mesh_from_sdf(sdf_canonical, resolution=256)
        delta_verts = decoder_deform(torch.from_numpy(mesh.vertices).float().to(device), latent_deform_init.unsqueeze(0).repeat(mesh.vertices.shape[0], 1))
        delta_verts = delta_verts.squeeze().detach().cpu().numpy()
        verts = mesh.vertices# + delta_verts
        faces = mesh.faces
        #verts = normalize_verts(verts)
        # now assemble loss function
        xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
        faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0'))

        """
        Differentiable Rendering back-propagating to mesh vertices
        """

        textures_dr = 0.7*torch.ones(faces_upstream.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
        # images_out, depth_out, silhouette_out = renderer(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0), textures_dr.unsqueeze(0))
        #mask = renderer.get_mask(Meshes(verts=[xyz_upstream.squeeze()], faces=[faces_upstream.squeeze()]))
        # require grads
        #mask_tensor = mask[:,:,:,3]
        #mask_out = torch.tensor(mask_out).float().to(device).requires_grad_(True)
        # loss_sillu = torch.mean((mask_tensor-mask_target[:,:,:,2])**2)
        loss_chamfer = chamfer_distance(xyz_upstream.unsqueeze(0), point_cloud_tensor)
        loss =loss_chamfer[0]#+torch.norm(latent_source, dim=-1)**2 # +loss_chamfer[0]
        # print losses
        print('loss_chamfer: {}'.format(loss_chamfer[0]))
        loss.backward()
        # now store upstream gradients
        dL_dx_i = xyz_upstream.grad

        # use vertices to compute full backward pass
        optimizer.zero_grad()
        xyz = torch.tensor(verts, requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
        # latent_expand = latent_shape_init.expand(xyz.shape[0], -1)

        #first compute normals 
        pred_sdf = decoder_shape(xyz[:,:2]*256, latent_shape_init.unsqueeze(0).repeat(xyz.shape[0], 1))
        # sdf_2d = sdf_2d.reshape(256, 256)
        # pred_sdf  = sdf2d_3d(sdf_2d)

        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)
        # normalization to take into account for the fact sdf is not perfect...
        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()
        dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        loss_backward = torch.sum(dL_ds_i * pred_sdf)
        loss_backward.backward()
        # and update params
        optimizer.step()
        mesh_refined = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_refined.export(f'refined_{i}.ply')
    return initial_mesh

if __name__ == "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = MeshRender(device=device)
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    
    # shape decoder initialization
    decoder_shape = UDFNetwork(d_in= CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                        d_out=CFG['shape_decoder']['decoder_out_dim'],
                        n_layers=CFG['shape_decoder']['decoder_nlayers'],
                        udf_type='sdf',
                        d_in_spatial=2,
                        use_mapping=CFG['shape_decoder']['use_mapping'])
    checkpoint = torch.load('checkpoints/2dShape/exp-sdf2d__300.tar')
    lat_idx_all = checkpoint['latent_idx_state_dict']['weight']
    #lat_idx= lat_idx_all[30] 
    decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    # deform decoder initialization
    decoder_deform = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False,
                         use_mapping=CFG['deform_decoder']['use_mapping'])
    checkpoint_deform = torch.load('checkpoints/exp-deform-dis__10000.tar')
    lat_deform_all = checkpoint_deform['latent_deform_state_dict']['weight']
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)
                                
                                
                                
    # dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # encoder = Imgencoder(dino,512)
    # encoder.to(device)
   # checkpoint_encoder =torch.load('checkpoints/inversion/inversion_2d_epoch_200.tar')
   # encoder.load_state_dict(checkpoint_encoder['encoder_state_dict'])
    # encoder.eval()
    # test single leaf
    mask_file = 'dataset/leaf_classification/images/Acer_Capillipes/201.jpg'
    target_mesh = 'dataset/leaf_classification/images/Acer_Capillipes/201_128.obj'
    mesh = trimesh.load(target_mesh)
    pds = mesh.sample(1000)
    mesh = fit_point_cloud(maskfile=mask_file,point_cloud=pds, device=device,
                           decoder_shape=decoder_shape , latent_shape=lat_idx_all,
                           decoder_deform=decoder_deform, latent_deform=lat_deform_all)
    