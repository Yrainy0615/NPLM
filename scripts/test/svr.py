import argparse
import json
import os
import random
import time
import torch
import numpy as np
from MeshSDF.lib.mesh import create_mesh
import yaml
from scripts.model.fields import UDFNetwork
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
import trimesh
from scripts.model.renderer import MeshRender
import cv2
from scripts.model.point_encoder import Imgencoder
from matplotlib import pyplot as plt

def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices

def fit_single_leaf(file, maskfile,encoder,decoder,device,latent):
    # read img
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (256, 256))
  img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
  # read mask
  mask = cv2.imread(maskfile)
  mask =cv2.resize(mask, (256, 256))
  # rotate mask for 90 degree
  mask = np.rot90(mask, 1,axes=(0, 1))
  mask_target = torch.tensor(mask/255, dtype=torch.float32).unsqueeze(0).cuda()
  
  #latent_init = encoder(mask_target.permute(0,3,1,2))
  latent_init = latent[6]
  #weights = encoder(mask_target.permute(0,3,1,2))
  
  #latent_init = torch.matmul(weights,latent)
  latent_copy= latent.detach().requires_grad_(True)
  verts_init, faces_init, _ , _ = create_mesh(decoder, latent_init, N=64, output_mesh = True)
  initial_mesh = trimesh.Trimesh(vertices=verts_init, faces=faces_init, process=False)
  # copy latent_init
  #weights_opt = weights.detach().requires_grad_(True)
  latent_source = latent_init.detach().requires_grad_(True)
  optimizer = torch.optim.Adam([latent_source], lr=1e-2)
  for i in range(400):
    optimizer.zero_grad()
    #lat_rep = torch.matmul(latent_source,latent_copy)
    verts, faces, samples, next_indices = create_mesh(decoder, latent_source, N=64, output_mesh = True)
    verts = normalize_verts(verts)
    # now assemble loss function
    xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
    faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0'))

    """
    Differentiable Rendering back-propagating to mesh vertices
    """

    textures_dr = 0.7*torch.ones(faces_upstream.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
    # images_out, depth_out, silhouette_out = renderer(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0), textures_dr.unsqueeze(0))
    mask = renderer.renderer_silhouette(Meshes(verts=[xyz_upstream.squeeze()], faces=[faces_upstream.squeeze()]))
    # require grads
    mask_tensor = mask[:,:,:,3]
    #mask_out = torch.tensor(mask_out).float().to(device).requires_grad_(True)
    loss_sillu = torch.mean((mask_tensor-mask_target[:,:,:,2])**2)
  #  loss_chamfer = chamfer_distance(xyz_upstream.unsqueeze(0), verts_tr)
    loss = loss_sillu *50   #+torch.norm(latent_source, dim=-1)**2 # +loss_chamfer[0]
    # print losses
    print(f'loss_sillu: {loss_sillu.item()}')
    loss.backward()
    # now store upstream gradients
    dL_dx_i = xyz_upstream.grad
    
    # use vertices to compute full backward pass
    optimizer.zero_grad()
    xyz = torch.tensor(verts.astype(float), requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))
    latent_expand = latent_source.expand(xyz.shape[0], -1)

    #first compute normals
    pred_sdf = decoder(torch.cat([latent_expand, xyz],dim=-1))
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
  # save mask
    save_mask = mask.detach().squeeze().cpu().numpy()
    # save_mask = save_mask*255
    if i % 10 == 0:
      plt.imsave(f'mask_{i}.png',save_mask)
    
  
  mesh_refined = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
  mesh_refined.export(f'refined_{i}.ply')
  return initial_mesh

if __name__ == "__main__":
    # set device
    os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = MeshRender(device=device)
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    
    decoder = UDFNetwork(d_in= CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                        d_out=CFG['shape_decoder']['decoder_out_dim'],
                        n_layers=CFG['shape_decoder']['decoder_nlayers'],
                        udf_type='abs')
    checkpoint = torch.load('checkpoints/2dShape/exp-2d-sdf__9000.tar')
    lat_idx_all = checkpoint['latent_idx_state_dict']['weight']
    #lat_idx= lat_idx_all[30] 
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    decoder.to(device)
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    encoder = Imgencoder(dino,512)
    encoder.to(device)
   # checkpoint_encoder =torch.load('checkpoints/inversion/inversion_2d_epoch_200.tar')
   # encoder.load_state_dict(checkpoint_encoder['encoder_state_dict'])
    encoder.eval()
    # test single leaf
    imgfile = 'test_img/Chinar_diseased_0002_aligned.JPG'
    maskfile = 'test_img/Lemon_healthy_0017_mask_aligned.JPG'
    mesh = fit_single_leaf(imgfile,maskfile,encoder=encoder,decoder=decoder,device=device, latent=lat_idx_all)
    