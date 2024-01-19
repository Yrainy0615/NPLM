import argparse
import json
import os
import random
import time
import torch
import numpy as np
from scripts.test.mesh import *
#from MeshSDF.lib.mesh import create_mesh
import yaml
from scripts.model.fields import UDFNetwork
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
import trimesh
from MeshUDF.optimize_chamfer_A_to_B import get_mesh_udf_fast
from scripts.model.renderer import MeshRender

def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices


if __name__ == "__main__":
    filename = 'dataset/LeafData/maple/healthy/Chinar_healthy_0001.ply'
    # set device
    os.environ['CUDA_VISIBLE_DEVICES']=str(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = MeshRender(device=device)
    config = 'NPLM/scripts/configs/npm_color.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    decoder = UDFNetwork(d_in= CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                        d_out=CFG['decoder']['decoder_out_dim'],
                        n_layers=CFG['decoder']['decoder_nlayers'],
                        udf_type='abs')
    checkpoint = torch.load('checkpoints/2dShape/exp-cg-sdf__30000.tar')
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    decoder.to(device)
    lat_all = checkpoint['latent_idx_state_dict']['weight']
    latent_init = lat_all[1]
    latent_init.requires_grad = True
    latent_target = lat_all[5]

    verts_target, faces_target, _ , _ = create_mesh(decoder, latent_target, N=64, output_mesh = True)
    verts_target = normalize_verts(verts_target)
    # savetrimesh
    mesh_target = trimesh.Trimesh(vertices=verts_target, faces=faces_target, process=False)
    mesh_target.export('target.ply')
    # visualize target stuff
    verts_tr = torch.tensor(verts_target[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces_tr = torch.tensor(faces_target[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_tr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join('./', "target.png")
    if not os.path.exists(os.path.dirname(image_filename)):
        os.makedirs(os.path.dirname(image_filename))
    img_target = renderer.render_rgb(Meshes(verts=[verts_tr.squeeze()], faces=[faces_tr.squeeze()]))
    mask_target = img_target[...,3]
    #store_image(image_filename, tgt_images_out, tgt_silhouette_out)

    # initialize and visualize initialization
    verts, faces, samples, next_indices = create_mesh(decoder, latent_init, N=64, output_mesh = True)
    # save trimesh
    mesh_init = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh_init.export('init.ply')
    verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()
    faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()
    textures_dr = 0.7*torch.ones(faces_dr.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()
    textures_dr = textures_dr.unsqueeze(0)
    image_filename = os.path.join('./', "initialization.png")
    mask = renderer.get_mask(Meshes(verts=[verts_dr.squeeze()], faces=[faces_dr.squeeze()]))

    #store_image(image_filename, images_out, alpha_out)

    lr= 1e-3
    #regl2 = 1000
    decreased_by = 1.5
    adjust_lr_every = 20
    # adjust learning rate
    optimizer = torch.optim.Adam([latent_init], lr=lr)

    print("Starting optimization:")
    decoder.eval()
    best_loss = None
    sigma = None
    images = []

    for e in range(400):

        optimizer.zero_grad() 

        # first extract iso-surface


        verts, faces, samples, next_indices = create_mesh(decoder, latent_init, N=64, output_mesh = True)
        verts = normalize_verts(verts)
        # now assemble loss function
        xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
        faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=torch.device('cuda:0'))

        """
        Differentiable Rendering back-propagating to mesh vertices
        """

        textures_dr = 0.7*torch.ones(faces_upstream.shape[0], 1, 1, 1, 3, dtype=torch.float32).cuda()
       # images_out, depth_out, silhouette_out = renderer(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0), textures_dr.unsqueeze(0))
        img = renderer.render_rgb(Meshes(verts=[xyz_upstream.squeeze()], faces=[faces_upstream.squeeze()]))
        # require grads
        mask = img[...,3]
        #mask_out = torch.tensor(mask_out).float().to(device).requires_grad_(True)
        loss_sillu = torch.mean((mask-mask_target)**2)
        loss_chamfer = chamfer_distance(xyz_upstream.unsqueeze(0), verts_tr)
        loss = loss_sillu  + torch.norm(latent_init, dim=-1)**2 +loss_chamfer[0]
        # print losses
        print(f'loss_sillu: {loss_sillu.item()}', f'loss_chamfer: {loss_chamfer[0].item()}')

        # now store upstream gradients
        loss.backward()
        dL_dx_i = xyz_upstream.grad
        
       # use vertices to compute full backward pass
        optimizer.zero_grad()
        xyz = torch.tensor(verts.astype(float), requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))
        latent_inputs = latent_init.expand(xyz.shape[0], -1)

        #first compute normals
        pred_sdf = decoder(torch.cat([latent_inputs, xyz],dim=-1))
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
    # save mesh
        if e %10 ==0:
          mesh_refined = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
          mesh_refined.export(f'refined_{e}.ply')
      

