import argparse
import yaml
import sys
sys.path.append('NPLM')
from scripts.model.reconstruction import deform_mesh, get_logits, mesh_from_logits,create_grid_points_from_bounds, sdf_from_latent, latent_to_mesh
import torch
import os
from matplotlib import pyplot as plt
import random
import numpy as np
import io
from PIL import Image
from scripts.model.fields import UDFNetwork
from scripts.dataset.img_to_3dsdf import sdf2d_3d, mesh_from_sdf
import trimesh
from scripts.model.renderer import MeshRender
from pytorch3d.structures import Meshes
from scipy.spatial import Delaunay

def img_to_leaf(mask, image_tensor=None):
    # Assuming mask is a numpy array and image_tensor is a torch tensor
    # Get the leaf indices from the mask
    # mask is numpy array, get leaf indices and vertices
    
    leaf_indices = np.argwhere(mask > 0)
    vertices_np = np.hstack((leaf_indices, np.zeros((leaf_indices.shape[0], 1))))


    
    # Calculate faces with Delaunay triangulation
    tri = Delaunay(vertices_np[:, :2])
    faces_np = tri.simplices
    
    # Function to check if a point is inside the mask
    def is_point_inside_mask(point, mask):
        x, y = int(point[0]), int(point[1])
        return mask[y, x] > 0
    
    # Filter faces where the centroid is inside the mask
    valid_faces = []
    for face in faces_np:
        centroid = vertices_np[face].mean(axis=0)
        if is_point_inside_mask(centroid, mask):
            valid_faces.append(face)
    valid_faces_np = np.array(valid_faces)
    


    mesh_tri = trimesh.Trimesh(vertices=vertices_np, faces=valid_faces_np)
 

    return mesh_tri

def save_mesh_image_with_camera(vertices, faces, filename="mesh.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], shade=True, color='grey')
    

    ax.view_init(elev=90, azim=180)  
    ax.dist = 8  
    ax.set_box_aspect([1,1,1.4]) 
    plt.axis('off')  
    
    plt.savefig(filename, dpi=300)
    print(f'figure saved to {filename}')
    plt.close()
def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices

def mesh_to_canonical(vertices):
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    vertices = np.matmul(vertices, rotation_matrix)
    normalize_vertices = normalize_verts(vertices)
    return normalize_vertices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize latent space')
    parser.add_argument('--config',type=str, default='NPLM/scripts/configs/npm_deform.yaml', help='config file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
        
    out_dir = 'results/viz_space'
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r'))
    
    
    # 3d shape decoder
    decoder_shape_3d = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                         d_out=CFG['shape_decoder']['decoder_out_dim'],
                         n_layers=CFG['shape_decoder']['decoder_nlayers'],
                         d_in_spatial=3,
                         udf_type='sdf')
    
    checkpoint_shape = torch.load('checkpoints/shape/latest_new.tar')
    lat_idx_all_3d = checkpoint_shape['latent_idx_state_dict']['weight']
    decoder_shape_3d.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape_3d.eval()
    decoder_shape_3d.to(device)

    # deform decoder initialization
    decoder_deform = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False,
                         use_mapping=CFG['deform_decoder']['use_mapping'])
    
    checkpoint_deform = torch.load('checkpoints/deform_final/latest_base.tar')
    lat_deform_all = checkpoint_deform['latent_deform_state_dict']['weight']
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)

    # renderer
    renderer = MeshRender(device=device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    mode =   'deform'
 
    if mode=='3dshape':
        # random 50 index from latent space
        ids = random.sample(range(0, lat_idx_all_3d.shape[0]), 10)
        for id in ids:
            latent = lat_idx_all_3d[id]
            mesh = latent_to_mesh(decoder_shape_3d,latent , device)
            save_file = out_dir + '/_{:04d}.obj'.format(id)
            mesh.export(save_file)
            print('{} saved'.format(save_file))
            
                

if mode == 'deform':
    save_dir = 'results/viz_new'
    random_shape_idx = random.sample(range(0, lat_idx_all_3d.shape[0]), 10)
    random_deform_idx = random.sample(range(0, lat_deform_all.shape[0]), 100)
    latent_shape = lat_idx_all_3d[2]
    mesh = trimesh.load('dataset/LeafData/Chinar/healthy/Chinar_healthy_0004_mask_aligned_128.obj')
    #mesh = latent_to_mesh(decoder_shape_3d,latent_shape , device)
    for i in range(20):
        shape_idx =1
        deform_idx = random_deform_idx[i]
        latent_deform = lat_deform_all[deform_idx]
        save_name = '{}.obj'.format(shape_idx)
        #mesh.export(os.path.join(save_dir, save_name))
        mesh.vertices =normalize_verts(mesh.vertices)
        # mesh_pytorch3d = Meshes(verts=torch.tensor(mesh.vertices).unsqueeze(0).float(), faces=torch.tensor(mesh.faces).unsqueeze(0))
        # mesh_pytorch3d = mesh_pytorch3d.to(device)
        # canonical_mask = renderer.get_mask(mesh_pytorch3d)
        # mesh_plane = img_to_leaf(canonical_mask)
        # mesh_plane.export(os.path.join(save_dir, '{}_plane.obj').format(shape_idx))
        #print('mesh {} saved'.format(save_name))
        latent_cond = torch.cat((latent_shape, latent_deform), dim=0)
        mesh_deform = deform_mesh(mesh, decoder_deform, latent_deform)
        save_name_deform = '{}_{}.obj'.format(shape_idx, deform_idx)
        mesh_deform.export(os.path.join(save_dir, save_name_deform))
        print('deformed mesh {} saved'.format(save_name_deform))

            

    




                

            
            
