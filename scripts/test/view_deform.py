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
    parser.add_argument('--config',type=str, default='NPLM/scripts/configs/npm_def.yaml', help='config file')
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
    
    checkpoint_shape = torch.load('checkpoints/3dShape/latest_3d_0126.tar')
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
    
    checkpoint_deform = torch.load('checkpoints/deform/exp-deform-dis__10000.tar')
    lat_deform_new = checkpoint_deform['latent_deform_state_dict']['weight']
    checkpoint_deform_old = torch.load('checkpoints/deform/deform_old.tar')
    lat_deform_old = checkpoint_deform_old['latent_deform_state_dict']['weight']
    lat_deform_all = torch.cat((lat_deform_new, lat_deform_old), dim=0)
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    grid_points = create_grid_points_from_bounds(mini, maxi, 128)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    # lat_def_all = checkpoint_deform['latent_deform_state_dict']['weight']
    # logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=2000)
    # mesh = mesh_from_logits(logits, mini, maxi,256)

    
    mode =   'deform'

            
    if mode=='3dshape':
        # random 50 index from latent space
        ids = random.sample(range(0, lat_idx_all_3d.shape[0]), 50)
        for id in ids:
            latent = lat_idx_all_3d[id]
            mesh = latent_to_mesh(decoder_shape_3d,latent , device)
            save_file = out_dir + '/_{:04d}.obj'.format(id)
            mesh.export(save_file)
            print('{} saved'.format(save_file))
            
                

if mode == 'deform':
    mesh_root = 'dataset/Mesh_colored'
    all_mesh = os.listdir(mesh_root)
    for mesh_file in all_mesh:
            ids = random.sample(range(0, lat_deform_all.shape[0]), 20)
            mesh = trimesh.load(os.path.join(mesh_root, mesh_file))
            print('>>>>> Loading {} >>>>>'.format(mesh_file))
            for id in ids:
                latent = lat_deform_all[id]
                mesh_deform = deform_mesh(mesh, decoder_deform, latent)
                mesh_deform.visual.vertex_colors = mesh.visual.vertex_colors
                save_name = mesh_file.split('.')[0] + '_d{}.obj'.format(id)
                save_file = os.path.join('dataset/Mesh_colored/deformed', save_name)
                mesh_deform.export(save_file, include_color=True)
                print('{} saved'.format(save_file))
            


                

            
            
