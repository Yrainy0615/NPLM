import argparse
import yaml
import sys
sys.path.append('NPLM')
from scripts.model.reconstruction import deform_mesh, get_logits, mesh_from_logits,create_grid_points_from_bounds, sdf_from_latent
import torch
import os
from matplotlib import pyplot as plt
import random
import numpy as np
import io
from PIL import Image
from scripts.model.fields import UDFNetwork
from scripts.dataset.img_to_3dsdf import sdf2d_3d, mesh_from_sdf

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
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize latent space')
    parser.add_argument('--config',type=str, default='NPLM/scripts/configs/npm_def.yaml', help='config file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
        
    out_dir = 'results/viz_space'
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r'))


    decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                         d_out=CFG['shape_decoder']['decoder_out_dim'],
                         n_layers=CFG['shape_decoder']['decoder_nlayers'],
                         d_in_spatial=2,
                         udf_type='sdf',
                         use_mapping=CFG['shape_decoder']['use_mapping'])
    
    checkpoint_shape = torch.load('checkpoints/2dShape/exp-sdf2d__300.tar')
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    

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

    
    mode =   'shape'
    if mode=='shape':
        # random 50 index from latent space
        ids = random.sample(range(0, lat_idx_all.shape[0]), 50)
        for id in ids:
            latent = lat_idx_all[id]
            sdf_3d = sdf_from_latent(decoder=decoder_shape, latent=latent, grid_size=256)
            mini = [-.95, -.95, -.95]
            maxi = [0.95, 0.95, 0.95]   
            mesh = mesh_from_sdf(sdf_3d,mini=mini, maxi=maxi , resolution=256)
            save_file = out_dir + '/_{:04d}.obj'.format(id)
            mesh.export(save_file)
            print('{} saved'.format(save_file))
            
                
        pass

    if mode == 'deform':
        images = []
        for i in range(7):
            lat_idx = torch.concat([lat_idx_all[i], lat_spc_all[i]]).to(device)
            logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=8000)
            mesh = mesh_from_logits(logits, mini, maxi,256)
            basename = out_dir + '/shape_{:04d}.ply'.format(i)
            mesh.export(basename)
            print(f'save mesh {basename}')
  

            if viz_deform:
                for j in range(lat_def_all.shape[0]):
                    deform_file =out_dir + '/shape_{:04d}_deform_{:04d}.ply'.format(i,j)
                    if not os.path.exists(deform_file):
                        lat_def = lat_def_all[j]
                        lat_rep = torch.cat([lat_idx.unsqueeze(0), lat_def.unsqueeze(0)], dim=-1)
                    #  logits = get_logits(decoder, lat_rep, grid_points=grid_points,nbatch_points=8000)
                
                        deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)

                        deform.export(deform_file)
                        img_name = out_dir + '/shape_{:04d}_deform_{:04d}.png'.format(i,j)
                        save_mesh_image_with_camera(deform.vertices, deform.faces, filename=img_name)

                        data_sample = {
                            'latent_shape': lat_idx,
                            'latent_def': lat_def,
                            'mesh_deform':  deform_file,
                            'img': img_name,
                            
                        }
                        np.save(out_dir + '/shape_{:04d}_deform_{:04d}.npy'.format(i,j), data_sample)


                

            
            
