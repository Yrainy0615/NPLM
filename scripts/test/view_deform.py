from scripts.model.deepSDF import DeepSDF
import argparse
import yaml
from scripts.model.reconstruction import deform_mesh, get_logits, mesh_from_logits,create_grid_points_from_bounds
import torch
import os
import pyvista as pv
from matplotlib import pyplot as plt
import random
import numpy as np
import io
from PIL import Image
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
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--config',type=str, default='NPLM/scripts/configs/npm_def.yaml', help='config file')
    parser.add_argument('--mode', type=str, default='deformation', choices=['shape', 'deformation','viz_shape'], help='training mode')
    parser.add_argument('--gpu', type=int, default=7, help='gpu index')
    parser.add_argument('--wandb', type=str, default='*', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r'))

    decoder = DeepSDF(lat_dim=512+200,
                    hidden_dim=1024,
                    geometric_init=False,
                    out_dim=3,
                    input_dim=3)
    decoder.lat_dim_expr = 200
    decoder_shape = DeepSDF(
            lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        )

    
    checkpoint_shape = torch.load('checkpoints/cgshape_bs7_map/cgshape_epoch_25000.tar')
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    lat_spc_all = checkpoint_shape['latent_spc_state_dict']['weight']
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    checkpoint_deform  = torch.load('checkpoints/deform_epoch_30000.tar')
    # decoder.load_state_dict(checkpoint_deform['decoder_state_dict'])
    # decoder.eval()
    # decoder = decoder.to(device)
    
    out_dir = 'sample_result/shape_cg_ep25000'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    grid_points = create_grid_points_from_bounds(mini, maxi, 256)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    decoder = decoder.to(device)
    lat_idx = lat_idx_all[1]
    lat_def_all = checkpoint_deform['latent_deform_state_dict']['weight']
    # logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=2000)
    # mesh = mesh_from_logits(logits, mini, maxi,256)

    
    viz = 'generation'
    viz_deform = False
    if viz == 'random':
    # generate deformation
        for i in range(lat_def_all.shape[1]):
            lat_def = lat_def_all[i]
            lat_idx = lat_idx_all[random.randint(0, 6)]
            lat_rep = torch.cat([lat_idx.unsqueeze(0), lat_def.unsqueeze(0)], dim=-1)
            logits = get_logits(decoder, lat_rep, grid_points=grid_points,nbatch_points=2000)
            print('starting mcubes')
            deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)
            deform.export(out_dir + '/deform_{:04d}.ply'.format(i))
            print('done mcubes')
            save_mesh_image_with_camera(deform.vertices, deform.faces, filename=out_dir + '/deform_{:04d}.png'.format(i))

    if viz == 'interpolation':
        out_dir = 'sample_result/deform_interpolation'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        alphs = torch.linspace(0,1, steps=10).cuda()
        interpolate = (1-alphs[:, None]) * lat_def_all[6] + alphs[:, None] * lat_def_all[7]
        for i in range(10):
            print('starting mcubes')
            deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=interpolate[i],anchors=None,lat_rep_shape=lat_idx)
            deform.export(out_dir + '/deform_{:04d}.ply'.format(i))
            print('done mcubes')
            save_mesh_image_with_camera(deform.vertices, deform.faces, filename=out_dir + '/deform_{:04d}.png'.format(i))

    if viz == 'generation':
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


                

            
            
