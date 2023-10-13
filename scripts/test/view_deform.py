from scripts.model.deepSDF import DeepSDF
import argparse
import yaml
from scripts.model.reconstruction import deform_mesh, get_logits, mesh_from_logits,create_grid_points_from_bounds
import torch
import os
import pyvista as pv
from matplotlib import pyplot as plt

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

    
    checkpoint_shape = torch.load('checkpoints/checkpoint_epoch_20000.tar')
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    checkpoint_deform  = torch.load('checkpoints/deform_epoch_5000.tar')
    decoder.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder.eval()
    decoder = decoder.to(device)
    
    out_dir = 'sample_result/deform'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    grid_points = create_grid_points_from_bounds(mini, maxi, 256)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    decoder = decoder.to(device)
    lat_idx = lat_idx_all[1]
    logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=2000)
    mesh = mesh_from_logits(logits, mini, maxi,256)
    lat_def_all = checkpoint_deform['latent_deform_state_dict']['weight']
    
    
    # generate deformation
    for i in range(lat_def_all.shape[1]):
        lat_def = lat_def_all[i]
        lat_idx = lat_idx_all[1]
        #lat_rep = torch.cat([lat_idx.unsqueeze(0), lat_def.unsqueeze(0)], dim=-1)
        #logits = get_logits(decoder, lat_rep, grid_points=grid_points,nbatch_points=2000)
        print('starting mcubes')
        deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)
        deform.export(out_dir + '/deform_{:04d}.ply'.format(i))
        print('done mcubes')
        save_mesh_image_with_camera(deform.vertices, deform.faces, filename=out_dir + '/deform_{:04d}.png'.format(i))
        # save
        # pl = pv.Plotter(off_screen=True)
        # pl.add_mesh(deform)
        # pl.reset_camera()
        # pl.camera.position = (0, 3, 0)
        # pl.camera.zoom(1.4)
        # pl.set_viewup((0, 1, 0))
        # pl.camera.view_plane_normal = (-0, -0, 1)
        # pl.show(screenshot=out_dir + '/deform_{:04d}.png'.format(i), auto_close=True)


        
     
    
