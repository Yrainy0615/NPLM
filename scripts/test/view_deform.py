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
from scripts.model.fields import UDFNetwork
from scripts.model.renderer import MeshRender

#os.environ['CUDA_LAUNCH_BLOCKING']= '1'

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
    parser.add_argument('--config',type=str, default='NPLM/scripts/configs/npm.yaml', help='config file')
    parser.add_argument('--mode', type=str, default='deformation', choices=['shape', 'deformation','viz_shape'], help='training mode')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--wandb', type=str, default='*', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r'))
    idx_spc = np.load('idx_spc.npy', allow_pickle=True)
    idx_spc = idx_spc.item()
    decoder = DeepSDF(lat_dim=512+200,
                    hidden_dim=1024,
                    geometric_init=False,
                    out_dim=3,
                    input_dim=3)
    decoder.lat_dim_expr = 200
    decoder = DeepSDF(
            lat_dim=512+200,
            hidden_dim=1024,
            geometric_init=True,
            out_dim=3,
        )
    decoder_shape = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=CFG['decoder']['decoder_out_dim'],
                         n_layers=CFG['decoder']['decoder_nlayers'],)
    
    checkpoint_shape = torch.load('checkpoints/2dShape/exp-cg-sdf__30000.tar')
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
   # lat_spc_all = checkpoint_shape['latent_spc_state_dict']['weight']
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    checkpoint_deform  = torch.load('checkpoints/deform_epoch_30000.tar')
    decoder.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder.eval()
    decoder = decoder.to(device)
    renderer = MeshRender(device=device)
    out_dir = 'sample_result/deform_new'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    grid_points = create_grid_points_from_bounds(mini, maxi, 256)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    decoder = decoder.to(device)
   # lat_idx = lat_idx_all[2]
    lat_def_all = checkpoint_deform['latent_deform_state_dict']['weight']
    # logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=2000)
    # mesh = mesh_from_logits(logits, mini, maxi,256)

    data_list = []
    viz = 'generation'
    viz_deform = False
    if viz == 'shape_interpolation':
        num_latents, latent_dim = lat_idx_all.shape
        id =0
    # generate deformation
        for i in range(10):
        
            indices = np.random.choice(num_latents, 2, replace=False)
            selected_latents = lat_idx_all[indices]
            alphs = torch.linspace(0,1, steps=4).cuda()
            interpolate = (1-alphs[:, None]) * selected_latents[0] + alphs[:, None] * selected_latents[1]
            for j in range(40):      
                id+=1
                lat_rep = lat_def_all[j]
                lat_shape = lat_idx_all[4]
                logits = get_logits(decoder_shape, lat_shape, grid_points=grid_points,nbatch_points=8000)
                print('starting mcubes')
                mesh = mesh_from_logits(logits, mini, maxi,256)
                deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_rep,anchors=None,lat_rep_shape=lat_shape)
                # deform.export(out_dir + '/deform_{:04d}.ply'.format(id))
                mesh_name=out_dir + '/shape_{:04d}.obj'.format(id)
                deform.export(mesh_name)
                    #normal_img = renderer.render_normal(mesh.to(device))
            #     for m in range():
            #         indices = np.random.choice(65, 2, replace=False)
            #         selected_latents = lat_def_all[indices]
            #         alphs = torch.linspace(0,1, steps=10).cuda()
            #         interpolate = (1-alphs[:, None]) * selected_latents[0] + alphs[:, None] * selected_latents[1]
            #         for n in range(10):      
            #             id+=1
            #             lat_def = interpolate[n]
            #             deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)
            #             deform.export(out_dir + '/deform_{:04d}.ply'.format(id))
            #             print('done mcubes')
            #             save_mesh_image_with_camera(deform.vertices, deform.faces, filename=out_dir + '/shape_{:04d}.png'.format(id))
            #             # save dict with lat_idx, lat_def, mesh_deform, img
            #             data_sample = {
            #                 'latent_shape': lat_idx,
            #                 'latent_def': lat_def,
            #                 'mesh_deform':  out_dir + '/deform_{:04d}.ply'.format(id),
            #                 'img': out_dir + '/shape_{:04d}.png'.format(id),
            #             }
            #             data_list.append(data_sample)
            # np.save(out_dir + '/deform_train.npy', data_list)
  
    if viz == 'generation':
        data_list = []
        id=0
        for i in range(lat_idx_all.shape[0]):
            lat_idx = lat_idx_all[4]
            logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=8000)
            mesh = mesh_from_logits(logits, mini, maxi,256)

            # index: len = 10 , range [0, lat_def_all.shape[0]-1] 
            index = np.random.choice(lat_def_all.shape[0], 20, replace=False)
            lat_def_sub = lat_def_all[index]
            mesh.export(out_dir + '/shape_{:04d}.ply'.format(i))
        pass
            # for lat_def in lat_def_sub:
            #     id+=1
            #     deform = deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)
            #     deform.export(out_dir + '/deform_{:04d}.ply'.format(id))
            #     print('done mcubes')
            #     save_mesh_image_with_camera(deform.vertices, deform.faces, filename=out_dir + '/shape_{:04d}.png'.format(id))
                        # save dict with lat_idx, lat_def, mesh_deform, img
            #data_sample = {
                        #    'latent_shape': lat_idx,
                         #   'mesh': out_dir + '/shape_{:04d}.ply'.format(i)
                           # 'latent_def': lat_def,
                           # 'mesh_deform':  out_dir + '/deform_{:04d}.ply'.format(id),
                           # 'img': out_dir + '/shape_{:04d}.png'.format(id),
                        
            #data_list.append(data_sample)
                
       # np.save(out_dir + '/deform_train_2d.npy', data_list)
    
    if viz == 'shape':
        images = []
        id=0
#         for i in range(200):
       
#             # lat_idx = torch.concat([lat_idx_all[i], lat_spc_all[idx_spc[str(i)]]]).to(device)
#             # linear combination of latent
#             data_list = []
    
#            # lat_idx = lat_idx_all[i]
#             index = np.random.choice(lat_idx_all.shape[0], 2, replace=False)
#             # linear interpolation
#             alphs = torch.linspace(0,1, steps=5).cuda()
#             interpolate = (1-alphs[:, None]) * lat_idx_all[index[0]] + alphs[:, None] * lat_idx_all[index[1]]
#             for j in range(5):
#                 id +=1
#                 if j <5:
#                     spc = index[0]
#                 else:
#                     spc = index[1]
#                 lat_idx = interpolate[j]
#                 logits = get_logits(decoder_shape, lat_idx, grid_points=grid_points,nbatch_points=8000)
#                 mesh = mesh_from_logits(logits, mini, maxi,64)
#                 basename = out_dir + '/shape_{:04d}.ply'.format(id)
#                 mesh.export(basename)
#                 data_sample = {
#                         'latent_shape': lat_idx,
#                         'mesh':basename ,
#                         'spc':spc
#                     }
#                 data_list.append(data_sample)
#                 print(f'save mesh {basename}')
#         np.save(out_dir + 'com.npy'.format(i,j), data_list)
    
# #                for j in range(lat_def_all.shape[0]):
        random_list=[]
      
        for x in range(500):
            # random combination of latent
            id +=1
            num_latents, latent_size = lat_idx_all.shape

            weights = torch.rand(num_latents)

            # Ensure one weight is greater than 0.5
            
            index =torch.randint(num_latents, (1,))
            weights[index] += 50
            # save this index
            
            # Normalize weights so they sum to 1
            weights /= weights.sum()
            weights = weights.to(device)
            random_combination = torch.matmul(weights, lat_idx_all)
            logits = get_logits(decoder_shape, random_combination, grid_points=grid_points,nbatch_points=8000)
            mesh = mesh_from_logits(logits, mini, maxi,64)
            basename = out_dir + '/shape_{:04d}.ply'.format(id)
            mesh.export(basename)
            data_sample = {
                    'latent_shape': random_combination,
                    'mesh':basename ,
                    'spc': index,
                    'weight': weights
                }
            data_list.append(data_sample)
            print(f'save mesh {basename}')
        np.save(out_dir + 'random.npy'.format(i,j), data_list)

  

            # if viz_deform:
            #     for j in range(lat_def_all.shape[0]):
            #         deform_file =out_dir + '/shape_{:04d}_deform_{:04d}.ply'.format(i,j)
            #         if not os.path.exists(deform_file):
            #             lat_def = lat_def_all[j]
            #             lat_rep = torch.cat([lat_idx.unsqueeze(0), lat_def.unsqueeze(0)], dim=-1)
            #         #  logits = get_logits(decoder, lat_rep, grid_points=grid_points,nbatch_points=8000)
                
            #             deform =  deform_mesh(mesh=mesh,deformer=decoder,lat_rep=lat_def,anchors=None,lat_rep_shape=lat_idx)

            #             deform.export(deform_file)
            #             img_name = out_dir + '/shape_{:04d}_deform_{:04d}.png'.format(i,j)
            #             save_mesh_image_with_camera(deform.vertices, deform.faces, filename=img_name)

            #             data_sample = {
            #                 'latent_shape': lat_idx,
            #                 'latent_def': lat_def,
            #                 'mesh_deform':  deform_file,
            #                 'img': img_name,
                            
            #             }
            #             np.save(out_dir + '/shape_{:04d}_deform_{:04d}.npy'.format(i,j), data_sample)


                

            
            
