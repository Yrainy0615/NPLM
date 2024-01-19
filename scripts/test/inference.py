import torch
import torch.nn as nn
import argparse
import os
import yaml
from scripts.model.deepSDF import DeepSDF
from scripts.model.point_encoder import PCAutoEncoder
from scripts.dataset.sample_surface import sample_surface
import point_cloud_utils as pcu
import numpy as np
from scripts.model.reconstruction import mesh_from_logits, get_logits, create_grid_points_from_bounds, deform_mesh
from scripts.model.renderer import MeshRender
from pytorch3d.io import load_ply, load_obj
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
from matplotlib import pyplot as plt
from scripts.model.fields import UDFNetwork

def load_mesh(path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh

def get_grid_points():
    mini = [-1.0, -1.0, -1.0]
    maxi = [1.0, 1.0, 1.0]
    grid_points = create_grid_points_from_bounds(mini, maxi, 256)
    grid_points = torch.from_numpy(grid_points).float().to(device)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    return grid_points


def inference(input,encoder, decoder_shape, decoder_deform,grid_points,device):
    mini = [-1.0, -1.0, -1.0]
    maxi = [1.0, 1.0, 1.0]
    #input_tensor = torch.from_numpy(input).float().to(device).permute(1,0)
    latent_shape, latent_deform = encoder(input.unsqueeze(0).permute(0,2,1))
    logits_shape = get_logits(decoder_shape, latent_shape, grid_points, nbatch_points=6000)
    mesh_base = mesh_from_logits(logits_shape, mini, maxi, 256)
    mesh_base.export('inference_base.ply')
    deform = deform_mesh(mesh=mesh_base, deformer=decoder_deform, lat_rep=latent_deform, anchors=None, lat_rep_shape=latent_shape)
    deform.export('inference.ply')
   
    return deform
    
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=3, help='gpu index')
    parser.add_argument('--wandb', type=str, default='inversion_latent_mse', help='run name of wandb')

    # setting
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    
    # initialize dataset
    render = MeshRender(device=device)
    
    # initialize for shape decoder
    decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                         d_out=1,
                         n_layers=CFG['shape_decoder']['decoder_nlayers'],
                         udf_type='sdf')
    checkpoint_shape = torch.load(CFG['training']['checkpoint_shape'])
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    # initialize for deformation decoder
    decoder_deform = DeepSDF(lat_dim=512+200,
                    hidden_dim=1024,
                    geometric_init=False,
                    out_dim=3,
                    input_dim=3)
    checkpoint_deform  = torch.load(CFG['training']['checkpoint_deform'])
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)
    
    
    # initialize for encoder
    encoder = PCAutoEncoder()
    checkpoint_encoder = torch.load(CFG['training']['checkpoint_encoder'])
    encoder.load_state_dict(checkpoint_encoder['encoder_state_dict'])
    encoder.eval()
    encoder.to(device)
    
    
    # input
    meshfile = 'sample_result/shape_deform/deform_0017.ply'
    #sample = sample_surface(mesh, n_samps=CFG['training']['n_samples'])
    verts, face= load_ply(meshfile)
    
    mesh = Meshes(verts=verts.unsqueeze(0), faces=face.unsqueeze(0)).to(device)
    depth = render.get_depth(mesh)
    # save depth img
    depth_img = depth.detach().cpu().squeeze().numpy()
    plt.imsave('inference_sample{i}.png', depth_img)
    
    point_cloud = render.depth_pointcloud(depth)
    # save point cloud image
    pts = point_cloud.points_packed().detach().cpu().numpy()
    figure = plt.figure()
    for pt in pts:
        plt.scatter(pt[0], pt[1], pt[2])
    plt.savefig('inference_point_cloud.png')

    grid_points = get_grid_points()
    output = inference(point_cloud.points_packed(), encoder, decoder_shape, decoder_deform, grid_points,device)
    output.export('inference.ply')