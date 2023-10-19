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
    input_tensor = torch.from_numpy(input).float().to(device).permute(1,0)
    latent_shape, latent_deform = encoder(input_tensor.unsqueeze(0))
    logits_shape = get_logits(decoder_shape, latent_shape, grid_points, nbatch_points=6000)
    mesh_base = mesh_from_logits(logits_shape, mini, maxi, 256)
    mesh_base.export('inference_base.ply')
    deform = deform_mesh(mesh=mesh_base, deformer=decoder_deform, lat_rep=latent_deform, anchors=None, lat_rep_shape=latent_shape)
    #deform.export('inference.ply')
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
  
    
    # initialize for shape decoder
    decoder_shape = DeepSDF(
            lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        ) 
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
    mesh = load_mesh('dataset/ScanData/Autumn_maple_leaf.001.obj')
    sample = sample_surface(mesh, n_samps=CFG['training']['n_samples'])
    points = sample['points']
    noise_index = np.random.choice(points.shape[0], CFG['training']['n_sample_noise'], replace=False)
    noise = points[noise_index] + np.random.randn(points[noise_index].shape[0], 3) * CFG['training']['sigma_near']
    input = np.concatenate([points, noise], axis=0)
    grid_points = get_grid_points()
    output = inference(input, encoder, decoder_shape, decoder_deform, grid_points,device)
    output.export('inference.ply')