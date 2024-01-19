import torch
from scripts.model.deepSDF import DeepSDF
from scripts.model.fields import UDFNetwork
import yaml
from MeshUDF.optimize_chamfer_A_to_B import get_mesh_udf_fast
from MeshUDF.custom_mc._marching_cubes_lewiner import udf_mc_lewiner
import os
from scripts.model.reconstruction import create_grid_points_from_bounds
from scripts.model.reconstruction import latent_to_mesh
import numpy as np


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open('NPLM/scripts/configs/npm.yaml', 'r'))
    decoder_shape = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=CFG['decoder']['decoder_out_dim'],
                         n_layers=CFG['decoder']['decoder_nlayers'],)
    checkpoint_shape = torch.load('checkpoints/cg_bs8/cgshape_bs8_udf_epoch__20500.tar')
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    lat_spc_all = checkpoint_shape['latent_spc_state_dict']['weight']
    
    # extract mesh from udf
    latent = torch.concat([lat_idx_all[0], lat_spc_all[0]]).to(device)
    #latent = lat_idx_all[0].to(device)
    
    _, _, udf_mesh,_,_ = get_mesh_udf_fast(decoder_shape, latent.unsqueeze(0))
    mc_mesh, mc_udf_mesh = latent_to_mesh(decoder_shape, latent, device)

    pass
    