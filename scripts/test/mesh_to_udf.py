import trimesh
import mesh2sdf
import time
import numpy as np
from scripts.model.reconstruction import mesh_from_logits, create_grid_points_from_bounds
from scripts.model.diff_operators import gradient   
from MeshUDF.custom_mc._marching_cubes_lewiner import udf_mc_lewiner
import torch
import skimage.measure
from MeshUDF.optimize_chamfer_A_to_B import get_mesh_udf_fast, optimize
from scripts.model.fields import UDFNetwork
import yaml
def refine_mesh(filename):

    mesh = trimesh.load(filename, force='mesh')
    mesh_scale = 0.8
    size = 128
    level = 2 / size 
    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) 

    # fix mesh
    t0 = time.time()
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    # calculate sdf normal

    t1 = time.time()
    # udf - marching cubes
    udf = np.abs(sdf)
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level=level)
    # scale mesh
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    udf_mc = trimesh.Trimesh(vertices, faces)
    return udf_mc
# scale mesh

if __name__ == "__main__":
    filename = 'test_2.obj'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_color.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    decoder = UDFNetwork(d_in= CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                        d_out=CFG['decoder']['decoder_out_dim'],
                        n_layers=CFG['decoder']['decoder_nlayers'],)
    checkpoint = torch.load('checkpoints/2dShape/exp-cg-sdf__2000.tar')
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    decoder.to(device)
    lat_all = checkpoint['latent_idx_state_dict']['weight']
    
    # load source
    latent_start = lat_all[0]
    latent_end = lat_all[3]
    _, _ ,mesh_start = get_mesh_udf_fast(decoder, latent_start.unsqueeze(0),gradient=False)
    _,_, mesh_target = get_mesh_udf_fast(decoder, latent_end.unsqueeze(0),gradient=False)

    optim_code, optim_mesh = optimize(
                    decoder,
                    20,
                    latent_start,
                    mesh_target,
                    lr=5e-3,
                    out_dir='test'
                ) 
    
    pass