import sys
sys.path.append('NPLM')
from scripts.geometry.dmtet import DMTet
import trimesh
import numpy as np
import torch
from scripts.model.fields import UDFNetwork
import yaml
from scripts.dataset.img_to_3dsdf import sdf2d_3d, mesh_from_sdf

# initialize
dmtet  = DMTet()
device = 'cuda:0'
config = 'NPLM/scripts/configs/npm_def.yaml'
CFG = yaml.safe_load(open(config, 'r'))


# load data
leaf_path = 'dataset/ScanData/canonical/ash_canonical.obj'
mesh  = trimesh.load(leaf_path)
vertices = mesh.vertices
sample_index = np.random.choice(vertices.shape[0], size=200, replace=False)
sample_points = vertices[sample_index]
sample_points_tensor = torch.from_numpy(sample_points).float().to(device)
sdf_file = 'dataset/leaf_classification/images/Acer_Capillipes/1196 .npy'
sdf_2d = np.load(sdf_file, allow_pickle=True)
sdf_3d = sdf2d_3d(sdf_2d)
sdf_3d = torch.from_numpy(sdf_3d).float().to(device)

 

# load decoder

decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                        d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                        d_out=CFG['shape_decoder']['decoder_out_dim'],
                        n_layers=CFG['shape_decoder']['decoder_nlayers'],
                        d_in_spatial=2)

checkpoint_shape = torch.load('checkpoints/2dShape/exp-sdf2d__10.tar')
lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
decoder_shape.eval()
decoder_shape.to(device)


#  get mesh
tets = np.load('dataset/32_tets.npz')

verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda:0') * 1
v_deformed =  torch.zeros_like(verts).to(device)
indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda:0')

verts, faces,_,_ = dmtet(v_deformed, sdf_3d.reshape(-1,1), indices)

