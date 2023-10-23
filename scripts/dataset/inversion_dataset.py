from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# from .sample_surface import sample_surface
import yaml
import point_cloud_utils as pcu
from scripts.model.renderer import MeshRender
from pytorch3d.io import load_objs_as_meshes, load_obj,load_ply
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
import torch

def uniform_ball(n_points, rad=1.0):
    angle1 = np.random.uniform(-1, 1, n_points)
    angle2 = np.random.uniform(0, 1, n_points)
    radius = np.random.uniform(0, rad, n_points)

    r = radius ** (1/3)
    theta = np.arccos(angle1) #np.pi * angle1
    phi = 2 * np.pi * angle2
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis=-1)
class InversionDataset(Dataset):
    def __init__(self, root_dir, n_samples,n_sample_noise, device,sigma_near=0.01):
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.n_sample_noise = n_sample_noise
        self.all_npy = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.sigma_near = sigma_near
        self.device = device
        self.renderer = MeshRender(device=device)
        
    def __len__(self):
        return len(self.all_npy)
    
    def __getitem__(self, index):
        trainfile = np.load(os.path.join(self.root_dir,self.all_npy[index]), allow_pickle=True)
        meshfile = trainfile.item()['mesh_deform']
        #mesh = self.load_mesh(meshfile)
        # sample = sample_surface(mesh, n_samps=self.n_samples)
        # points = sample['points']
        # noise_index = np.random.choice(points.shape[0], self.n_sample_noise, replace=False)
        # noise = points[noise_index] + np.random.randn(points[noise_index].shape[0], 3) * self.sigma_near
        verts,face = load_ply(meshfile)
        mesh = Meshes(verts=verts.unsqueeze(0), faces=face.unsqueeze(0)).to(self.device)
        depth = self.renderer.get_depth(mesh)
        point_cloud = self.renderer.depth_pointcloud(depth)
        pts = point_cloud._points_padded.squeeze(0)
        data = {
            'latent_shape': trainfile.item()['latent_shape'],
          #  'latent_spc': trainfile.item()['latent_spc'],
            'latent_deform' :trainfile.item()['latent_def'],
            'points': pts,
        }
        return data
    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh
        
        
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    dataset = InversionDataset(root_dir=CFG['training']['root_dir'],
                                 n_samples=CFG['training']['n_samples'],
                                 n_sample_noise=CFG['training']['n_sample_noise'],
                                 device=device,
                                 sigma_near=CFG['training']['sigma_near'])
    dataloader = DataLoader(dataset, batch_size=CFG['training']['batch_size'],shuffle=False, num_workers=2)
    batch = next(iter(dataloader))