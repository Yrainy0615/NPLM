from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .sample_surface import sample_surface
import yaml
import point_cloud_utils as pcu

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
    def __init__(self, root_dir, n_samples,n_sample_noise, sigma_near=0.01):
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.n_sample_noise = n_sample_noise
        self.all_npy = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.sigma_near = sigma_near
        
    def __len__(self):
        return len(self.all_npy)
    
    def __getitem__(self, index):
        trainfile = np.load(os.path.join(self.root_dir,self.all_npy[index]), allow_pickle=True)
        meshfile = trainfile.item()['mesh_deform']
        mesh = self.load_mesh(meshfile)
        sample = sample_surface(mesh, n_samps=self.n_samples)
        points = sample['points']
        noise_index = np.random.choice(points.shape[0], self.n_sample_noise, replace=False)
        noise = points[noise_index] + np.random.randn(points[noise_index].shape[0], 3) * self.sigma_near
        data = {
            'latent_shape': trainfile.item()['latent_shape'],
            'latent_deform' :trainfile.item()['latent_def'],
            'points_surface': points,
            'points_noise':  noise
        }
        return data
    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh
        
        
if __name__ == "__main__":
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    dataset = InversionDataset(root_dir=CFG['training']['root_dir'],
                                 n_samples=CFG['training']['n_samples'],
                                 n_sample_noise=CFG['training']['n_sample_noise'],
                                 sigma_near=CFG['training']['sigma_near'])
    dataloader = DataLoader(dataset, batch_size=CFG['training']['batch_size'],shuffle=False, num_workers=2)
    batch = next(iter(dataloader))