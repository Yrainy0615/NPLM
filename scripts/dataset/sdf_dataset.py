from torch.utils.data import Dataset, DataLoader
import numpy as np
from .DataManager import LeafScanManager, LeafImageManger
from typing import Literal
import os
import yaml
from .sample_surface import sample_surface
import json
import cv2

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

class LeafDeformDataset(Dataset):
    def __init__(self,
                 n_supervision_points_face: int,
                 root_dir: str):
        self.n_supervision_points_face = n_supervision_points_face
        self.root_dir = root_dir    
        self.all_sample = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.all_sample  = sorted(self.all_sample)
        label_file = os.path.join(root_dir, 'maple_label.json')
        with open(label_file, 'r') as f:
            self.label = json.load(f)

    def __len__(self):
        return len(self.all_sample)     
    
    def __getitem__(self, index):
        filename = self.all_sample[index]
        parts = filename.split('_')
        name = '_'.join(parts[:2])
        label = self.label[name]
        trainfile = np.load(os.path.join(self.root_dir,self.all_sample[index]), allow_pickle=True)
        valid = np.logical_not(np.any(np.isnan(trainfile), axis=-1))
        point_corresp = trainfile[valid,:].astype(np.float32)
        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points_face)
        sup_point_neutral = point_corresp[sup_idx,:3]
        sup_posed = point_corresp[sup_idx,3:] 
        neutral = sup_point_neutral
        pose = sup_posed
        return {
            'points_neutral': neutral,
            'points_posed': pose,
            'idx': np.array([index]),
            'label':label
        }


class LeafColorDataset(Dataset):
    def __init__(self,
                 mode: Literal['train','val'],
                 n_supervision_points_face: int,
                 n_supervision_points_non_face: int,
                 batch_size: int,
                 sigma_near: float,
                 root_dir: str):
        self.manager = LeafImageManger(root_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face  = n_supervision_points_non_face
        self.sigma_near = sigma_near
        self.all_mesh= self.manager.get_all_mesh()
        self.all_mask = self.manager.get_all_mask()
        self.species_to_idx = self.manager.get_species_to_idx()
    
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        mesh_file = self.all_mesh[index]
        rgb_file = mesh_file.replace('.obj', '_mask_aligned.JPG')
        rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (64,64))
        dict = self.manager.extract_info_from_meshfile(mesh_file)
        mesh = dict['mesh']
        sample = sample_surface(mesh,n_samps=2000)
        sup_points = sample['points']
      
        sup_grad_far = uniform_ball(self.n_supervision_points_face //8, rad=0.5)
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        points = np.concatenate([sup_points, sup_grad_far, sup_grad_near], axis=0)
        ret_dict = {'points': points,
                    'rgb': rgb,
                    'idx': np.array([index]),}
        return ret_dict


class LeafSDF2dFDataset(Dataset):
    def __init__(self, root_dir, num_samples) -> None:
        super().__init__()
        self.all_file = []
        self.root_dir = root_dir
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('_128.npy') and not 'sdf' in filename:
                    self.all_file.append(os.path.join(dirpath, filename))
        self.all_file.sort()
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.all_file)
    
    def __getitem__(self, index):
        trainfile = self.all_file[index]
        grid_size = 128
        sdf_2d = np.load(trainfile, allow_pickle=True)
        sdf_gt = sdf_2d.reshape(-1)
        # random sampling from sdf_grid
        x = np.linspace(0, grid_size, grid_size)
        y = np.linspace(0, grid_size, grid_size)
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        #points = self.sample_points_from_grid(sdf_2d.shape[0], self.num_samples)
        # sdf_gt = sdf_2d[points[:,0], points[:,1]]
        return {'points': points,
                'sdf_gt': sdf_gt,
                'index': index,}
        
    def sample_points_from_grid(self, n, x):
        if x > n * n:
            raise ValueError("over sampling")
        points = [(i, j) for i in range(n) for j in range(n)]
        sampled_points = np.random.choice(len(points), size=x, replace=False)
        
        return np.array(points)[sampled_points]
    

if __name__ == "__main__":
    cfg_path ='NPLM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    dataset = LeafDeformDataset(mode='train',
                                n_supervision_points_face=100,
                                n_supervision_points_non_face=0,
                                batch_size=1,
                                sigma_near=0.01,
                                root_dir=CFG['dataset']['root_dir'])                                                            
    
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=2)
    batch = next(iter(dataloader))
   
