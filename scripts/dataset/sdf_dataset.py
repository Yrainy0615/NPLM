import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from .DataManager import LeafScanManager, LeafImageManger
from typing import Literal
import os
import yaml
import igl
from .sample_surface import sample_surface

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

class LeafShapeDataset(Dataset):
    def __init__(self,
                 mode: Literal['train','val'],
                 n_supervision_points_face: int,
                 n_supervision_points_non_face: int,
                 batch_size: int,
                 sigma_near: float,
                 root_dir: str):
        self.manager = LeafScanManager(root_dir)
        self.mode = mode
        self.all_species = self.manager.get_all_species()
        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face  = n_supervision_points_non_face
        self.sigma_near = sigma_near
        self.all_pose = self.manager.get_all_pose()
        
    def __len__(self):
        #return len([f for f in os.listdir(self.manager.get_neutral_path()) if f.endswith('.obj')])
        return len(self.all_pose)
    
    def __getitem__(self, index):
        #species = self.all_species[index]
        (k ,v) , = self.all_pose[index].items()
        # train_file = np.load(self.manager.get_train_shape_file(species), allow_pickle=True)
        train_file = np.load(self.manager.get_train_pose_file(k,v), allow_pickle=True)
        points = train_file.item()['points']
        normals = train_file.item()['normals']
        #mesh_path = os.path.join(self.manager.get_neutral_path(),self.manager.get_neutral_pose(species))
        mesh_file = self.manager.mesh_from_npy_file(self.manager.get_train_pose_file(k,v))
        #mesh_file = mesh_path + '.obj'
        mesh = self.manager.load_mesh(mesh_file)
        # subsample points for supervision
        sup_idx = np.random.randint(0, points.shape[0], self.n_supervision_points_face)
        sup_points = points[sup_idx,:]
        sup_normals = normals[sup_idx, :]
        
        # subsample points for gradient-constraint (near surface &  random in space)
        sup_grad_far = uniform_ball(self.n_supervision_points_face //8, rad=0.5)
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        sup_grad_near_udf = np.abs(igl.signed_distance(sup_grad_near,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0])
        
        ret_dict = {'points': sup_points,
                    'normals': sup_normals,
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'sup_grad_near_udf': sup_grad_near_udf,
                    'idx': np.array([index])}
        return ret_dict


class LeafImageDataset(Dataset):
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
        self.species_to_idx = self.manager.get_species_to_idx()
    
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        train_file = self.all_mesh[index]
        dict = self.manager.extract_info_from_meshfile(train_file)
        mesh = dict['mesh']
        sample = sample_surface(mesh,n_samps=2000)
        sup_points = sample['points']
        sup_normals = sample['points']
        sup_grad_far = uniform_ball(self.n_supervision_points_face //8, rad=0.5)
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        sup_grad_near_udf = np.abs(igl.signed_distance(sup_grad_near,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0])
        ret_dict = {'points': sup_points,
                    'normals': sup_normals,
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'sup_grad_near_udf': sup_grad_near_udf,
                    'idx': np.array([index]),
                    'species':self.species_to_idx[dict['species']] }
        return ret_dict
        
     
if __name__ == "__main__":
    cfg_path = '/home/yang/projects/parametric-leaf/NPLM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    dataset = LeafImageDataset(mode='train',
                               n_supervision_points_face=CFG['training']['npoints_decoder'],
                               n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                               batch_size=CFG['training']['batch_size'],
                               sigma_near=CFG['training']['sigma_near'],
                               root_dir=CFG['training']['root_dir'])
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False, num_workers=2)
    batch = next(iter(dataloader))
    pass
