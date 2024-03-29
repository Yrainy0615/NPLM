from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal
import os
import yaml
import json
import cv2
import igl
import trimesh
import sys
sys.path.append('NPLM')
from scripts.model.reconstruction import create_grid_points_from_bounds
from scipy.spatial import cKDTree as KDTree
import torch
from scipy.ndimage import rotate
from matplotlib import pyplot as plt

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = np.sum(np.square(np.expand_dims(array1, axis=1) - np.expand_dims(array2, axis=0)), axis=-1)
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance(pointsA, pointsB):
    dist_A = av_dist(pointsA, pointsB)
    dist_B = av_dist(pointsB, pointsA) 
    return dist_A + dist_B

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
                 ):
        self.n_supervision_points_face = n_supervision_points_face
        self.all_sample = []
        self.root_dir = 'deformation'
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    self.all_sample.append(os.path.join(dirpath, filename))
        self.all_sample.sort()
    def __len__(self):
        return len(self.all_sample)     
    
    def __getitem__(self, index):
        filename = self.all_sample[index]
        name = os.path.basename(filename)
        shape_index = name.split('_')[0]
        deform_index = name.split('_')[1]
        trainfile = np.load(self.all_sample[index], allow_pickle=True)
        valid = np.logical_not(np.any(np.isnan(trainfile), axis=-1))
        point_corresp = trainfile[valid,:].astype(np.float32)
        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points_face)
        sup_point_neutral = point_corresp[sup_idx,:3]
        sup_posed = point_corresp[sup_idx,3:] 
        # visualize neutral and  3d posed points
        neutral = sup_point_neutral
        pose = sup_posed
        # chamfer distance between neutral and posed
        chamfer=chamfer_distance(neutral, pose)
        print(chamfer)
        return {
            'points_neutral': neutral,
            'points_posed': pose,
            'shape_idx': np.array([int(shape_index)]),
            'deform_idx': np.array([int(deform_index)])  
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
                'idx': index,}
        
    def sample_points_from_grid(self, n, x):
        if x > n * n:
            raise ValueError("over sampling")
        points = [(i, j) for i in range(n) for j in range(n)]
        sampled_points = np.random.choice(len(points), size=x, replace=False)
        
        return np.array(points)[sampled_points]
    
class LeafSDF3dDataset(Dataset):
    def __init__(self, root_dir, num_samples, 
                 num_sample_space, sigma_near) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.all_file = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.all_file.sort()
        self.num_samples = num_samples
        self.sigma_near = sigma_near
        self.num_samples_space = num_sample_space
        all_base = [f.split('.')[0] for f in self.all_file]
        all_shape_id = [int(f.split('_')[0]) for f in all_base]
        # all_deform_id = [int(f.split('_')[1]) for f in all_base]
        # map shape_id to index
        self.shape_id_to_idx = {k: v for v, k in enumerate(set(all_shape_id))}
        # self.deform_id_to_idx = {k: v for v, k in enumerate(set(all_deform_id))}
        
    def __len__(self):
        return len(self.all_file)
    
    def __getitem__(self, index):
        trainfile = os.path.join(self.root_dir, self.all_file[index])
        basename = self.all_file[index].split('.')[0]
        shape_id = int(basename.split('_')[0])
        shape_index = self.shape_id_to_idx[shape_id]
        # deform_id = int(basename.split('_')[1])
        # deform_index = self.deform_id_to_idx[deform_id]
        data = np.load(trainfile, allow_pickle=True)
        points = data.item()['points']
        normals = data.item()['normals']
        sup_idx = np.random.randint(0, points.shape[0], self.num_samples)
        sup_points = points[sup_idx,:]
        sup_normals = normals[sup_idx,:]
        
        # near surface & random in space
        sup_grad_far = uniform_ball(self.num_samples_space, rad=0.5)
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        ret_dict = {'points': sup_points,
                    'normals': sup_normals,
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'idx': np.array(shape_index),}
              #      'deform_idx': np.array(deform_index),}
        return ret_dict

class EncoderDataset(Dataset):
    def __init__(self, root_dir, resolution=128):
        self.root_dir = root_dir
        self.all_mesh = [f for f in os.listdir(root_dir) if '_' in f and f.endswith('.obj')]
        mini = [-.95, -.95, -.95]
        maxi = [0.95, 0.95, 0.95]
        self.resolution = resolution    
        self.grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
        self.transform = False
        
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        mesh_name = self.all_mesh[index]
        basename = mesh_name.split('.')[0]
        shape_index = int(basename.split('_')[0])
        deform_index = int(basename.split('_')[1])
        mesh_file = os.path.join(self.root_dir, mesh_name) 
        npy_file = mesh_file.replace('.obj', '.npy')
        mesh = trimesh.load_mesh(mesh_file)
        ind = np.random.randint(0, len(mesh.vertices), 2000)
        points = mesh.vertices[ind]

        if os.path.exists(npy_file):
            occupancy_grid = np.load(npy_file, allow_pickle=True)
            if self.transform:
                # R = self.random_rotation_matrix()
                # points = np.dot(points, R.T)
                occupancy_grid = self.random_rotate_3d(occupancy_grid)
        else:
            
            kdtree = KDTree(self.grid_points)
            occupancies = np.zeros(len(self.grid_points), dtype=np.int8)
            _, idx = kdtree.query(mesh.vertices)
            occupancies[idx] = 1
            occupancy_grid = occupancies.reshape(self.resolution, self.resolution, self.resolution)
            np.save(npy_file,  occupancy_grid)
            print('{} is saved'.format(npy_file))
        return {'occupancy_grid': occupancy_grid,
                'points'  : points,
                'mesh_file': mesh_name,
                'shape_idx': shape_index,
                'deform_idx': deform_index,}
        
    def random_rotation_matrix(self):
        """
        生成一个随机的三维旋转矩阵。
        """
        angle_x = np.random.uniform(0,  np.pi /2)
        angle_y = np.random.uniform(0,  np.pi/2)
        angle_z = np.random.uniform(0,  np.pi/2)
        
        # 分别绕 x, y, z 轴的旋转矩阵
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
        
        # 组合旋转
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def random_rotate_3d(self, occupancy_grid):

        # 随机定义旋转角度
        angle_x = np.random.uniform(0, 360)  # 绕 x 轴旋转的角度
        angle_y = np.random.uniform(0, 360)  # 绕 y 轴旋转的角度
        angle_z = np.random.uniform(0, 360)  # 绕 z 轴旋转的角度

        # 依次应用旋转
        rotated_grid = rotate(occupancy_grid, angle_x, axes=(1, 2), reshape=False, mode='nearest')
        rotated_grid = rotate(rotated_grid, angle_y, axes=(0, 2), reshape=False, mode='nearest')
        rotated_grid = rotate(rotated_grid, angle_z, axes=(0, 1), reshape=False, mode='nearest')
        
        return rotated_grid

if __name__ == "__main__":
    cfg_path ='NPLM/scripts/configs/npm_3d.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    dataset = LeafDeformDataset(n_supervision_points_face=2000)
    #dataset = LeafSDF3dDataset(root_dir=CFG['training']['root_dir'],
                            #    num_samples=2000,
                            #    num_sample_space=2000,
                            #    sigma_near=0.01)
    
    #encoder_dataset = EncoderDataset(root_dir='dataset/deformation')
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1)
    for data in dataloader:
        #print(data['points'].shape)
        print(data['deform_idx'])
    pass

   
