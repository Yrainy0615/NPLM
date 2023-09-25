import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd 
import trimesh
import numpy as np


def read_from_csv(path):
    df = pd.read_csv(path,usecols=['key','index'])
    grouped = df.groupby('key')['index'].apply(list).reset_index()
    dict = grouped.set_index('key').to_dict()['index']
    return dict  

import numpy as np


class LeafKeypointDataset(Dataset):
    def __init__(self, root_path, transform=None):
        super().__init__()
        self.root_path = root_path
        self.obj_file = [f for f in os.listdir(root_path) if f.endswith('.obj')]
        #annotation =read_from_csv(os.path.join(root_path, 'keypoint.csv'))
        annotation =pd.read_csv(os.path.join(root_path, 'keypoint.csv'))
        self.keypoints_all = annotation['index'].values
        self.transform = transform
    
    def random_scale(self,points, scale_range = (0.7,1.3)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        point_cloud = points * scale
        return point_cloud
    
    def random_rotation(self, points, rotation_range = ((-120,120))):
        angle = np.radians(np.random.uniform(rotation_range[0],rotation_range[1]))
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle),0],
            [0,0,1]
        ])
        return np.dot(points,rotation_matrix)
    
    def add_gaussian_noise(self, points, mean=0, std=0.01):
        return points + np.random.randn(*points.shape) * std
          
    def __getitem__(self, index):
        mesh = trimesh.load_mesh(os.path.join(self.root_path,self.obj_file[index]))
        vertices = np.array(mesh.vertices)
        if self.transform:
            vertices = self.random_scale( vertices)
            vertices = self.random_rotation(vertices)
            vertices = self.add_gaussian_noise(vertices)
        vertices = torch.tensor(vertices, dtype=torch.float32)
        # labels = np.zeros(len(vertices))
        # labels[ self.keypoints_all ] = 1
        # label = torch.tensor(labels)
        gt_pts = vertices[self.keypoints_all ]
        return   vertices, gt_pts
    
    def __len__(self):
        return len(self.obj_file)
    
if __name__ == "__main__":
    root_path = 'dataset/keypoints_detection'
    dataset = LeafKeypointDataset(root_path=root_path, transform=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4,shuffle=True,num_workers=2)
    verts, labels = next(iter(dataloader))
    pass