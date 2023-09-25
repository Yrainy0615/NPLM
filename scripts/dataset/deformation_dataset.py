import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DataManager import LeafDataManager
import os
import yaml
from typing import Literal


class LeafDeformationDataset(Dataset):
    def __init__(self, 
                 mode: Literal['train', 'val'],
                 n_supervision_points: int,
                 batch_size: int,
                 root_dir:str) :
        
        self.manager = LeafDataManager(root_dir)
        self.mode = mode
        self.all_species=self.manager.get_all_species()
        self.all_pose = self.manager.get_all_pose()
    
        
    def __len__(self):
        return len(self.all_pose)
    
    def __getitem__(self, index):
        pose_info = self.all_pose[index]
        pass
        return 

     
if __name__ == "__main__":
    cfg_path = '/home/yang/projects/parametric-leaf/NPM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    dataset = LeafDeformationDataset(mode='train',
                               n_supervision_points=CFG['training']['npoints_decoder'],
                               batch_size=CFG['training']['batch_size'],
                               root_dir=CFG['training']['root_dir'])
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=True, num_workers=2)
    batch = next(iter(dataloader))
    pass
