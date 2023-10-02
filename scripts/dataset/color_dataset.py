import torch
from torch.utils.data import Dataset, DataLoader
import os
from DataManager import LeafImageManger
import numpy as np
import yaml
from pytorch3d.io import load_objs_as_meshes
from image_processor import ImageProcessor

class ColorDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = cfg['training']['root_dir']
        self.manager = LeafImageManger(self.root_dir)
        self.processor  = ImageProcessor(self.root_dir)
        self.rgb_list = self.manager.get_all_rgb_healthy()
        self.mask_list = self. manager.get_all_mask_healthy()
        assert len(self.rgb_list) == len(self.mask_list)
        rgb_data = []
        mask_data = []
        for i in range(len(self.rgb_list)):
            rgb, mask, rgba = self.manager.cv2_read_rgba(self.rgb_list[i], self.mask_list[i])
            rgb_data.append(rgb)
            mask_data.append(mask)
        self.data = {
            'rgb':torch.tensor(np.stack(rgb_data, axis=0), dtype=torch.float32).permute(0,3,1,2) / 255.,# (n_img, 1, h, w)
            'alpha':torch.tensor(np.stack(mask_data, axis=0), dtype=torch.float32)[:,None,:,:] # (n_img, 1,h,w)
         }
            
    def __len__(self):
        return len(self.rgb_list)
        
        
    def gen_rays_at(self, img_idx, resolution_level):
        pass
    
    def __getitem__(self, index):
        rgb = self.data['rgb'][index]
        alpha = self.data['alpha'][index]
        rgb = rgb*alpha 
        dict = self.manager.extract_info_from_file(self.rgb_list[index])
        sdf = self.processor.mask_to_sdf(self.mask_list[index], save_path=None)
        data = {
            'image': rgb,
            'mask': alpha,
            'index': index,
            'species': dict['species'],
            'sdf': sdf
        } 
        return data    
    
    def get_dataloader(self, shuffle = True):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        
        return DataLoader(self, batch_size=self.cfg['training']['batch_size'],
                          num_workers=8,shuffle=shuffle)
    
        
    

if __name__ == "__main__":
    cfg_path  = 'NPLM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    dataset = ColorDataset(CFG) 
    dataloader = dataset.get_dataloader()
    batch = next(iter(dataloader))
    pass