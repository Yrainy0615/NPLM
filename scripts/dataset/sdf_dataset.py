import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from scripts.dataset.DataManager import LeafScanManager, LeafImageManger
from typing import Literal
import os
import yaml
import igl
from scripts.dataset.sample_surface import sample_surface
from scripts.dataset.sample_deformation import sample
import trimesh
import point_cloud_utils as pcu
import cv2
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.structures import Meshes
from scripts.test.leaf_to_mesh import LeaftoMesh

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
        self.all_neutral = self.manager.get_all_neutral()
        self.num_neutral = len(self.all_neutral)
        # create a dictionary to map species to index
        self.species_to_idx = {species:idx for idx, species in enumerate(self.all_species)}
        self.leaf_to_mesh = LeaftoMesh(root_dir)
    def __len__(self):
        #return len([f for f in os.listdir(self.manager.get_neutral_path()) if f.endswith('.obj')])
        return len(self.all_species)
    
    def __getitem__(self, index):
        neutral = self.all_neutral[index]
        species = os.path.splitext(os.path.basename(neutral))[0]
        species = species.split('_')[0]
        train_file = np.load(self.manager.get_train_shape_file(species), allow_pickle=True)
        #train_file = np.load(self.manager.get_train_pose_file(spe), allow_pickle=True)
        points = train_file.item()['points'] 
        normals = train_file.item()['normals']
        #mesh_path = os.path.join(self.manager.get_neutral_path(),self.manager.get_neutral_pose(species))
        
        #mesh_file = mesh_path + '.obj'
        mesh = self.manager.load_mesh(neutral)
        # subsample points for supervision
        sup_idx = np.random.randint(0, points.shape[0], self.n_supervision_points_face)
        sup_points = points[sup_idx,:]
        sup_normals = normals[sup_idx, :]
        
        # subsample points for gradient-constraint (near surface &  random in space)
        sup_grad_far = uniform_ball(self.n_supervision_points_non_face, rad=0.5)
        sup_grad_far_sdf = igl.signed_distance(sup_grad_far,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        sup_grad_near_sdf = igl.signed_distance(sup_grad_near,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        
        ret_dict = {'points': sup_points,
                    'normals': sup_normals,
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'sup_grad_near_sdf': sup_grad_near_sdf,
                    'sup_grad_far_sdf': sup_grad_far_sdf,
                    'idx': np.array([index]),
                    'spc': self.species_to_idx[species]}
        return ret_dict
       
class LeafDeformDataset(Dataset):
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
        self.n_supervision_points_face = n_supervision_points_face
        self.near_sigma = sigma_near
        self.all_neutral = self.manager.get_all_neutral()
        self.all_posed = [self.manager.get_all_pose()]
        self.num_species = len(self.all_species)

        self.checkpoint = torch.load('checkpoints/2dShape/exp-cg-sdf__15000.tar')
        self.lat_shape_all = self.checkpoint['latent_idx_state_dict']['weight']
        self.species_to_idx = {species:idx for idx, species in enumerate(self.all_species)}

    def __len__(self):
        return len(self.all_posed[0])     
    
    def __getitem__(self, index):
        posed = self.all_posed[0][index]
      
        species = list(posed.keys())[0]
        name  = posed[species]
        name = name.split('.obj')[0]
        name = name + '_deform.npy' 
        trainfile = np.load(name, allow_pickle=True)
        valid = np.logical_not(np.any(np.isnan(trainfile), axis=-1))
        point_corresp = trainfile[valid,:].astype(np.float32)
        
        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points_face)
        sup_point_neutral = point_corresp[sup_idx,:3]
        sup_posed = point_corresp[sup_idx,3:]
        neutral = sup_point_neutral
        pose = sup_posed
        if species == 'id2' or species == 'id1':
            species = 'yellow'
        return {
            'points_neutral': neutral,
            'points_posed': pose,
            'idx': np.array([index]),
            'species': self.species_to_idx[species],
            'species_to_idx': self.species_to_idx,
        }

class LeafColorDataset(Dataset):
    def __init__(self,
                 mode: Literal['train','val'],
                image_size:int,
                 batch_size: int,
                 root_dir: str):
        self.manager = LeafImageManger(root_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.all_mesh= self.manager.get_all_mesh()
        self.all_mesh = self.all_mesh
        self.all_mask = self.manager.get_all_mask()
        self.all_species = self.manager.get_all_species()
        self.species_to_idx = self.manager.get_species_to_idx()
        self.template = load_ply('dataset/leaf_uv_5sp.ply')
        self.leaf_to_mesh = LeaftoMesh(root_dir)

    
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
       # mesh_file = self.all_mesh[index]
        mesh_file = self.all_mesh[index]
        mesh = trimesh.load(mesh_file)
        vert_t = mesh.vertices
        vert_t = torch.from_numpy(vert_t).float()
        vert_t = vert_t - vert_t.mean(0)
        vert_t = vert_t/vert_t.abs().max()
        # mask_file = self.all_mask[index]  
        verts, faces = self.template[0], self.template[1]
        verts = verts - verts.mean(0)
        verts = verts/verts.abs().max()
        # assert shape is equal or print filename
        if verts.shape[0] == vert_t.shape[0]:
        
            delta_x = vert_t[:,:2]-verts[:,:2]
        else: 
            print(f'{mesh_file} shape is {vert_t.shape}')
        rgb_file = mesh_file.replace('.obj', '.png')
        rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size,self.image_size))
        rgb =rgb/255
        # mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask, (self.image_size,self.image_size))
        # verts_t = self.leaf_to_mesh.find_vertex(mask,verts[:, :2].numpy())
        # verts_t = torch.from_numpy(verts_t).float()
        # verts_t = verts_t - verts_t.mean(0)
        # verts_t = verts_t/verts_t.abs().max()
        # mask = mask/255
        # extract spcies from file name
        species = os.path.basename(mesh_file).split('_')[0]
        # dict = self.manager.extract_info_from_meshfile(mesh_file)
        # mesh = dict['mesh']
        # sample = sample_surface(mesh,n_samps=3000)
        # sup_points = sample['points']
        # rgb rotate 90 degree
        #rgb = np.rot90(rgb, k=1, axes=(0, 1))
        #dict = self.manager.extract_info_from_meshfile(mesh_file)
        # verts, face = load_ply(mesh_file)
        # mesh = Meshes(verts=[verts], faces=[face])
        # verts = mesh.verts_packed()

        ret_dict = {
            # 'mask': mask,
            'rgb':rgb,
            'idx': np.array([index]),
            'spc': self.species_to_idx[species],
            'verts': verts,
            'faces': faces ,
            'delta_x': delta_x
           # 'verts_t': verts_t,
            }
      
        
        return ret_dict
    def custom_collate_fn(self,batch):
        ret_dict = {
        'verts': [],
        'mesh': [],
        'rgb': [],
        'idx': []
    }

        for sample in batch:
            for key in ret_dict.keys():
                ret_dict[key].append(sample[key])

        # 如果您想将 'idx' 转为一个tensor，您可以在这里进行转换
        ret_dict['idx'] = torch.tensor(ret_dict['idx'])

        return ret_dict

    def get_loader(self):
        return DataLoader(self,batch_size=1, shuffle=False, num_workers=0, collate_fn=self.custom_collate_fn)

class Leaf2DShapeDataset(Dataset):
    def __init__(self,
                 mode: Literal['train','val'],
                 n_supervision_points_face: int,
                 batch_size: int,
                 sigma_near: float,
                 root_dir: str):
        self.manager = LeafImageManger(root_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.sigma_near = sigma_near
        self.all_species = self.manager.get_all_species()
        self.all_mesh= self.manager.get_all_mesh()
        self.all_mask = self.manager.get_all_mask()
        self.species_to_idx = self.manager.get_species_to_idx()
        # create a demo mesh set of all mesh ,where index %50 =0
        self.demo_mesh = [self.all_mesh[i] for i in range(len(self.all_mesh)) if i % 30 == 0]
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        mesh_file = self.all_mesh[index]
        # mask_file = mesh_file.replace('.obj', '_mask_aligned.JPG') 
        # mask = cv2.imread(mask_file)
        # mask = cv2.resize(mask, (64,64))
        dict = self.manager.extract_info_from_meshfile(mesh_file)
        mesh = dict['mesh']
        sample = sample_surface(mesh,n_samps=self.n_supervision_points_face)
        sup_points = sample['points']
      
        sup_grad_far = uniform_ball(self.n_supervision_points_non_face, rad=0.5)
        sup_grad_far_sdf = igl.signed_distance(sup_grad_far,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        sup_grad_near_sdf = igl.signed_distance(sup_grad_near,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        
        ret_dict = {'points':  sup_points ,
                    'normals': sample['normals'],
                    'sup_grad_far': sup_grad_far,
                    'sup_grad_near': sup_grad_near,
                    'sup_grad_near_sdf': sup_grad_near_sdf,
                    'sup_grad_far_sdf': sup_grad_far_sdf,
                    'idx': np.array([index]),
                    'spc': self.species_to_idx[dict['species']]}
        return ret_dict


class LeafDisplacementDataset(Dataset):
    def __init__(self,
                 mode: Literal['train','val'],
                 n_supervision_points_face: int,
                 batch_size: int,
                 sigma_near: float,
                 root_dir: str):
        self.manager = LeafImageManger(root_dir)
        self.scan_manager = LeafScanManager('dataset/ScanData/')
        self.mode = mode
        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.sigma_near = sigma_near
        self.all_species = len(self.manager.get_all_species())
        self.all_mesh= self.manager.get_all_mesh()
        self.all_mask = self.manager.get_all_mask()
        self.species_to_idx = self.manager.get_species_to_idx()
        normal_folder= os.listdir('dataset/normal_cg') 
        normalfile = [os.path.join('dataset/normal_cg', f) for f in normal_folder]
        # read all the image to rgb
        self.all_normal = [cv2.imread(f) for f in normalfile]
        # convert to rgb
        self.all_normal = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.all_normal]
        # resize the image
        self.all_normal = [cv2.resize(img, (64,64)) for img in self.all_normal]
        # normalize the image
        self.all_normal = [img/255.0 for img in self.all_normal]
        self.spc_normal = [f.split('/')[-1].split('_')[0] for f in normalfile]
        self.all_neutral = self.scan_manager.get_all_neutral()

    
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        mesh_file = self.all_mesh[index]
        mesh_2d = trimesh.load(mesh_file)
        #mask_file = mesh_file.replace('.obj', '_mask_aligned.JPG') 
        # mask = cv2.imread(mask_file)
        # mask = cv2.resize(mask, (64,64))
        dict = self.manager.extract_info_from_meshfile(mesh_file)
        neutral =self.scan_manager.get_neutral_pose(dict['species'])
        mesh_file = os.path.join('dataset/ScanData/',neutral)
        if mesh_file not in self.all_neutral:
            mesh_file =  os.path.join('dataset/ScanData/','canonical/ash_canonical.obj')
        mesh_neutral = trimesh.load(mesh_file)
        mesh = dict['mesh']
        p_neutral, sup_points, normals_neutral, normals = sample(mesh_neutral, mesh_2d, 0.01, self.n_supervision_points_face)
        # #self.viz_sample(p_neutral, normals_neutral,p, normals)
        # sup_grad_far = uniform_ball(self.n_supervision_points_non_face, rad=0.5)
        # sup_grad_far_sdf = igl.signed_distance(sup_grad_far,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        # sup_grad_near = sup_points + np.random.randn(sup_points.shape[0], 3) * self.sigma_near
        # sup_grad_near_sdf = igl.signed_distance(sup_grad_near,mesh.vertex_data.positions, mesh.face_data.vertex_ids)[0]
        verts = mesh.vertex_data.positions
        # random sample verts
        surf_index = np.random.randint(0, verts.shape[0], self.n_supervision_points_face)
        sup_points = verts[surf_index,:]
        spc = dict['species']
        # check if the species is in the name of 
        normal_map = self.all_normal[self.spc_normal.index(spc)]
        # ret_dict = {'points':  sup_points ,
        #             'normals': normals_neutral,
        #             'sup_grad_far': sup_grad_far,
        #             'sup_grad_near': sup_grad_near,
        #             'sup_grad_near_sdf': sup_grad_near_sdf,
        #             'sup_grad_far_sdf': sup_grad_far_sdf,
        #             'idx': np.array([index]),
        #             'spc': self.species_to_idx[dict['species']]}
        # return ret_dict   
        ret_dict = {
                    'normal_map':normal_map,
                    'mesh': mesh,
             #       'verts': verts,
                    'sup_points': sup_points,
                 #   'points_sdf': p_sdf,
                #    'neutral_normal': normals_neutral,
                    'surf_index': surf_index,
                    'idx': np.array([index]),}
        return ret_dict
    def custom_collate_fn(self, batch):
        merged_dict = {
            'normal_map': [],
            'mesh': [],
            'sup_points': [],
        #    'sup_normals' : [],
            'surf_index': [],
            'idx': [],
        #    'point_sdf': [],
       #     'neutral_normal': [],
       #     'verts': [],
        }
        for data in batch:
            for key, value in data.items():
                merged_dict[key].append(value)
        # to tensor
        merged_dict['sup_points'] = torch.tensor(merged_dict['sup_points'])
        merged_dict['surf_index'] = torch.tensor(merged_dict['surf_index'])
        merged_dict['idx'] = torch.tensor(merged_dict['idx'])
        merged_dict['normal_map'] = torch.tensor(merged_dict['normal_map'])
     #  merged_dict['neutral_normal'] = torch.tensor(merged_dict['neutral_normal'])
       # merged_dict['verts'] = torch.tensor(merged_dict['verts'])

       
        return merged_dict
    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(
        dataset=self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=self.custom_collate_fn
    ) 
    def viz_sample(self,pts_neutral, normal_neutral,pts_2d,normal_2d):
        fig = plt.figure(figsize=(12, 6))

        # Subplot for neutral points
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(pts_neutral[:, 0], pts_neutral[:, 1], normal_neutral[:, 2], c=normal_neutral[:, 0])
        ax1.set_title('Neutral Points')
        
        # Subplot for other points
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(pts_2d[:, 0], pts_2d[:, 1], pts_2d[:, 2], c=normal_2d[:, 0])
        ax2.set_title('Other Points')
        plt.savefig('sample.png', bbox_inches='tight')  # Save the figure
        plt.close(fig)  # Close the figure to free

if __name__ == "__main__":
    cfg_path ='NPLM/scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(cfg_path, 'r'))
    # dataset = LeafShapeDataset(mode='train',
    #                            n_supervision_points_face=CFG['training']['npoints_decoder'],
    #                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
    #                            batch_size=CFG['training']['batch_size'],
    #                            sigma_near=CFG['training']['sigma_near'],
    #                            root_dir=CFG['training']['root_dir'])
    # dataset = LeafDeformDataset(mode='train',
    #                            n_supervision_points_face=CFG['training']['npoints_decoder'],
    #                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
    #                            batch_size=CFG['training']['batch_size'],
    #                            sigma_near=CFG['training']['sigma_near'],
    #                            root_dir=CFG['training']['root_dir'])
    # dataset = LeafDisplacementDataset(mode='train',    
    #                              n_supervision_points_face=CFG['training']['npoints_decoder'],
    #                              batch_size=CFG['training']['batch_size'],
    #                              sigma_near=CFG['training']['sigma_near'],
    #                              root_dir=CFG['training']['root_dir_color'])
    dataset = LeafColorDataset(mode='train',
                                 n_supervision_points_face=CFG['training']['npoints_decoder'],
                                 n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                                 batch_size=CFG['training']['batch_size'],
                                 sigma_near=CFG['training']['sigma_near'],
                                 root_dir=CFG['training']['root_dir_color'])
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    #dataloader = dataset.get_loader(batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    pass
    
  
        
   
