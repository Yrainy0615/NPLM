import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
import os
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj , load_objs_as_meshes
import torch
import warnings 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data.dataloader import default_collate
import re
from collections import defaultdict
from PIL import Image
from transformers import ViTImageProcessor

warnings.filterwarnings('ignore', message='No mtl file provided')

def extract_info_from_masks(masks):
    plant_type_indices = defaultdict(int)
    for i, filename in enumerate(masks):
        plant_type = filename.split('_')[0]  
        if plant_type not in plant_type_indices:
            plant_type_indices[plant_type] = i
    return plant_type_indices

def extract_shape_index(canonical_name, plant_type_indices):
    pattern = re.compile(r'([A-Za-z]+)[_]?(\d+)')
    extracted_info = {}
    if '_d' in canonical_name:
        canonical_name = canonical_name.replace('_d','')
        match = pattern.search(canonical_name)
        plant_type = match.group(1)
        number = match.group(2)
        plant_type_index = plant_type_indices[plant_type] + 59 + int(number)
        
    else: 
        match = pattern.search(canonical_name)
        plant_type = match.group(1)
        number = match.group(2)
        plant_type_index = plant_type_indices[plant_type] + int(number)
    return plant_type_index, plant_type

def custom_collate_fn(batch):
    batch_points = [item['points'] for item in batch]
    batch_rgb = [item['rgb'] for item in batch]
    batch_canonical_verts = [item['canonical_verts'] for item in batch]
    batch_deformed_verts = [item['deformed_verts'] for item in batch]
    batch_camera_pose = [item['camera_pose'] for item in batch]
    batch_deform_index = [item['deform_index'] for item in batch]
    batch_shape_index = [item['shape_index'] for item in batch]
    batch_inputs = [item['inputs'] for item in batch]
    batch_canonical_rgb = [item['canonical_rgb'] for item in batch]
    batch_canonical_mask = [item['canonical_mask'] for item in batch]
    collated_data = {
        'rgb': default_collate(batch_rgb),
        'camera_pose': default_collate(batch_camera_pose),
        'deform_index': default_collate(batch_deform_index),
        'shape_index': default_collate(batch_shape_index),
        'inputs': default_collate(batch_inputs),
        'canonical_rgb': default_collate(batch_canonical_rgb),
        'canonical_mask': default_collate(batch_canonical_mask)
    }
    collated_data['points'] = batch_points
    collated_data['canonical_verts'] = batch_canonical_verts
    collated_data['deformed_verts'] = batch_deformed_verts
    
    return collated_data

def visualize_voxel(voxel_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k')

    plt.show()

def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices

def rgbd_to_point_cloud(rgb_file,depth_file, camera_info):
    rgb = o3d.io.read_image(rgb_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    cam2world_matrix = camera_info['cam2world_matrix'][1]
    cam2world_matrix = np.array(cam2world_matrix)
    image_width = 512
    image_height = 512
    sensor_width_mm = 32
    pixels_per_mm = image_width / sensor_width_mm
    fx = fy = 35 * pixels_per_mm
    cx = image_width / 2
    cy = image_height / 2
    intrinsics = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,intrinsics)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.01)
    # target_size = 128
    # target_voxel_grid = np.zeros((target_size, target_size, target_size))
    # min_bound = voxel_grid.get_min_bound()
    # max_bound = voxel_grid.get_max_bound()
    # scale_factor = (max_bound - min_bound) / target_size
    # for voxel in voxel_grid.get_voxels():
    #     x_idx = int((voxel.grid_index[0] * voxel_grid.voxel_size - min_bound[0]) / scale_factor[0])
    #     y_idx = int((voxel.grid_index[1] * voxel_grid.voxel_size - min_bound[1]) / scale_factor[1])
    #     z_idx = int((voxel.grid_index[2] * voxel_grid.voxel_size - min_bound[2]) / scale_factor[2])

    #     # if 0 <= x_idx < target_size and 0 <= y_idx < target_size and 0 <= z_idx < target_size:
    #     target_voxel_grid[x_idx, y_idx, z_idx] = 1

    pcd_points = np.asarray(pcd.points)
    pcd_points = normalize_verts(pcd_points)
    
    return pcd_points

class Point_cloud_dataset(Dataset):
    def __init__(self):
        self.root_dir = 'dataset/Mesh_colored'
        self.deformed_dir = os.path.join(self.root_dir,'deformed')
        self.all_deformed_file = os.listdir(self.deformed_dir)
        self.all_folders = os.listdir(os.path.join(self.root_dir, 'views'))
        self.all_depth = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(self.root_dir, 'views')):
            for file in filenames:
                if 'depth' in file and file.endswith('.png'):
                    full_path = os.path.join(dirpath, file)
                    self.all_depth.append(full_path)
        self.all_mask = []
        for dirpath, dirnames, filenames in os.walk('dataset/LeafData'):
            for file in filenames:
                if 'mask' in file and file.endswith('.JPG'):
                    self.all_mask.append(file)
        self.all_mask.sort()
        self.plant_type_indices  = extract_info_from_masks(self.all_mask)

    def __len__(self):
        return len(self.all_depth)
    def __getitem__(self, idx):
        depth_file = self.all_depth[idx]
        rgb_file = depth_file.replace('_depth','')
        deformed_name = depth_file.split('/')[-2]
        render_index = depth_file.split('/')[-1].split('_')[1].split('.')[0]
        deforn_index = depth_file.split('/')[-1].split('_')[1]
        deformed_mesh = load_obj(os.path.join(self.deformed_dir,deformed_name+'.obj'))
        deformed_verts = deformed_mesh[0]
        # name 
        last_index = deformed_name.rfind('_')
        canonical_name = deformed_name[:last_index]
        # get shape index from self.all_mask
        shape_index, plant = extract_shape_index(canonical_name, self.plant_type_indices)
        filename = self.all_mask[shape_index]
        if 'healthy' in filename:
            canonical_mask = os.path.join('dataset/LeafData',plant,'healthy', filename)
        else:
            canonical_mask = os.path.join('dataset/LeafData',plant,'diseased', filename)
        canonical_rgb_file = canonical_mask.replace('_mask', '')
        canonical_rgb = cv2.imread(canonical_rgb_file)
        canonical_rgb = cv2.cvtColor(canonical_rgb, cv2.COLOR_BGR2RGB)
        canonical_rgb = cv2.resize(canonical_rgb, (256, 256))
        canonical_mask_img = cv2.imread(canonical_mask)
        canonical_mask_img = cv2.resize(canonical_mask_img, (256, 256))
        canonical_mesh =load_obj(os.path.join(self.root_dir, canonical_name+'.obj'))
        canonical_verts = canonical_mesh[0]
        camera_file = os.path.join(self.root_dir,'views',deformed_name,'camera.json')
        point_savename  = depth_file.replace('_depth.png',' _points.npy')
        with open(camera_file) as f:
            camera_info = json.load(f)
        if os.path.exists(point_savename):
            point_cloud = np.load(point_savename)
            print('{} is loaded'.format(point_savename))
        else:
                point_cloud = rgbd_to_point_cloud(rgb_file,depth_file,camera_info)
                np.save(point_savename, point_cloud)
                print('{} is saved'.format(point_savename))
        
        # load rgb
        processor =  ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        rgb = Image.open(rgb_file).convert("RGB")
        
        inputs = processor(images=rgb, return_tensors="pt")

        azimuth = torch.tensor(camera_info['azimuth'][int(render_index)]).rad2deg()
        polar_angle = torch.tensor(camera_info['polar_angle'][int(render_index)]).rad2deg()
        data = {
           'points': point_cloud,
            'rgb': np.array(rgb.resize((256,256))),
            'deformed_verts': deformed_verts,
            'canonical_verts': canonical_verts,
           'camera_pose': np.array([azimuth, polar_angle]),
           'deform_index': deforn_index,
           'shape_index': shape_index,
           'inputs':inputs,
           'canonical_rgb': canonical_rgb,
           'canonical_mask': canonical_mask_img
        }
        return data
    
    
    
if __name__ == "__main__":
    dataset = Point_cloud_dataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print('Done')
