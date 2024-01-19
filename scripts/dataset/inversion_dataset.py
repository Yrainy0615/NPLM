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
import mesh2sdf
import trimesh
import open3d as o3d
from pytorch3d.ops import sample_points_from_meshes, points_to_volumes
from pytorch3d.ops.points_to_volumes import add_pointclouds_to_volumes
import cv2
from scripts.model.renderer import MeshRender
from matplotlib import pyplot as plt
from torchvision import transforms

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
        mesh = trimesh.load(meshfile, force='mesh')
        mesh_scale = 0.8
        size = 128
        level = 2 / size 
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) 
        sdf, mesh = mesh2sdf.compute(
            vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        # calculate sdf normal
        #mesh = self.load_mesh(meshfile)
        # sample = sample_surface(mesh, n_samps=self.n_samples)
        # points = sample['points']
        # noise_index = np.random.choice(points.shape[0], self.n_sample_noise, replace=False)
        # noise = points[noise_index] + np.random.randn(points[noise_index].shape[0], 3) * self.sigma_near
        verts_tensor = torch.from_numpy(mesh.vertices).to(self.device)
        faces_tensor = torch.from_numpy(mesh.faces).to(self.device)
        mesh = Meshes(verts=verts_tensor.unsqueeze(0), faces=faces_tensor.unsqueeze(0)).to(self.device)
        depth = self.renderer.get_depth(mesh)
        point_cloud = self.renderer.depth_pointcloud(depth)
        # embed point cloud to sdf
    
       
        data = {
            'latent_shape': trainfile.item()['latent_shape'],
          #  'latent_spc': trainfile.item()['latent_spc'],
            'latent_deform' :trainfile.item()['latent_def'],
            'sdf': sdf,
            'pts':pts
        }
        return data
    
    def depth_numpy_to_voxel(depth_numpy, voxel_size=0.05, depth_scale=1000.0, max_depth=3.0):
        # 从numpy数组创建Open3D的Image对象
        depth = o3d.geometry.Image(depth_numpy)

        # 使用默认的相机内参来生成点云
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth, 
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ), 
            depth_scale=depth_scale,
            depth_trunc=max_depth
        )
        
        # 转换点云到体素
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        return voxel_grid
        
    
    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh
        
class LeafInversion(Dataset):
    def __init__(self, root_dir, device):
        self.root_dir = root_dir
        self.trainfile = np.load(os.path.join(self.root_dir,'deform_train.npy'), allow_pickle=True)
        self.renderer = MeshRender(device=device)
        self.device = device

        
    def __len__(self):
        return len(self.trainfile)
    
    def __getitem__(self, index):
        trainfile = self.trainfile[index]
        meshfile = trainfile['mesh_deform']
        mesh = trimesh.load(meshfile, force='mesh')
        # mesh_scale = 0.8
        # vertices = mesh.vertices
        # size = 64
        # level = 2 / size 
        # bbmin = mesh.vertices.min(0)
        # bbmax = mesh.vertices.max(0)
        # center = (bbmin + bbmax) * 0.5
        # scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        # vertices = (vertices - center) 
        # sdf, mesh = mesh2sdf.compute(
        # vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        verts_tensor = torch.from_numpy(mesh.vertices).float()
        faces_tensor = torch.from_numpy(mesh.faces).float()
        mesh = Meshes(verts=verts_tensor.unsqueeze(0), faces=faces_tensor.unsqueeze(0)).to(self.device)
        depth = self.renderer.get_depth(mesh)
        pointcloud = self.renderer.depth_pointcloud(depth)
        
        # create volume
        initial_volume = Volumes(
            features=torch.zeros(1, 5, 32, 32, 32).to(self.device),
            voxel_size=1,
            densities=torch.ones(1, 1, 32, 32, 32).to(self.device),
            volume_translation=torch.tensor([0.0, 0.0, 0.0]),
        )
        # trilinear splatting
        updated_volume = add_pointclouds_to_volumes(
            initial_volumes=initial_volume,
            pointclouds=pointcloud)
        
        # change to open3d point cloud
        #pts = pointcloud.points_packed().detach().cpu().numpy()
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(pts)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.02)
        data = {
            'latent_shape': trainfile['latent_shape'],
          #  'latent_spc': trainfile.item()['latent_spc'],
            'latent_deform' :trainfile['latent_def'],
            'volume': updated_volume.features(),
        }
        return data
    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh


class Inversion_2d(Dataset):
    def __init__(self,  root_dir, device):
        self.root_dir = root_dir
        self.all_mesh = [f for f in os.listdir(root_dir) if f.endswith('.ply')]
        self.all_mask = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.renderer = MeshRender(device=device)
        self.device = device
        # random totation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),])
     #   self.weights_data = np.load(os.path.join(root_dir,'weights.npy'), allow_pickle=True)
    def __len__(self):
        return len(self.all_mesh)
    
    def __getitem__(self, index):
        # mesh_file = self.all_mesh[index]
        # verts, faces = load_ply(os.path.join(self.root_dir, mesh_file))
        # mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0)).to(self.device)
        # mask = self.renderer.renderer_silhouette(mesh)
        # mask_tensor = mask[:,:,:,3]
        # save_mask = mask.detach().squeeze().cpu().numpy()
        # savename = os.path.join(self.root_dir, mesh_file.split('.')[0]+'_mask.png')
        # plt.imsave(savename, save_mask)
        mask = self.all_mask[index]
        mask = cv2.imread(os.path.join(self.root_dir, mask))
        mask = mask /255
        #  torch.transform rotation for mask
        mask = self.transform(mask)
        
       
       #weights = self.weights_data[index]['weight']
        ret_dict = {
        #    'verts':verts,
          #  'mesh': mesh,
            'mask':mask,
            'idx': index}
        return ret_dict

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    cfgpath = 'NPLM/scripts/configs/inversion.yaml'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(cfgpath, 'r'))
    # dataset = InversionDataset(root_dir=CFG['training']['root_dir'],
    #                              n_samples=CFG['training']['n_samples'],
    #                              n_sample_noise=CFG['training']['n_sample_noise'],
    #                              device=device,
    #                              sigma_near=CFG['training']['sigma_near'])
    # dataset = LeafInversion(root_dir='sample_result/shape_deform',device=device)
    dataset = Inversion_2d(root_dir='sample_result/shape_new',device = device)
    dataloader = DataLoader(dataset, batch_size=16,shuffle=False)

    for batch in dataloader:
        print(batch)
  