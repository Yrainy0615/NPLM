import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import Voxel_dataset
from scripts.dataset.sdf_dataset import EncoderDataset
from torch.utils.data import DataLoader
from scripts.model.point_encoder import PCAutoEncoder, CameraNet
from scripts.model.fields import UDFNetwork
import yaml
import wandb
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender
from scripts.model.reconstruction import sdf_from_latent, latent_to_mesh, deform_mesh, create_grid_points_from_bounds
from scripts.model.inference_encoder import ShapeEncoder, PoseEncoder
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, FoVPerspectiveCameras
from matplotlib import pyplot as plt
import imageio  
import pandas as pd
from scripts.dataset.rgbd_dataset import points_to_occ, normalize_verts
import trimesh
import numpy as np
from scripts.registration.leaf_axis_determination import LeafAxisDetermination
from scripts.test.leaf_pose import find_rotation_matrix
import h5py
from scripts.dataset.rgbd_dataset import rgbd_to_voxel
from scripts.test.leaf_pose import visualize_points_and_axes

def crop_and_resize(mask, image,normal, depth, size=(256, 256)):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop_img = image[rmin:rmax+1, cmin:cmax+1]
    crop_mask = mask[rmin:rmax+1, cmin:cmax+1]
    crop_normal = normal[rmin:rmax+1, cmin:cmax+1]
    crop_depth = depth[rmin:rmax+1, cmin:cmax+1]
    new_img = np.zeros((size[0], size[1], 3), dtype=crop_img.dtype)
    new_mask = np.zeros((size[0], size[1]), dtype=crop_mask.dtype)
    new_normal = np.zeros((size[0], size[1], 3), dtype=crop_normal.dtype)
    new_depth = np.zeros((size[0], size[1]), dtype=crop_depth.dtype)
    center_row = (size[0] - (rmax - rmin + 1)) // 2
    center_col = (size[1] - (cmax - cmin + 1)) // 2

    # 将裁剪后的图像和 mask 放置在新画布中央
    new_img[center_row:center_row+crop_img.shape[0], center_col:center_col+crop_img.shape[1]] = crop_img[:,:,:3]
    new_mask[center_row:center_row+crop_mask.shape[0], center_col:center_col+crop_mask.shape[1]] = crop_mask[:,:,0]
    new_normal[center_row:center_row+crop_normal.shape[0], center_col:center_col+crop_normal.shape[1]] = crop_normal
    new_depth[center_row:center_row+crop_depth.shape[0], center_col:center_col+crop_depth.shape[1]] = crop_depth
    
    return new_img, new_mask, new_normal, new_depth

class Predictor(object):
    def __init__(self, encoder_shape, encoder_pose,encoder_2d, 
                 cameranet, trainloader, 
                 latent_shape, latent_deform,
                 decoder_shape, decoder_deform,
                 generator,
                 cfg, device):
        self.encoder_shape = encoder_shape
        self.encoder_pose = encoder_pose
        self.encoder_2d = encoder_2d
        self.cameranet = cameranet
        self.trainloader = trainloader
        self.decoder_shape = decoder_shape
        self.decoder_deform = decoder_deform
        self.device = device
        self.generator = generator
        self.renderer = MeshRender(device=device)
        self.cfg = cfg['training']
        self.latent_shape = latent_shape
        self.latent_deform = latent_deform
        R, t = look_at_view_transform(2,45, 0)
        self.deform_camera = FoVPerspectiveCameras(device=self.device, R=R, T=t)
    
    def predict(self, data, lat_shape_init=None):
            # initialization
        occupancy_grid = torch.from_numpy(data['occupancy_grid']).to(self.device).unsqueeze(0).float()
        points = torch.from_numpy(data['points']).to(self.device).unsqueeze(0).float()
        latent_shape_pred = self.encoder_shape(occupancy_grid)
        if lat_shape_init is not None:
            latent_shape_pred = lat_shape_init
        else:
            latent_shape_pred =self.encoder_shape(occupancy_grid)
        latent_pose_pred = self.encoder_pose(occupancy_grid)
        canonical_mesh = latent_to_mesh(self.decoder_shape, latent_shape_pred, self.device)
        deformed_mesh = deform_mesh(canonical_mesh, self.decoder_deform, latent_pose_pred)

        # canonical_mesh.export('{}_canonical.obj'.format(mesh_name))
        # deformed_mesh.export('{}.obj'.format(mesh_name))
        
        # optimization
        latent_shape_optimized, latent_deform_optimized, canonical_imgs, deform_imgs, = self.optim_latent(
                                    latent_shape_pred.detach().requires_grad_(), 
                                    latent_pose_pred.detach().requires_grad_(), points)
        
        # final output
        canonical_mesh_optimized = latent_to_mesh(self.decoder_shape, latent_shape_optimized, self.device)
        deformed_mesh_optimized = deform_mesh(canonical_mesh_optimized, self.decoder_deform, latent_deform_optimized)
        # canonical_mesh_optimized.export('{}_canonical_optimized.obj'.format(mesh_name))
        # deformed_mesh_optimized.export('{}_optimized.obj'.format(mesh_name))
        return canonical_mesh, deformed_mesh, canonical_mesh_optimized, deformed_mesh_optimized, canonical_imgs, deform_imgs
    
    def optim_latent(self, latent_shape_init, latent_deform_init, points):
        optimizer_shape = optim.Adam([latent_shape_init], lr=1e-3)
        optimizer_deform = optim.Adam([latent_deform_init], lr=1e-3)
        img_nps = []
        deform_nps = []
        for i in range(100):
            optimizer_shape.zero_grad()
            optimizer_deform.zero_grad()
            mesh = latent_to_mesh(self.decoder_shape, latent_shape_init, self.device)
            verts = mesh.vertices
            verts = normalize_verts(verts)
            xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
            delta_verts = self.decoder_deform(torch.from_numpy(verts).float().to(device), latent_deform_init.squeeze(0).repeat(mesh.vertices.shape[0], 1))
            """
            Differentiable Rendering back-propagating to mesh vertices
            """
            # generate a texture for the mesh in green 
            texture  = torch.ones_like(torch.from_numpy(mesh.vertices)).unsqueeze(0)
            texture[..., 0] = 0.0
            texture[..., 1] = 1
            texture[..., 2] = 0.0
            # create a Meshes object for the textured mesh
            textures = TexturesVertex(verts_features=texture.to(device).float())
            canonical_img = self.renderer.render_rgb(Meshes(verts=[xyz_upstream.squeeze()], faces=[torch.tensor(mesh.faces).squeeze().to(device)], textures=textures))
            img_np = canonical_img[:,:,:,:3].detach().squeeze().cpu().numpy().astype(np.uint8)
            loss_chamfer = chamfer_distance((xyz_upstream.unsqueeze(0)+delta_verts), points)
            deform_img =  self.renderer.renderer(Meshes(verts=[xyz_upstream.squeeze()+ delta_verts.squeeze()], faces=[torch.tensor(mesh.faces).squeeze().to(device)], textures=textures), camera = self.deform_camera)
            deform_img_np = deform_img[:,:,:,:3].detach().squeeze().cpu().numpy().astype(np.uint8)
            loss =loss_chamfer[0]#+torch.norm(latent_source, dim=-1)**2 # +loss_chamfer[0]
            # regularizer
            lat_reg_shape = torch.norm(latent_shape_init, dim=-1) ** 2
            lat_reg_deform = torch.norm(latent_deform_init, dim=-1) ** 2

            # print losses
            print('shape iter:{}  loss_chamfer: {}'.format(i,loss_chamfer[0]))
            loss_all = 10*loss + lat_reg_shape + lat_reg_deform
            loss_all.backward()
            optimizer_deform.step()
            # now store upstream gradients
            dL_dx_i = xyz_upstream.grad

            # use vertices to compute full backward pass
            optimizer_shape.zero_grad()
            xyz = torch.tensor(verts, requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
            #first compute normals 
            pred_sdf = decoder_shape(xyz, latent_shape_init.squeeze(0).repeat(xyz.shape[0], 1))
            loss_normals = torch.sum(pred_sdf)
            loss_normals.backward(retain_graph = True)
            # normalization to take into account for the fact sdf is not perfect...
            normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
            # now assemble inflow derivative
            optimizer_shape.zero_grad()
            dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
            # refer to Equation (4) in the main paper
            loss_backward = torch.sum(dL_ds_i * pred_sdf)
            loss_backward.backward()
            # and update params
            optimizer_shape.step()
            img_nps.append(img_np)
            deform_nps.append(deform_img_np)
        return latent_shape_init, latent_deform_init, img_nps, deform_nps

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='inference', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--save_mesh', action='store_true', help='save mesh')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    if args.use_wandb:
        wandb.init(project='NPLM', name =args.wandb)
        wandb.config.update(CFG)
    
    # dataset
    # trainset = EncoderDataset(root_dir='results/viz_space')
    # trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    trainloader = None
    # networl initialization
    checkpoint_encoder = torch.load('checkpoints/inference/encoder_soybean.tar')
    encoder_shape = ShapeEncoder()
    encoder_shape.load_state_dict(checkpoint_encoder['encoder_shape_state_dict'])
    encoder_shape.to(device)
    encoder_shape.eval()
    encoder_pose = PoseEncoder()
    encoder_pose.load_state_dict(checkpoint_encoder['encoder_pose_state_dict'])
    encoder_pose.to(device)
    encoder_pose.eval()
    
    encoder_2d = ViTModel.from_pretrained('facebook/dino-vitb16')
    encoder_2d.to(device)
    encoder_2d.eval()
    
    cameranet = CameraNet(feature_dim=768, hidden_dim=512)
    cameranet.to(device)
    cameranet.eval()
    # cameranet.load_state_dict(checkpoint_infer['cameranet_state_dict'])
    
    # load pretrained decoder 
        # shape decoder initialization
    decoder_shape = UDFNetwork(d_in= CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                        d_out=CFG['shape_decoder']['decoder_out_dim'],
                        n_layers=CFG['shape_decoder']['decoder_nlayers'],
                        udf_type='sdf',
                        d_in_spatial=3,)
    checkpoint = torch.load('checkpoints/3dShape/latest_3d_0126.tar')
    lat_idx_all = checkpoint['latent_idx_state_dict']['weight']
    decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    
    # deform decoder initialization
    decoder_deform = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False,
                         use_mapping=CFG['deform_decoder']['use_mapping'])
    checkpoint_deform = torch.load('checkpoints/deform/deform_soybean.tar')
    lat_deform_all = checkpoint_deform['latent_deform_state_dict']['weight']
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)
    
    # load generator
    generator = Generator(resolution=256)
    generator.to(device)
    generator.eval()
    # generator.load_state_dict(checkpoint_infer['generator_state_dict'])
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    resolution = 128
    grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
    # predict
    predictor = Predictor(encoder_shape, encoder_pose, encoder_2d, cameranet, trainloader, lat_idx_all, lat_deform_all, decoder_shape, decoder_deform, generator, CFG, device)
    data_source = 'soybean'
    z_axis_canonical = np.array([0, 0, 1])
    x_axis_canonical = np.array([1, 0, 0])
    y_axis_canonical = np.array([0, 1, 0])
    if data_source == 'dataset':
        for i, batch in enumerate(trainloader):
            canonical_mesh, deformed_mesh, canonical_mesh_optimized, deformed_mesh_optimized, canonical_img, deform_img = predictor.predict(batch)
    
    if data_source == 'raw':
        data_path = 'LeafSurfaceReconstruction/data/sugarbeet'
        points = []
        w_axis_canonical = np.array([0, 1, 0])
        l_axis_canonical = np.array([1, 0, 0])
        h_axis_canonical = np.array([0, 0, 1])
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                file_path = os.path.join(data_path, file)
                print(file_path)
                data = pd.read_csv(file_path, names=("x", "y", "z")).values
                points.append(data)
        for i, point_cloud in enumerate(points):  
            occupancy_grid = points_to_occ(point_cloud)
            point_cloud = normalize_verts(point_cloud)
            # move center to origin
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            
            # pca axis determination
            leafAxisDetermination = LeafAxisDetermination(point_cloud)
            w_axis, l_axis, h_axis, new_points = leafAxisDetermination.process()
            R_w2c = find_rotation_matrix(np.array([l_axis_canonical, w_axis_canonical, h_axis_canonical]).T, np.array([l_axis, w_axis, h_axis]).T)
            R_c2w  = np.linalg.inv(R_w2c)
            point_cloud_canonical= np.dot(new_points, R_w2c.T)
            
            # predict & inference
            data = {'occupancy_grid': occupancy_grid, 'points': point_cloud_canonical}
            canonical_mesh, deformed_mesh, canonical_mesh_optimized, deformed_mesh_optimized, canonical_img, deform_img = predictor.predict(data)
            # save results
            canonical_mesh.export('canonical_mesh_{}.obj'.format(i))
            deformed_mesh.export('deformed_mesh_{}.obj'.format(i))
            
            canonical_mesh_optimized.export('canonical_mesh_optimized_{}.obj'.format(i))
            deformed_mesh_optimized.export('deformed_mesh_optimized_{}.obj'.format(i))
            deformed_mesh_rot = trimesh.Trimesh(deformed_mesh.vertices.dot(R_c2w.T), deformed_mesh.faces, process=False)
            deformed_mesh_rot.export('deformed_mesh_rot_{}.obj'.format(i))
            origin= trimesh.points.PointCloud(point_cloud)
            origin_canonical = trimesh.points.PointCloud(point_cloud_canonical)
            origin.export(f'origin_pt{i}.ply')
            origin_canonical.export(f'origin_canonical_pt{i}.ply')
            imageio.mimsave('canonical_img_{}.gif'.format(i), canonical_img, 'GIF',fps=5)
            imageio.mimsave('deform_img_{}.gif'.format(i), deform_img, 'GIF', fps=5)
        
    if data_source == 'denseleaf':

        test_data= 'views/0.hdf5'
        latent_shape_init = lat_idx_all[100]
        with h5py.File(test_data, 'r') as f:
        # print items in the file
            color = np.array(f['colors'])
            depth = np.array(f['depth'])
            normal = np.array(f['normals'])
            categort_id_segmaps = np.array(f['category_id_segmaps'])
            instance_segmaps = np.array(f['instance_segmaps'])
            instance_attributes = np.array(f['instance_attribute_maps'])
        
        # get instance mask from instance_segmaps by different colors
        unique_values = np.unique(instance_segmaps)
        mask_ori = {}
        # Iterate over each unique value (excluding 0 if it's the background)
        for value in unique_values:
            if value != 0:  # Skip the background
                # Create a mask for the current value
                mask_ori[value] = (instance_segmaps == value)
        colors = []
        normals = []
        depths = []
        masks = []
        locations = []
        # get corresponding color for each instance mask
        for key in mask_ori:
            mask = mask_ori[key]
            mask = mask.astype(np.uint8)
            mask = mask[:,:, np.newaxis ]
            # get the color for the mask
            color_single  = color * mask
            normal_single = normal * mask
            depth_single = depth * mask.squeeze()
            color_resized, mask_resized, normal_resized, depth_resized = crop_and_resize(mask, color_single, normal_single, depth_single)
            colors.append(color_resized)
            normals.append(normal_resized)
            depths.append(depth_resized)
            # save location is 3*1 array , as (x,y,z) z is the depth
            location =np.ones([3])
            location[0], location[1] = np.mean(np.where(mask), axis=1)[0], np.mean(np.where(mask), axis=1)[1]
            location[2] = depth_single[int(location[0]), int(location[1])]
            locations.append(np.mean(np.where(mask), axis=1))
            

        for i in range(len(colors)):
            rgb = colors[i]
            depth = depths[i]
            normal = normals[i]
            occupancy_grid, point_cloud = rgbd_to_voxel(rgb=rgb, depth=depth,grid_points=grid_points)
            leafAxisDetermination = LeafAxisDetermination(point_cloud)
            y_axis, x_axis, z_axis, new_points = leafAxisDetermination.process()
            R_w2c = find_rotation_matrix(np.array([x_axis_canonical,  y_axis_canonical,z_axis_canonical]), np.array([x_axis,  y_axis,z_axis]))
            R_c2w  = np.linalg.inv(R_w2c)
            point_cloud_canonical= np.matmul(new_points, R_w2c.T)

            # predict & inference
            data = {'occupancy_grid': occupancy_grid, 'points': point_cloud_canonical}
            canonical_mesh, deformed_mesh, canonical_mesh_optimized, deformed_mesh_optimized, canonical_img, deform_img = predictor.predict(data, lat_shape_init=latent_shape_init)
            # visualize_points_and_axes(point_cloud_canonical, canonical_mesh.vertices,origin=np.mean(canonical_mesh.vertices, axis=0), x_axis=x_axis_canonical, y_axis=y_axis_canonical, z_axis=z_axis_canonical)

            # save results
            # canonical_mesh.export('canonical_mesh_{}.obj'.format(i))
            # deformed_mesh.export('deformed_mesh_{}.obj'.format(i))
            
            # canonical_mesh_optimized.export('canonical_mesh_optimized_{}.obj'.format(i))
            deformed_mesh_optimized.vertices = normalize_verts(deformed_mesh_optimized.vertices)
            deformed_mesh_optimized.export('deformed_canonical_{}.obj'.format(i))
            vertice_final = np.matmul(deformed_mesh_optimized.vertices, R_c2w) + np.mean(point_cloud, axis=0)
            deformed_mesh_rot = trimesh.Trimesh(vertice_final, deformed_mesh.faces, process=False)
            deformed_mesh_rot.export('deformed_ori_{}.obj'.format(i))
            origin= trimesh.points.PointCloud(point_cloud)
            origin_canonical = trimesh.points.PointCloud(point_cloud_canonical)
            origin.export(f'origin_pt{i}.ply')
            origin_canonical.export(f'origin_canonical_pt{i}.ply')
            imageio.mimsave('canonical_img_{}.gif'.format(i), canonical_img, 'GIF',fps=5)
            imageio.mimsave('deform_img_{}.gif'.format(i), deform_img, 'GIF', fps=5)
            
            pass
        
    if data_source == 'soybean':
        root = 'dataset/soybean'
        model_dir = os.path.join(root, 'model')
        annotation_dir = os.path.join(root, 'annotation')
        all_plants  = [f for f in os.listdir(annotation_dir) ]
        instance_dir = '/home/yang/projects/parametric-leaf/dataset/soybean/annotation/20180619_HN48/Annotations'
        for i,file in enumerate(os.listdir(instance_dir)):
            plant_ponts = []
            if file.endswith(".txt") and 'leaf' in file:
                file_path = os.path.join(instance_dir, file)
                print(file_path)
                data = pd.read_csv(file_path, sep=' ', header=None,
                 names=['x', 'y', 'z', 'r', 'g', 'b'],
                 dtype={'x': float, 'y': float, 'z': float, 'r': int, 'g': int, 'b': int})

                points = data[['x', 'y', 'z']].values
                colors = data[['r', 'g', 'b']].values
                # if nan in colors, replace with 0
                colors = np.nan_to_num(colors)
                point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
                point_cloud  = normalize_verts(point_cloud.vertices)
                # random select 1000 points
                point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 1000, replace=False), :]
                occupancy_grid = points_to_occ(point_cloud)
                
                # export origin shape
                origin= trimesh.points.PointCloud(point_cloud)
                # origin.export(f'origin_pt{i}.ply')


                # pca axis determination
                leafAxisDetermination = LeafAxisDetermination(point_cloud)
                y_axis, x_axis, z_axis, new_points = leafAxisDetermination.process()
                R_w2c = find_rotation_matrix(np.array([x_axis_canonical,  y_axis_canonical,z_axis_canonical]), np.array([x_axis,  y_axis,z_axis]))
                R_c2w  = np.linalg.inv(R_w2c)
                point_cloud_canonical= np.matmul(new_points, R_w2c.T)
                # visualize_points_and_axes(point_cloud_canonical,origin=np.mean(point_cloud_canonical, axis=0), x_axis=x_axis_canonical, y_axis=y_axis_canonical, z_axis=z_axis_canonical)
                origin_canonical = trimesh.points.PointCloud(point_cloud_canonical)
                origin_canonical.export(f'origin_canonical_pt{i}.ply')


                # predict & inference
                data = {'occupancy_grid': occupancy_grid, 'points': point_cloud_canonical}
                canonical_mesh, deformed_mesh, canonical_mesh_optimized, deformed_mesh_optimized, canonical_img, deform_img = predictor.predict(data, lat_shape_init=None)
                deformed_mesh_optimized.vertices = normalize_verts(deformed_mesh_optimized.vertices)
                deformed_mesh_optimized.export('deformed_canonical_{}.obj'.format(i))
                vertice_final = np.matmul(deformed_mesh_optimized.vertices, R_c2w) 
                deformed_mesh_rot = trimesh.Trimesh(vertice_final, deformed_mesh.faces, process=False)
                # deformed_mesh_rot.export('deformed_ori_{}.obj'.format(i))
