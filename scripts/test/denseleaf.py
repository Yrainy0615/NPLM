import h5py
import numpy as np
from matplotlib import pyplot as plt
import cv2
import open3d as o3d
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import rgbd_to_voxel
import torch
from scripts.model.reconstruction import latent_to_mesh, deform_mesh, create_grid_points_from_bounds
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender
from scripts.model.fields import UDFNetwork
from scripts.model.point_encoder import PCAutoEncoder, CameraNet
import argparse
import os
import yaml
from pytorch3d.structures import Meshes
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from transformers import ViTImageProcessor

def canonical_uv_mapping(canonical_mask, texture):
    
    pass

def predict_single_leaf(points, rgb, encoder_3d, encoder_2d, processor, CameraNet, 
                        decoder_shape, decoder_deform, renderer,device):
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    rgb_tensor = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    
    # canonical shape inference
    latent_shape_pred, latent_deform_pred = encoder_3d(points_tensor.unsqueeze(0).permute(0,2,1))
    canonical_mesh = latent_to_mesh(decoder_shape, latent_shape_pred, device)
    canonical_mesh_tensor = Meshes(verts=torch.tensor(canonical_mesh.vertices).float().unsqueeze(0), faces=torch.tensor(canonical_mesh.faces).float().unsqueeze(0))    
    canonical_mask = renderer.get_mask_tensor(canonical_mesh_tensor.to(device))
    mask_pil = to_pil_image(canonical_mask)
    
    # texture inference 
    input_mask = processor(mask_pil.convert('RGB'), return_tensors="pt").to(device)
    outputs_mask = encoder_2d(input_mask['pixel_values'])
    feat_mask = outputs_mask.pooler_output
    input_rgb = processor(rgb_tensor, return_tensors="pt").to(device)
    outputs_rgb = encoder_2d(input_rgb['pixel_values'])
    feat_rgb = outputs_rgb.pooler_output
    camera_pose = CameraNet(feat_rgb)
    texture = generator(torch.cat([feat_rgb, feat_mask], dim=1))
    texture_pil = to_pil_image(texture.squeeze())
    canonical_mesh_textured = canonical_uv_mapping(canonical_mask, texture)
    
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
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference on single leaf')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    
    # networl initialization
    checkpoint_infer = torch.load('checkpoints/inference/latest.tar')
    encoder_3d = PCAutoEncoder(point_dim=3)
    encoder_3d.to(device)
    encoder_3d.eval()
    encoder_3d.load_state_dict(checkpoint_infer['encoder3d_state_dict'])
    
    encoder_2d = ViTModel.from_pretrained('facebook/dino-vitb16')
    encoder_2d.to(device)
    encoder_2d.eval()
    processor =  ViTImageProcessor.from_pretrained('facebook/dino-vitb16')

    
    cameranet = CameraNet(feature_dim=768, hidden_dim=512)
    cameranet.to(device)
    cameranet.eval()
    cameranet.load_state_dict(checkpoint_infer['cameranet_state_dict'])
    
    # load pretrained decoder 
        # shape decoder initialization
    decoder_shape = UDFNetwork(d_in= CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                        d_out=CFG['shape_decoder']['decoder_out_dim'],
                        n_layers=CFG['shape_decoder']['decoder_nlayers'],
                        udf_type='sdf',
                        d_in_spatial=3,)
    checkpoint = torch.load('checkpoints/3dShape/latest.tar')
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
    checkpoint_deform = torch.load('checkpoints/deform/exp-deform-dis__10000.tar')
    lat_deform_all = checkpoint_deform['latent_deform_state_dict']['weight']
    decoder_deform.load_state_dict(checkpoint_deform['decoder_state_dict'])
    decoder_deform.eval()
    decoder_deform.to(device)
    
    # load generator
    generator = Generator(resolution=256)
    generator.to(device)
    generator.eval()
    checkpint_texture = torch.load('checkpoints/Texture/latest.tar')
    generator.load_state_dict(checkpint_texture['generator_state_dict'])
    
    renderer = MeshRender(device=device)
    predictor = Predictor(encoder_3d, encoder_2d, processor, cameranet, decoder_shape, decoder_deform, generator, renderer, device)
    # create grid points 
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    resolution = 128
    grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
    
    test_data= 'views/0.hdf5'
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
        locations.append(np.mean(np.where(mask), axis=1))
        

    for i in range(len(colors)):
        rgb = colors[i]
        depth = depths[i]
        normal = normals[i]
        mask = mask[i]
        occupancy_grid, points = rgbd_to_voxel(rgb=rgb, depth=depth,grid_points=grid_points)
        # predict
        

    pass 