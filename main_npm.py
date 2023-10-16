from scripts.model.deepSDF import DeepSDF, DeformationNetwork
import argparse
import torch
from torch.utils.data import DataLoader
from scripts.dataset.sdf_dataset import LeafShapeDataset, LeafDeformDataset, LeafColorDataset
import yaml
from scripts.model.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from scripts.training.trainer_shape import ShapeTrainer
from scripts.training.trainer_deform import DeformTrainer
from scripts.training.trainer_color import ColorTrainer
from scripts.model.discriminator import Discriminator
import math
from skimage.measure import marching_cubes
import trimesh
from matplotlib import pyplot as plt
from scripts.model.reconstruction import mesh_from_logits, get_logits, create_grid_points_from_bounds
import numpy as np
import pyvista as pv
import os
import wandb



parser = argparse.ArgumentParser(description='RUN Leaf NPM')
parser.add_argument('--mode', type=str, default='color', choices=['shape', 'deformation','viz_shape', 'color'], help='training mode')
parser.add_argument('--gpu', type=int, default=7, help='gpu index')
parser.add_argument('--wandb', type=str, default='*', help='run name of wandb')
parser.add_argument('--output', type=str, default='shape', help='output directory')
# setting

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.mode == "shape":
        config = 'NPLM/scripts/configs/npm.yaml'
        CFG = yaml.safe_load(open(config, 'r'))
        wandb.init(project='NPLM', name =args.wandb)
        trainset = LeafShapeDataset(mode='train',
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            root_dir=CFG['training']['root_dir'])
        trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=False, num_workers=2)
        decoder = DeepSDF(
            lat_dim=CFG['decoder']['decoder_lat_dim'],
            hidden_dim=CFG['decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
            )

        decoder = decoder.to(device)
        trainer = ShapeTrainer(decoder, CFG, trainset,trainloader, device)
        trainer.train(30001)
    
if args.mode == "deformation":
        config = 'NPLM/scripts/configs/npm_def.yaml'
        CFG = yaml.safe_load(open(config, 'r'))
        wandb.init(project='NPLM', name =args.wandb)
        trainset = LeafDeformDataset(mode='train',
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            root_dir=CFG['training']['root_dir'])
        trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=False, num_workers=2)

        decoder_shape = DeepSDF(
            lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        )
        decoder_shape.mlp_pos = None
        decoder = DeepSDF(lat_dim=512+200,
                      hidden_dim=1024,
                      geometric_init=False,
                      out_dim=3,
                      input_dim=3)
        decoder.lat_dim_expr = 200

        decoder = decoder.to(device)
        decoder_shape = decoder_shape.to(device)
        trainer = DeformTrainer(
                                decoder=decoder,
                                decoder_shape=decoder_shape,
                                cfg=CFG, 
                                trainset=trainset,trainloader=trainloader, 
                                device=device)
        trainer.train(30001)

if args.mode == "color":
        config = 'NPLM/scripts/configs/npm_color.yaml'
        CFG = yaml.safe_load(open(config, 'r'))

        wandb.init(project='NPLM', name =args.wandb)
        trainset = LeafColorDataset(mode='train',
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            root_dir=CFG['training']['root_dir_color'])
        trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=False, num_workers=2)

        decoder_shape = DeepSDF(
            lat_dim=CFG['shape_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['shape_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        )
        checkpoint_shape = 'checkpoints/shape_epoch_25000.tar'
        ckpt_shape = torch.load(checkpoint_shape)
        
        decoder_shape.mlp_pos = None
        decoder_shape.load_state_dict(ckpt_shape['decoder_state_dict'])
        decoder = DeepSDF(lat_dim=200,
                      hidden_dim=1024,
                      geometric_init=False,
                      out_dim=3,
                      input_dim=3)
        decoder.lat_dim_expr = 200
        discriminator = Discriminator(channel=3, size=64)
        discriminator = discriminator.to(device)
        decoder = decoder.to(device)
        decoder_shape = decoder_shape.to(device)
        trainer = ColorTrainer(
                                decoder=decoder,
                                decoder_shape=decoder_shape,
                                cfg=CFG, 
                                trainset=trainset,trainloader=trainloader, 
                                discriminator=discriminator,
                                device=device)
        trainer.train(30001)
    
    
    
    
if args.mode == "viz_shape":
        trainset = LeafShapeDataset(mode='train',
                        n_supervision_points_face=CFG['training']['npoints_decoder'],
                        n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                        batch_size=CFG['training']['batch_size'],
                        sigma_near=CFG['training']['sigma_near'],
                        root_dir=CFG['training']['root_dir'])
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
        decoder = DeepSDF(
            lat_dim=CFG['decoder']['decoder_lat_dim'],
            hidden_dim=CFG['decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
            )
        def generate_random_latent(device):
                return torch.normal(mean=0, std=0.1/math.sqrt(512), size=(512,)).to(device)

        checkpoint = torch.load('checkpoints/checkpoint_epoch_25000.tar')
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.eval()
        step =0
        out_dir =os.path.join('sample_result', args.output)
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        mini = [-.95, -.95, -.95]
        maxi = [0.95, 0.95, 0.95]
        grid_points = create_grid_points_from_bounds(mini, maxi, 256)
        grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
        grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
        decoder = decoder.to(device)
        #lat_mean = torch.from_numpy(np.load('dataset/npm_lat_mean.npy'))
        lat_idx_all = checkpoint['latent_idx_state_dict']['weight']
        # lat_spc_all = checkpoint['latent_spc_state_dict']['weight']
        # lat_combined = torch.cat((lat_idx_all, lat_spc_all), dim=1)
        # lat_mean = torch.mean(lat_all,dim=0).to(device)
        # lat_std = torch.std(lat_all, dim=0).to(device)        
        # alphs = torch.linspace(0,1, steps=10).cuda()
        # interpolate = (1-alphs[:, None]) * lat_all[4] + alphs[:, None] * lat_all[5]
        # lat_std = torch.from_numpy(np.load('dataset/npm_lat_std.npy'))

        for j in range(lat_idx_all.shape[0]):
                lat_rep = lat_idx_all[j]
                logits = get_logits(decoder, lat_rep, grid_points=grid_points,nbatch_points=2000)
                print('starting mcubes')
                mesh = mesh_from_logits(logits, mini, maxi,256)
                print('done mcubes')
                # pv.start_xvfb()
                # pl = pv.Plotter(off_screen=True)
                # pl.add_mesh(mesh)
                # pl.reset_camera()
                # pl.camera.position = (0, 3, 0)
                # pl.camera.zoom(1.4)
                # pl.set_viewup((0, 1, 0))
                # pl.camera.view_plane_normal = (-0, -0, 1)
                # pl.show(screenshot=out_dir + '/step_{:04d}.png'.format(step), auto_close=True)
                mesh.export(out_dir + '/mesh_{:04d}.ply'.format(step))
                #print(pl.camera)
                step += 1

        
                
