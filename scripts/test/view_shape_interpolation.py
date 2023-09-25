from model.deepSDF import DeepSDF
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.sdf_dataset import LeafShapeDataset
import yaml
from training.trainer_shape import ShapeTrainer
import math

parser = argparse.ArgumentParser(description='RUN Leaf NPM')
parser.add_argument('--config',type=str, default='/home/yang/projects/parametric-leaf/NPM/scripts/configs/npm.yaml', help='config file')
parser.add_argument('--mode', type=str, default='shape', choices=['shape', 'deformation'], help='training mode')


# setting
device = torch.device('cuda')
args = parser.parse_args()
CFG = yaml.safe_load(open(args.config, 'r'))

if args.mode == "shape":
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

    decoder = decoder.to(device)
    decoder.eval()
    batch = next(iter(trainloader))
    pts = batch['points']
    lat1 = torch.nn.Embedding(len(trainloader), decoder.lat_dim, max_norm = 1.0, sparse=True, device = device).float()
    torch.nn.init.normal_(lat1.weight.data, 0.0, 0.1/math.sqrt(decoder.lat_dim))
    sdf = decoder(pts, lat1)
    
    
    
    
