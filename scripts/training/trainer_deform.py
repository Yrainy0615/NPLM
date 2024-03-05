import torch
import torch.optim as optim
import math
from glob import glob
import sys
sys.path.append('NPLM')
from scripts.model.loss_functions import compute_loss, compute_loss_corresp_forward
import os
import numpy as np
import wandb
import argparse

import warnings
warnings.filterwarnings("ignore")
import yaml
from scripts.dataset.sdf_dataset import  LeafDeformDataset
from scripts.model.fields import UDFNetwork
from torch.utils.data import DataLoader
import random
from scripts.model.reconstruction import latent_to_mesh, deform_mesh, save_mesh_image_with_camera
import trimesh

CUDA_LAUNCH_BLOCKING=1

class DeformTrainer(object):
    def __init__(self, decoder,cfg, latent_shape,
                 trainset,trainloader,device, args):
        self.decoder = decoder
        self.cfg = cfg['training']
        self.trainset = trainset
        self.args = args
        self.latent_shape = latent_shape
        self.latent_shape.requires_grad = False
        self.latent_deform = torch.nn.Embedding(812, 128, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(self.latent_deform.weight.data, 0.0, 0.01)
        print('deform latent loaded with dims:{}'.format(self.latent_deform.weight.data.shape))
        
        self.trainloader = trainloader
        self.device = device
        self.optimizer_decoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.optimizer_latent = optim.SparseAdam(params= list(self.latent_deform.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']
        self.checkpoint_path = self.cfg['save_path']
        # add a scalar parameter and need to optimize it
        self.phi = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
        self.optimizer_phi = optim.Adam(params=[self.phi], lr=0.01)
        

        
    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints)==0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0
        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        if 'ckpt' in self.cfg and self.cfg['ckpt'] is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['ckpt'])
        else:
            print('LOADING', checkpoints[-1])
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer_encoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
        self.optimizer_lat.load_state_dict(checkpoint['optimizer_latent_state_dict'])
        #self.optimizer_lat_val.load_state_dict(checkpoint['optimizer_lat_val_state_dict'])
        self.latent_codes.load_state_dict(checkpoint['latent_state_dict'])
        #self.latent_codes_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
        epoch = checkpoint['epoch']
        for param_group in self.optimizer_decoder.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr']))
            param_group['lr'] = self.cfg['lr']
        for param_group in self.optimizer_latent.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr_lat']))
            param_group['lr'] = self.cfg['lr_lat']
        if self.cfg['lr_decay_interval'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_decoder.param_groups:
                param_group["lr"] = self.lr * self.cfg['lr_decay_factor']**decay_steps
        if self.cfg['lr_decay_interval_lat'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_latent.param_groups:
                param_group["lr"] = lr
        return epoch
    
    def reduce_lr(self, epoch):
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_decoder.param_groups:
                param_group["lr"] = lr

        if epoch > 1000 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_latent.param_groups:
                param_group["lr"] = lr
            # for param_group in self.optimizer_lat_val.param_groups:
            #     param_group["lr"] = lr 
        
    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest_dis_wo_shape.tar'
            torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),  
                        'latent_deform_state_dict': self.latent_deform.state_dict(),
                        'phi': self.phi,},
                       path)
       
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                        'latent_deform_state_dict': self.latent_deform.state_dict(),
                        'phi': self.phi,
                       },
                       path)
       
    def train_step(self, batch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        self.optimizer_phi.zero_grad()
        loss_dict, pred_posed, posed = compute_loss_corresp_forward(batch,
                                decoder=self.decoder, device=self.device, latent_shape=self.latent_shape,
                                latent_deform=self.latent_deform,phi=self.phi,cfg = self.cfg)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.latent_deform.parameters(), max_norm=self.cfg['grad_clip_lat'])
        self.optimizer_decoder.step()
        self.optimizer_latent.step()
        self.optimizer_phi.step()

        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}

        loss_dict.update({'loss': loss_total.item()})
           

        return loss_dict    , pred_posed, posed
    
    

    def train(self, epochs):
        loss = 0
        # start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict, pred_posed, posed = self.train_step(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                if self.args.use_wandb:
                    wandb.log(loss_values)
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0 and epoch >0:
                self.save_checkpoint(epoch,self.cfg['save_name'])
                # save points
            self.save_checkpoint(epoch, save_name='latest')
            # points = pred_posed[0].squeeze().detach().cpu().numpy()
            # gt = posed[0].squeeze().detach().cpu().numpy()
            # # save point cloud
            # clouds = trimesh.points.PointCloud(points)
            # clouds.export(os.path.join(self.cfg['save_path'],'deform_points_{}.ply'.format(epoch)))
            # gt = trimesh.points.PointCloud(gt)
            # gt.export(os.path.join(self.cfg['save_path'],'gt_points_{}.ply'.format(epoch)))
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
            print('phi:', self.phi.item())
                
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='deform', help='run name of wandb')
    parser.add_argument('--output', type=str, default='deform', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    # setting

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_deform.yaml'
    CFG = yaml.safe_load(open(config, 'r'))
    if args.use_wandb:
        wandb.init(project='NPLM', name =args.wandb)
    
    # dataset & dataloader
    trainset = LeafDeformDataset(
                        n_supervision_points_face=CFG['training']['npoints_decoder'])
    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=2)

    # shape latent
    checkpoint_shape = torch.load('checkpoints/shape/Shape_final__12000.tar')
    latent_shape = checkpoint_shape['latent_idx_state_dict']['weight']
    latent_shape.to(device)
    
    decoder = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False,
                         use_mapping=CFG['deform_decoder']['use_mapping'])
   

    decoder = decoder.to(device)
    
    # trainer initialization
    trainer = DeformTrainer(
                            decoder=decoder,
                            cfg=CFG, 
                            latent_shape=latent_shape,
                            trainset=trainset,trainloader=trainloader, 
                            device=device,
                            args=args)
    # train
    trainer.train(10001)
