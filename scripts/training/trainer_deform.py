import torch
import torch.optim as optim
import math
from glob import glob
from scripts.model.loss_functions import compute_loss, compute_loss_corresp_forward
import os
import numpy as np
import wandb
import argparse

import warnings
warnings.filterwarnings("ignore")
import yaml
from scripts.dataset.sdf_dataset import LeafShapeDataset, Leaf2DShapeDataset, LeafDeformDataset
from scripts.model.fields import SDFNetwork ,UDFNetwork
from torch.utils.data import DataLoader
import random
from scripts.model.reconstruction import latent_to_mesh, deform_mesh, save_mesh_image_with_camera

class DeformTrainer(object):
    def __init__(self, decoder, decoder_shape,cfg, 
                 trainset,trainloader,device):
        self.decoder = decoder
        self.decoder_shape = decoder_shape
        self.cfg = cfg['training']
        # latent initializaiton
        self.trainset = trainset
        self.latent_shape = torch.nn.Embedding(7, decoder_shape.lat_dim, max_norm = 1.0, sparse=True, device = device).float()
        self.latent_shape.requires_grad_  = False
        self.latent_deform = torch.nn.Embedding(len(trainset), 200, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(self.latent_deform.weight.data, 0.0, 0.01)
        print(self.latent_deform.weight.data.shape)
        print(self.latent_deform.weight.data.norm(dim=-1).mean())
        self.init_shape_state(self.cfg['shape_path'])
        
        self.trainloader = trainloader
        self.device = device
        self.optimizer_decoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.optimizer_latent = optim.SparseAdam(params= list(self.latent_deform.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']

        self.checkpoint_path = self.cfg['save_path']
        
        
    def init_shape_state(self, path):
        checkpoint = torch.load(path)
        self.decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
        self.latent_shape.load_state_dict(checkpoint['latent_idx_state_dict'])
        print('Train shape space loaded with dims: ')
        print(self.latent_shape.weight.shape)
  
        print('Loaded checkpoint from: {}'.format(path))
        
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
            # for param_group in self.optimizer_lat_val.param_groups:
            #     param_group["lr"] = lr
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
        
    def save_checkpoint(self, epoch, save_name):
        path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                      #  'optimizer_lat_val_state_dict': self.optimizer_lat_val.state_dict(),
                        'latent_deform_state_dict': self.latent_deform.state_dict(),
                  
                       # 'latent_codes_val_state_dict': self.latent_codes_val.state_dict()
                       },
                       path)
       
    def train_step(self, batch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        loss_dict = compute_loss_corresp_forward(batch, decoder_shape=self.decoder_shape,
                                decoder=self.decoder, device=self.device,
                                latent_codes=self.latent_deform, latent_codes_shape=self.latent_shape)
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

        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}

        loss_dict.update({'loss': loss_total.item()})
           

        return loss_dict    
    
    

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
                loss_dict = self.train_step(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}

                wandb.log(loss_values)
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0:
                self.save_checkpoint(epoch,self.cfg['save_name'])
            if epoch % 10 ==0:
                lat_shape = self.latent_shape.weight[random.randint(0,6)].to(self.device)
                mesh_mc, _ = latent_to_mesh(self.decoder_shape, lat_shape,self.device,size=64)
                lat_def = self.latent_deform.weight[random.randint(0,len(self.trainset)-1)].to(self.device)
                deform = deform_mesh(mesh_mc, self.decoder, lat_rep=lat_def,lat_rep_shape=lat_shape )
                img_deform= save_mesh_image_with_camera(deform.vertices, deform.faces)
                wandb.log({'deform': wandb.Image(img_deform)})
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
                
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7, help='gpu index')
    parser.add_argument('--wandb', type=str, default='2d_sdf', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    # setting

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                         d_out=CFG['shape_decoder']['decoder_out_dim'],
                         n_layers=CFG['shape_decoder']['decoder_nlayers'],
                         udf_type='sdf')
    decoder_shape.mlp_pos = None
    decoder = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False)
   

    decoder = decoder.to(device)
    decoder_shape = decoder_shape.to(device)
    trainer = DeformTrainer(
                            decoder=decoder,
                            decoder_shape=decoder_shape,
                            cfg=CFG, 
                            trainset=trainset,trainloader=trainloader, 
                            device=device)
    trainer.train(30001)

