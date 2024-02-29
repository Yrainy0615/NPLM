import torch
import torch.optim as optim
import math
from glob import glob
import sys
sys.path.append('NPLM')
from scripts.model.loss_functions import end_to_end_loss
import os
import numpy as np
import wandb
import argparse

import warnings
warnings.filterwarnings("ignore")
import yaml
from scripts.dataset.sdf_dataset import  LeafSDF3dDataset
from scripts.model.fields import UDFNetwork
from torch.utils.data import DataLoader
import random
from scripts.model.reconstruction import latent_to_mesh_e2e, deform_mesh, save_mesh_image_with_camera

class ShapeDeformTrainer(object):
    def __init__(self, decoder_deform,decoder_shape,cfg, 
                 trainset,trainloader,device, args):
        self.decoder_deform = decoder_deform
        self.decoder_shape = decoder_shape
        self.cfg = cfg['training']
        self.trainset = trainset
        self.args = args
        self.latent_deform = torch.nn.Embedding(168, 256, max_norm = 1.0, sparse=True, device = device).float()
        self.latent_shape = torch.nn.Embedding(326, 512, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(self.latent_deform.weight.data, 0.0, 0.01)
        print('Latent Deform', self.latent_deform.weight.data.shape)
        print('Latent Shape', self.latent_shape.weight.data.shape)
        torch.nn.init.normal_(self.latent_shape.weight.data, 0.0, 0.01)
        self.trainloader = trainloader
        self.device = device
        self.optimizer_decoder_deform = optim.AdamW(params=list(decoder_deform.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.optimizer_decoder_shape = optim.AdamW(params=list(decoder_shape.parameters()),
                                                lr = self.cfg['lr'],
                                                weight_decay= self.cfg['weight_decay'])
        self.optimizer_latent_deform = optim.SparseAdam(params= list(self.latent_deform.parameters()), lr=self.cfg['lr_lat'])
        self.optimizer_latent_shape = optim.SparseAdam(params= list(self.latent_shape.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']
        self.checkpoint_path = self.cfg['save_path']
        

        
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
            for param_group in self.optimizer_decoder_deform.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_decoder_shape.param_groups:
                param_group["lr"] = lr

        
    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest.tar'
            torch.save({'epoch': epoch,
                        'decoder_shape_state_dict': self.decoder_shape.state_dict(),
                        'optimizer_decoder_shape_state_dict': self.optimizer_decoder_shape.state_dict(),
                        'optimizer_lat_shape_state_dict': self.optimizer_latent_shape.state_dict(),  
                        'optimizer_lat_deform_state_dict': self.optimizer_latent_deform.state_dict(),
                        'latent_deform_state_dict': self.latent_deform.state_dict(),
                        'latent_shape_state_dict': self.latent_shape.state_dict(),
                        'decoder_deform_state_dict': self.decoder_deform.state_dict(),
                        'optimizer_decoder_deform_state_dict': self.optimizer_decoder_deform.state_dict(),},
                       path)
       
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
            if not os.path.exists(path):
                torch.save({'epoch': epoch,
                            'decoder_shape_state_dict': self.decoder_shape.state_dict(),
                            'optimizer_decoder_shape_state_dict': self.optimizer_decoder_shape.state_dict(),
                            'optimizer_lat_shape_state_dict': self.optimizer_latent_shape.state_dict(),  
                            'optimizer_lat_deform_state_dict': self.optimizer_latent_deform.state_dict(),
                            'latent_deform_state_dict': self.latent_deform.state_dict(),
                            'latent_shape_state_dict': self.latent_shape.state_dict(),
                            'decoder_deform_state_dict': self.decoder_deform.state_dict(),
                            'optimizer_decoder_deform_state_dict': self.optimizer_decoder_deform.state_dict(),},
                        path)
       
    def train_step(self, batch):
        self.decoder_shape.train()
        self.decoder_deform.train()
        self.optimizer_decoder_shape.zero_grad()
        self.optimizer_decoder_deform.zero_grad()
        self.optimizer_latent_deform.zero_grad()
        self.optimizer_latent_shape.zero_grad()
        loss_dict = end_to_end_loss(batch,
                                decoder_shape=self.decoder_shape, latent_shape=self.latent_shape,
                                decoder_deform=self.decoder_deform, latent_deform=self.latent_deform,
                                device=self.device) 
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder_shape.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder_deform.parameters(), max_norm=self.cfg['grad_clip'])
        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.latent_deform.parameters(), max_norm=self.cfg['grad_clip_lat'])
            torch.nn.utils.clip_grad_norm_(self.latent_shape.parameters(), max_norm=self.cfg['grad_clip_lat'])
        self.optimizer_decoder_shape.step()
        self.optimizer_latent_shape.step()
        self.optimizer_latent_deform.step()
        self.optimizer_decoder_deform.step()

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
                if self.args.use_wandb:
                    wandb.log(loss_values)
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0 and epoch >0:
                self.save_checkpoint(epoch,self.cfg['save_name'])
            self.save_checkpoint(epoch, save_name='latest')

            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
                
def test():
    pass 
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--wandb', type=str, default='end2end', help='run name of wandb')
    parser.add_argument('--output', type=str, default='deform', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    # setting
    mode = 'test'
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_deform.yaml'
    CFG = yaml.safe_load(open(config, 'r'))
    if args.use_wandb:
        wandb.init(project='NPLM', name =args.wandb)
    
    # dataset & dataloader
    trainset = LeafSDF3dDataset(
                        num_samples=CFG['training']['npoints_decoder'],
                        root_dir = CFG['training']['root_dir'],
                        sigma_near=CFG['training']['sigma_near'],)
    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=2)


    decoder_deform = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False,
                         use_mapping=CFG['deform_decoder']['use_mapping'])
   
    decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                            d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                            d_out=CFG['shape_decoder']['decoder_out_dim'],
                            n_layers=CFG['shape_decoder']['decoder_nlayers'],
                            udf_type='sdf',
                            d_in_spatial=3,
                            geometric_init=False,
                            use_mapping=CFG['shape_decoder']['use_mapping'])
    decoder_deform = decoder_deform.to(device)
    decoder_shape = decoder_shape.to(device)
    
    # trainer initialization
    trainer = ShapeDeformTrainer(
                            decoder_deform=decoder_deform,
                            decoder_shape=decoder_shape,
                            cfg=CFG, 
                            trainset=trainset,trainloader=trainloader, 
                            device=device,
                            args=args)
    # train
    if mode == 'train':
        trainer.train(10001)
    
    else:
        decoder_deform.eval()
        decoder_shape.eval()
        decoder_deform.load_state_dict(torch.load('checkpoints/shape_deform/exp-end2end__2000.tar')['decoder_deform_state_dict'])
        decoder_shape.load_state_dict(torch.load('checkpoints/shape_deform/exp-end2end__2000.tar')['decoder_shape_state_dict'])
        latent_shape = torch.load('checkpoints/shape_deform/exp-end2end__2000.tar')['latent_shape_state_dict']['weight']
        latent_deform = torch.load('checkpoints/shape_deform/exp-end2end__2000.tar')['latent_deform_state_dict']['weight']
        latent = torch.cat((latent_shape[8],latent_deform[8]), dim=0)
        batch = next(iter(trainloader))
        points = batch['points'].to(device)
        points_c  = decoder_deform(points[0].float(), latent.unsqueeze(0).repeat(points[0].shape[0],1).float())
        points_sdf = decoder_shape(points_c, latent_shape[8].unsqueeze(0).repeat(points_c.shape[0],1).float())
        latent_to_mesh_e2e(points_c,points_sdf,device)
        


