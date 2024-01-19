import torch
import torch.optim as optim
import math
from glob import glob
from scripts.model.loss_functions import compute_loss, compute_diff_loss
import os
import numpy as np
import wandb
from scripts.model.reconstruction import latent_to_mesh, save_mesh_image_with_camera
from matplotlib import pyplot as plt
import io
from PIL import Image
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")
import yaml
from scripts.dataset.sdf_dataset import LeafShapeDataset, Leaf2DShapeDataset, LeafColorDataset
from scripts.model.fields import SDFNetwork ,UDFNetwork
from torch.utils.data import DataLoader
import random
from MeshUDF.optimize_chamfer_A_to_B import get_mesh_udf_fast, optimize
from scripts.model.renderer import MeshRender
from torchvision import utils as vutils

def save_grid_image(tensor, nrow=1,padding=1):
       # images = tensor.permute(0, 3, 1, 2)
    #tensor = torch.clamp(tensor, 0, 1)
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding)

    im = grid.detach().cpu().numpy().transpose(1,2,0)
    if im.shape[-1] == 4:
        im = np.clip(im, 0, 1)
        im = im[:,:,:3]
   # im = Image.fromarray(grid)

   # im.save('mask.png')
    return im


class ShapeTrainer(object):
    def __init__(self, decoder, cfg, trainset,trainloader,device):
        self.decoder = decoder
        self.cfg = cfg['training']
        self.latent_idx = torch.nn.Embedding(len(trainset), decoder.lat_dim//2, max_norm = 1.0, sparse=False, device = device).float()
        torch.nn.init.normal_(
            self.latent_idx.weight.data, 0.0, 0.1/math.sqrt(decoder.lat_dim//2)
        )
        self.latent_spc = torch.nn.Embedding(len(trainset.all_species), decoder.lat_dim//2, max_norm = 1.0, sparse=False, device = device).float()
        torch.nn.init.normal_(
            self.latent_spc.weight.data, 0.0, 0.1/math.sqrt(decoder.lat_dim//2)
        )   
        print(self.latent_idx.weight.shape)
        print(self.latent_spc.weight.shape)
        self.trainloader = trainloader

        self.device = device
        self.renderer = MeshRender(device)
        self.optimizer_decoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.combined_para = list(self.latent_idx.parameters()) + list(self.latent_spc.parameters())
        self.optimizer_latent = optim.Adam(self.combined_para, lr=self.cfg['lr_lat'])
      #self.optimizer_latent = optim.Adam(params= self.combined_para, lr=self.cfg['lr_lat'])
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
            path = self.checkpoint_path + 'shape_epoch_{}.tar'.format(self.cfg['ckpt'])
        else:
            print('LOADING', checkpoints[-1])
            path = self.checkpoint_path + 'shape_epoch_{}.tar'.format(checkpoints[-1])

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
        
    def save_checkpoint(self, epoch,save_name):
        path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                      #  'optimizer_lat_val_state_dict': self.optimizer_lat_val.state_dict(),
                        'latent_idx_state_dict': self.latent_idx.state_dict(),
                       'latent_spc_state_dict': self.latent_spc.state_dict(),
                  
                       # 'latent_codes_val_state_dict': self.latent_codes_val.state_dict()
                       },
                       path)
       
    def train_step(self, batch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        loss_dict, rgb, mask_pred = compute_diff_loss(batch, self.decoder, self.latent_idx, self.latent_spc,self.device, self.renderer)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        rgb = save_grid_image(rgb.permute(0,3,1,2))
        mask_pred = save_grid_image(mask_pred.permute(0,3,1,2))
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.combined_para, max_norm=self.cfg['grad_clip_lat'])
        self.optimizer_decoder.step()
        self.optimizer_latent.step()

        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}

        loss_dict.update({'loss': loss_total.item()})
           

        return loss_dict   , rgb, mask_pred
    
    

    def train(self, epochs,wandb_flag):
        loss = 0
       # start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict, mask, mask_pred = self.train_step(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                # use wandb to log mask and mask_pred, tensor image (b,64,64,4)
                if wandb_flag:
                    wandb.log({'mask': [wandb.Image(mask, caption='mask')],
                            'mask_pred': [wandb.Image(mask_pred, caption='mask_pred')]})
                    
                    wandb.log(loss_values)
                
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0:
                self.save_checkpoint(epoch, save_name=CFG['training']['save_name'])
         
                
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
                
if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=4, help='gpu index')
    parser.add_argument('--wandb', type=str, default='*', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    # setting

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    config = 'NPLM/scripts/configs/npm_color.yaml'
    CFG = yaml.safe_load(open(config, 'r'))
    if args.wandb != '*':
        wandb.init(project='NPLM', name =args.wandb)
        wandb.config.update(CFG)
    trainset = LeafColorDataset(mode='train',
                                batch_size=CFG['training']['batch_size'],
                                image_size=CFG['training']['image_size'],
                        root_dir=CFG['training']['root_dir_color'])
    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=False, num_workers=2)

    decoder = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=CFG['decoder']['decoder_out_dim'],
                         n_layers=CFG['decoder']['decoder_nlayers'],
                         udf_type='sdf')

    decoder = decoder.to(device)
    trainer = ShapeTrainer(decoder, CFG, trainset,trainloader, device)
    wandb_flag = True if args.wandb != '*' else False
    trainer.train(10001, wandb_flag)
            
        
        
