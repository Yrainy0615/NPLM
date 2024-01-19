import torch
import torch.optim as optim
import math
from glob import glob
from scripts.model.loss_functions import compute_loss, compute_verts, compute_normal
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
from scripts.dataset.sdf_dataset import LeafShapeDataset, Leaf2DShapeDataset, LeafDisplacementDataset
from scripts.model.fields import SDFNetwork ,UDFNetwork
from torch.utils.data import DataLoader
import random
from scripts.model.discriminator import Discriminator, weights_init
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
from scripts.model.renderer import MeshRender
from scripts.training.trainer_color import save_grid_image
from scripts.test.mesh_to_udf import refine_mesh

class ShapeTrainer(object):
    def __init__(self, decoder, cfg, trainset,trainloader,device, latent_shape_all):
        self.decoder = decoder
        self.cfg = cfg['training']
        self.latent_shape_all = latent_shape_all
        self.latent_idx = torch.nn.Embedding(len(trainset), decoder.lat_dim, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(
            self.latent_idx.weight.data, 0.0, 0.1/math.sqrt(decoder.lat_dim)
        )
        self.latent_spc = torch.nn.Embedding(trainset.all_species, decoder.lat_dim//2, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(
            self.latent_spc.weight.data, 0.0, 0.1/math.sqrt(decoder.lat_dim//2))
        
        print(self.latent_idx.weight.shape)
        print(self.latent_spc.weight.shape)
        self.trainset = trainset

        self.init_latent()
        self.trainloader = trainloader
        self.device = device
        self.optimizer_decoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.combined_para = list(self.latent_idx.parameters()) + list(self.latent_spc.parameters())
        self.optimizer_latent = optim.SparseAdam(params=list(self.latent_idx.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']

        self.checkpoint_path = self.cfg['save_path']
        
        # discriminator
        self.discriminator = Discriminator(3,64)
        self.discriminator.apply(weights_init)
        self.discriminator.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer_D = optim.Adam(self.discriminator.parameters(),self.cfg['lr_dis'], betas=(0.5, 0.999))
    
        # renderer
        self.renderer = MeshRender(device)
    
    def init_latent(self):
        # init latent codes
        all_mesh = self.trainset.all_mesh
        all_neutral = self.trainset.all_neutral
        all_neutral = [os.path.splitext(os.path.basename(specie))[0] for specie in all_neutral if '.obj' in specie]
        all_neutral = [name.split('_')[0] for name in all_neutral]
        for i in range(len(all_mesh)):
            spc = all_mesh[i].split('/')[-3]
            # from spc get index in all_neutral
            if spc not in all_neutral:
                weight_clone = self.latent_idx.weight.clone()
                weight_clone[i] = self.latent_shape_all[2]
                self.latent_idx.weight.data = weight_clone
            else:
                weight_clone = self.latent_idx.weight.clone()
                weight_clone[i] = self.latent_shape_all[all_neutral.index(spc)]
                self.latent_idx.weight.data[i] = weight_clone[i]
        
        

        #self.latent_codes_val.weight.data = self.latent_codes_val.weight.data
        print('init latent codes')
    
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
        
    def save_checkpoint(self, epoch, savename):
        path = self.checkpoint_path + '/{}__{}.tar'.format(savename,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                      #  'optimizer_lat_val_state_dict': self.optimizer_lat_val.state_dict(),
                        'latent_idx_state_dict': self.latent_idx.state_dict(),
                       # 'latent_spc_state_dict': self.latent_spc.state_dict(),
                  
                       # 'latent_codes_val_state_dict': self.latent_codes_val.state_dict()
                       },
                       path)
    def train_normal(self, batch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        new_mesh = compute_verts(batch, self.decoder, self.latent_idx,self.device)
        # change verts face to float tyoe
        #new_verts = new_verts.to(torch.float32)
       # pred_mesh = Meshes(new_verts.to(torch.float32), batch['faces'].to(self.device).to(torch.float32))
        normal_pred = self.renderer.render_normal(new_mesh.to(device))
    
    def train_dis(self, batch, epoch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()

        new_mesh, loss_reg = compute_verts(batch, self.decoder, self.latent_idx,self.device)
            # change verts face to float tyoe
            #new_verts = new_verts.to(torch.float32)
        # pred_mesh = Meshes(new_verts.to(torch.float32), batch['faces'].to(self.device).to(torch.float32))
        normal_pred = self.renderer.render_normal(new_mesh.to(device)) # h,w ,4
            # remove channel 4 ->(b,h,w,3)
        normal_pred = normal_pred[ :,:, :,:3]
        normal_gt = batch['normal_map'].to(self.device)
        normal_gt.requires_grad = True
       # normal = compute_normal(decoder)
        # mse loss
        if epoch<2:
            normal_pred_tensor = torch.tensor(normal_pred, dtype=torch.float64, device=self.device)
            loss_mse = torch.nn.functional.mse_loss(normal_pred_tensor, normal_gt)
            loss = loss_mse  + loss_reg
            loss.backward()
            self.optimizer_decoder.step()
            self.optimizer_latent.step()
            loss_dict = {'normal_mse': loss_mse.item(),
                         'loss_reg': loss_reg.item()}
            fake_result = save_grid_image(normal_pred_tensor.permute(0,3,1,2),nrow=8)
            return loss_dict, fake_result   
        else:
        # adverserial loss
            # train discriminator
            self.optimizer_D.zero_grad()
            normal_pred_tensor = torch.tensor(normal_pred, dtype=torch.float32, device=self.device)
            pred_fake = self.discriminator(normal_pred_tensor.permute(0,3,1,2))
            loss_D_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
            pred_real = self.discriminator(normal_gt.permute(0,3,1,2).float())
            loss_D_real = self.criterion(pred_real, torch.ones_like(pred_real))
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            self.optimizer_D.step()
            loss_dict = {
                'loss_D_real': loss_D_real.item(),
                'loss_D_fake': loss_D_fake.item(),
            }
            
            # train generator for 10 times
            for i in range(10):
                self.optimizer_decoder.zero_grad()
                self.optimizer_latent.zero_grad()
                new_mesh, loss_reg = compute_verts(batch, self.decoder, self.latent_idx,self.device)
                normal_pred = self.renderer.render_normal(new_mesh.to(device)) # h,w ,4
                normal_pred = normal_pred[:, :, :,:3]
                normal_pred_tensor = torch.tensor(normal_pred, dtype=torch.float32, device=self.device)

                pred_fake = self.discriminator(normal_pred_tensor.permute(0,3,1,2))
                loss_G = self.criterion(pred_fake, torch.ones_like(pred_fake))
                # retain graph?
                
                loss = loss_G + loss_reg
                loss.backward()
                loss_dict.update({'loss_G': loss_G.item()})
                loss_dict.update({'loss_reg': loss_reg.item()})

                
                
                if self.cfg['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

                if self.cfg['grad_clip_lat'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.combined_para, max_norm=self.cfg['grad_clip_lat'])
                self.optimizer_decoder.step()
                self.optimizer_latent.step()

            fake_result = save_grid_image(normal_pred_tensor.permute(0,3,1,2),nrow=8)
            #real_result = save_grid_image(normal_gt.permute(0,3,1,2),nrow=8)
            
     
            return loss_dict  ,  fake_result
    
    def train_step(self, batch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        loss_dict = compute_loss(batch, self.decoder, self.latent_idx, self.latent_spc,self.device)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

        if self.cfg['grad_clip_lat'] is not None:
            torch.nn.utils.clip_grad_norm_(self.latent_idx.parameters(), max_norm=self.cfg['grad_clip_lat'])
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
                loss_dict, fake = self.train_dis(batch,epoch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                wandb.log(loss_values)
                #log images
                wandb.log({'fake': [wandb.Image(fake)]})
     
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0:
                self.save_checkpoint(epoch,self.cfg['savename'])
                
            # if epoch % 5 ==0:
            #     #lat = torch.concat([self.latent_idx.weight[0], self.latent_spc.weight[0]]).to(self.device)
            #     lat = self.latent_idx.weight[random.randint(0,len(trainset)-1)].to(self.device)
            #    # _, _ ,mesh_udf = get_mesh_udf_fast(decoder, lat.unsqueeze(0),gradient=False)
            #     #print(mesh_udf)
            #     mesh_mc, mesh_udf = latent_to_mesh(self.decoder, lat,self.device,size=64)
            #     if mesh_mc is not None and mesh_mc.faces.shape[0]>0:
            #         img_mc = save_mesh_image_with_camera(mesh_mc.vertices, mesh_mc.faces)
            #         wandb.log({'sdf': wandb.Image(img_mc)})
                
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
                
if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=3, help='gpu index')
    parser.add_argument('--wandb', type=str, default='*', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    # setting

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    config = 'NPLM/scripts/configs/npm_color.yaml'
    CFG = yaml.safe_load(open(config, 'r'))
    wandb.init(project='NPLM', name =args.wandb)
    wandb.config.update(CFG)
    trainset = LeafDisplacementDataset(mode='train',
                        n_supervision_points_face=CFG['training']['npoints_decoder'],
                        batch_size=CFG['training']['batch_size'],
                        sigma_near=CFG['training']['sigma_near'],
                        root_dir=CFG['training']['root_dir_color'])
   # trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=2)
    trainloader = trainset.get_loader(batch_size=CFG['training']['batch_size'])
    # all_mesh = trainset.all_mesh
    # for i in range(len(all_mesh)):
    #     mesh = refine_mesh(all_mesh[i])
    #     mesh.export(all_mesh[i].replace('.obj','.ply'))
    #     print('refine mesh {}'.format(all_mesh[i]))
    #trainloader = trainset.get_loader(batch_size=CFG['training']['batch_size'])
    decoder = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=CFG['decoder']['decoder_out_dim'],
                         n_layers=CFG['decoder']['decoder_nlayers'],
                         udf_type='sdf')
    checkpoint_shape = torch.load('checkpoints/2dShape/exp-cg-sdf__30000.tar')
    decoder = decoder.to(device)
    #decoder.load_state_dict(checkpoint_shape['decoder_state_dict'])
    print(f'load decoder from checkpoints')
    latent_shape_all = checkpoint_shape['latent_idx_state_dict']['weight']
    latent_shape_all = latent_shape_all.to(device)
    trainer = ShapeTrainer(decoder, CFG, trainset,trainloader, device,latent_shape_all)
    trainer.train(30001)
            
        
        
