import torch
import torch.optim as optim
import math
from glob import glob
from scripts.model.loss_functions import compute_loss, compute_loss_corresp_forward, compute_color_forward
import os
import numpy as np
import wandb
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from PIL import Image
import torchvision.utils as vutils
import torch.nn as nn

def compute_gradient_penalty(real, fake, discriminator, lambda_pen,device):
    # Compute the sample as a linear combination
    alpha = torch.rand(real.shape[0], 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)
    x_hat = alpha * real + (1 - alpha) * fake
    # Compute the output
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    out = discriminator(x_hat)
    # compute the gradient relative to the new sample
    gradients = torch.autograd.grad(
        outputs=out,
        inputs=x_hat,
        grad_outputs=torch.ones(out.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # Reshape the gradients to take the norm
    # gradients = gradients.view(gradients.shape[0], -1)
    gradients = gradients.contiguous().view(gradients.shape[0], -1)
    # Compute the gradient penalty
    penalty = (gradients.norm(2, dim=1) - 1) ** 2
    penalty = penalty * lambda_pen
    return penalty

def save_grid_image(tensor, nrow=8, padding=2):
   # images = tensor.permute(0, 3, 1, 2)
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding)

    grid = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()

    im = Image.fromarray(grid)

    # im.save('grid_image.png')
    return im

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ColorTrainer(object):
    def __init__(self, decoder, decoder_shape,cfg, 
                 trainset,trainloader,discriminator,device):
        self.decoder = decoder
        self.decoder_shape = decoder_shape
        self.cfg = cfg['training']
        # latent initializaiton
        self.latent_shape = torch.nn.Embedding(7, decoder_shape.lat_dim, max_norm = 1.0, sparse=True, device = device).float()
        self.latent_shape.requires_grad_  = False
        self.latent_color = torch.nn.Embedding(len(trainset), self.decoder.lat_dim_expr, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(self.latent_color.weight.data, 0.0, 0.01)
        print(self.latent_color.weight.data.shape)
        print(self.latent_color.weight.data.norm(dim=-1).mean())
        self.init_shape_state(cfg['training']['shape_ckpt'], 'checkpoints/')
        
        self.trainloader = trainloader
        self.device = device
        self.optimizer_decoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr = self.cfg['lr'],
                                             weight_decay= self.cfg['weight_decay'])
        self.optimizer_latent = optim.SparseAdam(params= list(self.latent_color.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']

        self.checkpoint_path = self.cfg['save_path']
        
        # discriminator
        self.discriminator = discriminator
        self.discriminator.apply(weights_init)
        self.criterion = torch.nn.BCELoss()
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        
        # renderer
        R, t = look_at_view_transform(0.00001, 34, 138)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=t)
        raster_settings = PointsRasterizationSettings(
            image_size=64, 
            radius=0.01, 
            points_per_pixel=1
        )
        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )
        
        
    def init_shape_state(self, ckpt, path):
        path = path + 'shape_epoch_{}.tar'.format(ckpt)
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
        
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'color_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                      #  'optimizer_lat_val_state_dict': self.optimizer_lat_val.state_dict(),
                        'latent_color_state_dict': self.latent_color.state_dict(),
                  
                       # 'latent_codes_val_state_dict': self.latent_codes_val.state_dict()
                       },
                       path)
       
    def train_step(self, batch):
        self.decoder.train()
        self.decoder_shape.eval()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        # real_label_value = torch.rand(batch['rgb'].shape[0], device=self.device) * 0.1 + 0.9  # Random values between 0.9 and 1.0
        # fake_label_value = torch.rand(batch['rgb'].shape[0], device=self.device) * 0.1  # Random values between 0 and 0.1
        # real_label = torch.full((batch['rgb'].shape[0],), real_label_value, dtype=torch.float, device=self.device)
        # fake_label = torch.full((batch['rgb'].shape[0],), fake_label_value, dtype=torch.float, device=self.device)        
        color = compute_color_forward(batch, decoder_shape=self.decoder_shape,
                                decoder=self.decoder, device=self.device,
                                latent_codes=self.latent_color, latent_codes_shape=self.latent_shape)
        point_cloud = Pointclouds(points=batch['points'].to(self.device).to(dtype=torch.float32), features=color)

        # real 
        self.discriminator.zero_grad()
        real = batch['rgb'].to(self.device)
        real = real.permute(0,3,1,2).float()
       # label = torch.full((batch['rgb'].shape[0],), real_label, dtype=torch.float, device=self.device)
        output_real = self.discriminator(real).view(-1)
        # errD_real = self.criterion(output_real, label)
        # errD_real.backward()

        # fake
        fake = self.renderer(point_cloud)

      #  label.fill_(fake_label)
        fake = fake.permute(0,3,1,2)
        output_fake = self.discriminator(fake.detach()).view(-1)
        # errD_fake = self.criterion(output_fake, label)
        # errD_fake.backward()
        gradient_penalty = compute_gradient_penalty(real, fake, self.discriminator, self.cfg['lambda_pen'],device=self.device)
        loss = torch.mean(output_fake - output_real + gradient_penalty)
        loss.backward()
        self.optimizer_D.step()
        wdist = torch.mean(output_real - output_fake)
        loss_dict = {'w_distance': wdist,
                     'loss_discriminator': loss,
                     'gradient_penalty': gradient_penalty.mean()}

        
        # train generator
        for i in range(5):
            self.decoder.zero_grad()
            color = compute_color_forward(batch, decoder_shape=self.decoder_shape,
                                decoder=self.decoder, device=self.device,
                                latent_codes=self.latent_color, latent_codes_shape=self.latent_shape)
            point_cloud = Pointclouds(points=batch['points'].to(self.device).to(dtype=torch.float32), features=color)
            fake = self.renderer(point_cloud)   
            fake = fake.permute(0,3,1,2)
      #      label.fill_(real_label)
            output_fake = self.discriminator(fake).view(-1)
            loss_G = - torch.mean(output_fake)
            if i <4:
                loss_G.backward(retain_graph=True)
            else:
                loss_G.backward()
            # if self.cfg['grad_clip'] is not None:
            #     torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

            if self.cfg['grad_clip_lat'] is not None:
                torch.nn.utils.clip_grad_norm_(self.latent_color.parameters(), max_norm=self.cfg['grad_clip_lat'])

            self.optimizer_decoder.step()
            self.optimizer_latent.step()
        fake_result = save_grid_image(fake)
        loss_dict.update({'loss_G': loss_G})
        

        return loss_dict    , fake_result
    
    

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
                loss_dict, fake = self.train_step(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
  
                wandb.log(loss_values)
    
                wandb.log({'fake':wandb.Image(fake)})
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0:
                self.save_checkpoint(epoch)
                
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)
                
            
                
            
        

