import torch
import torch.optim as optim
import math
from glob import glob
from scripts.model.loss_functions import perceptual_loss, compute_loss_corresp_forward, compute_color_forward
import os
import numpy as np
import wandb
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    MeshRenderer, 
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointLights,
    SoftPhongShader,
    MeshRasterizer,
    AlphaCompositor,
    TexturesVertex
)
from PIL import Image
import torchvision.utils as vutils
import torch.nn as nn
import torchvision
import yaml
from scripts.dataset.sdf_dataset import LeafShapeDataset, LeafColorDataset
from torch.utils.data import DataLoader
from scripts.model.fields import UDFNetwork
import argparse
from scripts.model.discriminator import Discriminator
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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

def save_grid_image(tensor, nrow=1, padding=2):
   # images = tensor.permute(0, 3, 1, 2)
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding)

    grid = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()

 

    #im.save('grid_image.png')
    return grid

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ColorTrainer(object):
    def __init__(self, decoder,cfg, 
                 trainset,trainloader,device):
        self.decoder = decoder

        self.cfg = cfg['training']
        # latent initializaiton
    
        self.latent_color = torch.nn.Embedding(len(trainset), self.decoder.lat_dim, max_norm = 1.0, sparse=True, device = device).float()
        torch.nn.init.normal_(self.latent_color.weight.data, 0.0, 0.01)
        print(self.latent_color.weight.data.shape)
        #print(self.latent_color.weight.data.norm(dim=-1).mean())
       # self.init_shape_state(cfg['training']['shape_ckpt'], 'checkpoints/')
        
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
        self.discriminator = Discriminator(3,64)
        self.discriminator.apply(weights_init)
        self.discriminator = self.discriminator.to(device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.mse = torch.nn.MSELoss()
        self.vgg = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        # renderer
        R, T = look_at_view_transform(1.5, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=64, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
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
       
    def train_step(self, batch,epoch):
        self.decoder.train()
        self.optimizer_decoder.zero_grad()
        self.optimizer_latent.zero_grad()
        
        # train discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad() 
      
        # color,lat_mean = compute_color_forward(batch, 
        #                         decoder=self.decoder, device=self.device,
        #                         latent_codes=self.latent_color)
        # batch['mesh'][0].textures = TexturesVertex(verts_features=color)
        
        if epoch < 0:
            real = batch['rgb'][0]
            real[real==0] = 255
            real_tensor = torch.from_numpy(real/255).float().to(device)
            fake = self.renderer(batch['mesh'][0].to(device))
            loss_mse =   (fake.squeeze(0)[:,:,:3]- real_tensor).pow(2).mean()
            loss_mse.backward()
            self.optimizer_decoder.step()
            self.optimizer_latent.step()
            loss_dict = {'loss_mse': loss_mse}
            fake_result = save_grid_image(fake.permute(0,3,1,2))
            real_result = save_grid_image(real_tensor.unsqueeze(0).permute(0,3,1,2))
            return loss_dict, fake_result, real_result
        else:
            # train discriminator
            real = batch['rgb'][0]
            real[real==0] = 255
            # float type
            
            real_tensor = torch.from_numpy(real/255).float().unsqueeze(0).to(self.device)
            color = compute_color_forward(batch,decoder=self.decoder, 
                                          latent_codes=self.latent_color ,device=self.device)
            
            batch['mesh'][0].textures = TexturesVertex(verts_features=color)
            fake = self.renderer(batch['mesh'][0].to(self.device))
            real_label = torch.rand(real_tensor.shape[0], device=self.device) * 0.1 + 0.9  # Random values between 0.9 and 1.0
            fake_label = torch.rand(real_tensor.shape[0], device=self.device) * 0.1 +0.01  # Random values between 0 and 0.1
            output_real = self.discriminator(real_tensor.permute(0,3,1,2)).view(-1)
            errD_real = self.criterion(output_real, real_label)

            # fake
            fake = self.renderer(batch['mesh'][0].to(self.device))

            # label.fill_(fake_label)
            fake = fake[:,:,:,:3].permute(0,3,1,2)
            output_fake = self.discriminator(fake).view(-1) 
            if torch.isnan(output_fake).any():
                raise ValueError("NaN detected in forward pass")

            errD_fake = self.criterion(output_fake, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            self.optimizer_D.step()
        # wdist = torch.mean(output_real - output_fake)
            loss_dict = {'loss_D_real': errD_real,
                        'loss_D_fake': errD_fake,}
                    #  'gradient_penalty': gradient_penalty.mean()}

            
            # train generator

            for p in self.discriminator.parameters():  # reset requires_grad
                p.requires_grad = False
            self.optimizer_D.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.latent_color.zero_grad()
            color= compute_color_forward(batch, decoder=self.decoder,
                                device=self.device,
                                latent_codes=self.latent_color )
            batch['mesh'][0].textures = TexturesVertex(verts_features=color)
            fake = self.renderer(batch['mesh'][0].to(self.device))   
            fake = fake[:,:,:,:3].permute(0,3,1,2)

            # label.fill_(real_label)
            output_fake = self.discriminator(fake).view(-1)
            if torch.isnan(output_fake).any():
                raise ValueError("NaN detected in forward pass")
            loss_G = self.criterion(output_fake, real_label)

            #output_fake = self.discriminator(fake).view(-1)
            gradient_penalty = compute_gradient_penalty(real_tensor.permute(0,3,1,2), fake, self.discriminator, self.cfg['lambda_pen'],device=self.device)
            loss_w = torch.mean(fake -real_tensor.permute(0,3,1,2) + gradient_penalty)
            
            # perceptual loss
            loss_per = perceptual_loss(vgg=self.vgg, fake=fake, real=real_tensor.permute(0,3,1,2))
            
            # color smoothness
            color_gradients = torch.sqrt(torch.sum((color[0][:-1] - color[0][1:]) ** 2, dim=1)+1e-8)
            color_smoothness_loss = torch.mean(color_gradients)
            
            #loss_G = - torch.mean(output_fake)
            loss = loss_G*0.3  + loss_w *0.1  + loss_per+ color_smoothness_loss *10
            loss.backward()
            # if self.cfg['grad_clip'] is not None:
            #     torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])

            if self.cfg['grad_clip_lat'] is not None:
                torch.nn.utils.clip_grad_norm_(self.latent_color.parameters(), max_norm=self.cfg['grad_clip_lat'])

            self.optimizer_decoder.step()
            self.optimizer_latent.step()
            fake_result = save_grid_image(fake)
            real_result = save_grid_image(real_tensor.permute(0,3,1,2))

            loss_dict.update({'loss_G': loss_G,
                            'loss_per': loss_per,
                            'loss_w': loss_w,
                            'color_smoothness_loss': color_smoothness_loss})
            return loss_dict  , fake_result, real_result
    
    

    def train(self, epochs):
        loss = 0
        # start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict, fake, real = self.train_step(batch,epoch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
  
                wandb.log(loss_values)
    
                wandb.log({'fake':wandb.Image(fake)})
                wandb.log({'real':wandb.Image(real)})
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
    trainset = LeafColorDataset(mode='train',
                        batch_size=CFG['training']['batch_size'],
                        root_dir=CFG['training']['root_dir_color'])
#    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=0)
    trainloader = trainset.get_loader()

    decoder = UDFNetwork(d_in=CFG['decoder']['decoder_lat_dim'],
                         d_hidden=CFG['decoder']['decoder_hidden_dim'],
                         d_out=CFG['decoder']['decoder_out_dim'],
                         n_layers=CFG['decoder']['decoder_nlayers'],
                         udf_type='sdf')

    decoder = decoder.to(device)
    trainer = ColorTrainer(decoder, CFG, trainset,trainloader,device)
    trainer.train(30001)
            
                
            
        

