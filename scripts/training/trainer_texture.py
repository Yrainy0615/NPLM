import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import Point_cloud_dataset, custom_collate_fn
from torch.utils.data import DataLoader
from scripts.model.point_encoder import PCAutoEncoder, CameraNet
from scripts.model.fields import UDFNetwork
import yaml
import glob
import wandb
from scripts.model.loss_functions import rgbd_loss, texture_loss
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender

class TextureTrainer(object):
    def __init__(self, encoder_3d, encoder_2d, 
                 cameranet, trainloader, 
                 latent_shape, latent_deform,
                 decoder_shape, decoder_deform,
                 generator,
                 cfg, device):
        self.encoder_3d = encoder_3d
        self.encoder_2d = encoder_2d
        self.cameranet = cameranet
        self.trainloader = trainloader
        self.decoder_shape = decoder_shape
        self.decoder_deform = decoder_deform
        self.device = device
        self.generator = generator
        self.renderer = MeshRender(device=device)
        self.cfg = cfg['training']
        self.latent_shape = latent_shape
        self.latent_deform = latent_deform
        self.optimizer_cameranet = optim.Adam(self.cameranet.parameters(), lr=1e-4)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optimizer_encoder3d = optim.Adam(self.encoder_3d.parameters(), lr=1e-4)
        self.checkpoint_path = self.cfg['save_path']

        
    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints)==0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0
        path = os.path.join(self.checkpoint_path, 'latest.tar')


        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
        self.optimizer_latent.load_state_dict(checkpoint['optimizer_lat_state_dict'])
        self.latent_idx.load_state_dict(checkpoint['latent_idx_state_dict'])
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

        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_cameranet.param_groups:
                param_group["lr"] = lr
            
    
    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest.tar'
            torch.save({'epoch': epoch,
                        'encoder3d_state_dict': self.encoder_3d.state_dict(),
                        'cameranet_state_dict': self.cameranet.state_dict(),
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_encoder3d_state_dict': self.optimizer_encoder3d.state_dict(),
                        'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                        'optimizer_cameranet_state_dict': self.optimizer_cameranet.state_dict(),},
                       path)
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'encoder3d_state_dict': self.encoder_3d.state_dict(),
                        'cameranet_state_dict': self.cameranet.state_dict(),
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_encoder3d_state_dict': self.optimizer_encoder3d.state_dict(),
                        'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                        'optimizer_cameranet_state_dict': self.optimizer_cameranet.state_dict(),},
                       path)


    def train(self, epochs):
        loss = 0
        if args.continue_train:
            start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        ckp_vis = self.cfg['ckpt_vis']
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict, texture_gt, texture_pred = self.train_step(batch, epoch,args)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                if args.use_wandb:
                    wandb.log(loss_values)
                    wandb.log({'texture_gt': wandb.Image(texture_gt),
                               'texture_pred': wandb.Image(texture_pred)})
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0 and epoch >0:
                self.save_checkpoint(epoch, save_name=CFG['training']['save_name'])
            # save as latest
            self.save_checkpoint(epoch, save_name='latest')
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)

    def train_step(self, batch, epoch, args):
        self.optimizer_cameranet.zero_grad()
        self.generator.zero_grad()
        loss_dict, texture_gt, texture_pred = texture_loss(batch, cameranet=self.cameranet, encoder_3d=self.encoder_3d,
                                 decoder_deform=self.decoder_deform, decoder_shape=self.decoder_shape,
                                 latent_deform=self.latent_deform, latent_shape=self.latent_shape,
                                 encoder_2d=self.encoder_2d, 
                                 epoch=epoch, cfg=self.cfg,
                                 generator=self.generator, renderer=self.renderer,
                                 device=self.device)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder_3d.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.cameranet.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.cfg['grad_clip'])
        self.optimizer_cameranet.step()
        self.optimizer_generator.step()

        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}

        loss_dict.update({'loss': loss_total.item()})
           

        return loss_dict    , texture_gt, texture_pred

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='texture', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--save_mesh', action='store_true', help='save mesh')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    if args.use_wandb:
        wandb.init(project='NPLM', name =args.wandb)
        wandb.config.update(CFG)
    
    # dataset
    trainset = Point_cloud_dataset(mode='texture')
    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=0)
    print('data loaded: {} samples'.format(len(trainloader)))
    # model initialization
    encoder_3d = PCAutoEncoder(point_dim=3)
    encoder_3d.to(device)
    encoder_3d.train()
    
    encoder_2d = ViTModel.from_pretrained('facebook/dino-vitb16')
    encoder_2d.to(device)
    encoder_2d.eval()
    
    cameranet = CameraNet(feature_dim=768, hidden_dim=512)
    cameranet.to(device)
    cameranet.train()
    
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
    
    trainer = TextureTrainer(encoder_3d=encoder_3d, encoder_2d=encoder_2d,
                            cameranet=cameranet, trainloader=trainloader,
                            decoder_shape=decoder_shape, decoder_deform=decoder_deform,
                            latent_deform=lat_deform_all, latent_shape=lat_idx_all,
                            generator=generator, 
                            cfg=CFG, device=device)
    trainer.train(10001)
    