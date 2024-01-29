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
from scripts.model.loss_functions import rgbd_loss

class PointsTrainer(object):
    def __init__(self, encoder_3d, encoder_2d, 
                 cameranet, trainloader, 
                 latent_shape, latent_deform,
                 decoder_shape, decoder_deform,
                 cfg, device):
        self.encoder_3d = encoder_3d
        self.encoder_2d = encoder_2d
        self.cameranet = cameranet
        self.trainloader = trainloader
        self.decoder_shape = decoder_shape
        self.decoder_deform = decoder_deform
        self.device = device
        self.cfg = cfg['training']
        self.latent_shape = latent_shape
        self.latent_deform = latent_deform
        self.optimizer_encoder3d = optim.Adam(self.encoder_3d.parameters(), lr=1e-3)
        self.optimizer_cameranet = optim.Adam(self.cameranet.parameters(), lr=1e-3)
        
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
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder3d.param_groups:
                param_group["lr"] = lr

        if epoch > 1000 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_cameranet.param_groups:
                param_group["lr"] = lr
            
            # if self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            #     decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            #     lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            #     print('Reducting LR for latent codes to {}'.format(lr))
            #     for param_group in self.generator.param_groups:
            #         param_group["lr"] = lr
    
    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest.tar'
            torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                        'latent_idx_state_dict': self.latent_idx.state_dict(),
                       },
                       path)
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),
                        'optimizer_lat_state_dict': self.optimizer_latent.state_dict(),
                        'latent_idx_state_dict': self.latent_idx.state_dict(),},
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
                loss_dict = self.train_step(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                if args.use_wandb:
                    wandb.log(loss_values)
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

    def train_step(self, batch):
        self.optimizer_encoder3d.zero_grad()
        self.optimizer_cameranet.zero_grad()
        loss_dict = rgbd_loss(batch, cameranet=self.cameranet, encoder_3d=self.encoder_3d,
                                 decoder_deform=self.decoder_deform, decoder_shape=self.decoder_shape,
                                 latent_deform=self.latent_deform, latent_shape=self.latent_shape,
                                 encoder_2d=self.encoder_2d, 
                                 device=self.device)
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

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='2d_sdf', help='run name of wandb')
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
    trainset = Point_cloud_dataset()
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    # model initialization
    encoder_3d = PCAutoEncoder(point_dim=3)
    encoder_3d.to(device)
    encoder_3d.train()
    
    encoder_2d = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    encoder_2d.to(device)
    encoder_2d.eval()
    
    cameranet = CameraNet(feature_dim=512, hidden_dim=512)
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
    checkpoint = torch.load('checkpoints/3dShape/latest_3d_0126.tar')
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
    
    trainer = PointsTrainer(encoder_3d=encoder_3d, encoder_2d=encoder_2d,
                            cameranet=cameranet, trainloader=trainloader,
                            decoder_shape=decoder_shape, decoder_deform=decoder_deform,
                            latent_deform=lat_deform_all, latent_shape=lat_idx_all,
                            cfg=CFG, device=device)
    trainer.train(10001)
    