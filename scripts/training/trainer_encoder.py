import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import Voxel_dataset, custom_collate_fn
from torch.utils.data import DataLoader
from scripts.model.point_encoder import PCAutoEncoder
from scripts.model.fields import UDFNetwork
import yaml
import glob
import wandb
from scripts.model.loss_functions import rgbd_loss
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender
from scripts.model.inference_encoder import ShapeEncoder, PoseEncoder, CameraEncoder
from PIL import Image

class VoxelTrainer(object):
    def __init__(self, encoder_shape, encoder_pose, encoder_2d,
                 encoder_camera, trainloader, 
                 latent_shape, latent_deform,
                 decoder_shape, decoder_deform,
                 generator,
                 cfg, device):
        self.encoder_shape = encoder_shape
        self.encoder_pose = encoder_pose
        self.encoder_2d = encoder_2d
        self.encoder_camera = encoder_camera
        self.trainloader = trainloader
        self.decoder_shape = decoder_shape
        self.decoder_deform = decoder_deform
        self.device = device
        self.generator = generator
        self.renderer = MeshRender(device=device)
        self.cfg = cfg['training']
        self.latent_shape = latent_shape
        self.latent_deform = latent_deform
        self.optimizer_encoder_shape = optim.Adam(self.encoder_shape.parameters(), lr=1e-4)
        self.optimizer_encoder_pose = optim.Adam(self.encoder_pose.parameters(), lr=1e-4)
        self.optimizer_encoder_camera = optim.Adam(self.encoder_camera.parameters(), lr=1e-4)
        self.checkpoint_path = self.cfg['save_path']
        self.renderer = MeshRender(device=device)

        
    def load_checkpoint(self):
        path = os.path.join(self.checkpoint_path, 'latest.tar')
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.encoder_shape.load_state_dict(checkpoint['encoder_shape__state_dict'])
        self.encoder_pose.load_state_dict(checkpoint['encoder_pose_state_dict'])
        self.encoder_camera.load_state_dict(checkpoint['encoder_camera_state_dict'])
        self.optimizer_encoder_camera.load_state_dict(checkpoint['optimizer_encoder_camera_state_dict'])
        self.optimizer_encoder_shape.load_state_dict(checkpoint['optimizer_encoder_shape_state_dict'])
        self.optimizer_encoder_pose.load_state_dict(checkpoint['optimizer_encoder_pose_state_dict'])
        for param_group in self.optimizer_encoder3d.param_groups:
            param_group["lr"] = self.cfg['lr']
        epoch = checkpoint['epoch']
        
        return epoch

    def reduce_lr(self, epoch):
        if self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            lr_cam = self.cfg['lr_cam'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder_pose.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_encoder_shape.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_encoder_camera.param_groups:
                param_group["lr"] = lr


    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest.tar'
            torch.save({'epoch': epoch,
                        'encoder_shape_state_dict': self.encoder_shape_.state_dict(),
                        'encoder_pose_state_dict': self.encoder_pose.state_dict(),
                        'encoder_camera_state_dict': self.encoder_camera.state_dict(),
                        'optimizer_encoder_pose_state_dict': self.optimizer_encoder_pose.state_dict(),
                        'optimizer_encoder_shape_state_dict': self.optimizer_encoder_shape.state_dict(),
                        'optimizer_encoder_camera_state_dict': self.optimizer_encoder_camera.state_dict(),},
                       path)
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'encoder_shape_state_dict': self.encoder_shape_.state_dict(),
                        'encoder_pose_state_dict': self.encoder_pose.state_dict(),
                        'encoder_camera_state_dict': self.encoder_camera.state_dict(),
                        'optimizer_encoder_pose_state_dict': self.optimizer_encoder_pose.state_dict(),
                        'optimizer_encoder_shape_state_dict': self.optimizer_encoder_shape.state_dict(),
                        'optimizer_encoder_camera_state_dict': self.optimizer_encoder_camera.state_dict(),},
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
                loss_dict,canonical_mask_pred, canonical_mask_gt= self.train_step(batch, epoch,args)
                combined_image = Image.new('L', (canonical_mask_pred.shape[2] * 2, canonical_mask_pred.shape[1]))
                gt_image = Image.fromarray(canonical_mask_gt[0])  # 假设是单通道的，取第0个通道
                pred_image = Image.fromarray(canonical_mask_pred[0])

                combined_image.paste(gt_image, (0, 0))
                combined_image.paste(pred_image, (gt_image.width, 0))
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                if args.use_wandb:
                    wandb.log(loss_values)
                    if canonical_mask_gt is not None:
                        wandb.log({'mask': wandb.Image(combined_image)})
                                
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
        self.optimizer_encoder_pose.zero_grad()
        self.optimizer_encoder_shape.zero_grad()
        self.optimizer_encoder_camera.zero_grad()
        loss_dict, canonical_mask_pred, canonical_mask_gt = rgbd_loss(batch, self.encoder_shape, self.encoder_pose, self.encoder_camera,
                        self.latent_shape, self.latent_deform, 
                        self.decoder_shape, self.decoder_deform, self.renderer,
                        device=self.device)
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()
        
        
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder_shape.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.encoder_pose.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.encoder_camera.parameters(), max_norm=self.cfg['grad_clip'])
        self.optimizer_encoder_shape.step()
        self.optimizer_encoder_pose.step()
        self.optimizer_encoder_camera.step()

        loss_dict = {k: loss_dict[k].item() for k in loss_dict.keys()}

        loss_dict.update({'loss': loss_total.item()})
        return loss_dict  , canonical_mask_pred, canonical_mask_gt, input_rgb, canonical_rgb , canonical_rotated

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='inference', help='run name of wandb')
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
    
    
    # dataset
    trainset = Voxel_dataset(mode='shape')
    trainloader = DataLoader(trainset, batch_size=CFG['training']['batch_size'], shuffle=True, num_workers=2)
    print('data loaded: {} samples'.format(len(trainloader)))
    # model initialization
    # encoder_3d = PCAutoEncoder(point_dim=3)
    # encoder_3d.to(device)
    # encoder_3d.train(
    encoder_shape  = ShapeEncoder(output_dim=lat_idx_all.shape[0])
    encoder_pose = PoseEncoder(output_dim=lat_idx_all.shape[0])
    encoder_shape = encoder_shape.to(device)
    encoder_pose = encoder_pose.to(device)
    encoder_shape.train()
    encoder_pose.train()
    
    encoder_2d = ViTModel.from_pretrained('facebook/dino-vitb16')
    encoder_2d.to(device)
    encoder_2d.eval()
    
    encoder_camera = CameraEncoder()
    encoder_camera.to(device)
    encoder_camera.train()
    

    # load generator
    generator = Generator(resolution=256)
    generator.to(device)
    
    trainer = VoxelTrainer(encoder_shape=encoder_shape,
                           encoder_pose=encoder_pose, encoder_2d=encoder_2d,
                            encoder_camera=encoder_camera, trainloader=trainloader,
                            decoder_shape=decoder_shape, decoder_deform=decoder_deform,
                            latent_deform=lat_deform_all, latent_shape=lat_idx_all,
                            generator=generator, 
                            cfg=CFG, device=device)
    trainer.train(10001)
    