import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import Voxel_dataset, custom_collate_fn
from torch.utils.data import DataLoader
from scripts.model.point_encoder import PCAutoEncoder, CameraNet
from scripts.model.fields import UDFNetwork
import yaml
import wandb
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender
from scripts.model.reconstruction import sdf_from_latent, latent_to_mesh, deform_mesh


class Predictor(object):
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
    
    def predict(self):
        for i, batch in enumerate(self.trainloader):
            batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            points = batch_cuda['points'][0]
            points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
            rgb =batch_cuda['rgb']
            deform_name = batch_cuda['deformed_name'][0]
            latent_shape_gt = self.latent_shape[int(batch_cuda['shape_index'][0])]
            latent_deform_gt = self.latent_deform[int(batch_cuda['deform_index'][0])]
            input = batch_cuda['inputs'].to(device)
            # encode 3d
            latent_shape_pred, latent_deform_pred = self.encoder_3d(points_tensor.unsqueeze(0).permute(0,2,1))
            canonical_gt = latent_to_mesh(self.decoder_shape, latent_shape_gt, device)
            canonical_mesh = latent_to_mesh(self.decoder_shape, latent_shape_pred, device)
            deformed_mesh = deform_mesh(canonical_mesh, self.decoder_deform, latent_deform_pred)
            # encode 2d
            deformed_gt =deform_mesh(canonical_gt, self.decoder_deform, latent_deform_gt)
            canonical_mesh.export('{}_canonical.obj'.format(deform_name))
            canonical_gt.export('{}_canonical_gt.obj'.format(deform_name))
            deformed_mesh.export('{}.obj'.format(deform_name))
            deformed_gt.export('{}_gt.obj'.format(deform_name))

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
    
    # dataset
    trainset = Point_cloud_dataset(mode='shape')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    
    checkpoint_infer = torch.load('checkpoints/inference/latest.tar')
    # model initialization
    encoder_3d = PCAutoEncoder(point_dim=3)
    encoder_3d.to(device)
    encoder_3d.eval()
    encoder_3d.load_state_dict(checkpoint_infer['encoder3d_state_dict'])
    
    encoder_2d = ViTModel.from_pretrained('facebook/dino-vitb16')
    encoder_2d.to(device)
    encoder_2d.eval()
    
    cameranet = CameraNet(feature_dim=768, hidden_dim=512)
    cameranet.to(device)
    cameranet.eval()
    # cameranet.load_state_dict(checkpoint_infer['cameranet_state_dict'])
    
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
    generator.eval()
    # generator.load_state_dict(checkpoint_infer['generator_state_dict'])
    
    # predict
    predictor = Predictor(encoder_3d, encoder_2d, cameranet, trainloader, lat_idx_all, lat_deform_all, decoder_shape, decoder_deform, generator, CFG, device)
    predictor.predict()
