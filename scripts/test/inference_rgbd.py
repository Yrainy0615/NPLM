import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import Voxel_dataset, custom_collate_fn
from scripts.dataset.sdf_dataset import EncoderDataset
from torch.utils.data import DataLoader
from scripts.model.point_encoder import PCAutoEncoder, CameraNet
from scripts.model.fields import UDFNetwork
import yaml
import wandb
from transformers import ViTModel
from scripts.model.generator import Generator
from scripts.model.renderer import MeshRender
from scripts.model.reconstruction import sdf_from_latent, latent_to_mesh, deform_mesh
from scripts.model.inference_encoder import ShapeEncoder, PoseEncoder
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, FoVPerspectiveCameras
from matplotlib import pyplot as plt
import imageio  


class Predictor(object):
    def __init__(self, encoder_shape, encoder_pose,encoder_2d, 
                 cameranet, trainloader, 
                 latent_shape, latent_deform,
                 decoder_shape, decoder_deform,
                 generator,
                 cfg, device):
        self.encoder_shape = encoder_shape
        self.encoder_pose = encoder_pose
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
        R, t = look_at_view_transform(2,45, 0)
        self.deform_camera = FoVPerspectiveCameras(device=self.device, R=R, T=t)
    
    def predict(self):
        for i, batch in enumerate(self.trainloader):
            batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            occupancy_grid = batch_cuda['occupancy_grid']
            mesh_file = batch_cuda['mesh_file'][0]
            mesh_name = mesh_file.split('.')[0]
            points = batch_cuda['points']            # encode 3d
            
            # initialization
            latent_shape_pred = self.encoder_shape(occupancy_grid)
            latent_pose_pred = self.encoder_pose(occupancy_grid)
            canonical_mesh = latent_to_mesh(self.decoder_shape, latent_shape_pred, device)
            deformed_mesh = deform_mesh(canonical_mesh, self.decoder_deform, latent_pose_pred)

            canonical_mesh.export('{}_canonical.obj'.format(mesh_name))
            deformed_mesh.export('{}.obj'.format(mesh_name))
            
            # optimization
            latent_shape_optimized, latent_deform_optimized, canonical_imgs, deform_imgs, = self.optim_latent(
                                        latent_shape_pred.detach().requires_grad_(), 
                                        latent_pose_pred.detach().requires_grad_(), points)
            
            # final output
            canonical_mesh_optimized = latent_to_mesh(self.decoder_shape, latent_shape_optimized, device)
            deformed_mesh_optimized = deform_mesh(canonical_mesh_optimized, self.decoder_deform, latent_deform_optimized)
            canonical_mesh_optimized.export('{}_canonical_optimized.obj'.format(mesh_name))
            deformed_mesh_optimized.export('{}_optimized.obj'.format(mesh_name))
            # save gif
            gif_filename = '{}_deform.gif'.format(mesh_name)
            imageio.mimsave(gif_filename, deform_imgs)
            imageio.mimsave('{}_canonical.gif'.format(mesh_name), canonical_imgs)
    
    def optim_latent(self, latent_shape_init, latent_deform_init, points):
        optimizer_shape = optim.Adam([latent_shape_init], lr=1e-3)
        optimizer_deform = optim.Adam([latent_deform_init], lr=1e-3)
        img_nps = []
        deform_nps = []
        for i in range(40):
            optimizer_shape.zero_grad()
            optimizer_deform.zero_grad()
            mesh = latent_to_mesh(self.decoder_shape, latent_shape_init, self.device)
            verts = mesh.vertices
            xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
            delta_verts = self.decoder_deform(torch.from_numpy(verts).float().to(device), latent_deform_init.squeeze(0).repeat(mesh.vertices.shape[0], 1))
            """
            Differentiable Rendering back-propagating to mesh vertices
            """
            # generate a texture for the mesh in green 
            texture  = torch.ones_like(torch.from_numpy(mesh.vertices)).unsqueeze(0)
            texture[..., 0] = 0.0
            texture[..., 1] = 1
            texture[..., 2] = 0.0
            # create a Meshes object for the textured mesh
            textures = TexturesVertex(verts_features=texture.to(device).float())
            canonical_img = self.renderer.render_rgb(Meshes(verts=[xyz_upstream.squeeze()], faces=[torch.tensor(mesh.faces).squeeze().to(device)], textures=textures))
            img_np = canonical_img[:,:,:,:3].detach().squeeze().cpu().numpy()/255
            loss_chamfer = chamfer_distance((xyz_upstream.unsqueeze(0)+delta_verts), points)
            deform_img =  self.renderer.renderer(Meshes(verts=[xyz_upstream.squeeze()+ delta_verts.squeeze()], faces=[torch.tensor(mesh.faces).squeeze().to(device)], textures=textures), camera = self.deform_camera)
            deform_img_np = deform_img[:,:,:,:3].detach().squeeze().cpu().numpy()/255
            loss =loss_chamfer[0]#+torch.norm(latent_source, dim=-1)**2 # +loss_chamfer[0]
            # print losses
            print('shape iter:{}  loss_chamfer: {}'.format(i,loss_chamfer[0]))
            loss.backward()
            optimizer_deform.step()
            # now store upstream gradients
            dL_dx_i = xyz_upstream.grad

            # use vertices to compute full backward pass
            optimizer_shape.zero_grad()
            xyz = torch.tensor(verts, requires_grad = True, dtype=torch.float32, device=torch.device('cuda:0'))
            #first compute normals 
            pred_sdf = decoder_shape(xyz, latent_shape_init.squeeze(0).repeat(xyz.shape[0], 1))
            loss_normals = torch.sum(pred_sdf)
            loss_normals.backward(retain_graph = True)
            # normalization to take into account for the fact sdf is not perfect...
            normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
            # now assemble inflow derivative
            optimizer_shape.zero_grad()
            dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
            # refer to Equation (4) in the main paper
            loss_backward = torch.sum(dL_ds_i * pred_sdf)
            loss_backward.backward()
            # and update params
            optimizer_shape.step()
            img_nps.append(img_np)
            deform_nps.append(deform_img_np)
        return latent_shape_init, latent_deform_init, img_nps, deform_nps

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
    trainset = EncoderDataset(root_dir='results/viz_space')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    
    # networl initialization
    checkpoint_encoder = torch.load('checkpoints/inference/inference_0208.tar')
    encoder_shape = ShapeEncoder()
    encoder_shape.load_state_dict(checkpoint_encoder['encoder_shape_state_dict'])
    encoder_shape.to(device)
    encoder_shape.eval()
    encoder_pose = PoseEncoder()
    encoder_pose.load_state_dict(checkpoint_encoder['encoder_pose_state_dict'])
    encoder_pose.to(device)
    encoder_pose.eval()
    
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
    
    # load generator
    generator = Generator(resolution=256)
    generator.to(device)
    generator.eval()
    # generator.load_state_dict(checkpoint_infer['generator_state_dict'])
    
    # predict
    predictor = Predictor(encoder_shape, encoder_pose, encoder_2d, cameranet, trainloader, lat_idx_all, lat_deform_all, decoder_shape, decoder_deform, generator, CFG, device)
    predictor.predict()
