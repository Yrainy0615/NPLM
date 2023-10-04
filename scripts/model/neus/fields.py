import os.path
import torch.distributed as dist
import torch.nn as nn
import torch
from ..stylesdf.volume_renderer import SirenGenerator
from ..stylesdf.model import MappingLinear

class ShapeNetwork(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super().__init__()

        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(kwargs['style_dim'], kwargs['style_dim'], activation="leaky_relu")
            )
        style = nn.Sequential(*layers)
        network = SirenGenerator(**kwargs)
        self.style = style
        self.pts_linears = network.pts_linears
        self.sigma_linear = network.sigma_linear

        if checkpoint_path is not None:
            # logger.info(f'loading from {checkpoint_path}')

            # hack
            if dist.is_initialized():
                device = f'cuda:{dist.get_rank()}'
                self.to(device)
            else:
                device = 'cuda'
            # logger.info(f'loading to device: {device}')
            state_dict = torch.load(checkpoint_path, map_location=device)
            # check_cfg_consistency(kwargs, state_dict['cfg']['model']['generator']['kwargs']['sdf_network']['kwargs'],
            #                       ignore_keys=['checkpoint_path',])
            self.load_state_dict(state_dict['sdf_network'])

        # g = nn.Module()
        # g.style = style
        # g.renderer = nn.Module()
        # g.renderer.network = network
        #
        # path = './intrinsics/third_party/stylesdf/pretrained_renderer/sphere_init.pt'
        # state_dict = torch.load(path)
        # g.load_state_dict(state_dict['g'], strict=False)

    def forward(self, x, z, w=None):
        ray_bs = x.shape[0]
        if w is not None:
            bs = w.shape[0]
        else:
            bs = z.shape[0]
        x = x.reshape(bs, ray_bs//bs, 1, 1, *x.shape[1:])

        if w is None:
            latent = self.style(z)
        else:
            latent = w

        mlp_out = x.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, latent)

        sdf = self.sigma_linear(mlp_out)
        # print(torch.linalg.norm(sdf[:, :2], dim=-1).flatten(), torch.linalg.norm(x[:, :2], dim=-1).flatten())

        outputs = torch.cat([sdf, mlp_out], -1)
        return outputs.flatten(0, 3)

    def sdf(self, x, z, w=None):
        return self.forward(x, z=z, w=w)[:, :1]

    def gradient(self, x, z, w=None, second_order=False):
        return gradient(x, lambda pts: self.sdf(pts, z=z, w=w).squeeze(-1),
                        second_order=second_order)