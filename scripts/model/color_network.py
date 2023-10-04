from .stylesdf.volume_renderer import SirenGenerator
import torch
import torch.nn as nn

class ColorNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        network = SirenGenerator()
        self.views_linears = network.views_linears
        self.rgb_linear = network.rgb_linear
        self.style_dim = cfg['style_dim']   
        self.w_dim = cfg['W']

    def forward(self, points, normals, feature_vectors, z, w):
        ray_bs = points.shape[0]
        bs = w.shape[0]
        feature_vectors = feature_vectors.reshape(bs, ray_bs // bs, 1, 1, *feature_vectors.shape[1:])
        normals = normals.reshape(bs, ray_bs // bs, 1, 1, *normals.shape[1:])

        mlp_out = torch.cat([feature_vectors, normals], -1)
        out_features = self.views_linears(mlp_out, w)
        rgb = self.rgb_linear(out_features)
        rgb = torch.sigmoid(rgb)

        rgb = rgb.flatten(0, 3)
        return rgb
    

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.3):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device='cuda') * torch.exp(self.variance * 10.0)