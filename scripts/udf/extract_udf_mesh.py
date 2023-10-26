import torch
from scripts.model.deepSDF import DeepSDF
import yaml
from MeshUDF.optimize_chamfer_A_to_B import get_mesh_udf_fast
import os
from scripts.model.reconstruction import latent_to_mesh


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open('NPLM/scripts/configs/npm.yaml', 'r'))
    decoder_shape = DeepSDF(
            lat_dim=CFG['decoder']['decoder_lat_dim'],
            hidden_dim=CFG['decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
            map=True
        )
    checkpoint_shape = torch.load('checkpoints/cg_bs1/cgshape_epoch_30000.tar')
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder_shape.eval()
    decoder_shape.to(device)
    lat_idx_all = checkpoint_shape['latent_idx_state_dict']['weight']
    lat_spc_all = checkpoint_shape['latent_spc_state_dict']['weight']
    
    # extract mesh from udf
    latent = torch.concat([lat_idx_all[0], lat_spc_all[0]]).to(device)
    #latent = lat_idx_all[0].to(device)
    _, _, udf_mesh,_,_ = get_mesh_udf_fast(decoder_shape, latent.unsqueeze(0))
    sdf_mesh = latent_to_mesh(decoder_shape, latent, device)
    pass
    