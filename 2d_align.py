from scripts.model.renderer import MeshRender
import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply, save_obj
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
import cv2
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance
from matplotlib import pyplot as plt
from pytorch3d.renderer import TexturesVertex

class align_mask():
    def __init__(self, renderer, device) -> None:
        self.renderer = renderer
        self.template = load_ply('dataset/leaf_uv_5sp.ply')
        self.device = device
       # self.shape_decoder = decoder
        
        #self.optimizer = torch.optim.Adam(self.shape_decoder.parameters(), lr=0.01)
        
    def align(self,mask):
        verts_t, face_t = self.template[0], self.template[1]
        # move to center
        verts_t = verts_t - verts_t.mean(0)
        verts_t = verts_t/verts_t.abs().max()
        # copy a verts_t for optimization
        
        # template = Meshes(verts=[verts_t], faces=[face_t])
        # template = template.to(self.renderer.device)
        mask_temp = torch.from_numpy(mask).float().to(self.renderer.device)
        mask_tensor = mask_temp.unsqueeze(0)/255
        # silhouette loss
        delta_x = torch.nn.Parameter(torch.randn_like(verts_t[:, :2], device=self.device) * 0.01)
        optimizer = torch.optim.Adam([delta_x], lr=0.4)


        for i in range(1000):
            optimizer.zero_grad()
            # create a copy of verts_t
            deform_verts = verts_t.clone().to(self.device)
            #deform_verts = torch.from_numpy(verts_t).float().to(self.device)
            deform_verts[:, :2] = deform_verts[:, :2] + delta_x
            deform_verts = deform_verts.to(self.device)
        
            #chamfer = chamfer_distance(deform_verts.unsqueeze(0), verts_t.unsqueeze(0).to(self.device))            # Create a new mesh with the updated vertices
            mesh = Meshes(verts=[deform_verts], faces=[face_t.to(self.device)])
            mesh = mesh.to(self.renderer.device)
            # generate  a random texture
            N = mesh.verts_packed().shape[0]
            vertex_colors = torch.ones((N, 3), device=self.renderer.device)            
            texture = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))
            mesh.textures = texture
            # Compute the predicted mask
            img_render = self.renderer.renderer(mesh) # [1,64,64,4]
            pred = img_render.detach().cpu().float().squeeze().numpy()
            pred_clipped = np.clip(pred, 0, 1)

            plt.imsave('pred.png', pred_clipped[:,:,:3])
            # Compute the silhouette loss
            loss_silh = torch.mean((mask_tensor[:, :, :, 0] - img_render[:,:,:,0])**2)
           # loss_lap = mesh_laplacian_smoothing(mesh, method="uniform")

            loss =loss_silh # + loss_lap
            print("loss: ", loss.item())
            loss.backward()
            optimizer.step()
            
        mesh = Meshes(verts=[deform_verts], faces=[face_t])         
        return mesh                                              
        



if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.autograd.set_detect_anomaly(True)
    renderer  = MeshRender(device)
    aligner = align_mask(renderer,device)
    mask = cv2.imread('dataset/LeafData/Lemon/healthy/Lemon_healthy_0002_mask_aligned.JPG')
    mask = cv2.resize(mask, (64,64))
    mesh_deformed = aligner.align(mask)
    save_obj('deformed.obj', mesh_deformed.verts_packed(), mesh_deformed.faces_packed())
    pass