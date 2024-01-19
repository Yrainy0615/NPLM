from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    PointsRasterizationSettings,
    PointsRasterizer,
    blending,
    Textures,
    PointsRenderer,
    AlphaCompositor,
    SoftSilhouetteShader,
    softmax_rgb_blend
)
import torch.nn.functional as F
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
import os
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
import torch
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
from pytorch3d.loss import mesh_laplacian_smoothing
from matplotlib import pyplot as plt
import numpy as np
from pytorch3d.ops import add_pointclouds_to_volumes
from pytorch3d.ops.marching_cubes import marching_cubes_naive

class MeshRender():
    def __init__(self,device):
        R, t = look_at_view_transform(2,0, 0)
        self.R = torch.nn.Parameter(R)
        self.t = torch.nn.Parameter(t)
        self.device = device
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                            ambient_color=[[1, 1, 1]],
                            specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        raster_settings = RasterizationSettings(
            image_size=1024,
            blur_radius=0.0,
            faces_per_pixel=1,)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=t)
        blend_params = blending.BlendParams(background_color=[0, 0, 0])
        self.renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=lights,
            blend_params=blend_params
        )
    )
        raster_settings_silhouette = RasterizationSettings(
        image_size=64, 
        blur_radius=0, 
        faces_per_pixel=1, )
        self.renderer_silhouette = MeshRenderer( 
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()        )
        #self.render = MeshRenderer(rasterizer=raster_settings, shader=self.shader)
        
    def render_rgb(self, mesh):
        # if mesh.textures is None, generates a texture for each face
        if mesh.textures is None:
            mesh.textures = Textures(verts_rgb=torch.ones_like(mesh.verts_padded()))
        return self.renderer(mesh)
    
    def rasterize(self, mesh):
        mesh.to(self.device)
        return self.renderer.rasterizer(mesh)
    
    def get_mask(self, mesh):
        fragments = self.renderer.rasterizer(mesh)
        mask = fragments.zbuf > 0
        #mask = mask.detach().cpu().squeeze(0).numpy()
        return mask
    
    def get_depth(self, mesh):
        fragments = self.renderer.rasterizer(mesh)
        depth = fragments.zbuf
        return depth
    
    def get_intrinsic(self):
        return self.renderer.rasterizer.cameras.get_projection_transform()._matrix
    
    def viz_depth(self,depth_data,):
        depth_data = depth_data.detach().cpu().squeeze(0).numpy()
        plt.imshow(depth_data, cmap='gray')  # 使用灰度图
        plt.colorbar()
        plt.title("Depth Visualization")
        plt.show()
    
    
    def depth_pointcloud(self, depth):
        depth = depth.detach().cpu().squeeze().numpy()
        fov =60
        width , height = depth.shape[0], depth.shape[1]
        cx = width / 2
        cy = height / 2
        fx = cx / np.tan(fov / 2)
        fy = cy / np.tan(fov / 2)

        row = height
        col = width
        # TODO check whether u or v is the column. depth[v, u] ???
        v, u = np.indices((row, col))

        # Calculate X_ and Y_ coordinates
        X_ = (u - cx) / fx
        Y_ = (v - cy) / fy * depth

        # Create a mask to exclude infinity values from depth
        mask = depth > -1

        # Apply mask to X_, Y_, and depth
        X_ = X_[mask]
        Y_ = Y_[mask]
        depth_ = depth[mask]

        X = X_ * depth_
        Y = Y_ * depth_
        Z = depth_

        coords_g = np.stack([X, Y, Z])  # shape: num_points * 3
        coords_g = torch.tensor(coords_g, dtype=torch.float32, device=self.device)
        point_cloud = Pointclouds(points=coords_g.permute(1,0).unsqueeze(0), 
                                  features=torch.rand(1,coords_g.shape[1], 5).to(self.device))
        return point_cloud
    
    def pts_volume(self, pts):
        initial_volumes = Volumes(features=torch.zeros(1,5,256,256,256),
                                  densities=torch.zeros(1,1,256,256,256),
                                  volume_translation=[-0.5, -0.5, -0.5],
                                  voxel_size=1/256)
        
        updated_volumes = add_pointclouds_to_volumes(
                        pointclouds=pts,
                        initial_volumes=initial_volumes.to(self.device),
                        mode='trilinear',)
        return updated_volumes
    
    def phong_normal_shading(self, meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        return pixel_normals
    
    def render_normal(self, mesh):
        # if mesh.textures is None, generates a texture for each face
        if mesh.textures is None:
            mesh.textures = Textures(verts_rgb=torch.ones_like(mesh.verts_padded()))
        fragments = self.renderer.rasterizer(mesh)
        normal = self.phong_normal_shading(mesh, fragments)
        images = softmax_rgb_blend(normal, fragments, blend_params=blending.BlendParams(background_color=[128, 128, 255]))
        normals = F.normalize(images, dim=3)
        img = normals.detach().cpu().squeeze().numpy()
        img = (img + 1) / 2.0
        return img
        


if  __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    mesh_path = 'dataset/ScanData/canonical'
    meshlist = os.listdir(mesh_path)
    #meshfile = [os.path.join(mesh_path, file) for file in meshlist if file.endswith('.obj')]
    #meshfile = 'dataset/leaf_uv_5sp.ply'
    meshfile = 'test.ply'
    verts, faces = load_ply(meshfile)
    verts = verts - verts.mean(0)
    verts = verts/verts.abs().max()
    mesh = Meshes(verts=[verts], faces=[faces], textures=None)
    mesh = mesh.to(device)
    if mesh.textures is None:
        mesh.textures = Textures(verts_rgb=torch.ones_like(mesh.verts_padded()))
    renderer  = MeshRender(device=device)
    # add a z-axis displacement to mesh 1000 of vertex
    index = torch.randint(0, len(mesh.verts_padded()), (1000,))
    mask = renderer.renderer_silhouette(mesh)
    mask = mask.detach().cpu().squeeze().numpy()
    plt.imsave('target.png',mask[:,:,:])
    mesh.verts_padded()[:, :, 2][index] += 0.05
    normals = renderer.render_normal(mesh)
    plt.imshow(normals)
    plt.show()
    # fragments = renderer.rasterize(mesh)
    # mask  =renderer.get_mask(mesh)
    # normal = renderer.phong_normal_shading(mesh, fragments)
    # images = softmax_rgb_blend(normal, fragments, blend_params=blending.BlendParams(background_color=[128, 128, 255]))
    # normals = F.normalize(images, dim=3)
    # # save each image
    # for i in range(len(normals)):
    #     img = normals[i].detach().cpu().squeeze().numpy()

    #     normal = (img + 1) / 2.0

    #     plt.imsave(os.path.join('dataset/ScanData/rgb', meshlist[i].replace('.obj','.png')), normal )
    #     pass
    