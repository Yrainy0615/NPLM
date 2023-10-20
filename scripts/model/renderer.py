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
    blending,
    Textures
)
from pytorch3d.io import load_objs_as_meshes, load_obj
import torch
from pytorch3d.structures import Meshes,  Pointclouds
from pytorch3d.loss import mesh_laplacian_smoothing
from matplotlib import pyplot as plt
import numpy as np
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.ops import add_pointclouds_to_volumes



class MeshRender():
    def __init__(self,device):
        R, t = look_at_view_transform(1, 0, 0)
        self.R = torch.nn.Parameter(R)
        self.t = torch.nn.Parameter(t)
        self.device = device
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                            ambient_color=[[1, 1, 1]],
                            specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=t)
        blend_params = blending.BlendParams(background_color=[255, 255, 255])
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
    def render_rgb(self, mesh):
        return self.renderer(mesh)
    
    def rasterize(self, mesh):
        mesh.to(self.device)
        return self.renderer.rasterizer(mesh)
    
    def get_mask(self, mesh):
        fragments = self.renderer.rasterizer(mesh)
        mask = fragments.zbuf > 0
        mask = mask.detach().cpu().squeeze(0).numpy()
        return mask
    
    def get_depth(self, mesh):
        fragments = self.renderer.rasterizer(mesh)
        depth = fragments.zbuf
        return depth
    
    def get_intrinsic(self):
        return self.renderer.rasterizer.cameras.get_projection_transform()._matrix
    
    def viz_depth(depth_data,):
        plt.imshow(depth_data, cmap='gray')  # 使用灰度图
        plt.colorbar()
        plt.title("Depth Visualization")
        plt.show()
    
    def depth_pts(self, depth):
        width , height = depth.shape[0], depth.shape[1]
        depth = depth.detach().cpu().squeeze().numpy()
        fov =60
        cx = width / 2
        cy = height / 2
        fx = cx / np.tan(fov / 2)
        fy = cy / np.tan(fov / 2)

        row = height
        col = width
        # TODO check whether u or v is the column. depth[v, u] ???
        u = np.array(list(np.ndindex((row, col)))).reshape(row, col, 2)[:, :, 0].squeeze()
        v = np.array(list(np.ndindex((row, col)))).reshape(row, col, 2)[:, :, 0].squeeze()

        X_ = (u - cx) / fx
        X_ = X_[depth > -1]  # exclude infinity
        Y_ = (v - cy) / fy * depth
        Y_ = Y_[depth > -1]  # exclude infinity
        depth_ = depth[depth > -1]  # exclude infinity

        X = X_ * depth_
        Y = Y_ * depth_
        Z = depth_

        coords_g = np.stack([X, Y, Z])  # shape: num_points * 3
        pcd = Pointclouds(points=[coords_g])
        # pcd.points = o3d.utility.Vector3dVector(coords_g.T)
        # o3d.visualization.draw_geometries([pcd])
        

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    meshfile = '/home/yang/projects/parametric-leaf/dataset/ScanData/maple/Autumn_maple_leaf.005.obj'
    mesh = load_objs_as_meshes([meshfile],device=device)
    renderer  = MeshRender(device=device)
    #fragments = renderer.rasterize(mesh)

    mask = renderer.get_mask(mesh)
    depth = renderer.get_depth(mesh)
    #viz_depth(depth)
    #sdf_grid = renderer.depth_sdf(depth)
   # K = renderer.get_intrinsic()
    #point_cloud = renderer.fast_from_depth_to_pointcloud(depth)
    point_cloud = renderer.depth_pts(depth)
    fig = plot_scene({
    "Pointcloud": {
        "person": Pointclouds([point_cloud])
    }
})
    fig.show()
    pass