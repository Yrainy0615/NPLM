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
    AlphaCompositor
)
from pytorch3d.io import load_objs_as_meshes, load_obj
import torch
from pytorch3d.structures import Meshes,  Pointclouds, Volumes
from pytorch3d.loss import mesh_laplacian_smoothing
from matplotlib import pyplot as plt
import numpy as np
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.ops import add_pointclouds_to_volumes
import open3d as o3d
import mcubes
from pytorch3d.ops.marching_cubes import marching_cubes_naive

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
        
        raster_settings = PointsRasterizationSettings(
        image_size=256, 
        radius = 0.01,
        points_per_pixel = 5
    )
        # PointsRasterizer
        rasterizer_point = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.point_renderer = PointsRenderer(
            rasterizer=rasterizer_point,
            compositor=AlphaCompositor(background_color=(0,0,0))
        ).cuda()
        
        
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


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    meshfile = 'dataset/ScanData/Autumn_maple_leaf.001.obj'
    mesh = load_objs_as_meshes([meshfile],device=device)
    renderer  = MeshRender(device=device)
    #fragments = renderer.rasterize(mesh)

    mask = renderer.get_mask(mesh)
    depth = renderer.get_depth(mesh)
    # renderer.viz_depth(depth)
    #sdf_grid = renderer.depth_sdf(depth)
   # K = renderer.get_intrinsic()
    point_cloud = renderer.depth_pointcloud(depth)
   # images_PC = renderer.point_renderer(point_cloud)
   # images_PC = images_PC.detach().cpu().squeeze(0).numpy()
    volume = renderer.pts_volume(point_cloud)
    coords = volume.get_coord_grid().view(1,-1,3).detach().cpu().numpy()
   # point_cloud = renderer.depth_pts(depth)
    #point_cloud=point_cloud.view(-1,3)
   # verts , faces = marching_cubes_naive(coords)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], alpha=0.6, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    fig = plot_scene({
    "Pointcloud": {
        "person": coords
    }
})
    fig.show()
    pass