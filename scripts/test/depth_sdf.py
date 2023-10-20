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
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing

def mesh_to_depth(mesh, renderer):
   

if __name__ == "__main__":
    file = 'dataset/ScanData/maple/Autumn_maple_leaf.004.obj'
    MeshRenderer
