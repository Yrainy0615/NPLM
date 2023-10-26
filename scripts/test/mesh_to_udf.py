import trimesh
import mesh2sdf
import time
import numpy as np
from scripts.model.reconstruction import mesh_from_logits, create_grid_points_from_bounds
from scripts.model.diff_operators import gradient   
from MeshUDF.custom_mc._marching_cubes_lewiner import udf_mc_lewiner
import torch
import skimage.measure

filename = 'dataset/ScanData/canonical/ash_canonical.obj'
mesh = trimesh.load(filename, force='mesh')
mesh_scale = 0.8
size = 128
level = 2 / size
# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
#scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) 

# fix mesh
t0 = time.time()
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
# calculate sdf normal

t1 = time.time()
udf = np.abs(sdf)
mini = [-.95, -.95, -.95]
maxi = [0.95, 0.95, 0.95]
# grid_points = create_grid_points_from_bounds(mini, maxi, 128)
# sdf_tensor = torch.tensor(sdf,requires_grad=True)
# grid_points_tensor = torch.tensor(grid_points,requires_grad=True)
# normal = gradient(sdf_tensor.view(-1,1), grid_points_tensor)
#mesh_sdf = mesh_from_logits(sdf, mini, maxi, 128)
verts, faces,_,_ = skimage.measure.marching_cubes(udf, 0.015)

udf_mesh = trimesh.Trimesh(verts, faces)
pass