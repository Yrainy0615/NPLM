import numpy as np
import trimesh
import matplotlib.pyplot as pt

#from mesh_to_depth import log

params = []

params.append({
    'cam_pos': [1, 90, 180], 'cam_lookat': [0, 0, 0], 'cam_up': [0, 1, 0],
    'x_fov': 0.349,  # End-to-end field of view in radians
    'near': 0.1, 'far': 10,
    'height': 256, 'width': 256,
    'is_depth': True,  # If false, output a ray displacement map, i.e. from the mesh surface to the camera center.
})
# Append more camera parameters if you want batch processing.

# Load triangle mesh data. See python/resources/airplane/models/model_normalized.obj
mesh = trimesh.load('sample_result/shape_ep5000/mesh_0001.ply')
vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces.astype(np.uint32)

depth_maps = m2d.mesh_to_depth(vertices, faces, params, empty_pixel_value=np.nan)
pt.imshow(depth_maps[0], interpolation='none')
pt.colorbar()