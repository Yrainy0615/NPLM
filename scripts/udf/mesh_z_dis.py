import trimesh
from trimesh.sample import sample_surface
import numpy as np

filename = ''
neutral_mesh = trimesh.load(filename, force='mesh')
samples, _, colors = sample_surface(neutral_mesh, 5000, sample_color=True)
# generate a z-dim displacement field from random
displacement = np.random.normal(0, 0.01, samples.shape[0])
# apply displacement to z-dim
samples[:, 2] += displacement
pcd = trimesh.points.PointCloud(samples, colors.astype(np.uint8))
pcd.export("pcd.ply")
pass
