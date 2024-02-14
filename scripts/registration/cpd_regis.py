import copy
import numpy as np
import open3d as o3
from probreg import cpd
import trimesh
import pandas as pd

# load source and target point cloud
source_dir = 'dataset/leaf_classification/images/Quercus_Phillyraeoides/43_128.obj'
source  = trimesh.load(source_dir)
target_path = 'dataset/soybean/annotation/20180612_DN251/Annotations/leaf_1.txt'
data = pd.read_csv(target_path, sep=' ', header=None,
                 names=['x', 'y', 'z', 'r', 'g', 'b'],
                 dtype={'x': float, 'y': float, 'z': float, 'r': int, 'g': int, 'b': int})

target = data[['x', 'y', 'z']].values
# transform target point cloud



# compute cpd registration
tf_param, _, _ = cpd.NonRigidCPD(source.vertices)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])