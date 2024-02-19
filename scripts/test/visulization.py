import numpy as np
import open3d as o3d
import os
import trimesh
import pandas as pd

root = 'dataset/soybean'
model_dir = os.path.join(root, 'model')
annotation_dir = os.path.join(root, 'annotation')
all_plants  = [f for f in os.listdir(annotation_dir) ]
plant = []
plant_color = []
instance_dir = '/home/yang/projects/parametric-leaf/dataset/soybean/annotation/20180619_HN48/Annotations'
for i,file in enumerate(os.listdir(instance_dir)):
    plant_ponts = []
    if file.endswith(".txt") and 'leaf' in file:
        file_path = os.path.join(instance_dir, file)
        print(file_path)
        data = pd.read_csv(file_path, sep=' ', header=None,
            names=['x', 'y', 'z', 'r', 'g', 'b'],
            dtype={'x': float, 'y': float, 'z': float, 'r': int, 'g': int, 'b': int})
        points = data[['x', 'y', 'z']].values
        colors = data[['r', 'g', 'b']].values        
        points = points[np.random.choice(points.shape[0], 5000, replace=False), :]
        colors = np.array(plant_color)
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

        # if nan in colors, replace with 0
        colors = np.nan_to_num(colors)
        plant.extend(points)
        plant_color.extend(colors)
# create pcd
points = np.array(plant)
# random sample 1000
points = points[np.random.choice(points.shape[0], 5000, replace=False), :]
colors = np.array(plant_color)
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
pass
