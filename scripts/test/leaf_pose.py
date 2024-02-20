import numpy as np
import os
import pandas as pd
import sys
sys.path.append('NPLM/scripts')
from registration.helper_functions import *
from registration.leaf_axis_determination import LeafAxisDetermination
from registration.leaf_flattening import LeafFlattening
import trimesh


def find_rotation_matrix(A, B):
    # A and B are 3x3 matrices where columns are the vectors of the two bases
    assert A.shape == B.shape == (3, 3), "A and B must be 3x3 matrices"
    
    # Step 1: Compute the covariance matrix
    H = A @ B.T
    
    # Step 2: Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Step 3: Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
       Vt[2, :] *= -1
       R = Vt.T @ U.T
    
    return R

def normalize_verts(verts):
      bbmin = verts.min(0)
      bbmax = verts.max(0)
      center = (bbmin + bbmax) * 0.5
      scale = 2.0 * 0.8 / (bbmax - bbmin).max()
      vertices = (verts - center) *scale
      return vertices

def visualize_points_and_axes(points , origin, x_axis,y_axis, z_axis):
    """
    visualize the point cloud and the axes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    # ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='g', marker='o')
    # ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2], c='r', marker='o')

    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

if __name__ == "__main__":
    data_path = 'LeafSurfaceReconstruction/data/sugarbeet'
    root = 'dataset/deform_soybean'
    for i in os.listdir(root):
        if not i.endswith('.obj'):
            continue
        target_path = os.path.join(root, i)
        #target_path = '/home/yang/projects/parametric-leaf/dataset/deform_soybean/20190722_DN252_leaf_33.obj'

        target_mesh = trimesh.load(target_path)
        points = target_mesh.vertices
        w_axis_canonical = np.array([0, 1, 0])
        l_axis_canonical = np.array([1, 0, 0])
        h_axis_canonical = np.array([0, 0, 1])

        # random sample 1000 points
        points_sample = points[np.random.choice(points.shape[0], 1000, replace=False)]
        leafAxisDetermination = LeafAxisDetermination(points_sample)
            
        w_axis, l_axis, h_axis, new_points = leafAxisDetermination.process()
        # visualize_points_and_axes(new_points, np.mean(new_points, axis=0),l_axis, w_axis, h_axis)
        # find rotation matrix
        R_w2c = find_rotation_matrix(np.array([l_axis_canonical, w_axis_canonical, h_axis_canonical]), np.array([l_axis, w_axis, h_axis]))
        points_canonical = points @ R_w2c.T
        canonical_mesh = trimesh.Trimesh(points_canonical, target_mesh.faces)
        canonical_mesh.export(target_path)
        print('{} is rotated'.format(target_path))
    # visualize_points_and_axes(points_canonical, np.mean(points_canonical, axis=0),l_axis_canonical, w_axis_canonical, h_axis_canonical)
    
    # R_w2c_gt = np.linalg.inv(R_c2w)
    # points_ori_gt = new_points @ RT_w2c
    # points_ori = new_points @ R_w2c.T
    
    
    
