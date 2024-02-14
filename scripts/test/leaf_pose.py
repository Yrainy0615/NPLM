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
    plt.show()

if __name__ == "__main__":
    data_path = 'LeafSurfaceReconstruction/data/sugarbeet'
    points = []
    canonical_shape_path = 'dataset/Mesh_colored/Bael_0.obj'
    canonical_shape = trimesh.load(canonical_shape_path)
    leafAxisDetermination = LeafAxisDetermination(canonical_shape.vertices)
    w_axis_canonical, l_axis_canonical, h_axis_canonical, canonical_points = leafAxisDetermination.process()
    # random sample 1000
    canonical_points = canonical_points[np.random.choice(canonical_points.shape[0], 1000, replace=False), :]
    w_axis_canonical = np.array([0, 1, 0])
    l_axis_canonical = np.array([1, 0, 0])
    h_axis_canonical = np.array([0, 0, 1])
    
    
    # read txt point cloud data
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            file_path = os.path.join(data_path, file)
            print(file_path)
            data = pd.read_csv(file_path, names=("x", "y", "z")).values
            points.append(data)

    for point_cloud in points:
        print('current point cloud shape is {}'.format(point_cloud.shape))
        point_cloud = normalize_verts(point_cloud)
        point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        leafAxisDetermination = LeafAxisDetermination(point_cloud)
        
        w_axis, l_axis, h_axis, new_points = leafAxisDetermination.process()
        # normalize points to(-1,1)
        R = find_rotation_matrix(np.array([l_axis_canonical, w_axis_canonical, h_axis_canonical]).T, np.array([l_axis, w_axis, h_axis]).T)
        new_points_rotated = new_points @ R.T
        visualize_points_and_axes(new_points, new_points_rotated, canonical_points,np.mean(new_points, axis=0), l_axis_canonical, w_axis_canonical, h_axis_canonical)
        # visualize_points_and_axes(canonical_points, np.mean(canonical_points, axis=0), l_axis_canonical, w_axis_canonical, h_axis_canonical)
        # visualize_points_and_axes(new_points_rotated, np.mean(new_points_rotated, axis=0), l_axis_canonical, w_axis_canonical, h_axis_canonical)
        # pass

            