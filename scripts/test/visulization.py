import numpy as np
# import open3d as o3d
import os
import trimesh
import pandas as pd
import pyvista as pv
import torch
from pytorch3d.loss import chamfer_distance, mesh_normal_consistency
import sys
sys.path.append('NPLM')
from scipy.spatial import KDTree
from scripts.dataset.sample_surface import sample_surface, load_mesh
from scripts.registration.leaf_axis_determination import LeafAxisDetermination

def plot_soybean():
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

def plot_mesh(mesh_dir):
    # use pyvista save plot of mesh
    mesh = pv.read(mesh_dir)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, lighting=False)
    plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
    image = plotter.screenshot()

def chamfer_and_normal_consistency(mesh1,mesh2, num_samples=10000):
    # Compute Chamfer distance
    chamfer_dist, _ = chamfer_distance(mesh1.verts_packed().unsqueeze(0), mesh2.verts_packed().unsqueeze(0))
    normal_consistency = mesh_normal_consistency(mesh2)



    return chamfer_dist.item(), 1-normal_consistency.item()

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product

def eval_pointcloud(pointcloud_pred,
                    pointcloud_gt,
                    normals_pred=None,
                    normals_gt=None,
                    return_error_pcs=True,
                    metric_space = False,
                    subject = None,
                    expression = None):

    if not metric_space:
        thresholds = [0.005, 0.01, 0.015, 0.02]
    else:
        thresholds = [1, 5, 10, 20] # scale in mm

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)



    # Completeness: how far are the points of the target point cloud
    # from the predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness_pc = completeness
    completeness_pc_normals = completeness_normals
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are the points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy_pc = accuracy
    accuracy_pc_normals = accuracy_normals
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()


    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals consistency': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'chamfer_l1': chamfer_l1,
        'f_score_05': F[0], # threshold: metric: 1mm,  otherwise 0.005
        'f_score_10': F[1], # threshold: metric: 5mm,  otherwise 0.01
        'f_score_15': F[2], # threshold: metric: 10mm, otherwise 0.015
        'f_score_20': F[3], # threshold: metric: 12mm, otherwise 0.020
    }

    if return_error_pcs:
        return out_dict, {'completeness': completeness_pc,
                          'accuracy': accuracy_pc,
                          'completeness_normals': completeness_pc_normals,
                          'accuracy_normals': accuracy_pc_normals}
    else:
        return out_dict

def error_map(result_dir):
    gt_dir = 'dataset/testset'
    gt_mesh= [f for f in os.listdir(gt_dir) if f.endswith('.obj')]
    #plot_mesh(mesh_dir)
    chamfers = []
    normal_consistency = []
    SHOW_VIZ = True
    for gt in gt_mesh:
        gt_path = os.path.join(gt_dir, gt)
        basename = gt.split('.')[0]
        mesh_gt = load_mesh(gt_path)
        mesh_pred = load_mesh(os.path.join(result_dir, basename + '.obj'))
        data_pred= sample_surface(mesh_pred, n_samps=10000)
        points, normals = data_pred['points'], data_pred['normals']
        data_gt = sample_surface(mesh_gt, n_samps=10000)
        points_GT, normals_GT = data_gt['points'], data_gt['normals']
        #chamfer, normal = chamfer_and_normal_consistency(mesh1, mesh2)
        metric_tmp, per_point_errors= eval_pointcloud(points, points_GT, normals, normals_GT)
        pv.start_xvfb()
        if SHOW_VIZ:
                        pl = pv.Plotter(shape=(1, 2), off_screen=True)

                        pl.subplot(0, 0)
                        clim = (0, 0.03)  # 定义颜色标度的范围

                        pl.subplot(0, 0)
                        pl.add_points(points, scalars=per_point_errors['accuracy'], clim=clim)
                        actor = pl.add_title('Pred', font='courier', color='k', font_size=10)

                        pl.subplot(0, 1)
                        pl.add_points(points_GT, scalars=per_point_errors['completeness'], clim=clim)
                        actor = pl.add_title('GT', font='courier', color='k', font_size=10)


                        pl.link_views()
                        pl.camera_position = (0, 0, 10)
                        pl.camera.zoom(1)
                        pl.camera.roll = 90
                        pl.camera.up = (0, 1, 0)
                        pl.camera.focal_point = (0, 0.15, 0)
                      
                       # pl.show()
                        pl.screenshot(result_dir + basename + '_pred.png')
                        pl.close()
def meshrender(result_dir):
    mesh_list = [f for f in os.listdir(result_dir) if f.endswith('.obj')]
    camera_list = [(0, 0, 10),
                  (10, 0, 0),
                (0,10,0)  ]
    pv.start_xvfb()

    for mesh in mesh_list:
        basename = mesh.split('.')[0]

        mesh_path = os.path.join(result_dir, mesh)
        mesh = pv.read(mesh_path)
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh, lighting=True, color='grey')
        pl.camera.zoom(1)
        pl.camera.roll = 0
        pl.camera.up = (0, 1, 0)
        pl.camera.focal_point = (0, 0.15, 0)
        for i in range(3):
            pl.camera_position = camera_list[i]
            pl.screenshot(result_dir +'/' +basename + '_view{}.png'.format(i))

        pl.close()

def visualize_alignment(mesh_file):
        mesh = pv.read(mesh_file)
        vertices = np.array(mesh.points)
        leafAxisDetermination = LeafAxisDetermination(vertices)
        w, l, h ,y, z = leafAxisDetermination.process()
        x= l 
        # visualize the point cloud and the axes using pyvista
        # random 1000 points
        random_index  = np.random.choice(vertices.shape[0], 1000, replace=False)
        # pv.start_xvfb()

        pl = pv.Plotter(shape=(1,2),off_screen=False)
        mean = np.mean(vertices, axis=0)
        pl.subplot(0, 0)
        pl.add_points(vertices[random_index], color='grey')
        # make arrow longer
        pl.add_arrows(mean, x, mag=0.5, color='r')
        # pl.add_arrows(mean, y, mag=0.5, color='g')
        # pl.add_arrows(mean, z, mag=0.5, color='b')
        pl.add_axes()

        pl.subplot(0, 1)
        # pl.add_mesh(mesh, color='grey')
        pl.add_points(vertices[random_index], color='grey')
        pl.add_arrows(mean, l, mag=0.5, color='r')
        pl.add_arrows(mean, w*-1, mag=0.5, color='g')
        pl.add_arrows(mean, h, mag=0.5, color='b')


        pl.link_views()
        pl.camera_position = (-5, 10, -10)
        pl.camera.zoom(1.2)
        pl.camera.roll = 90
        pl.camera.up = (0, 1, 0)
        pl.camera.focal_point = (0, 0.15, 0)
        
        pl.show()
        pl.screenshot('axis.png')
        
def draw_correspondence(mesh1, mesh2):
    # Read the meshes
    mesh1 = pv.read(mesh1)
    mesh2 = pv.read(mesh2)
    mesh1 =mesh1.translate([0, 1.2, 0])
    mesh1 = mesh1.rotate_x(45)

   # mesh2 = mesh2.rotate_y()

    # Create a plotter
    pl = pv.Plotter()

    # Add the meshes to the plotter
    pl.add_mesh(mesh1, color='cefad0', opacity=0.5)
    pl.add_mesh(mesh2, color='cefad0', opacity=0.5)

    # Randomly select 300 points from each mesh
    indices = np.random.choice(mesh1.points.shape[0], 50, replace=False)
    points1 = mesh1.points[indices]
    points2 = mesh2.points[indices]

    for p1, p2 in zip(points1, points2):
        # Generate a random color
        color = np.random.rand(3)
        # Add the points and lines to the plotter
        pl.add_points(np.array([p1, p2]), color=color, render_points_as_spheres=True,point_size=10)
      #  pl.add_lines(np.array([p1, p2]), color=color, render_points_as_spheres=True, point_size=10)

    # Draw lines between corresponding points
    for p1, p2 in zip(points1, points2):
        pl.add_lines(np.array([p1, p2]), color='grey')

    # Show the plot
    pl.show()


if __name__ == '__main__':
    #meshrender('results/transfer')
    #mesh_file = 'results/interpolation/deform_7.obj'
   #mesh1 = '/home/yang/Desktop/teaser/16_690.obj'
    # mesh1 = 'deformation/1_614.obj'
    # mesh2 = 'dataset/leaf_classification/canonical_mesh/450.obj'
    meshrender('deformation')
    #visualize_alignment(mesh1)
    #draw_correspondence(mesh1, mesh2)
    #error_map('results/test_ours')
    #meshrender('results/test_ours')