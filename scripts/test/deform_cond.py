import os
from sklearn.cluster import KMeans
import trimesh
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
sys.path.append('NPLM')
from scripts.dataset.rgbd_dataset import normalize_verts    
from scripts.registration.leaf_axis_determination import LeafAxisDetermination
import cv2
from probreg import cpd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cupy as cp
use_cuda = True
if use_cuda:
    # set gpu 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

def process(index, all_mesh, all_canonical_mask, points_sample, save_dir):                
        mask_name = all_canonical_mask[index].split('.')[0]
        if '_128' in mask_name:
            mask_name = mask_name.split('_')[0]
        # find mask name in all_mesh via string matching
        mesh_path  = [f for f in all_mesh if mask_name in f]
        mesh_canonical = trimesh.load(mesh_path[0])
        source_pt = cp.asarray(mesh_canonical.vertices, dtype=cp.float32)

        target_pt = cp.asarray(source_pt, dtype=cp.float32)
        target_pt = target_pt[np.random.choice(target_pt.shape[0], 300, replace=False), :]

        acpd = cpd.NonRigidCPD(source_pt, use_cuda=True)
        tf_param, _, _ = acpd.registration(target_pt)
        result = tf_param.transform(source_pt)
        registed_mesh = trimesh.Trimesh(vertices=to_cpu(result), faces=mesh_canonical.faces)
        save_name = mask_name + '_{}'.format(index) + '.obj'
        registed_mesh.export(os.path.join(save_dir, save_name))
        mesh_canonical.export(os.path.join(save_dir, mask_name + '_canonical.obj'))
        print('save to {}'.format(save_name))
method = 'l-w ratio'
if method == 'pca':
    root = 'dataset/deform_soybean/done'
    all_files = os.listdir(root)
    all_files.sort()
    save_dir = os.path.join(root, 'group')
    all_points = []
    for i in all_files:
        if i.endswith('.obj'):
            target_path = os.path.join(root, i)
            target_mesh = trimesh.load(target_path)
            points = target_mesh.vertices
            points = normalize_verts(points)
            
            # random sample 1000 points
            points_sample = points[np.random.choice(points.shape[0], 1000, replace=False)]
            all_points.append(points_sample)
    # use pca to get feature vectors dim=50
    pca = PCA(n_components=3)
    features = []
    for i in range(len(all_points)):
        pca.fit(all_points[i])
        features.append(np.concatenate((pca.components_[0].reshape(-1, 1), pca.components_[1].reshape(-1, 1), pca.components_[2].reshape(-1, 1)), axis=1))
    features_matrix = np.array(features).reshape(len(features), -1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features_matrix)
    # get index of each group
    group_index = kmeans.labels_
    # get index   of the mean of each grou

    group_mean = kmeans.cluster_centers_
    # get index of mean
    # save mean 
    for i in range(group_mean.shape[0]):
        # find the nearest point to the mean
        mean_index = np.argmin(np.linalg.norm(features_matrix - group_mean[i]))
        target_mesh = trimesh.load(os.path.join(root, all_files[mean_index]))
        save_folder = os.path.join(save_dir, str(i))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        target_mesh.export(os.path.join(save_folder, 'mean.obj'))    
    for i in range(features_matrix.shape[0]):
        target_path = os.path.join(root, all_files[i])
        target_mesh = trimesh.load(target_path)
        target_mesh.vertices = normalize_verts(target_mesh.vertices)
        save_folder = os.path.join(save_dir, str(group_index[i]))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        target_mesh.export(os.path.join(save_folder, all_files[i]))


if method == 'l-w ratio':
    root = 'dataset/deform_soybean/done'
    all_files = os.listdir(root)
    all_files.sort()
    save_dir = 'dataset/deformation'
    all_points = []
    all_mesh = []
    ratios = []
    canonical_mask_root = 'dataset/leaf_classification/canonical_mask'
    all_canonical_mask = os.listdir(canonical_mask_root)
    canonical_mesh_root = 'dataset/leaf_classification/images'
    mask_ratios = []
    for dirpath, dirnames, filenames in os.walk(canonical_mesh_root):
        for filename in filenames:
            if filename.endswith('_128.obj'):
                all_mesh.append(os.path.join(dirpath, filename))
    all_mesh.sort()
    all_canonical_mask.sort()
    for mask in all_canonical_mask:
        mask_path = os.path.join(canonical_mask_root, mask)
        mask = cv2.imread(mask_path, 0)
        # calculate the ratio of the mask
        # mask = mask / 255
        mask = mask.astype(np.uint8)
        # weight and height of mask region
        contours, _= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)

        #cv2.drawContours(mask, contours[0], -1, (255, 255, 255), 1) 
        x,y,w,h = cv2.boundingRect(cnt)
        ratio = h/w
        mask_ratios.append(ratio)

    for i in all_files:
        if i.endswith('.obj') and not 'maple' in i:
            target_path = os.path.join(root, i)
            target_mesh = trimesh.load(target_path)
            points = target_mesh.vertices
            points = normalize_verts(points)
            # random sample 1000 points
            points_sample = points[np.random.choice(points.shape[0], 1000, replace=False)]
            leafAxisDetermination = LeafAxisDetermination(points_sample)         
            w_axis, l_axis, h_axis, new_points = leafAxisDetermination.process()
            x_axis = l_axis
            y_axis = w_axis
            projection_1 = np.dot(points, x_axis)
            length_1 = projection_1.max() - projection_1.min()
            projection_2 = np.dot(points, y_axis)
            length_2 = projection_2.max() - projection_2.min()
            ratio = length_1 / length_2
            # find the nearest  5 mask 
            mask_five = np.argsort(np.abs(np.array(mask_ratios) - ratio))[:5]
            for index in mask_five:
                process(index, all_mesh, all_canonical_mask, points_sample, save_dir)
                # mask_name = all_canonical_mask[index].split('.')[0]
                # if '_128' in mask_name:
                #     mask_name = mask_name.split('_')[0]
                # # find mask name in all_mesh via string matching
                # mesh_path  = [f for f in all_mesh if mask_name in f]
                # mesh_canonical = trimesh.load(mesh_path[0])
                # acpd = cpd.NonRigidCPD(mesh_canonical.vertices, use_cuda=False)
                # tf_param, _, _ = acpd.registration(points_sample)
                # result = tf_param.transform(mesh_canonical.vertices)
                # registed_mesh = trimesh.Trimesh(vertices=result, faces=mesh_canonical.faces)
                # save_name = mask_name + '_{}'.format(i)
                # registed_mesh.export(os.path.join(save_dir, save_name))
                # mesh_canonical.export(os.path.join(save_dir, mask_name + '_canonical.obj'))
                # print('save to {}'.format(save_name))

            # with ProcessPoolExecutor(max_workers=2) as executor:
            #     # Wrap the executor with tqdm for progress visualization
            #     futures = [executor.submit(process, index, all_mesh, all_canonical_mask, points_sample, save_dir) for index in mask_five]
            #     for future in as_completed(futures):
            #         future.result()

                
                
                                
          
            
       