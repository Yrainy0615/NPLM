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
    save_dir = os.path.join(root, 'group')
    all_points = []
    ratios = []
    canonical_mask_root = 'dataset/leaf_classification/canonical_mask'
    all_canonical_mask = [f for f in os.listdir(canonical_mask_root) ]
    mask_ratios = []
    for mask in all_canonical_mask:
        mask_path = os.path.join(canonical_mask_root, mask)
        mask = cv2.imread(mask_path, 0)
        # calculate the ratio of the mask
        mask = mask / 255
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        # draw the bounding box
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 2)
        ratio = w / h
        mask_ratios.append(ratio)

    for i in all_files:
        if i.endswith('.obj'):
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
            # find the nearest mask ratio
            nearest_mask_ratio = np.argmin(np.abs(np.array(mask_ratios) - ratio))
            
    pass