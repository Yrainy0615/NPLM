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
from multiprocessing import Process, Pool
import multiprocessing
from matplotlib import pyplot as plt 

def process(mask_ratios,root,deform_file,all_mesh, deform_index, save_dir,all_canonical_mask,use_cuda=True):
    target_path = os.path.join(root, deform_file)
    target_mesh = trimesh.load(target_path)
    points = target_mesh.vertices
    points = normalize_verts(points)
    # random sample 1000 points
    points_sample = points[np.random.choice(points.shape[0], 1000, replace=False)]
    # leafAxisDetermination = LeafAxisDetermination(points_sample)         
    # w_axis, l_axis, h_axis, new_points = leafAxisDetermination.process()
    # x_axis = l_axis
    # y_axis = w_axis
    # projection_1 = np.dot(points, x_axis)
    # length_1 = projection_1.max() - projection_1.min()
    # projection_2 = np.dot(points, y_axis)
    # length_2 = projection_2.max() - projection_2.min()
    # ratio = length_1 / length_2
    # indicies = np.argsort(np.abs(np.array(mask_ratios) - ratio))[:3]
    all_maple_index = [0,3,4,7,21,23,25,30,35,43,48,64,70,87,90,117,125,143,151,163,180,199,223,225,230,234,257,263,265,271,291,304,328,330,332,342,360,397,409,411,440,443,445,449,451,459,484,487,499,512,518,534,566,584,592,602,605,619,646,651,659,704,712,744,766,787,790,797,811,832,838,852,883,894,919,927,952,958,961,963,967]
    # random slet 3 maple leaf
    indicies = np.random.choice(all_maple_index, 5, replace=False)
    for index in indicies:
        # canonical_name = all_canonical_mask[index]
        # shape_index = int(canonical_name.split('.')[0])
        save_name = str(index) + '_{}'.format(deform_index) + '.obj'
        if not os.path.exists(os.path.join(save_dir,save_name)):
            import cupy as cp
            if use_cuda:
                to_cpu = cp.asnumpy
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            else:
                cp = np
                to_cpu = lambda x: x
            source_file = os.path.join('dataset/leaf_classification/canonical_mesh', str(index) + '.obj')
            source = trimesh.load(source_file)
            source_pt = cp.asarray(source.vertices, dtype=cp.float32)
            source_pt = normalize_verts(source_pt)
            target_pt = cp.asarray(target_mesh.vertices, dtype=cp.float32)
            target_pt = target_pt[np.random.choice(target_pt.shape[0], 1000, replace=False), :]
            target_pt = normalize_verts(target_pt)   
            acpd = cpd.NonRigidCPD(source_pt, use_cuda=True)
            tf_param, _, _ = acpd.registration(target_pt)
            result = tf_param.transform(source_pt)
            registed_mesh = trimesh.Trimesh(vertices=to_cpu(result), faces=source.faces)
            registed_mesh.export(os.path.join(save_dir, save_name))
            print('{} is saved'.format(save_name))



if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)

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
        all_files = [f for f in all_files if not 'maple' in f]
        all_files.sort()
        save_dir = 'dataset/deformation_maple'
        all_points = []
        all_maple = 'dataset/data/maple_align'
        all_maple_file = os.listdir(all_maple)
        all_maple_file.sort()
        ratios = []
        canonical_mask_root = 'dataset/leaf_classification/canonical_mask'
        all_canonical_mask = os.listdir(canonical_mask_root)
        all_canonical_mask.sort()
        canonical_mesh_root = 'dataset/leaf_classification/canonical_mesh'
        all_maple_index = []
        all_mesh = os.listdir(canonical_mesh_root)
        mask_ratios = []
        all_mesh.sort()
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
            
        for index, deform_file in enumerate(all_maple_file):
            deform_index = index + len(all_files) 
            process(mask_ratios, all_maple, deform_file, all_mesh, deform_index, save_dir, all_canonical_mask)
        # save_dir_multi = 'dataset/deformation_multi'
        # with Pool(4) as pool:
        #     for deform_index, deform_file in enumerate(all_files):   
        #         pool.apply_async(process, args=(mask_ratios,root, deform_file, all_mesh, deform_index, save_dir_multi,all_canonical_mask))
        #     print('waiting for all subprocesses done...')
        #     pool.close()
        #     pool.join()
        #     print('done')
         
            

                    
                
                                
          
            
       