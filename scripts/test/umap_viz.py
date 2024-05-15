from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import torch
import json
from sklearn.preprocessing import MinMaxScaler

#import umap.plot
# with open('dataset/ScanData/deformation/deformation/maple_label.json') as f:
#     label = json.load(f)

# # create a array of label (dict to array, only perserve the value)
# label = np.array(list(label.values()))
# deformation_checkpoint without nce
chenkpoint_wo_nce = torch.load('checkpoints/deform.tar')
latent_wo_nce = chenkpoint_wo_nce['latent_deform_state_dict']['weight']
latent_wo_nce = latent_wo_nce.cpu().numpy()

# deformation_checkpoint with nce
# chenkpoint_w_nce = torch.load('checkpoints/deform_wo_dis.tar')
# latent_w_nce = chenkpoint_w_nce['latent_deform_state_dict']['weight']
# latent_w_nce = latent_w_nce.cpu().numpy()

 
# umap 
standard_embedding = umap.UMAP().fit_transform(latent_wo_nce)
kmeans_labels = cluster.KMeans(n_clusters=3).fit_predict(latent_wo_nce)
plt.figure()
plt.title('deformation_wo_NTXtent') 
# use different color to represent different label
# plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c='r', s=8, cmap='Spectral')
mean_latent = np.mean(latent_wo_nce, axis=0)

# Calculate the distance of each point from the mean
distances = np.sqrt(np.sum((latent_wo_nce - mean_latent)**2, axis=1))

# Normalize the distances to the range [0, 1] for coloring
distances = MinMaxScaler().fit_transform(distances.reshape(-1, 1))

# umap 
standard_embedding = umap.UMAP().fit_transform(latent_wo_nce)
mean_embedding = np.mean(standard_embedding, axis=0)

standard_embedding = standard_embedding - mean_embedding
min_index = np.argmin(distances)

# Create a scatter plot with colors representing the distance from the mean
scatter = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=distances, s=8, cmap='Spectral')
plt.scatter(standard_embedding[min_index, 0], standard_embedding[min_index, 1], c='black', s=50)

# Add a color bar

cbar = plt.colorbar(scatter, orientation='horizontal')
cbar.set_label('Chamfer distance between base shape and deformed shape', fontsize=20)
plt.title('Learned Deformation Space', fontsize=20)
plt.text(0.2, 0.2, 'Mean', ha='center', va='center', color='black', fontsize=20)
plt.axis('off')

plt.show()

# nce_embedding = umap.UMAP().fit_transform(latent_w_nce)
# # kmeans_labels = cluster.KMeans(n_clusters=3).fit_predict(latent_w_nce)
# plt.subplot(1,2,2)
# plt.title('deformation_w_NTXtent')
# plt.scatter(nce_embedding[:, 0], nce_embedding[:, 1], c='b', s=5, cmap='Spectral')
# plt.show()
pass