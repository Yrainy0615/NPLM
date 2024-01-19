from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import torch
import json
#import umap.plot
with open('dataset/ScanData/deformation/deformation/maple_label.json') as f:
    label = json.load(f)

# create a array of label (dict to array, only perserve the value)
label = np.array(list(label.values()))
# deformation_checkpoint without nce
chenkpoint_wo_nce = torch.load('checkpoints/exp-deform-dis__10000.tar')
latent_wo_nce = chenkpoint_wo_nce['latent_deform_state_dict']['weight']
latent_wo_nce = latent_wo_nce.cpu().numpy()

# deformation_checkpoint with nce
chenkpoint_w_nce = torch.load('checkpoints/exp-deform-nce-dis__10000.tar')
latent_w_nce = chenkpoint_w_nce['latent_deform_state_dict']['weight']
latent_w_nce = latent_w_nce.cpu().numpy()

 
# umap 
standard_embedding = umap.UMAP().fit_transform(latent_wo_nce)
kmeans_labels = cluster.KMeans(n_clusters=3).fit_predict(latent_wo_nce)
plt.subplot(1,2,1)
plt.title('deformation_wo_NTXtent') 
# use different color to represent different label
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=label, s=8, cmap='Spectral')


nce_embedding = umap.UMAP().fit_transform(latent_w_nce)
kmeans_labels = cluster.KMeans(n_clusters=3).fit_predict(latent_w_nce)
plt.subplot(1,2,2)
plt.title('deformation_w_NTXtent')
plt.scatter(nce_embedding[:, 0], nce_embedding[:, 1], c=label, s=5, cmap='Spectral')
plt.show()
pass