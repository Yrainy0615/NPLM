from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from scripts.dataset.DataManager import LeafImageManger, LeafScanManager

tsne = TSNE(n_components=2, random_state=42,perplexity=100)
manager = LeafImageManger(root_dir='dataset/LeafData/')
all_species = manager.species
species_to_idx = manager.get_species_to_idx()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
all_mesh = manager.get_all_mesh()
idx_list = []
for mesh in all_mesh:
    train_dict = manager.extract_info_from_meshfile(mesh)
    spc = train_dict['species']
    idx = species_to_idx[spc]
    idx_list.append(idx)

checkpoint_path = 'checkpoints/2dShape/exp-2d-sdf__3000.tar'
checkpoint = torch.load(checkpoint_path)



latent_code_all = checkpoint['latent_idx_state_dict']['weight']
latent_2d = tsne.fit_transform(latent_code_all.detach().cpu().numpy())
 



# Plot the 2D latent codes
plt.figure(figsize=(8, 6))
unique_labels = set()
for i in range(len(latent_2d)):
    label = all_species[idx_list[i]]
    if label not in unique_labels:
        plt.scatter(latent_2d[i, 0], latent_2d[i, 1], alpha=0.5, label=label, c=colors[idx_list[i]])
        unique_labels.add(label)
    else:
        plt.scatter(latent_2d[i, 0], latent_2d[i, 1], alpha=0.5, c=colors[idx_list[i]])
plt.legend()

#save figure
plt.savefig('tsne.png')
plt.show()
pass
