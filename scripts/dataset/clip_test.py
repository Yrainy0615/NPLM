import torch
import clip
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import figure

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

leaf_names = []
data_root = 'dataset/leaf_classification/images'

# get all sub foler names
for root, dirs, files in os.walk(data_root):
    for dir in dirs:
        leaf_names.append(dir)

text_tokens = []
for leaf_name in leaf_names:
    text_tokens.append(clip.tokenize(leaf_name))


text_features = []
with torch.no_grad():
    for i in range(len(text_tokens)):
        text_features.append(model.encode_text(text_tokens[i].to(device)))
        
# show cosine similarity between each pair of leaf names use a fuse matrix
cosine_similarity_matrix = torch.zeros((len(text_features), len(text_features)))
for i in range(len(text_features)):
    for j in range(len(text_features)):
        cosine_similarity_matrix[i][j] = torch.cosine_similarity(text_features[i], text_features[j], dim=1)

# visualize the cosine similarity matrix and label with leaf names


figure(figsize=(8, 6), dpi=80)
plt.imshow(cosine_similarity_matrix, cmap='hot', interpolation='nearest')
plt.show()
