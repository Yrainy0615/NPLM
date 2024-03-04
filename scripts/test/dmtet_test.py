import sys
sys.path.append('NPLM')

import trimesh
import numpy as np
import torch
from scripts.model.fields import UDFNetwork
import yaml
from scripts.dataset.img_to_3dsdf import sdf2d_3d, mesh_from_sdf

# # initialize
deform_mesh = trimesh.load('dataset/deformation/2_179.obj')
canonical_mesh = trimesh.load('dataset/leaf_classification/canonical_mesh/2.obj')
target_mesh = trimesh.load('dataset/leaf_classification/canonical_mesh/179.obj')
delta_x = deform_mesh.vertices - canonical_mesh.vertices

canonical_mesh.vertices+=delta_x
canonical_mesh.export('deformed.obj')