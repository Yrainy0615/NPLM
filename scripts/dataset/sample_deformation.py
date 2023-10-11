import trimesh
import numpy as np
import os
from DataManager import LeafScanManager
from multiprocessing import Pool
import pyvista as pv


def sample(m_neutral, m_deformed, std, n_samps):
    p_neutral, idx_neutral = m_neutral.sample(n_samps, return_index=True)
    normals_neutral = m_neutral.face_normals[idx_neutral, :]
    faces = m_neutral.faces[idx_neutral]
    faces_lin = faces.reshape(-1)
    triangles_neutral = m_neutral.vertices[faces_lin].reshape(-1, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(triangles_neutral, p_neutral, method = 'cross')
    offsets = np.random.randn(p_neutral.shape[0]) * std
    offsets = np.expand_dims(offsets, axis=-1)
    p_neutral += offsets * normals_neutral
    
    
    faces = m_deformed.faces[idx_neutral]
    normals= m_deformed.face_normals[idx_neutral, :]
    faces_lin = faces.reshape(-1)
    triangles = m_deformed.vertices[faces_lin, :]
    triangles = triangles.reshape(-1, 3, 3)
    p = trimesh.triangles.barycentric_to_points(triangles, bary)
    p += offsets * normals
    return p_neutral, p, normals_neutral, normals

def main(n_samples):
    manager = LeafScanManager('dataset/ScanData')
    all_species = manager.get_all_species()
    for species in all_species:
        neutral_pose = manager.get_neutral_pose(species)
        neutral_pose = os.path.join(manager.neutral_path, neutral_pose)
        m_neutral = trimesh.load(neutral_pose)
        all_deformed = manager.get_poses(species)
        for deform in all_deformed:
            deform = os.path.join(manager.get_species_path(species), deform)
            m_deformed = trimesh.load(deform)
            p_neutral, p, normals_neutral, normals = sample(m_neutral, m_deformed, 0.01, n_samples)
            pass

if __name__ == "__main__":
    VIZ=True
    n_samples = 250000
    if not VIZ:
        p = Pool(10)
        p.map(main, n_samples)
        p.close()
        p.join()
    
    else:
        main(n_samples)