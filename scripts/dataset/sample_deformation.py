import trimesh
import numpy as np
import os
from DataManager import LeafScanManager
from multiprocessing import Pool
import pyvista as pv
from matplotlib import pyplot as plt

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
            p_neutral2, p2, normals_neutral2, normals2 = sample(m_deformed, m_neutral, 0.01, n_samples)#0.01)
            p_neutral = np.concatenate([p_neutral, p2], axis=0)
            p = np.concatenate([p, p_neutral2], axis=0)
            normals_neutral = np.concatenate([normals_neutral, normals2], axis=0)
            normals = np.concatenate([normals, normals_neutral2], axis=0)

            p_neutral_tight, p_tight, normals_neutral_tight, normals_tight = sample(m_neutral, m_deformed, 0.002, n_samples)#0.002)
            p_neutral_tight2, p_tight2, normals_neutral_tight2, normals_tight2 = sample(m_deformed, m_neutral, 0.002, n_samples)#0.002)
            p_neutral_tight = np.concatenate([p_neutral_tight, p_tight2], axis=0)
            p_tight = np.concatenate([p_tight, p_neutral_tight2], axis=0)
            normals_neutral_tight = np.concatenate([normals_neutral_tight, normals_tight2], axis=0)
            normals_tight = np.concatenate([normals_tight, normals_neutral_tight2], axis=0)          
                                           
            all_p_neutral = np.concatenate([p_neutral, p_neutral_tight], axis=0)
            all_normals_neutral = np.concatenate([normals_neutral, normals_neutral_tight], axis=0)
            all_p = np.concatenate([p, p_tight], axis=0)
            all_normals = np.concatenate([normals, normals_tight], axis=0)
            perm = np.random.permutation(all_p.shape[0])
            all_p_neutral = all_p_neutral[perm, :]
            all_normals_neutral = all_normals_neutral[perm, :]
            all_p = all_p[perm, :]
            all_normals = all_normals[perm, :]
            if np.any(np.isnan(all_p)) or np.any(np.isnan(all_normals)):
                print('DONE')
                break    
                
                
            # if VIZ:
            #     pv.start_xvfb()
            #     pl = pv.Plotter(shape=(1, 2))
            #     pl.subplot(0, 0)
            #     pl.add_points(all_p_neutral, scalars=all_normals_neutral[:, 0])
            #     pl.subplot(0, 1)
            #     pl.add_points(all_p, scalars=all_normals_neutral[:, 0])
            #     pl.link_views()
            #     pl.show()
            
            if VIZ:
                fig = plt.figure(figsize=(12, 6))

                # Subplot for neutral points
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.scatter(all_p_neutral[:, 0], all_p_neutral[:, 1], all_p_neutral[:, 2], c=all_normals_neutral[:, 0])
                ax1.set_title('Neutral Points')
                
                # Subplot for other points
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.scatter(all_p[:, 0], all_p[:, 1], all_p[:, 2], c=all_normals_neutral[:, 0])
                ax2.set_title('Other Points')

                plt.show()
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