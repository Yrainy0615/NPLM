import sys
from .DataManager import LeafScanManager
import point_cloud_utils as pcu
import pyvista as pv
import trimesh
import PIL
import os
from matplotlib import pyplot as plt
import numpy as np

def sample_surface(mesh, n_samps, viz=False):
    verts = mesh.vertex_data.positions
    faces = mesh.face_data.vertex_ids
    normal = pcu.estimate_mesh_vertex_normals(verts, faces)
    pts, bc = pcu.sample_mesh_random(mesh.vertex_data.positions, mesh.face_data.vertex_ids, num_samples=n_samps)
    surf_points = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, pts, bc, mesh.vertex_data.positions)
    surf_normals = pcu.interpolate_barycentric_coords(mesh.face_data.vertex_ids, pts, bc, normal)
    # sdfs, fi, bc = pcu.signed_distance_to_mesh(mesh.vertex_data.positions, mesh.face_data.vertex_ids)
    if viz:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection ='3d')
        ax.plot_trisurf(verts[:, 0],verts[:,1], triangles=faces, Z=verts[:,2]) 
        ax.scatter(surf_points[:, 0], surf_points[:, 1], surf_points[:, 2], s=0.5)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.show()
    return {'points':surf_points, 'normals':surf_normals}
    

def run_species(manager):
    all_neutral = manager.get_all_neutral()
    
    for neutral in all_neutral:
        #neutral_id = manager.get_neutral_pose(neutral)
        neutral_name = os.path.splitext(os.path.basename(neutral))[0]
        neutral_name = neutral_name.split('_')[0]
        # sample surface points from neural
    
        neutral_mesh = manager.load_mesh(neutral)
        result = sample_surface(neutral_mesh, n_samps=250000)
        out_dir = os.path.join(manager.get_neutral_path(),'train_file')
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir,f"{neutral_name}_neutral.npy"), result)
        
        # sample surface points and deformation field of poses
        

def run_poses(manager):
    all_pose = manager.get_all_pose()
    for pose in all_pose:
        (k ,v) , = pose.items()
        mesh = manager.load_mesh(v)
        result = sample_surface(mesh, n_samps=2500000)
        species_dir = manager.get_species_path(k)
        output_dir = os.path.join(species_dir,'train')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.splitext(os.path.basename(v))[0] + '.npy'
        np.save(os.path.join(output_dir, filename),result)
        print(f'File {filename} saved')
        pass


if __name__ == "__main__":
    root_path = 'dataset/ScanData'
    manager = LeafScanManager(root_path)
    run_species(manager)
    #run_poses(manager)
    
