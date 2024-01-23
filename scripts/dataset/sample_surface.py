import point_cloud_utils as pcu
import trimesh
import PIL
import os
from matplotlib import pyplot as plt
import numpy as np

def load_mesh(path):
    mesh = pcu.TriangleMesh()
    v, f = pcu.load_mesh_vf(path)
    mesh.vertex_data.positions = v
    mesh.face_data.vertex_ids = f
    return mesh

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
    

def run_species(all_mesh):
    #all_neutral = manager.get_all_neutral()
    
    for mesh_file in all_mesh:
        #neutral_id = manager.get_neutral_pose(neutral)
        save_name = mesh_file.replace('.obj', '_3d.npy')
        mesh =load_mesh(mesh_file)
        result = sample_surface(mesh, n_samps=250000)
     
        np.save(save_name, result)
        print('{} is saved'.format(save_name))
        


def run_poses(manager):
    all_pose = manager.get_all_pose()
    for pose in all_pose:
        (k ,v) , = pose.items()
        mesh = manager.load_mesh(v)
        result = sample_surface(mesh, n_samps=250000)
        species_dir = manager.get_species_path(k)
        output_dir = os.path.join(species_dir,'train')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.splitext(os.path.basename(v))[0] + '.npy'
        np.save(os.path.join(output_dir, filename),result)
        print(f'File {filename} saved')
        pass


if __name__ == "__main__":
    root = 'dataset/leaf_classification/images'    
    save_mesh = True
    all_mesh = []
    for dirpath , dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.obj'):
                all_mesh.append(os.path.join(dirpath, filename))
    all_mesh.sort()
    run_species(all_mesh)
    #run_poses(manager)
    
