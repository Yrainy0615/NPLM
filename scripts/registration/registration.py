import os
import sys
sys.path.append('NPLM/scripts')
from dataset.DataManager import LeafImageManager
import trimesh
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import igl
import csv

def show_contour(mesh, keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='gray', alpha=0.6)
    for i,idx in enumerate(keypoints):
        point = mesh.vertices[idx]
        if i%10==0:
            ax.scatter3D(point[0], point[1], point[2], c='red', s=5)
            ax.text(point[0], point[1], point[2], str(i), fontsize=10, color='black')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show() 

def show_mesh(mesh, keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='gray', alpha=0.6)
    for key,idx in keypoints.items():
        point = mesh.vertices[idx]
        ax.scatter3D(point[0], point[1], point[2], c='red', s=5)
        ax.text(point[0], point[1], point[2], key, fontsize=5, color='black')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show() 

def icp_correspondence(template, target, template_contour):
    # 使用子集构建KDTree
    tree = cKDTree(template[template_contour])
    
    # 创建一个映射，将KDTree索引映射回原始template的索引
    index_mapping = {i: original_idx for i, original_idx in enumerate(template_contour)}
    
    idx_list = []
    for pts in target:
        _, idx = tree.query(pts)
        # 使用映射找到原始的索引
        original_index = index_mapping[idx]
        idx_list.append(original_index)
    return idx_list

def compare_keypoint(mesh1, keypoints1, mesh2, keypoints2):
    # Create a new figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    keypoint_color = 'red'

    # Plot the first mesh faces
    ax.plot_trisurf(mesh1.vertices[:, 0], mesh1.vertices[:, 1], mesh1.vertices[:, 2], triangles=mesh1.faces, color='gray', alpha=0.6)

    # Plot keypoints for the first mesh
    for key,idx in keypoints1.items():
        point = mesh1.vertices[idx]
        ax.scatter3D(point[0], point[1], point[2], c=keypoint_color, s=5)
        ax.text(point[0], point[1], point[2], key, fontsize=5, color='black')
        
    # Plot the second mesh faces
    ax.plot_trisurf(mesh2.vertices[:, 0], mesh2.vertices[:, 1], mesh2.vertices[:, 2], triangles=mesh2.faces, color='lightblue', alpha=0.6)

    # Plot keypoints for the second mesh
    for key,idx in keypoints2.items():
        point = mesh2.vertices[idx]
        ax.scatter3D(point[0], point[1], point[2], c=keypoint_color, s=5)
        ax.text(point[0], point[1], point[2], key, fontsize=5, color='black')


    # Connect keypoints with same index from both meshes
    for key in keypoints1.keys():
        idx1 = keypoints1[key]
        idx2 = keypoints2[key]
        point1 = mesh1.vertices[idx1]
        point2 = mesh2.vertices[idx2]
        ax.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'b-') # using blue color for lines

    # Set axis limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
def compare_mesh(mesh1, keypoints1, mesh2, keypoints2):
    # Create a new figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    keypoint_color = 'red'

    # Plot the first mesh faces
    ax.plot_trisurf(mesh1.vertices[:, 0], mesh1.vertices[:, 1], mesh1.vertices[:, 2], triangles=mesh1.faces, color='gray', alpha=0.6)

    # Plot keypoints for the first mesh
    for i,idx in enumerate(keypoints1):
        if idx % 10 == 0:
            point = mesh1.vertices[idx]
            ax.scatter3D(point[0], point[1], point[2], c=keypoint_color, s=5)
            ax.text(point[0], point[1], point[2], str(i), fontsize=5, color='black')


    # Plot the second mesh faces
        ax.plot_trisurf(mesh2.vertices[:, 0], mesh2.vertices[:, 1], mesh2.vertices[:, 2], triangles=mesh2.faces, color='lightblue', alpha=0.6)
     
    # Plot keypoints for the second mesh
    for j,idx in enumerate(keypoints2):
        point = mesh2.vertices[idx]    
        if idx%10==0:
            ax.scatter3D(point[0], point[1], point[2], c=keypoint_color, s=5)
            ax.text(point[0], point[1], point[2], str(j), fontsize=10, color='black')


    # Connect keypoints with same index from both meshes
    for idx1, idx2 in zip(keypoints1, keypoints2):
        if idx1 % 10 == 0:
            point1 = mesh1.vertices[idx1]
            point2 = mesh2.vertices[idx2]
            ax.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'b-')
        
    # Set axis limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])

    # Display the plot
    plt.show()
def read_vertex_groups_from_csv(filepath):
    group_dict = {}

    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header row

        for row in csvreader:
            group_name, index = row[0], int(row[1])
            if group_name not in group_dict:
                group_dict[group_name] = []
            group_dict[group_name].append(index)

    return group_dict


class LeafRegistration():
    def __init__(self, root_dir, template,vertex_info):
        self.root_dir = root_dir
        self.manager = LeafImageManager(root_dir)
        self.species = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.all_mesh = self.manager.get_all_mesh()
        self.template = trimesh.load_mesh(template)
        self.vertex_info = read_vertex_groups_from_csv(vertex_info)
        
    def raw_to_canonical(self,mesh):
        t = -mesh.centroid
        mesh.apply_translation(t)
        max_extent = mesh.extents.max()
        scale_factor = 1 / max_extent
        mesh.apply_scale(scale_factor)
        return mesh

    def get_boundary(self, mesh):
        # Find the boundary vertices
        unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        boundary_vertex_indices = np.unique(unique_edges)
        return boundary_vertex_indices

    def get_species_mesh(self, species):
        """
        return input kind of species mesh from all_mesh
        """
        return [mesh for mesh in self.all_mesh if species in mesh]                
                                          
    def rigid_registration(self,path):
        target_mesh =trimesh.load_mesh(path)
        
        template = self.id1_temp
        # extract principal axes
        pca_temp = self.find_principai_axis(template)
        pca_target = self.find_principai_axis(target_mesh)
     
        # extract extreme_points as keypoint
        verts_temp = self.find_extreme_points(template,pca_temp)
        verts_target = self.find_extreme_points(target_mesh,pca_target)
        # self.visualize_axis(target_mesh, pca_target,verts_target)
        # self.visualize_axis(template, pca_temp,verts_temp)

        # icp registration
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(verts_target)
        target.points = o3d.utility.Vector3dVector(verts_temp)
        result = o3d.pipelines.registration.registration_icp(source,target, max_correspondence_distance=1)
        transformation = np.asarray(result.transformation)
        target_mesh.apply_transform(transformation)
        return target_mesh
    
    def find_pca(self,mesh):
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        return pca
    
    def find_keypoints(self,mesh,pca):
        def project_to_axis(vertices, axis):
            return np.dot(vertices, axis)

        def get_extremal_points_for_axis(vertices, axis):
            projections = project_to_axis(vertices, axis)
            max_idx = np.argmax(projections)
            min_idx = np.argmin(projections)
            return max_idx, min_idx 

        top_idx, base_idx = get_extremal_points_for_axis(mesh.vertices, pca.components_[0])
    
        # Extremal points for the second principal component
        left_idx, right_idx = get_extremal_points_for_axis(mesh.vertices, pca.components_[1])
        keypoints = {
            'top': top_idx,
            'base': base_idx,
            'left': left_idx,
            'right': right_idx
        }
        return keypoints

    def get_correspondence(self, mesh_temp, mesh_target, keypoints_temp, keypoints_target):
        # Helper function to compute distance between keypoints
        def compute_distance(idx1, idx2, mesh1, mesh2):
            return np.linalg.norm(mesh1.vertices[idx1] - mesh2.vertices[idx2])

        # Compute distances
        dist_top = compute_distance(keypoints_temp['top'], keypoints_target['top'], mesh_temp, mesh_target)
        dist_base = compute_distance(keypoints_temp['top'], keypoints_target['base'], mesh_temp, mesh_target)
        
        dist_left = compute_distance(keypoints_temp['left'], keypoints_target['left'], mesh_temp, mesh_target)
        dist_right = compute_distance(keypoints_temp['left'], keypoints_target['right'], mesh_temp, mesh_target)

        # Adjust the keypoints order for target based on distances
        if dist_top > dist_base:
            keypoints_target['top'], keypoints_target['base'] = keypoints_target['base'], keypoints_target['top']
        
        if dist_left > dist_right:
            keypoints_target['left'], keypoints_target['right'] = keypoints_target['right'], keypoints_target['left']

        # Return the correspondence (since each keypoint is already its vertex index)
        correspondence = {
            'top': keypoints_target['top'],
            'base': keypoints_target['base'],
            'left': keypoints_target['left'],
            'right': keypoints_target['right']
        }
        
        return correspondence

        # Helper function to sample points on a line segment
    
    def sampling_keypoints(self,mesh, keypoints):
        def sample_line_segment(mesh, start, end, n_samples=24):
            pts = np.linspace(start, end, n_samples)[:-1]
            # find nearest point on surface and return mesh vertex index
            # idx =icp_correspondence(template=mesh.vertices, target=pts)
            return pts

        # Sample points on the top edge
        top_edge = sample_line_segment(mesh, mesh.vertices[keypoints['top']], mesh.vertices[keypoints['right']])

        # Sample points on the left edge
        left_edge = sample_line_segment(mesh, mesh.vertices[keypoints['right']], mesh.vertices[keypoints['base']])

        # Sample points on the bottom edge
        bottom_edge = sample_line_segment(mesh, mesh.vertices[keypoints['base']], mesh.vertices[keypoints['left']])

        # Sample points on the right edge
        right_edge = sample_line_segment(mesh, mesh.vertices[keypoints['left']], mesh.vertices[keypoints['top']])

        # Stack all points together
        keypoints_samples = np.concatenate((top_edge, left_edge, bottom_edge, right_edge))
        return keypoints_samples
     
    def ARAP_rigistration(self,target_path):
        template = self.raw_to_canonical(self.template)  
        template.fill_holes()
        #template = self.ICP_rigid_registration(target_path)
        target = trimesh.load_mesh(target_path)
        target = self.raw_to_canonical(target)
        target.fill_holes()
        pca_target = self.find_pca(target)
        keypoints_target = self.find_keypoints(target,pca_target)
        keypoints_temp = {
            'top':self.vertex_info['top'][0],
            'base':self.vertex_info['base'][0],
            'left':self.vertex_info['left'][0],
            'right':self.vertex_info['right'][0]
        }
        template_contour = self.vertex_info['contour']
        # get boundary index of target mesh
        target_contour = self.get_boundary(target)
        keypoints_target = self.get_correspondence(template, target, keypoints_temp, keypoints_target)
        
        keypoints_target_up = self.sampling_keypoints(target, keypoints_target)
        keypoints_temp_up = self.sampling_keypoints(template, keypoints_temp)
        keypoints_target_surf = icp_correspondence(target.vertices, keypoints_target_up, template_contour=target_contour)        
        keypoints_temp_surf = icp_correspondence(template.vertices, keypoints_temp_up, template_contour=template_contour)
        # find correspondence
        # show_mesh(template, keypoints_temp)
        # show_mesh(target, keypoints_target)
        # compare_keypoint(template, keypoints_temp, target, keypoints_target)
        
        # show contour
        # show_contour(template, keypoints_temp_surf)
        # show_contour(target, keypoints_target_surf)    
       #  compare_mesh(template, template_contour, target, keypoints_target_surf)
        
        
        # Use libigl to perform non-rigid deformation
        v, f = template.vertices, template.faces


        # Define the anchor positions
        b = np.array(template_contour)

        # Precompute ARAP (assuming 3D vertices)
        arap = igl.ARAP(v, f, 3, b)

        # Set the positions of anchors in the target to their corresponding positions
        bc = np.array([target.vertices[i] for i in keypoints_target_surf])

        # Perform ARAP deformation
        vn = arap.solve(bc, v)
        deformed_mesh = trimesh.Trimesh(vertices=vn, faces=f)
        # compare_mesh(deformed_mesh, template_contour, target, keypoints_target_surf)
        return deformed_mesh
    
    def ICP_rigid_registration(self, target_path):
        target = trimesh.load_mesh(target_path)
        template = self.raw_to_canonical(self.template)
        target = self.raw_to_canonical(target)
        aligned_mesh =template.copy()
        transform, cost = trimesh.registration.mesh_other(template, target, samples=100,
                                                          icp_first=10,icp_final=50,scale=False, return_cost=True)
        aligned_mesh.apply_transform(transform)
        return aligned_mesh
        

if __name__ == '__main__':
    root_dir = 'dataset/LeafData'
    template = 'dataset/LeafData/Jamun/healthy/Jamun_healthy_0019.obj'
    vertex_info = 'dataset/vertex.csv'
    registrater  =LeafRegistration(root_dir, template, vertex_info)
    target = 'dataset/ScanData/yellow/Leaf_yellow.003.obj'
    registrater.ARAP_rigistration(target)
    #registrater.ICP_rigid_registration(target)
