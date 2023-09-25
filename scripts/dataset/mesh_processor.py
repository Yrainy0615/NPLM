import trimesh
import os
import o3d
from sklearn.decomposition import PCA
import numpy as np 
import igl

class mesh_processor():
    def __init__(self, root_dir):
        self.template_path = os.path.join(root_dir,'template')
        self.id1_temp = trimesh.load_mesh(os.path.join(self.template_path, 'id1_template.obj'))
        self.idx_info = self.read_from_csv(os.path.join(self.template_path, 'id1_vertex.csv'))
        self.base_idx =self.idx_info['base']
        self.apex_idx = self.idx_info['apex']
        self.left_idx = self.idx_info['left']
        self.right_idx = self.idx_info['right']
        self.contour_idx = self.idx_info['contour']
        self.main_axis_idx = self.idx_info['main-axis']       
        
    def raw_to_canonical(self,path):
        mesh = trimesh.load_mesh(path)
        t = -mesh.centroid
        mesh.apply_translation(t)
        max_extent = mesh.extents.max()
        scale_factor = 1 / max_extent
        mesh.apply_scale(scale_factor)
        return mesh
    
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
        
    def non_rigid_rigistration(self,path):
        template = self.id1_temp
        #verts_temp =  template.vertices[self.id1_kps_idx]
        target = trimesh.load_mesh(path)
        pca_temp = self.find_principai_axis(template)
        pca_target = self.find_principai_axis(target)
        temp_idx = np.unique(np.concatenate((self.contour_idx, self.main_axis_idx)))
        verts_temp = template.vertices[temp_idx]
        
        # find correspondence
        verts_target = self.find_keypoints(template,pca_temp)
        target_tree = cKDTree(template.vertices)
        closest_vertex_idx = target_tree.query(verts_target)[1]
        #verts_target = self.find_keypoints(target, pca_target)     
        
        # visualize
        #self.visualize_axis(template,pca=pca_temp,verts=verts_temp)
        self.visualize_axis(template,pca_temp, target.vertices[closest_vertex_idx])
        
        # get vertex index
        anchor_indices_template = self.find_indices_of_vertex(template, verts_temp)
        anchor_indices_target = self.find_indices_of_vertex(target, verts_target)
        
        #  Use libigl to perform non-rigid deformation
        v, f = template.vertices, template.faces
        #  Use the anchor indices to define the anchor positions
        b = np.array(anchor_indices_template)  
        # Precompute ARAP (assuming 3D vertices)
        arap = igl.ARAP(v, f, 3, b)
        # Set the positions of anchors in the target to their corresponding positions
        bc = np.array([target.vertices[i] for i in anchor_indices_target]).squeeze(axis=1)
        # Perform ARAP deformation
        vn = arap.solve(bc, v)
        deformed_mesh = trimesh.Trimesh(vertices=vn, faces=f)