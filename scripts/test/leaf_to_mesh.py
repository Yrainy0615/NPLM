import cv2
import numpy as np
from skimage.measure import subdivide_polygon
from sklearn.decomposition import PCA
import os
from scipy.spatial import Delaunay
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
import pytorch3d
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.io import save_obj
from pytorch3d.io import IO
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

def normalize_to_center(data):
   
    mean_val = np.mean(data, axis=0)

    centered_data = data - mean_val
    max_abs_val = np.max(np.abs(centered_data))
    normalized_data = centered_data / max_abs_val
    return normalized_data


class LeaftoMesh():
    def __init__(self,root) -> None:
        self.root_dir = root
        self.all_images = [i for i in os.listdir(root) if i.endswith('.JPG') and not 'mask' in i]
        self.all_masks = [i for i in os.listdir(root) if i.endswith('.JPG') and 'mask' in i]
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=128, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
            perspective_correct=False, 
        )
        self.source_index =[115, 120, 121, 116,  13,  12,  17,  18,  83,  82,  87,  88, 153,
       148, 147, 152, 123, 118, 117, 122,  53,  52,  57,  58,  23,  22,
        27,  28,  93,  92,  97,  98, 163, 158, 157, 162,   0,   2,   3,
         4, 133, 128, 127, 132, 179, 178, 184,  15,  16,  11,  10,  63,
        62,  67,  68, 155, 160, 161, 156,  33,  32,  37,  38,  55,  56,
        51,  50, 103, 102, 107, 108,  95,  96,  91,  90, 173, 168, 167,
       172, 135, 140, 141, 136,  73,  72,  77,  78,  35,  36,  31,  30,
       143, 138, 137, 142, 175, 180, 181, 176, 113, 112,  43,  42,  47,
        48, 145, 150, 151, 146,  45,  46,  41,  40, 125, 130, 131, 126,
        65,  66,  61,  60,   6,  85,  86,  81,  80, 165, 170, 171, 166,
        75,  76,  71,  70,  25,  26,  21,  20, 105, 106, 101, 100,   1,
       186, 185, 110, 111, 109, 104,  99,  24,  19,  29,  74,  69,  79,
       164, 169, 159,  84,  89,   9,   5,  64,  59, 124, 129, 119,  44,
        39,  49, 144, 149, 139, 114, 174,  34, 134, 177,  94,  54, 154,
        14, 182, 183,   8,   7]
        self.device = torch.device("cuda:0")
        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=90)
        camera  = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        # Differentiable soft renderer using per vertex RGB colors for texture
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=self.device, 
                cameras=camera,
                lights=lights)
        )
        
    def icp_correspondence(self, source, target):
        '''
        source: leaf
        target: template
        '''
        source_norm = normalize_to_center(source)
        target_norm = normalize_to_center(target)
        # ICP correspondence return new index of source
        def find_correspondences(source, target):
            cost_matrix = np.sqrt(((source[:, np.newaxis] - target[np.newaxis, :])**2).sum(axis=2))
            source_indices, target_indices = linear_sum_assignment(cost_matrix)
            return source_indices, target_indices
        target_index, source_index = find_correspondences( target_norm[:,:2], source_norm)
        sorted_target = target_norm[target_index]
        sorted_source = source_norm[source_index]
        plt.scatter(sorted_source[:, 0], sorted_source[:, 1], c='r', label='Source')
        plt.scatter(sorted_target[:, 0], sorted_target[:, 1], c='b', label='Target')
        # label index for each point
        for i, (s, t) in enumerate(zip(source_norm, target_norm)):
            plt.text(s[0], s[1], str(i), fontsize=8)
           # plt.text(t[0], t[1], str(i), fontsize=8)
        plt.plot
        plt.xlim(-1.3, 1.3)
        plt.ylim(-1.3, 1.3)
        for s, t in zip(sorted_source, sorted_target):
            plt.plot([s[0], t[0]], [s[1], t[1]], 'k-')

        return source[source_index]
        
    def uv_mapping(self, mesh, img):
        image_width, image_height = img.shape[1], img.shape[0]
        uv_coords = np.zeros_like(mesh.vertices)
        uv_coords[:, 0] = mesh.vertices[:, 0]  / image_width  #u
        uv_coords[:, 1] = 1 - mesh.vertices[:, 1]  /image_height  #  v
        uv_coords = torch.from_numpy(uv_coords).unsqueeze(0).float()
        faces = torch.from_numpy(mesh.faces).unsqueeze(0).long()
        image_tensor = torch.from_numpy(img).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        texture = Textures(verts_uvs=uv_coords[:,:,:2], faces_uvs=faces,maps=image_tensor)
        verts = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
        mesh = Meshes(verts=verts, faces=faces, textures=texture)
    
        return mesh
    
        
        
    
    def leaftomesh(self, index, verts,face):
        # Read the image and mask
        img_path = os.path.join(self.root_dir, self.all_images[index])
        #mask_path = os.path.join(self.root_dir, self.all_masks[index])
        mask_path = img_path.replace('_aligned', '_mask_aligned')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        # Threshold the mask to get the leaf area
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours and approximate the contour with less points
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # draw the contour on the original image
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)


        # PCA to get the major axis
        pca = PCA(n_components=1)
        pca.fit(contours[0].squeeze())
        major_axis_vector = pca.components_[0]
        centroid = pca.mean_

        # print rgb value of contour

        # find top & base
        projection = np.dot(contours-centroid, major_axis_vector)
        min_index = np.argmin(projection)
        max_index = np.argmax(projection)   
        endpoint1 = contours[0][min_index]
        endpoint2 = contours[0][max_index]
        if endpoint1[0][1] > endpoint2[0][1]:
            endpoint1, endpoint2 = endpoint2, endpoint1
        
        # Subdivide the contour into smaller segments
        num_divisions = 39  # 18 divisions create 20 segments
        points_on_major_axis = np.linspace(endpoint1, endpoint2, num_divisions ).squeeze()  # Exclude first and last

        # Each line will be defined by a point on the major axis and the direction perpendicular to the major axis
        perpendicular_direction = np.array([-major_axis_vector[1], major_axis_vector[0]]).T

        # Initialize a list to hold all intersection points
        intersection_points = []
        vertex_list = []
        vertex_list.append(tuple(endpoint1[0].astype(np.int16)))
        vertex_list.append(tuple(endpoint2[0].astype(np.int16)))
        hull = cv2.convexHull(contours[0])
        for point in points_on_major_axis[1:-1]:
            line_point1 = point + perpendicular_direction * 300  
            line_point2 = point - perpendicular_direction * 300  
            line = np.array([line_point1, line_point2], dtype=np.float32)
            # plot line
           # cv2.line(img, tuple(line_point1.astype(np.int16)), tuple(line_point2.astype(np.int16)), (0, 0, 255), 1)
            _,intersection = cv2.intersectConvexConvex(line, hull)
            # delete the same point
            rounded_intersection = np.round(intersection).astype(int)
            intersection = np.unique(rounded_intersection, axis=0)
            intersection_points.append(intersection)
        # draw intersection points
        for point in intersection_points:
            if len(point) == 2:        
                coordinate_1 = tuple(abs(point[0]).astype(np.int16))
                coordinate_2 = tuple(abs(point[1]).astype(np.int16))
                # vertex_list.append(coordinate_1[0])
                # vertex_list.append(coordinate_2[0])
            else:
                coordinate_1 = tuple(abs(point[0]).astype(np.int16))
                coordinate_2 = tuple(abs(point[1]).astype(np.int16))
            sample_points = np.linspace(coordinate_1, coordinate_2, 5).astype(np.int16)
            for point in sample_points:
                vertex_list.append(point[0])
                    
            # vertex_list.append(coordinate_1[0])
            # vertex_list.append(coordinate_2[0])

        # draw vertex and index
        # for i, vertex in enumerate(vertex_list):
        #     cv2.circle(img, vertex, 2, (0, 0, 255), -1)
        #     #cv2.putText(img, str(i), vertex, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite('leaf.jpg', img)
        vertexs = np.array(vertex_list)
        #tri = Delaunay(vertexs)
        vertexs_sorted = self.icp_correspondence(vertexs, verts)
        #vertexs_sorted = vertexs[self.source_index]
        mesh = trimesh.Trimesh(vertices=np.hstack((vertexs_sorted, np.zeros((vertexs.shape[0], 1)))), 
                       faces=face)
        # uv mapping to mesh

        mesh = self.uv_mapping(mesh, img)
        return vertexs_sorted
    def find_vertex(self,mask, verts):
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours and approximate the contour with less points
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw the contour on the original image
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)


        # PCA to get the major axis
        pca = PCA(n_components=1)
        pca.fit(contours[0].squeeze())
        major_axis_vector = pca.components_[0]
        centroid = pca.mean_

        # print rgb value of contour

        # find top & base
        projection = np.dot(contours-centroid, major_axis_vector)
        min_index = np.argmin(projection)
        max_index = np.argmax(projection)   
        endpoint1 = contours[0][min_index]
        endpoint2 = contours[0][max_index]
        if endpoint1[0][1] > endpoint2[0][1]:
            endpoint1, endpoint2 = endpoint2, endpoint1
        
        # Subdivide the contour into smaller segments
        num_divisions = 39  # 18 divisions create 20 segments
        points_on_major_axis = np.linspace(endpoint1, endpoint2, num_divisions ).squeeze()  # Exclude first and last

        # Each line will be defined by a point on the major axis and the direction perpendicular to the major axis
        perpendicular_direction = np.array([-major_axis_vector[1], major_axis_vector[0]]).T

        # Initialize a list to hold all intersection points
        intersection_points = []
        vertex_list = []
        vertex_list.append(tuple(endpoint1[0].astype(np.int16)))
        vertex_list.append(tuple(endpoint2[0].astype(np.int16)))
        hull = cv2.convexHull(contours[0])
        for point in points_on_major_axis[1:-1]:
            line_point1 = point + perpendicular_direction * 300  
            line_point2 = point - perpendicular_direction * 300  
            line = np.array([line_point1, line_point2], dtype=np.float32)
            # plot line
           # cv2.line(img, tuple(line_point1.astype(np.int16)), tuple(line_point2.astype(np.int16)), (0, 0, 255), 1)
            _,intersection = cv2.intersectConvexConvex(line, hull)
            # delete the same point
            rounded_intersection = np.round(intersection).astype(int)
            intersection = np.unique(rounded_intersection, axis=0)
            intersection_points.append(intersection)
        # draw intersection points
        for point in intersection_points:
            if len(point) == 2:        
                coordinate_1 = tuple(abs(point[0]).astype(np.int16))
                coordinate_2 = tuple(abs(point[1]).astype(np.int16))
                # vertex_list.append(coordinate_1[0])
                # vertex_list.append(coordinate_2[0])
            else:
                coordinate_1 = tuple(abs(point[0]).astype(np.int16))
                coordinate_2 = tuple(abs(point[1]).astype(np.int16))
            sample_points = np.linspace(coordinate_1, coordinate_2, 5).astype(np.int16)
            for point in sample_points:
                vertex_list.append(point[0])
                    
            # vertex_list.append(coordinate_1[0])
            # vertex_list.append(coordinate_2[0])

        # draw vertex and index
        # for i, vertex in enumerate(vertex_list):
        #     cv2.circle(img, vertex, 2, (0, 0, 255), -1)
        #     #cv2.putText(img, str(i), vertex, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite('leaf.jpg', img)
        vertexs = np.array(vertex_list)
        #tri = Delaunay(vertexs)
        vertexs_sorted = self.icp_correspondence(vertexs, verts)
        #vertexs_sorted = vertexs[self.source_index]
        # mesh = trimesh.Trimesh(vertices=np.hstack((vertexs_sorted, np.zeros((vertexs.shape[0], 1)))), 
        #                faces=face)
        # uv mapping to mesh

        #mesh = self.uv_mapping(mesh, img)
        return vertexs_sorted
        


if __name__ == "__main__":
    root_directory = 'dataset/LeafData/Lemon/healthy'  # Update this path
    leaf_to_mesh = LeaftoMesh(root_directory)
    mesh = trimesh.load('uvtest/leaf_uv_5sp.ply')
    # delete the same vertex in template
    for i in range(len(mesh.vertices)):
        new_mesh = leaf_to_mesh.leaftomesh(i, mesh.vertices, mesh.faces)  # Index of the image and mask pair
        IO().save_mesh(new_mesh, f"uvtest/data/lemon/lemon_{i}.obj" )
        print(f'mesh{i} is saved')