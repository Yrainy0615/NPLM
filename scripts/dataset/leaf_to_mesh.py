import cv2
import numpy as np
from skimage.measure import subdivide_polygon
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import os
from scipy.spatial import Delaunay , Voronoi, voronoi_plot_2d, ConvexHull, transform
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
    TexturesVertex)
from matplotlib import pyplot as plt

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
        self.all_images.sort()
        self.all_masks = [i for i in os.listdir(root) if i.endswith('.JPG') and 'mask' in i]
        self.all_masks.sort()
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=128, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
            perspective_correct=False, 
        )
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
    
    def visualize_mesh(self, mesh,deform, rotation_axis):
        # move vertices to center and (-1,1)
        mesh.vertices = normalize_to_center(mesh.vertices)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # plot vertex
        ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], c='r', label='Source')
        #ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
        # plot deform
        ax.scatter(deform[:, 0], deform[:, 1], deform[:, 2], c='b', label='Target')
        # plot rotation_axis
        ax.quiver(0, 0, 0, rotation_axis[0], rotation_axis[1], rotation_axis[2], length=1, normalize=True)
        
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
    
    def get_corner(self, mask):
        # 获取mask的多个角点
        mask = cv2.resize(mask,(256,256))
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #corners = cv2.goodFeaturesToTrack(thresh, maxCorners=7, qualityLevel=0.01, minDistance=30)
        dst = cv2.cornerHarris(thresh,2,3,0.1)
        mask[dst>0.01*dst.max()]=[0,0,255]
       # plot corner
        for i in corners:
            x, y = i.ravel()[0], i.ravel()[1]
            center = (int(x), int(y))
            cv2.circle(mask, center, 5, 255, -1)
        plt.imshow(mask)   
        

    def get_vertex(self,mask):
        mask = cv2.resize(mask,(256,256))
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours and approximate the contour with less points
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # draw the contour on the original image
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        
        # get a approximate contour
        contour = np.vstack(contours).squeeze()

        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 1, True)

        approx  = approx.squeeze()

        # PCA to get the major axis
        pca = PCA(n_components=1)
        pca.fit(contour)
        #major_axis_vector = pca.components_[0]
        s = np.array([mask.shape[0]//2,mask.shape[1]//2]) 
        e = np.array([mask.shape[0]//2,mask.shape[1]//2+1])
        major_axis_vector = e - s    

        centroid = pca.mean_
        # 
        # print rgb value of contour
        try:     
        # 计算主轴与轮廓的交点
            intersection = np.array([np.cross(major_axis_vector, contour[i] - centroid) for i in range(len(contour))])
            projection = np.dot(contour-centroid, major_axis_vector)
        except ValueError:
            print('ValueError')
            return None,None
        # if value error break and continue next image
     
        min_index = np.argmin(projection)
        max_index = np.argmax(projection)   
        endpoint1 =contour[min_index]
        endpoint2 = contour[max_index]
                # plot endpoints
        # plt.figure()
        # plt.scatter(endpoint1[0], endpoint1[1], c='r', label='Source')        
        # plt.scatter(endpoint2[0], endpoint2[1], c='r', label='Source')
        # plt.scatter(contour[:,0], contour[:,1], c='b', label='Source')
        # plot major axis major_axis_vector
        
        if endpoint1[1] > endpoint2[1]:
            endpoint1, endpoint2 = endpoint2, endpoint1
        
        # Subdivide the contour into smaller segments
        num_divisions = 10  # 18 divisions create 20 segments
        points_on_major_axis = np.linspace(endpoint1, endpoint2, num_divisions ).squeeze()  # Exclude first and last
        

        # combine the points on major axis and contour
        points = np.vstack((points_on_major_axis, approx))
        
        # Each line will be defined by a point on the major axis and the direction perpendicular to the major axis
        perpendicular_direction = np.array([-major_axis_vector[1], major_axis_vector[0]]).T

        # Initialize a list to hold all intersection points
        intersection_points = []
        vertex_list = []
        vertex_list.append(tuple(endpoint1.astype(np.int16)))
        vertex_list.append(tuple(endpoint2.astype(np.int16)))
        hull = cv2.convexHull(contour)
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
        return  approx, points_on_major_axis 


    def compute_perpendicular_bisector_3d(self, A, B):
        AB = B - A
        # midpoint of AB
        midpoint = (A + B) / 2
        # normal vector of AB
        normal = np.array([AB[1], -AB[0], 0])
        # normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        return  normal
        
    def to_mesh(self, mask, image):
        # read img and mask
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512,512))
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512,512))
        assert image.shape[:2] == mask.shape, "Image and mask must have the same dimensions."
        
        # get vertex from mask
        leaf_indices = np.where(mask > 0)
        vertices = np.column_stack((leaf_indices[1], leaf_indices[0], np.zeros_like(leaf_indices[0])))
        tri = Delaunay(vertices[:, :2])
        faces = tri.simplices
        vertex_colors = image[leaf_indices[0], leaf_indices[1]]
        valid_faces = []
        # delete faces which is not in maskl
        def is_point_inside_mask(point, mask):
            x, y = int(point[0]), int(point[1])
            return mask[y, x] != 0
        for face in faces:
            centroid = np.mean([vertices[vertex] for vertex in face], axis=0)
            if is_point_inside_mask(centroid, mask):
                valid_faces.append(face)
        valid_faces = np.array(valid_faces)
        # Create a mesh with only the faces that are completely inside the mask
        mesh = trimesh.Trimesh(vertices=vertices, faces=valid_faces,vertex_colors=vertex_colors)

        # Remove the extraneous faces from the mesh

        return mesh


        
        

        
    
        


if __name__ == "__main__":
    root_directory = 'dataset/LeafData/Lemon/healthy'  # Update this path
    leaf_to_mesh = LeaftoMesh(root_directory)
    
    for i in range(len(leaf_to_mesh.all_masks)):
        mask_path = os.path.join(root_directory, leaf_to_mesh.all_masks[i])
        image_path = mask_path.replace('_mask_aligned', '_aligned')
    
        mesh = leaf_to_mesh.to_mesh(mask_path, image_path)
        #mesh = repair_mesh(mesh)
        mesh.export(f'dataset/Mesh_colored/Lemon{i}.obj', include_color=True)
        print('{} is saved'.format(f'dataset/Mesh_colored/Lemon{i}.obj') )

        

