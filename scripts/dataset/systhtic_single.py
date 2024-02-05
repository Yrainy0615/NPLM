import blenderproc as bproc
import numpy as np
import os
bproc.init()

leaf_path = "dataset/TestData/test_obj/bael"
leaf_name = os.listdir(leaf_path)
leaf_paths = [os.path.join(leaf_path, p) for p in leaf_name]
# fix seed for reproducibility
np.random.seed(0)
leaf_objects = [bproc.loader.load_obj(p)[0] for p in leaf_paths]

for leaf in leaf_objects:
    leaf.set_location(np.random.uniform(low=[-1, -1, -1], high=[1, 1, 1]))
    leaf.set_rotation_euler(np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi]))
    mat = bproc.types.Material()
    vertex_color = leaf.get_vertex_color()
    if vertex_color is None:
        mat.set_vertex_colors(vertex_color)
    leaf.set_material(mat)
# set camera
lens = 35  
sensor_width = 32  
image_width_in_pixels = 640  
image_height_in_pixels = 480  
focal_length_in_pixels = (image_width_in_pixels * lens) / sensor_width
c_x = image_width_in_pixels / 2
c_y = image_height_in_pixels / 2
K = np.array([
    [focal_length_in_pixels, 0, c_x],
    [0, focal_length_in_pixels, c_y],
    [0, 0, 1]
])
bproc.camera.set_intrinsics_from_K_matrix(K, image_width_in_pixels, image_height_in_pixels)

ratation_matrix = bproc.camera.rotation_from_forward_vec([-1, 0, 0])
cam2world = bproc.math.build_transformation_mat([5, 0, 0], ratation_matrix)    
bproc.camera.add_camera_pose(cam2world)

# set light
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 0, 5])
light.set_energy(1000)



bproc.renderer.enable_depth_output(activate_antialiasing=False, convert_to_distance=False)
data = bproc.renderer.render()
bproc.writer.write_hdf5("output", data)
