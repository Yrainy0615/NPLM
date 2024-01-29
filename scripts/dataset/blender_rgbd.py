import blenderproc as bproc
import numpy as np
import argparse
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('object', nargs='?', default="dataset/Mesh_colored/deformed/Bael_0_d6.obj", help="Path to the model file")
parser.add_argument('camera_dist', nargs='?', type=float, default=2, help="Camera distance from the object")
parser.add_argument('output_dir', nargs='?', default="views", help="Path to where the final files will be saved")
args = parser.parse_args()

bproc.init()
context = bpy.context
scene = context.scene
render = scene.render

render.engine = 'CYCLES'
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


# load aobj
obj = bproc.loader.load_obj(args.object)[0]
obj.set_cp('category_id', 1)
# 使用顶点颜色进行纹理映射
for mat in obj.get_materials():
    mat.map_vertex_color()

# 设置光源
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)
def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, args.camera_dist, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint
# intrisic matrix
# bproc.camera.set_intrinsics_from_K_matrix(
#     [[35, 0, 16],  
#      [0, 35, 16],  
#      [0, 0, 1]], 32, 32  
# )
def sample_camera_loc(phi=None, theta=None, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])
cam, cam_constraint = setup_camera()

np.random.seed(0)
polar_angles = np.radians(np.random.uniform(15, 75, 24))
azimuths = np.radians(np.random.uniform(180, 360, 24))
for i in range(24):
    location = sample_camera_loc(polar_angles[i], azimuths[i], args.camera_dist)
    rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, 0] - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# 启用深度渲染
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# 渲染整个管线
data = bproc.renderer.render()

# 以BOP格式写入数据
bproc.writer.write_bop(args.output_dir, [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)
