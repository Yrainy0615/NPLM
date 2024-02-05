import numpy as np
import os
import bpy
import mathutils
from mathutils import Vector
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# set camera 
def setup_camera():
    lens = 35  
    sensor_width = 32  
    image_width_in_pixels = 640 
    image_height_in_pixels = 480  
    focal_length_in_pixels = (image_width_in_pixels * lens) / sensor_width  # 焦距（像素）

    # intrinsic
    c_x = image_width_in_pixels / 2
    c_y = image_height_in_pixels / 2
    K = np.array([
        [focal_length_in_pixels, 0, c_x],
        [0, focal_length_in_pixels, c_y],
        [0, 0, 1]
    ])
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_data.lens = lens
    cam_data.sensor_width = sensor_width
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # 设置相机位置和朝向
    cam_location = [10, 0, 0]  
    look_at = Vector((0,0,0)) 
    direction = look_at - cam_obj.location
    cam_obj.location = cam_location
    bpy.context.view_layer.update()
    
def setup_light():
    light = bpy.data.lights.new(name= 'light', type="POINT")
    light.energy = 1000
    light_object = bpy.data.objects.new(name='light', object_data=light)
    light_object.location = [4,2,3]
    bpy.context.collection.objects.link(light_object)
    
def read_material(obj):
    # load material from vertex color
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    attr = nodes.new(type='ShaderNodeAttribute')
    attr.attribute_name = 'Color'  

    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    links = mat.node_tree.links
    links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    obj.data.materials.append(mat)

def set_obj_rotation(obj, rotation_euler):
    obj.rotation_euler[0] = rotation_euler[0]  
    obj.rotation_euler[1] = rotation_euler[1]  
    obj.rotation_euler[2] = rotation_euler[2]  
    bpy.context.view_layer.update()

def set_obj_rotation(obj, rotation_euler):
    obj.rotation_euler[0] = rotation_euler[0]  
    obj.rotation_euler[1] = rotation_euler[1]  
    obj.rotation_euler[2] = rotation_euler[2]  
    bpy.context.view_layer.update()

# setup camera & light
setup_camera()
setup_light()

cam = bpy.data.objects['Camera']

# set track
track_to = cam.constraints.new(type = 'TRACK_TO')
empty = bpy.data.objects.new("Empty", None)
bpy.context.collection.objects.link(empty)
track_to.target = empty
bpy.context.view_layer.update()

# load data
root_path = ('/home/yang/projects/parametric-leaf/dataset/TestData/test_obj/bael/')
np.random.seed(6)
mesh_file = os.listdir(root_path)
mesh_file.sort()
for file in mesh_file:
    if '.obj' in file:
        mesh_path = os.path.join(root_path, file)
        bpy.ops.wm.obj_import(filepath = mesh_path )
        leaf = bpy.context.selected_objects[-1]
        read_material(leaf)


        # rotate and translate obj
        location = np.random.uniform(low=[-2, -3, -1], high=[2, 3, 1])
        rotation = np.radians(np.random.uniform(low = [60,60,60], high=[90,90,90]))
        set_obj_rotation(leaf, rotation)
        leaf.location = location




# rendering
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480

# Set up render layers
render_layers = bpy.context.scene.view_layers["ViewLayer"]
render_layers.use_pass_z = True  # Enable depth pass

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
nodes = tree.nodes

# Clear default nodes
for node in nodes:
    nodes.remove(node)

# Create Render Layers node
render_layers_node = nodes.new(type='CompositorNodeRLayers')

# Create Composite node (for RGB)
comp_node = nodes.new(type='CompositorNodeComposite')
comp_node.location = 0, -200

# Link Render Layers Image to Composite Image
links = tree.links
link = links.new(render_layers_node.outputs['Image'], comp_node.inputs['Image'])

# Create File Output node (for Depth)
file_output_node = nodes.new(type='CompositorNodeOutputFile')
file_output_node.location = 200, -200
file_output_node.base_path = os.path.join(root_path, 'output')  # Set your desired output path here
file_output_node.file_slots[0].path = 'depth_'  # Set your desired file name here

# Link Render Layers Depth to File Output node
link = links.new(render_layers_node.outputs['Depth'], file_output_node.inputs[0])

bpy.ops.render.render(write_still=True)
