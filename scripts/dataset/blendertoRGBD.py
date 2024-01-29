import blenderproc as bproc
import os
import time
import math
import argparse
import numpy as np
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.scripts.saveAsImg import save_array_as_image
from blenderproc.python.writer.BopWriterUtility import _BopWriterUtility 

"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.
"""
import json
import bpy
from mathutils import Vector


def add_lighting() -> None:
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 70000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
        mesh = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.material_slot_add()
        mat_slot = mesh.material_slots[0]
        # Create a new material and set up its shader nodes
        mat = bpy.data.materials.new(name="Vertex")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        attr_node = nodes.new(type='ShaderNodeAttribute')
        attr_node.attribute_name = "Color"  # Replace with the name of your vertex color layer
        # Connect the nodes
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        links.new(attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        # Assign the material to the object
        mat_slot.material = mat
        # Switch back to object mode and deselect everything
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(format="glb"):
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, args.camera_dist, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def sample_camera_loc(phi=None, theta=None, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(object_file).replace(".obj", "")
    # load the object
    load_object(object_file)
    # bproc.utility.reset_keyframes()
    object_uid = os.path.basename(object_file).split(".")[0]
    object_format = os.path.basename(object_file).split(".")[-1]
    normalize_scene(object_format)
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
   # fix random seed
    np.random.seed(0)
    img_ids = [f"{view_num}.png" for view_num in range(24)]
    # polar_angles = np.radians([60] * 12 + [90] * 12)
    polar_angles = np.radians(np.random.uniform(15, 75, 24))
    azimuths = np.radians(np.random.uniform(180, 360, 24))

    location_list = []
    rotation_list = []
    cam2world_list = []
    for i in range(len(img_ids)):
        # Sample random camera location around the object
        location = sample_camera_loc(polar_angles[i], azimuths[i], args.camera_dist)
        location_list.append(location.tolist())
        # Compute rotation based on vector going from location towards the location of the object
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - location)
        rotation_list.append(rotation_matrix.tolist())
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        cam2world_list.append(cam2world_matrix.tolist())
        print(bproc.camera.add_camera_pose(cam2world_matrix))
    
    RendererUtility.set_cpu_threads(20)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False, convert_to_distance=False)
    bproc.renderer.set_output_format(enable_transparency=True)

    data = bproc.renderer.render(verbose=True)
    #bproc.writer.write_hdf5(args.output_dir, data)
    for index, (image, depth, normal) in enumerate(zip(data["colors"], data["depth"],data['normals'])):
        save_folder = os.path.join(args.output_dir, filename)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = "render_" + str(index) + ".png"
        render_path = os.path.join(save_folder, save_name)
        save_array_as_image(image, "colors", render_path)
        depth_mm = 1000.0 * depth  # [m] -> [mm]
        _BopWriterUtility.save_depth(render_path.replace('.png', '_depth.png'), im=depth_mm)
    # save camera pose as dict to json file


    # 定义包含 NumPy 数组的字典
    camera = {
        "location": location_list,
        "rotation_matrix": rotation_list,
        "cam2world_matrix": cam2world_list,
        "polar_angle": polar_angles.tolist(),
        "azimuth": azimuths.tolist(),
    }
    for key, value in camera.items():
        print(key, ":",type(value))
    camera_path = os.path.join(save_folder,"camera.json")
    with open(camera_path, 'w') as f:
        json.dump(camera, f, indent=4)

        print("{} is  saved".format(camera_path))


    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        default='/home/yang/projects/parametric-leaf/dataset/Mesh_colored/deformed',
        help="Path to the object file",
    )
    parser.add_argument("--output_dir", type=str, default="/home/yang/projects/parametric-leaf/views")
    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--camera_dist", type=float, default=1.5)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    bproc.init()
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = args.engine
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
    # scene.view_settings.view_transform = "Standard"
    print("display device", scene.display_settings.display_device)
    print("view type", scene.view_settings.view_transform)
    print("view exposure", scene.view_settings.exposure)
    print("view gamma", scene.view_settings.gamma)
    args = parser.parse_args()
    start_i = time.time()    
    mesh_path = os.path.join(args.object_path, args.mesh)
    print("{} is loaded".format(mesh_path))
    save_images(mesh_path)
    end_i = time.time()
    print("Finished", args.object_path, "in", end_i - start_i, "seconds")
