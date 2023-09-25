import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import os

# import deep_sdf.utils
def sample_shape_space(CFG):

    if CFG['local_shape']:
        out_dir = 'nphm_shape_space_samples_085'
    else:
        out_dir = 'npm_shape_space_samples_085'
    print(f'Saving Random Samples in {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    step = 0


    if False and latent_codes_shape is not None:
        lat_mean = torch.mean(latent_codes_shape.weight, dim=0)
        lat_std = torch.std(latent_codes_shape.weight, dim=0)
    else:
        if CFG['local_shape']:
            lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
            lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))
        else:
            lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'npm_lat_mean.npy'))
            lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'npm_lat_std.npy'))
    for i in range(100):
        lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).cuda()

        logits = get_logits(decoder_shape, lat_rep, grid_points, nbatch_points=25000)
        print('starting mcubes')

        mesh = mesh_from_logits(logits, mini, maxi, args.resolution)
        print('done mcubes')

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh)
        pl.reset_camera()
        pl.camera.position = (0, 0, 3)
        pl.camera.zoom(1.4)
        pl.set_viewup((0, 1, 0))
        pl.camera.view_plane_normal = (-0, -0, 1)
        #pl.show()
        pl.show(screenshot=out_dir + '/step_{:04d}.png'.format(step))
        mesh.export(out_dir + '/mesh_{:04d}.ply'.format(step))
        #print(pl.camera)
        step += 1





# deepsdf
def create_mesh(
    
     latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename



    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
          latent_vec
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )