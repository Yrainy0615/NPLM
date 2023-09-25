import numpy as np
import trimesh
import mcubes
import torch

def create_grid_points_from_bounds(minimun, maximum, res, scale=None):
    if scale is not None:
        res = int(scale * res)
        minimun = scale * minimun
        maximum = scale * maximum
    x = np.linspace(minimun[0], maximum[0], res)
    y = np.linspace(minimun[1], maximum[1], res)
    z = np.linspace(minimun[2], maximum[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def mesh_from_logits(logits, mini, maxi, resolution):
    logits = np.reshape(logits, (resolution,) * 3)

    logits *= -1

    # padding to ba able to retrieve object close to bounding box bondary
    # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1000)
    threshold = 0.0
    vertices, triangles = mcubes.marching_cubes(logits, threshold)

    # rescale to original scale
    step = (np.array(maxi) - np.array(mini)) / (resolution - 1)
    vertices = vertices * np.expand_dims(step, axis=0)
    vertices += [mini[0], mini[1], mini[2]]

    return trimesh.Trimesh(vertices, triangles)

def get_logits(decoder,
               encoding,
               grid_points,
               nbatch_points=100000,
               return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            logits, anchors = decoder(points, encoding.repeat(1, points.shape[1], 1), None)
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits, anchors
    else:
        return logits


def get_logits_backward(decoder_shape,
                        decoder_expr,
                        encoding_shape,
                        encoding_expr,
                        grid_points,
                        nbatch_points=100000,
                        return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            # deform points
            if encoding_expr is not None:
                offsets, _ = decoder_expr(points, encoding_expr.repeat(1, points.shape[1], 1), None)
                points_can = points + offsets
            else:
                points_can = points
            # query geometry in canonical space
            logits, anchors = decoder_shape(points_can, encoding_shape.repeat(1, points.shape[1], 1), None)
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits, anchors
    else:
        return logits


def deform_mesh(mesh,
                deformer,
                lat_rep,
                anchors,
                lat_rep_shape=None):
    points_neutral = torch.from_numpy(np.array(mesh.vertices)).float().unsqueeze(0).to(lat_rep.device)

    with torch.no_grad():
        grid_points_split = torch.split(points_neutral, 5000, dim=1)
        delta_list = []
        for split_id, points in enumerate(grid_points_split):
            if lat_rep_shape is None:
                glob_cond = lat_rep.repeat(1, points.shape[1], 1)
            else:
                glob_cond = torch.cat([lat_rep_shape, lat_rep], dim=-1)
                glob_cond = glob_cond.repeat(1, points.shape[1], 1)
            if anchors is not None:
                d, _ = deformer(points, glob_cond, anchors.unsqueeze(1).repeat(1, points.shape[1], 1, 1))
            else:
                d, _ = deformer(points, glob_cond, None)
            delta_list.append(d.detach().clone())

            torch.cuda.empty_cache()
        delta = torch.cat(delta_list, dim=1)

    pred_posed = points_neutral[:, :, :3] + delta.squeeze()
    verts = pred_posed.detach().cpu().squeeze().numpy()
    mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

    return mesh_deformed

