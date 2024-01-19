import numpy as np
import trimesh
import mcubes
import torch
from matplotlib import pyplot as plt
import skimage.measure
from MeshUDF.custom_mc._marching_cubes_lewiner import udf_mc_lewiner
import io
from PIL import Image
from pytorch3d.structures import Meshes,  Pointclouds, Volumes

def save_mesh_image_with_camera(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))
    fig.set_facecolor((1, 1, 1))
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], shade=True, color='blue')

    ax.view_init(elev=90, azim=180)  
    ax.dist = 8  
    ax.set_box_aspect([1,1,1.4]) 
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    plt.axis('off')  
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close()
    return img
def latent_to_mesh(decoder, latent_idx,device,size=64):
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]
    grid_points = create_grid_points_from_bounds(mini, maxi, size)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    logits = get_logits(decoder, latent_idx, grid_points=grid_points,nbatch_points=2000)
    try:
        mesh_sdf = mesh_from_logits(logits, mini, maxi,size)
    except:
        print('no mesh found for sdf')
        mesh_sdf = None
    try:
        verts, faces ,_,_ = udf_mc_lewiner(np.abs(logits.reshape(size,size,size).astype(np.float32)), grid_points.detach().cpu().numpy().reshape(size,size,size,3).astype(np.float32))
        mesh_udf = trimesh.Trimesh(verts, faces, process=False)
    except: 
        print('no mesh found for udf')
        mesh_udf = None
    return mesh_sdf ,  mesh_udf

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

    #logits *= -1

    # padding to ba able to retrieve object close to bounding box bondary
    # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1000)
    threshold = 0.015
    # vertices, triangles = mcubes.marching_cubes(logits, threshold
    try:
        vertices, triangles,_,_ = skimage.measure.marching_cubes(logits, threshold)
    except:
        return None
        
    # rescale to original scale
    step = (np.array(maxi) - np.array(mini)) / (resolution - 1)
    vertices = vertices * np.expand_dims(step, axis=0)
    vertices += [mini[0], mini[1], mini[2]]
    # return pytorch3d mesh
    #return Meshes(verts=[torch.from_numpy(vertices.copy()).float()], faces=[torch.from_numpy(triangles.copy()).int()])
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
            logits = decoder(torch.cat([points, encoding.repeat(1, points.shape[1], 1)],dim=-1))
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits
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
                anchors=None,
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
                d = deformer(points, glob_cond)
            delta_list.append(d.detach().clone())

            torch.cuda.empty_cache()
        delta = torch.cat(delta_list, dim=1)

    pred_posed = points_neutral[:, :, :3] + delta.squeeze()
    verts = pred_posed.detach().cpu().squeeze().numpy()
    mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

    return mesh_deformed

