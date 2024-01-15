import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement

# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Jtr,):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)