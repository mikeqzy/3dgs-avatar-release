#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def augm_rots(roll_range=90, pitch_range=90, yaw_range=90):
    """ Get augmentation for rotation matrices.
    Args:
        roll_range (int): roll angle sampling range (train mode) or value (test mode)
        pitch_range (int): pitch angle sampling range (train mode) or value (test mode)
        yaw_range (int): yaw angle sampling range (train mode) or value (test mode)
    Returns:
        rot_mat (4 x 4 float numpy array): homogeneous rotation augmentation matrix.
    """
    # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
    # Roll
    rot_x = min(2*roll_range,
            max(-2*roll_range, np.random.randn()*roll_range))

    sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
    rot_x = np.eye(3)
    rot_x[1, 1] = cs
    rot_x[1, 2] = -sn
    rot_x[2, 1] = sn
    rot_x[2, 2] = cs

    rot_y = min(2*pitch_range,
            max(-2*pitch_range, np.random.rand()*pitch_range))

    # Pitch
    sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
    rot_y = np.eye(3)
    rot_y[0, 0] = cs
    rot_y[0, 2] = sn
    rot_y[2, 0] = -sn
    rot_y[2, 2] = cs

    rot_z = min(2*yaw_range,
            max(-2*yaw_range, np.random.randn()*yaw_range))

    # Yaw
    sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
    rot_z = np.eye(3)
    rot_z[0, 0] = cs
    rot_z[0, 1] = -sn
    rot_z[1, 0] = sn
    rot_z[1, 1] = cs

    rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))

    return rot_mat