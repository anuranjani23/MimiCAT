"""Rotation helpers and correspondence-driven initialisation of target transformations."""
import torch
import torch.nn.functional as F

IDENT6 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def matrix_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def blend_rotations(R, W, matched):
    """R (T,Js,3,3), W (Jt,Js), matched (Jt,) -> (T,Jt,3,3).
    Weighted sum of source rotations projected back to SO(3) (Gram-Schmidt, differentiable);
    unmatched target joints keep the identity."""
    Rb = torch.einsum("ji,tiab->tjab", W, R)
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    Rb = torch.where(matched[None, :, None, None], Rb, I)
    return rot6d_to_matrix(matrix_to_rot6d(Rb))


def geodesic_deg(R1, R2):
    c = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)))
