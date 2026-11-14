"""Pose transfer + cycle-consistency error."""
import torch

from ..correspondence.transfer import blend_rotations
from ..geometry.lbs import pose_vertices_torch


def transfer_pose(src, tgt, R_src, provider, H=None):
    """-> R_tgt (T,Jt,3,3), R_init, delta (None without H), matched"""
    W, matched, _ = provider(src, tgt)
    R_init = blend_rotations(R_src, W, matched)
    if H is None:
        return R_init, R_init, None, matched
    R_out, delta = H(tgt.feats, tgt.pts, src.feats, src.pts, R_src, R_init, matched)
    return R_out, R_init, delta, matched


def vertex_error(ch, R_a, R_b):
    """Mean vertex distance between two poses of the same character, / bbox size."""
    z = torch.zeros(R_a.shape[0], 3, dtype=ch.v.dtype, device=ch.v.device)
    xa = pose_vertices_torch(ch.v, ch.joints, ch.parents, ch.weights, R_a, z)
    xb = pose_vertices_torch(ch.v, ch.joints, ch.parents, ch.weights, R_b, z)
    return (xa - xb).norm(dim=-1).mean() / ch.scale


def cycle_error(a, b, R_a, provider_ab, provider_ba=None, H=None):
    """A -> B -> A. Returns (error, R_b, deltas)."""
    provider_ba = provider_ba or provider_ab
    R_b, _, d1, _ = transfer_pose(a, b, R_a, provider_ab, H)
    R_a2, _, d2, _ = transfer_pose(b, a, R_b, provider_ba, H)
    return vertex_error(a, R_a, R_a2), R_b, [d for d in (d1, d2) if d is not None]
