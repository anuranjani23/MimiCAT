"""Forward kinematics + linear blend skinning, mirroring quick_demo.py exactly.

Conventions (from quick_demo.py):
  * pose is axis-angle, shape (J, 3) per frame
  * L_j = [R_j | c_j - R_j c_j]  with c_j the REST joint position
  * G_j = G_parent @ L_j  (parents precede children)
  * skinning: blended = tensordot(weights, G); no inverse-rest matrix
  * translation is added after skinning
"""
import numpy as np
import torch


def pose_vertices(vertices, joints, parents, weights, pose_rotvec, transl):
    """numpy reference (same maths as quick_demo.pose_vertices)."""
    from scipy.spatial.transform import Rotation
    J = joints.shape[0]
    rots = Rotation.from_rotvec(pose_rotvec).as_matrix().astype(np.float32)
    G = np.zeros((J, 4, 4), dtype=np.float32)
    G[:, 3, 3] = 1.0
    for j in range(J):
        L = np.eye(4, dtype=np.float32)
        L[:3, :3] = rots[j]
        L[:3, 3] = joints[j] - rots[j] @ joints[j]
        p = parents[j]
        G[j] = L if p < 0 else G[p] @ L
    blended = np.tensordot(weights, G, axes=(1, 0))
    vh = np.concatenate([vertices, np.ones((len(vertices), 1), np.float32)], 1)
    posed = np.einsum("nij,nj->ni", blended, vh)[:, :3]
    return posed + transl.astype(np.float32)


def rotvec_to_mat(r):
    """Rodrigues. r: (...,3) -> (...,3,3)."""
    th = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / th
    K = torch.zeros(*r.shape[:-1], 3, 3, dtype=r.dtype, device=r.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    I = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def pose_vertices_torch(v, joints, parents, weights, rotmats, transl):
    """Batched over frames.  v (V,3) joints (J,3) parents (J,) weights (V,J)
    rotmats (T,J,3,3) transl (T,3)  ->  (T,V,3)"""
    T, J = rotmats.shape[:2]
    dt, dev = v.dtype, v.device
    G = [None] * J
    eye = torch.eye(4, dtype=dt, device=dev)
    for j in range(J):
        L = eye.repeat(T, 1, 1)
        L[:, :3, :3] = rotmats[:, j]
        L[:, :3, 3] = joints[j] - rotmats[:, j] @ joints[j]
        p = int(parents[j])
        G[j] = L if p < 0 else G[p] @ L
    G = torch.stack(G, dim=1)                                   # (T,J,4,4)
    blended = torch.einsum("vj,tjab->tvab", weights, G)
    vh = torch.cat([v, torch.ones(len(v), 1, dtype=dt, device=dev)], 1)
    return torch.einsum("tvab,vb->tva", blended, vh)[..., :3] + transl[:, None]
