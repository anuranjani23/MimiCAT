"""Name-free geometric joint features (inputs to G and H). No joint names are used here."""
import numpy as np

FEAT_DIM = 18


def joint_features(v, joints, parents, weights):
    """Returns feats (J, FEAT_DIM), center (3,), scale (float).
    Coordinates are normalised by the character's bounding box (centre, longest side)."""
    v = np.asarray(v, np.float64); joints = np.asarray(joints, np.float64)
    w = np.asarray(weights, np.float64)
    lo, hi = v.min(0), v.max(0)
    center = (lo + hi) / 2
    scale = float((hi - lo).max()) + 1e-8
    j = (joints - center) / scale
    vn = (v - center) / scale
    J = len(joints)

    depth = np.zeros(J); n_child = np.zeros(J); size = np.ones(J)
    for k in range(J):
        p = parents[k]
        if p >= 0:
            depth[k] = depth[p] + 1
            n_child[p] += 1
    for k in range(J - 1, 0, -1):
        if parents[k] >= 0:
            size[parents[k]] += size[k]

    par = np.where(parents >= 0, parents, np.arange(J))
    offset = j - j[par]
    bone = np.linalg.norm(offset, axis=1)

    mass = w.sum(0)
    wn = w / (mass[None, :] + 1e-8)
    cen = wn.T @ vn
    var = np.clip(wn.T @ (vn ** 2) - cen ** 2, 0, None)
    std = np.sqrt(var)
    frac = mass / (mass.sum() + 1e-8)

    feats = np.concatenate([
        j,                                   # 3 position
        offset * 10,                         # 3 offset to parent
        np.log1p(bone * 20)[:, None],        # 1 bone length
        (depth / 10)[:, None],               # 1 depth
        (n_child / 4)[:, None],              # 1 children
        (n_child == 0).astype(float)[:, None],  # 1 leaf
        (size / J)[:, None],                 # 1 subtree size
        np.log1p(frac * J)[:, None],         # 1 skin mass
        (cen - j) * 5,                       # 3 skin centroid offset
        std * 5,                             # 3 skin spread
    ], axis=1)
    assert feats.shape[1] == FEAT_DIM
    return feats.astype(np.float32), center.astype(np.float32), scale


def subsample_points(v, center, scale, n=512, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=min(n, len(v)), replace=False)
    return ((v[idx] - center) / scale).astype(np.float32)
