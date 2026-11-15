"""Name-free geometric joint features (inputs to G and H). No joint names are used here.

Columns 0..17 are identical to v1 (anything indexing them keeps working).
Columns 18..39 add skeleton-topology and symmetry descriptors that mirror what the
label targets depend on: chain position, siblings, branching, subtree extent.
Assumes parents[k] < k (same assumption as v1)."""
import numpy as np

FEAT_DIM = 40


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def joint_features(v, joints, parents, weights):
    """Returns feats (J, FEAT_DIM), center (3,), scale (float).
    Coordinates are normalised by the character's bounding box (centre, longest side)."""
    v = np.asarray(v, np.float64); joints = np.asarray(joints, np.float64)
    w = np.asarray(weights, np.float64)
    parents = np.asarray(parents).astype(int)
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

    has_p = parents >= 0
    par = np.where(has_p, parents, np.arange(J))
    offset = j - j[par]
    bone = np.linalg.norm(offset, axis=1)

    mass = w.sum(0)
    wn = w / (mass[None, :] + 1e-8)
    cen = wn.T @ vn
    var = np.clip(wn.T @ (vn ** 2) - cen ** 2, 0, None)
    std = np.sqrt(var)
    frac = mass / (mass.sum() + 1e-8)

    # ---------------- topology / symmetry descriptors (new) ----------------
    D = np.linalg.norm(j[:, None, :] - j[None, :, :], axis=-1)
    lb = np.log1p(bone * 20)
    u = _unit(offset)

    height = np.zeros(J); n_leaf = np.zeros(J); path = np.zeros(J)
    for k in range(J - 1, -1, -1):                 # children before parents
        if n_child[k] == 0:
            n_leaf[k] = 1
        p = parents[k]
        if p >= 0:
            height[p] = max(height[p], height[k] + 1)
            n_leaf[p] += n_leaf[k]
    for k in range(J):                             # geodesic distance from root
        if parents[k] >= 0:
            path[k] = path[parents[k]] + bone[k]

    extent = np.zeros(J)                           # farthest descendant (spatial)
    for k in range(J):
        x = parents[k]
        while x >= 0:
            extent[x] = max(extent[x], D[x, k])
            x = parents[x]

    hops = np.zeros(J)                             # hops since last branching ancestor
    for k in range(J):
        x, h = parents[k], 0
        while x >= 0 and n_child[x] == 1:
            h += 1
            x = parents[x]
        hops[k] = h

    seen = np.zeros(J); sib_rank = np.zeros(J)
    for k in range(J):
        p = parents[k]
        if p >= 0:
            sib_rank[k] = seen[p]
            seen[p] += 1
    n_sib = np.where(has_p, n_child[par] - 1, 0)
    sib_rank_n = sib_rank / np.maximum(n_sib, 1)

    cos_par = np.zeros(J)                          # bend vs. parent bone
    for k in range(J):
        p = parents[k]
        if p >= 0 and parents[p] >= 0:
            cos_par[k] = float(u[k] @ u[p])
    child_cos = np.zeros(J); child_bone = np.zeros(J)
    for k in range(J):
        p = parents[k]
        if p >= 0:
            child_cos[p] += cos_par[k]
            child_bone[p] += lb[k]
    nc = np.maximum(n_child, 1)
    child_cos /= nc
    child_bone /= nc

    par_nchild = np.where(has_p, n_child[par] / 4, 0.0)
    par_bone = np.where(has_p, lb[par], 0.0)
    root = int(np.argmax(parents < 0))
    rel_root = j - j[root]
    total_leaves = max(1, int((n_child == 0).sum()))

    feats = np.concatenate([
        # ---- v1 columns (0..17) ----
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
        # ---- new columns (18..39) ----
        u,                                   # 3 unit bone direction
        np.abs(j),                           # 3 |position| (left/right symmetric)
        (height / 10)[:, None],              # 1 hops to deepest leaf
        (n_leaf / total_leaves)[:, None],    # 1 fraction of leaves below
        path[:, None],                       # 1 path length from root
        (hops / 10)[:, None],                # 1 hops since last branch (chain position)
        (n_child > 1).astype(float)[:, None],   # 1 is branch point
        (n_sib / 4)[:, None],                # 1 number of siblings
        sib_rank_n[:, None],                 # 1 sibling rank
        par_nchild[:, None],                 # 1 parent's child count
        par_bone[:, None],                   # 1 parent's bone length
        cos_par[:, None],                    # 1 bend vs parent bone
        child_bone[:, None],                 # 1 mean child bone length
        child_cos[:, None],                  # 1 mean child bend
        extent[:, None],                     # 1 subtree spatial extent
        rel_root * 3,                        # 3 position relative to root joint
    ], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape[1] == FEAT_DIM, feats.shape
    return feats.astype(np.float32), center.astype(np.float32), scale


def subsample_points(v, center, scale, n=512, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=min(n, len(v)), replace=False)
    return ((v[idx] - center) / scale).astype(np.float32)