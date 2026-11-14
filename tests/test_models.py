"""Shape / gradient smoke tests that need no dataset."""
import numpy as np, torch
from mimicat.models.tokens import joint_features, subsample_points, FEAT_DIM
from mimicat.models.corr_transformer import CorrTransformer
from mimicat.models.pose_transformer import PoseTransformer
from mimicat.correspondence.soft import (row_col_probs, mutual, log_sinkhorn, weights_from_M,
                                         gt_probs, corr_loss)
from mimicat.correspondence.transfer import blend_rotations, rot6d_to_matrix, matrix_to_rot6d
from mimicat.correspondence.labels import build_M_gt
from mimicat.geometry.lbs import rotvec_to_mat, pose_vertices_torch, pose_vertices


def fake_char(J, V=300, seed=0):
    rng = np.random.default_rng(seed)
    parents = np.array([-1] + [int(rng.integers(0, j)) for j in range(1, J)])
    joints = rng.normal(0, .3, (J, 3)).astype(np.float32)
    v = rng.normal(0, .3, (V, 3)).astype(np.float32)
    w = rng.random((V, J)).astype(np.float32); w /= w.sum(1, keepdims=True)
    f, c, s = joint_features(v, joints, parents, w)
    return dict(feats=torch.from_numpy(f), pts=torch.from_numpy(subsample_points(v, c, s, 128)),
                v=torch.from_numpy(v), joints=torch.from_numpy(joints), parents=parents,
                weights=torch.from_numpy(w))


def test_rot6d_roundtrip():
    R = rotvec_to_mat(torch.randn(5, 3))
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(R)), R, atol=1e-5)


def test_blend_identity():
    R = rotvec_to_mat(torch.randn(2, 6, 3))
    out = blend_rotations(R, torch.eye(6), torch.ones(6, dtype=torch.bool))
    assert torch.allclose(out, R, atol=1e-5)


def test_G_shapes_and_grad():
    a, b = fake_char(12, seed=1), fake_char(20, seed=2)
    G = CorrTransformer()
    S = G(a["feats"], a["pts"], b["feats"], b["pts"])
    assert S.shape == (13, 21)
    na = ["pm1_00", "origin", "waist", "hips"] + [f"spine_{k:02d}" for k in range(8)]
    nb = ["pm2_00", "origin", "waist", "hips"] + [f"tail_{k:02d}" for k in range(16)]
    _, M_aug, _, _ = build_M_gt(na, nb)
    pr, pc = gt_probs(M_aug)
    loss = corr_loss(S, pr, pc); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in G.parameters() if p.grad is not None)
    p_row, p_col = row_col_probs(S.detach())
    assert mutual(p_row, p_col).shape == (12, 20) and log_sinkhorn(S.detach()).shape == (12, 20)


def test_H_starts_at_init_and_grads():
    a, b = fake_char(12, seed=1), fake_char(20, seed=2)
    H = PoseTransformer()
    R_src = rotvec_to_mat(torch.randn(3, 12, 3) * .3)
    W, matched = weights_from_M(torch.rand(12, 20))
    R_init = blend_rotations(R_src, W, matched)
    R_out, delta = H(b["feats"], b["pts"], a["feats"], a["pts"], R_src, R_init, matched)
    assert R_out.shape == (3, 20, 3, 3) and torch.allclose(R_out, R_init, atol=1e-5)
    (R_out.sum() + delta.sum()).backward()


def test_lbs_torch_vs_numpy():
    c = fake_char(10, seed=3)
    rv = (torch.randn(2, 10, 3) * .4)
    out = pose_vertices_torch(c["v"], c["joints"], c["parents"], c["weights"], rotvec_to_mat(rv), torch.zeros(2, 3))
    ref = np.stack([pose_vertices(c["v"].numpy(), c["joints"].numpy(), c["parents"], c["weights"].numpy(),
                                  rv[i].numpy(), np.zeros(3, np.float32)) for i in range(2)])
    assert np.abs(out.numpy() - ref).max() < 1e-4
