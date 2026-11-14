"""Gate test: our batched torch LBS must reproduce the official demo's pose_vertices."""
import sys, numpy as np, torch, pytest
sys.path.insert(0, ".")
from mimicat.geometry.lbs import pose_vertices_torch, rotvec_to_mat, pose_vertices


def test_parent_order(sample):
    p = sample["parents"]
    assert p[0] < 0 and all(p[j] < j for j in range(1, len(p)))


def test_numpy_matches_official(sample):
    official = pytest.importorskip("quick_demo").pose_vertices
    s = sample
    for f in (0, 10, len(s["body_pose"]) - 1):
        a = official(s["v"], s["joints"], s["parents"], s["weights"], s["body_pose"][f], s["transl"][f])
        b = pose_vertices(s["v"], s["joints"], s["parents"], s["weights"], s["body_pose"][f], s["transl"][f])
        assert np.abs(a - b).max() < 1e-4


def test_torch_matches_numpy(sample):
    s = sample
    frames = [0, 5, len(s["body_pose"]) - 1]
    ref = np.stack([pose_vertices(s["v"], s["joints"], s["parents"], s["weights"],
                                  s["body_pose"][f], s["transl"][f]) for f in frames])
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    R = rotvec_to_mat(t(s["body_pose"][frames]))
    out = pose_vertices_torch(t(s["v"]), t(s["joints"]), s["parents"], t(s["weights"]),
                              R, t(s["transl"][frames])).numpy()
    assert np.abs(out - ref).max() < 1e-3
