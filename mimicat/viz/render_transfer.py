"""Render source frames next to transferred target frames (matplotlib point clouds).

    python -m mimicat.viz.render_transfer --src pm0025_17_00 --tgt pm0100_00_00 \
        --corr oracle --frames 4 --out outputs/transfer_demo.png
    (add --corr g --g-ckpt ... and --h-ckpt ... to use trained models)
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import DATA_ROOT
from ..data.tensors import CharStore
from ..data.dataset import load_motion
from ..geometry.lbs import pose_vertices_torch, rotvec_to_mat
from ..correspondence.provider import CorrProvider
from ..eval.cycle import transfer_pose
from ..train.common import load_G, load_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--motion", default=None, help="motion file stem (default: first one)")
    ap.add_argument("--corr", default="oracle", choices=CorrProvider.MODES)
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt"); ap.add_argument("--h-ckpt", default=None)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--up", default="y", choices=["y", "z"])
    ap.add_argument("--out", default="outputs/transfer_demo.png")
    a = ap.parse_args()

    store = CharStore(DATA_ROOT, "cpu")
    S, T_ = store.get(a.src), store.get(a.tgt)
    mp = next((p for p in S.motion_paths if a.motion is None or p.stem == a.motion), None)
    bp, tr = load_motion(mp)
    idx = np.linspace(0, len(bp) - 1, a.frames).astype(int)
    R_s = rotvec_to_mat(torch.from_numpy(bp[idx]))
    G = load_G(a.g_ckpt) if a.corr in ("g", "g_sinkhorn") else None
    H = load_H(a.h_ckpt) if a.h_ckpt else None
    prov = CorrProvider(a.corr, G=G)
    with torch.no_grad():
        R_t, _, _, matched = transfer_pose(S, T_, R_s, prov, H)
        z = torch.zeros(a.frames, 3)
        Xs = pose_vertices_torch(S.v, S.joints, S.parents, S.weights, R_s, torch.zeros(a.frames, 3)).numpy()
        Xt = pose_vertices_torch(T_.v, T_.joints, T_.parents, T_.weights, R_t, z).numpy()
    order = {"y": (0, 2, 1), "z": (0, 1, 2)}[a.up]
    fig = plt.figure(figsize=(3 * a.frames, 6))
    for r, (X, ch, title) in enumerate([(Xs, S, f"source {a.src}"), (Xt, T_, f"target {a.tgt} ({a.corr}{'+H' if H else ''})")]):
        lo, hi = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
        c, s = (lo + hi) / 2, (hi - lo).max() / 2
        for f in range(a.frames):
            ax = fig.add_subplot(2, a.frames, r * a.frames + f + 1, projection="3d")
            P = ((X[f] - c) / s)[::2][:, order]
            ax.scatter(*P.T, s=0.4)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_axis_off()
            if f == 0:
                ax.set_title(title, fontsize=8, loc="left")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out, "| driven target joints:", int(matched.sum()), "of", len(matched))


if __name__ == "__main__":
    main()
