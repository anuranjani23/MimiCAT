"""Render source frames next to transferred target frames (matplotlib point clouds).

Single variant:
    python -m mimicat.viz.render_transfer --src pm0025_17_00 --tgt pm0100_00_00 \
        --corr g --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h.pt --out outputs/demo1.png
Side-by-side comparison (one row per variant; "+h" adds H):
    python -m mimicat.viz.render_transfer --src ... --tgt ... \
        --variants hard,g,g+h,oracle --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h.pt
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
from ..correspondence.provider import CorrProvider, load_threshold
from ..eval.cycle import transfer_pose
from ..train.common import load_G, load_H


def parse_variant(spec):
    spec = spec.strip().lower()
    use_h = spec.endswith("+h")
    mode = spec[:-2] if use_h else spec
    if mode not in CorrProvider.MODES:
        raise SystemExit(f"unknown variant '{spec}' (modes: {CorrProvider.MODES}, add '+h' for H)")
    return mode, use_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--motion", default=None, help="motion file stem (default: first one)")
    ap.add_argument("--corr", default="oracle", choices=CorrProvider.MODES)
    ap.add_argument("--variants", default="", help="comma list, e.g. hard,g,g+h,oracle")
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt"); ap.add_argument("--h-ckpt", default=None)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--up", default="y", choices=["y", "z"])
    ap.add_argument("--out", default="outputs/transfer_demo.png")
    a = ap.parse_args()

    specs = [s for s in a.variants.split(",") if s.strip()] or \
            [a.corr + ("+h" if a.h_ckpt else "")]
    variants = [parse_variant(s) for s in specs]

    store = CharStore(DATA_ROOT, "cpu")
    S, T_ = store.get(a.src), store.get(a.tgt)
    mp = next((p for p in S.motion_paths if a.motion is None or p.stem == a.motion), None)
    if mp is None:
        raise SystemExit(f"motion '{a.motion}' not found for {a.src}")
    bp, tr = load_motion(mp)
    idx = np.linspace(0, len(bp) - 1, a.frames).astype(int)
    R_s = rotvec_to_mat(torch.from_numpy(bp[idx]))

    need_G = any(m.startswith("g") for m, _ in variants)
    need_H = any(h for _, h in variants)
    if need_H and not a.h_ckpt:
        raise SystemExit("a '+h' variant needs --h-ckpt")
    G = load_G(a.g_ckpt, "cpu") if need_G else None
    H = load_H(a.h_ckpt, "cpu") if need_H else None
    thr_g = load_threshold(a.g_ckpt) if need_G else 0.2

    provs = {}
    for m, _ in variants:
        if m not in provs:
            thr = thr_g if m in ("g", "g_hungarian") else 0.2
            provs[m] = CorrProvider(m, G=G, thr=thr)

    rows = []   # (X, title)
    with torch.no_grad():
        z = torch.zeros(a.frames, 3)
        Xs = pose_vertices_torch(S.v, S.joints, S.parents, S.weights, R_s, z).numpy()
        rows.append((Xs, f"source {a.src}"))
        for (m, use_h), spec in zip(variants, specs):
            R_t, _, _, matched = transfer_pose(S, T_, R_s, provs[m], H if use_h else None)
            Xt = pose_vertices_torch(T_.v, T_.joints, T_.parents, T_.weights, R_t, z).numpy()
            rows.append((Xt, f"target {a.tgt} [{spec}]  driven {int(matched.sum())}/{len(matched)}"))
            print(f"{spec:12s} driven target joints: {int(matched.sum())} of {len(matched)}")

    order = {"y": (0, 2, 1), "z": (0, 1, 2)}[a.up]
    nr = len(rows)
    fig = plt.figure(figsize=(3 * a.frames, 3 * nr))
    for r, (X, title) in enumerate(rows):
        lo, hi = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
        c, s = (lo + hi) / 2, (hi - lo).max() / 2
        for f in range(a.frames):
            ax = fig.add_subplot(nr, a.frames, r * a.frames + f + 1, projection="3d")
            P = ((X[f] - c) / s)[::2][:, order]
            ax.scatter(*P.T, s=0.4)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_axis_off()
            if f == 0:
                ax.set_title(title, fontsize=8, loc="left")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()