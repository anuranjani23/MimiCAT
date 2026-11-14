"""Create a small FAKE dataset in the real layout so the whole pipeline can be smoke-tested in
seconds (no download needed).

    python scripts/make_synthetic_data.py            # -> data/synthetic/data
    export MIMICAT_DATA=data/synthetic/data
"""
import argparse
from pathlib import Path
import numpy as np


def build_char(rng, plan, cid):
    names, parents, pos = [f"pm{cid:04d}_00_00", "origin", "waist", "hips"], [-1, 0, 1, 2], \
        [np.zeros(3), np.zeros(3), np.array([0, .45, 0.]), np.array([0, .42, 0.])]
    def add(n, p, off):
        names.append(n); parents.append(p); pos.append(pos[p] + np.asarray(off, float)); return len(names) - 1
    quad = plan in ("quad", "quad_wing")
    prev, n_sp = 3, int(rng.integers(1, 4))
    for k in range(n_sp):
        prev = add(f"spine_{k+1:02d}", prev, [0, 0.08, 0.06 if quad else 0])
    spine_top = prev
    neck = add("neck", spine_top, [0, 0.07, 0.04 if quad else 0]) if rng.random() < .6 else spine_top
    add("head", neck, [0, 0.08, 0.05 if quad else 0])
    for side, sx in (("left", .1), ("right", -.1)):
        p = add(f"{side}_leg_01", 3, [sx, -.05, 0])
        p = add(f"{side}_leg_02", p, [0, -.12, 0]); add(f"{side}_foot", p, [0, -.12, .03])
        if quad:
            p = add(f"{side}_arm_01", spine_top, [sx, -.02, .0])
            p = add(f"{side}_arm_02", p, [0, -.12, 0]); add(f"{side}_hand", p, [0, -.12, .03])
        else:
            s = add(f"{side}_shoulder", spine_top, [sx * .5, .0, 0])
            p = add(f"{side}_arm_01", s, [sx * 1.2, 0, 0])
            p = add(f"{side}_arm_02", p, [sx * 1.2, -.02, 0]); add(f"{side}_hand", p, [sx * .8, -.02, 0])
        if plan in ("wing", "quad_wing"):
            p = add(f"{side}_wing_a_01", spine_top, [sx * 1.5, .05, -.05])
            for k in (2, 3):
                p = add(f"{side}_wing_a_{k:02d}", p, [sx * 1.5, .03, -.02])
    if rng.random() < .7:
        p = 3
        for k in range(int(rng.integers(2, 5))):
            p = add(f"tail_{k+1:02d}", p, [0, -.02, -.1])
    for side, sx in (("left", .04), ("right", -.04)):
        if rng.random() < .5:
            p = add(f"{side}_feeler_a_01", names.index("head"), [sx, .08, 0])
            for k in (2, 3):
                p = add(f"{side}_feeler_a_{k:02d}", p, [sx, .06, 0])
    return names, np.array(parents), np.stack(pos).astype(np.float32)


def make(root, n_chars, seed):
    rng = np.random.default_rng(seed)
    plans = ["biped", "wing", "quad", "quad_wing"]
    root = Path(root)
    for cid in range(n_chars):
        name = "pm0025_17_00" if cid == 0 else f"pm{cid:04d}_00_00"
        names, parents, joints = build_char(rng, plans[cid % 4], cid)
        J = len(names)
        pts = []
        for _ in range(700):
            j = int(rng.integers(J))
            pts.append(joints[j] + rng.normal(0, .03, 3))
        v = np.array(pts, np.float32)
        d2 = ((v[:, None, :] - joints[None]) ** 2).sum(-1)
        w = np.exp(-d2 / 0.004); w /= w.sum(1, keepdims=True) + 1e-9
        cdir = root / name; (cdir / "motion").mkdir(parents=True, exist_ok=True)
        np.savez(cdir / f"{name}.npz", v=v, f=rng.integers(0, len(v), (900, 3)).astype(np.int32),
                 joints=joints, joint_names=np.array(names), parents=parents.astype(np.int64),
                 weights=w.astype(np.float32))
        for m in range(4):
            T = 30
            t = np.linspace(0, 2 * np.pi, T)[:, None, None]
            ph = rng.uniform(0, 6.28, (1, J, 3)); amp = rng.uniform(.1, .5, (1, J, 3))
            bp = (amp * np.sin(t * (m + 1) + ph)).astype(np.float32)
            np.savez(cdir / "motion" / ("attack01_0" if m == 0 else f"move{m:02d}_0") ,
                     body_pose=bp, transl=(0.01 * np.sin(t[:, 0, :] * np.ones((1, 3)))).astype(np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synthetic/data")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.root, a.n, a.seed)
    print("wrote", a.n, "synthetic characters to", a.root)
