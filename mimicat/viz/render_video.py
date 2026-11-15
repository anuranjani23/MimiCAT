"""Side-by-side mp4: source motion + transferred target for several correspondence variants.
Same pyrender look as quick_demo.py.

    python -m mimicat.viz.render_video --src pm0449_00_00 --tgt pm0311_00_00 \
        --variants hard,g,oracle --g-ckpt checkpoints/g.pt --out outputs/video_main.mp4
("g+h" adds H; needs --h-ckpt.)  Translation is zeroed so characters move in place.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm

from ..paths import DATA_ROOT
from ..data.tensors import CharStore
from ..data.dataset import load_motion
from ..geometry.lbs import pose_vertices_torch, rotvec_to_mat
from ..correspondence.provider import CorrProvider, load_threshold
from ..eval.cycle import transfer_pose
from ..train.common import load_G, load_H
from .render_transfer import parse_variant
from ..paths import DATA_ROOT as _DATA_ROOT

DATA_ROOT = Path(_DATA_ROOT)

def build_camera_pose(target, max_extent):
    fov_y = np.pi / 3.5
    dist = max_extent / (2.0 * np.tan(fov_y / 2.0)) * 1.3
    eye = np.array([target[0], target[1] + max_extent * 0.1, target[2] + dist], np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up_c = np.cross(right, fwd)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up_c, -fwd, eye
    return pose


def create_scene(camera_pose):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.94, 0.94, 0.94, 1.0])
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5), pose=np.array(
        [[0.707, -0.408, 0.577, 0], [0, 0.816, 0.577, 0],
         [-0.707, -0.408, 0.577, 0], [0, 0, 0, 1]], np.float32))
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0), pose=np.array(
        [[-0.707, 0.408, -0.577, 0], [0, 0.816, 0.577, 0],
         [0.707, 0.408, -0.577, 0], [0, 0, 0, 1]], np.float32))
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=camera_pose)
    return scene


def load_faces(name):
    with np.load(DATA_ROOT / name / f"{name}.npz", allow_pickle=True) as z:
        f = z["f"].astype(np.int32)
    return f - f.min()


def normalise(X, rest_v, up):
    """X (T,V,3) posed vertices -> centred/scaled by the character's rest bbox (y-up for the camera)."""
    c = (rest_v.min(0) + rest_v.max(0)) / 2
    e = max(float((rest_v.max(0) - rest_v.min(0)).max()), 1e-8)
    Xn = (X - c) / e
    if up == "z":                                   # z-up data -> y-up
        Xn = Xn[..., [0, 2, 1]] * np.array([1, 1, -1], np.float32)
    return Xn.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--motion", default=None, help="motion file stem (default: first one)")
    ap.add_argument("--variants", default="hard,g,oracle")
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt"); ap.add_argument("--h-ckpt", default=None)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--res", type=int, default=384, help="pixels per panel")
    ap.add_argument("--up", default="y", choices=["y", "z"])
    ap.add_argument("--out", default="outputs/video_demo.mp4")
    a = ap.parse_args()

    specs = [s for s in a.variants.split(",") if s.strip()]
    variants = [parse_variant(s) for s in specs]
    need_G = any(m.startswith("g") for m, _ in variants)
    need_H = any(h for _, h in variants)
    if need_H and not a.h_ckpt:
        raise SystemExit("a '+h' variant needs --h-ckpt")

    store = CharStore(DATA_ROOT, "cpu")
    S, T_ = store.get(a.src), store.get(a.tgt)
    mp = next((p for p in S.motion_paths if a.motion is None or p.stem == a.motion), None)
    if mp is None:
        raise SystemExit(f"motion '{a.motion}' not found for {a.src}")
    bp, _ = load_motion(mp)
    n = min(len(bp), a.max_frames)
    idx = np.linspace(0, len(bp) - 1, n).astype(int)
    fps = max(8, round(a.fps * n / len(bp)))        # keep roughly the original duration
    R_s = rotvec_to_mat(torch.from_numpy(bp[idx]))

    G = load_G(a.g_ckpt, "cpu") if need_G else None
    H = load_H(a.h_ckpt, "cpu") if need_H else None
    thr_g = load_threshold(a.g_ckpt) if need_G else 0.2
    provs = {}
    for m, _ in variants:
        if m not in provs:
            provs[m] = CorrProvider(m, G=G, thr=thr_g if m in ("g", "g_hungarian") else 0.2)

    # ---- pose all panels up front ----
    panels = []   # dict(Xn, faces, label)
    with torch.no_grad():
        z = torch.zeros(n, 3)
        Xs = pose_vertices_torch(S.v, S.joints, S.parents, S.weights, R_s, z).numpy()
        panels.append(dict(Xn=normalise(Xs, S.v.cpu().numpy(), a.up),
                           faces=load_faces(a.src), label=f"source {a.src}"))
        tgt_faces, tgt_rest = load_faces(a.tgt), T_.v.cpu().numpy()
        for (m, use_h), spec in zip(variants, specs):
            R_t, _, _, matched = transfer_pose(S, T_, R_s, provs[m], H if use_h else None)
            Xt = pose_vertices_torch(T_.v, T_.joints, T_.parents, T_.weights, R_t, z).numpy()
            panels.append(dict(Xn=normalise(Xt, tgt_rest, a.up), faces=tgt_faces,
                               label=f"{spec} ({int(matched.sum())}/{len(matched)} joints)"))
            print(f"{spec:12s} driven target joints: {int(matched.sum())} of {len(matched)}")

    # ---- cameras / scenes ----
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.6, 0.6, 0.6, 1.0], metallicFactor=0.05, roughnessFactor=0.7)
    for p in panels:
        lo, hi = p["Xn"].reshape(-1, 3).min(0), p["Xn"].reshape(-1, 3).max(0)
        p["scene"] = create_scene(build_camera_pose((lo + hi) / 2, max(float((hi - lo).max()), 1e-3)))
        p["node"] = None

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    W = a.res * len(panels)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, a.res))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {a.out}")
    renderer = pyrender.OffscreenRenderer(a.res, a.res)
    try:
        for f in tqdm(range(n), desc="rendering", unit="frame"):
            tiles = []
            for p in panels:
                mesh = trimesh.Trimesh(vertices=p["Xn"][f], faces=p["faces"],
                                       process=False, maintain_order=True)
                if p["node"] is not None:
                    p["scene"].remove_node(p["node"])
                p["node"] = p["scene"].add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
                color, _ = renderer.render(p["scene"])
                tile = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.putText(tile, p["label"], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (30, 30, 30), 1, cv2.LINE_AA)
                tiles.append(tile)
            writer.write(np.hstack(tiles))
    finally:
        writer.release()
        renderer.delete()
    print("wrote", a.out, f"({n} frames @ {fps} fps)")


if __name__ == "__main__":
    main()