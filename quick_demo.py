import argparse
from pathlib import Path
import heapq

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
import os
import zlib


DATA_ROOT = Path(__file__).resolve().parent / "PokeAnimDB" / "data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "example"
FPS = 24
RESOLUTION = 512


def load_sample(sample_dir):
    rig = np.load(sample_dir / f"{sample_dir.name}.npz", allow_pickle=True)
    return {
        "v": rig["v"].astype(np.float32, copy=False),
        "f": rig["f"].astype(np.int32, copy=False),
        "joints": rig["joints"].astype(np.float32, copy=False),
        "parents": rig["parents"].astype(np.int64, copy=False),
        "weights": rig["weights"].astype(np.float32, copy=False),
    }


def pick_longest_motion(sample_dir):
    top_motions = []
    for motion_path in sorted((sample_dir / "motion").glob("*.npz")):
        with np.load(motion_path, allow_pickle=True) as motion:
            if "body_pose" not in motion:
                continue
            frame_count = int(motion["body_pose"].shape[0])
        top_motions.append((frame_count, motion_path))
    if not top_motions:
        raise FileNotFoundError(
            f"No motion with body_pose found in {sample_dir / 'motion'}"
        )
    top_motions.sort(key=lambda item: item[0], reverse=True)
    candidates = top_motions[:10]
    rng = np.random.default_rng(zlib.crc32(sample_dir.name.encode("utf-8")))
    return candidates[int(rng.integers(len(candidates)))][1]


def resolve_motion_path(sample_dir, motion_name=None):
    if motion_name is None:
        return pick_longest_motion(sample_dir)

    motion_path = sample_dir / "motion" / f"{motion_name}.npz"
    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Motion '{motion_name}' not found in {sample_dir / 'motion'}"
        )
    return motion_path


def build_camera_pose(target, max_extent):
    fov_y = np.pi / 3.5
    camera_distance = max_extent / (2.0 * np.tan(fov_y / 2.0)) * 1.3
    eye = np.array(
        [target[0], target[1] + max_extent * 0.1, target[2] + camera_distance],
        dtype=np.float32,
    )
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(right, forward)

    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up_corrected
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = eye
    return camera_pose


def pose_vertices(vertices, joints, parents, weights, pose_rotvec, transl):
    joint_count = joints.shape[0]
    rotations = (
        Rotation.from_rotvec(pose_rotvec).as_matrix().astype(np.float32, copy=False)
    )

    transforms = np.zeros((joint_count, 4, 4), dtype=np.float32)
    transforms[:, 3, 3] = 1.0
    for joint_idx in range(joint_count):
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = rotations[joint_idx]
        local[:3, 3] = joints[joint_idx] - rotations[joint_idx] @ joints[joint_idx]
        parent = parents[joint_idx]
        transforms[joint_idx] = local if parent < 0 else transforms[parent] @ local

    blended = np.tensordot(weights, transforms, axes=(1, 0))
    vertices_h = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1
    )
    posed = np.einsum("nij,nj->ni", blended, vertices_h)[:, :3]
    return posed + transl.astype(np.float32, copy=False)


def create_scene(camera_pose):
    scene = pyrender.Scene(
        ambient_light=[0.3, 0.3, 0.3],
        bg_color=[0.94, 0.94, 0.94, 1.0],
    )
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5),
        pose=np.array(
            [
                [0.707, -0.408, 0.577, 0.0],
                [0.0, 0.816, 0.577, 0.0],
                [-0.707, -0.408, 0.577, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0),
        pose=np.array(
            [
                [-0.707, 0.408, -0.577, 0.0],
                [0.0, 0.816, 0.577, 0.0],
                [0.707, 0.408, -0.577, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=camera_pose)
    return scene


def render_sample(sample_dir, output_root, fps, resolution, motion_name=None):
    sample = load_sample(sample_dir)
    motion_path = resolve_motion_path(sample_dir, motion_name)
    motion = np.load(motion_path, allow_pickle=True)
    body_pose = motion["body_pose"].astype(np.float32, copy=False)
    transl = motion["transl"].astype(np.float32, copy=False)

    rest_min = sample["v"].min(axis=0)
    rest_max = sample["v"].max(axis=0)
    cano_center = (rest_min + rest_max) * 0.5
    cano_extent = max(float(np.max(rest_max - rest_min)), 1.0)
    transl_min = transl.min(axis=0)
    transl_max = transl.max(axis=0)
    scene_min = (rest_min + transl_min - cano_center) / cano_extent
    scene_max = (rest_max + transl_max - cano_center) / cano_extent
    center = (scene_min + scene_max) * 0.5
    extent = max(float(np.max(scene_max - scene_min)), 1.0)
    camera_pose = build_camera_pose(center, extent)

    output_dir = output_root / sample_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{motion_path.stem}.mp4"

    if os.path.exists(output_path):
        print(f"Skipping {output_path}")
        return

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (resolution, resolution),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.6, 0.6, 0.6, 1.0],
        metallicFactor=0.05,
        roughnessFactor=0.7,
    )
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    scene = create_scene(camera_pose)
    mesh_node = None

    try:
        faces = sample["f"] - sample["f"].min()
        for frame_idx in range(body_pose.shape[0]):
            posed_vertices = pose_vertices(
                sample["v"],
                sample["joints"],
                sample["parents"],
                sample["weights"],
                body_pose[frame_idx],
                transl[frame_idx],
            )
            posed_vertices = (posed_vertices - cano_center) / cano_extent
            mesh = trimesh.Trimesh(
                vertices=posed_vertices,
                faces=faces,
                process=False,
                maintain_order=True,
            )
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            mesh_node = scene.add(
                pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
            )
            color, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.delete()
        motion.close()

    print(f"Rendered {sample_dir.name}: {motion_path.name} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render sample motions into quick demo videos."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing sample folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to write rendered videos to.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help="Square output resolution in pixels.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Render only the named sample directory.",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Render a specific motion file name without the .npz suffix.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all sample directories instead of one quick demo sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        sample_dir for sample_dir in args.data_root.iterdir() if sample_dir.is_dir()
    )
    if args.sample is not None:
        sample_dirs = [
            sample_dir for sample_dir in sample_dirs if sample_dir.name == args.sample
        ]
        if not sample_dirs:
            raise FileNotFoundError(
                f"Sample '{args.sample}' not found in {args.data_root}"
            )
    elif not args.all:
        if not sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {args.data_root}")
        sample_dirs = [sample_dirs[0]]

    for sample_dir in sample_dirs:
        render_sample(
            sample_dir,
            args.output_root,
            args.fps,
            args.resolution,
            args.motion,
        )


if __name__ == "__main__":
    main()
