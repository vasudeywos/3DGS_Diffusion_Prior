"""Seed Scaffold-GS anchors by backprojecting teacher depth maps.

This utility is intentionally conservative: it requires external depth maps
for teacher images rather than inventing depth from RGB. Depth files should be
`.npy` arrays in teacher pixel units, named like the matching teacher PNG stem.
"""

import json
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = THIS_DIR / "Scaffold-GS-main"
if str(SCAFFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAFFOLD_ROOT))

from arguments import ModelParams
from scene import GaussianModel
from utils.general_utils import inverse_sigmoid


def load_stage_config(model_path):
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    return eval(cfg_path.read_text(), {"Namespace": Namespace})


def point_cloud_dir(model_path, iteration):
    path = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    if not (path / "point_cloud.ply").is_file():
        raise FileNotFoundError(f"Missing point cloud: {path / 'point_cloud.ply'}")
    return path


def camera_world_points(record, pixels_xy, depth):
    rotation = np.asarray(record["R"], dtype=np.float32)
    translation = np.asarray(record["T"], dtype=np.float32)
    fx, fy = float(record["fx"]), float(record["fy"])
    cx, cy = float(record["cx"]), float(record["cy"])
    x = (pixels_xy[:, 0] - cx) / fx * depth
    y = (pixels_xy[:, 1] - cy) / fy * depth
    z = depth
    cam = np.stack([x, y, z], axis=1).astype(np.float32)
    return (rotation @ (cam - translation).T).T.astype(np.float32)


def voxel_unique(points, voxel_size):
    coords = np.round(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def sample_depth_points(depth, max_points, stride, min_depth, max_depth):
    valid = np.isfinite(depth) & (depth > min_depth)
    if max_depth > 0:
        valid &= depth < max_depth
    ys, xs = np.nonzero(valid[::stride, ::stride])
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xs = xs * stride
    ys = ys * stride
    values = depth[ys, xs].astype(np.float32)
    if len(xs) > max_points:
        rng = np.random.default_rng(123)
        chosen = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys, values = xs[chosen], ys[chosen], values[chosen]
    pixels = np.stack([xs, ys], axis=1).astype(np.float32)
    return pixels, values


def copy_mlp_files(src_dir, dst_dir):
    for path in src_dir.iterdir():
        if path.name == "point_cloud.ply":
            continue
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.add_argument("--stage_iteration", type=int, default=10000)
    parser.add_argument("--teacher_cache_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--output_iteration", type=int, default=10000)
    parser.add_argument("--max_points_per_teacher", type=int, default=512)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--max_depth", type=float, default=0.0)
    parser.add_argument("--new_anchor_opacity", type=float, default=0.05)
    args = parser.parse_args()

    saved = load_stage_config(args.model_path)
    for key, value in vars(saved).items():
        if not hasattr(args, key):
            continue
        setattr(args, key, value)
    dataset = model.extract(args)

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
        dataset.update_depth, dataset.update_init_factor,
        dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio,
        dataset.add_opacity_dist, dataset.add_cov_dist,
        dataset.add_color_dist,
    )
    source_dir = point_cloud_dir(args.model_path, args.stage_iteration)
    gaussians.load_ply_sparse_gaussian(str(source_dir / "point_cloud.ply"))

    teacher_cache = Path(args.teacher_cache_dir).resolve()
    depth_dir = Path(args.depth_dir).resolve()
    new_points = []
    for metadata_path in sorted((teacher_cache / "metadata").glob("*.json")):
        record = json.loads(metadata_path.read_text())
        teacher_stem = Path(record["teacher_path"]).stem
        depth_path = depth_dir / f"{teacher_stem}.npy"
        if not depth_path.is_file():
            continue
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (int(record["height"]), int(record["width"])):
            raise RuntimeError(
                f"{depth_path.name} has shape {depth.shape}; expected "
                f"{(int(record['height']), int(record['width']))}."
            )
        pixels, values = sample_depth_points(
            depth,
            max_points=args.max_points_per_teacher,
            stride=args.stride,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        if len(values) == 0:
            continue
        new_points.append(camera_world_points(record, pixels, values))

    if not new_points:
        raise RuntimeError(
            "No depth-backed teacher points found. Ensure depth_dir contains "
            ".npy files named after teacher image stems."
        )
    points = voxel_unique(np.concatenate(new_points, axis=0), dataset.voxel_size)
    device = gaussians._anchor.device
    new_anchor = torch.from_numpy(points).float().to(device)
    mean_feat = gaussians._anchor_feat.detach().mean(dim=0, keepdim=True)
    new_feat = mean_feat.repeat(new_anchor.shape[0], 1)
    new_offsets = torch.zeros(
        new_anchor.shape[0], dataset.n_offsets, 3, device=device
    )
    scale_value = max(float(dataset.voxel_size), 1e-6)
    new_scaling = torch.log(
        torch.ones(new_anchor.shape[0], 6, device=device) * scale_value
    )
    new_rotation = torch.zeros(new_anchor.shape[0], 4, device=device)
    new_rotation[:, 0] = 1.0
    new_opacity = inverse_sigmoid(
        torch.ones(new_anchor.shape[0], 1, device=device)
        * args.new_anchor_opacity
    )

    gaussians._anchor = torch.nn.Parameter(torch.cat([gaussians._anchor, new_anchor], dim=0))
    gaussians._anchor_feat = torch.nn.Parameter(torch.cat([gaussians._anchor_feat, new_feat], dim=0))
    gaussians._offset = torch.nn.Parameter(torch.cat([gaussians._offset, new_offsets], dim=0))
    gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, new_scaling], dim=0))
    gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, new_rotation], dim=0))
    gaussians._opacity = torch.nn.Parameter(torch.cat([gaussians._opacity, new_opacity], dim=0))

    output_dir = (
        Path(args.output_model_path)
        / "point_cloud"
        / f"iteration_{args.output_iteration}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(output_dir / "point_cloud.ply"))
    copy_mlp_files(source_dir, output_dir)
    cfg_src = Path(args.model_path) / "cfg_args"
    if cfg_src.is_file():
        Path(args.output_model_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_src, Path(args.output_model_path) / "cfg_args")
    print(
        f"Added {new_anchor.shape[0]} depth-backprojected anchors. "
        f"Saved seeded model to {output_dir}."
    )


if __name__ == "__main__":
    main()
