"""COLMAP scene dataset for open-scene EscherNet adaptation.

This dataset mirrors the keys returned by the original ObjaverseData class:
image_input, image_target, pose_in, pose_in_inv, pose_out, pose_out_inv.
It samples local camera neighborhoods from COLMAP scenes and normalizes each
scene's camera centers so EscherNet sees a bounded pose distribution.
"""

import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _load_scaffold_colmap_loader(scaffold_root):
    root = Path(scaffold_root).expanduser().resolve()
    loader_path = root / "scene" / "colmap_loader.py"
    if not loader_path.is_file():
        raise FileNotFoundError(f"Missing Scaffold-GS COLMAP loader: {loader_path}")
    module_name = "_scaffold_colmap_loader"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_colmap(scene_dir, images_name, scaffold_root):
    colmap_loader = _load_scaffold_colmap_loader(scaffold_root)
    qvec2rotmat = colmap_loader.qvec2rotmat
    read_extrinsics_binary = colmap_loader.read_extrinsics_binary
    read_extrinsics_text = colmap_loader.read_extrinsics_text
    read_intrinsics_binary = colmap_loader.read_intrinsics_binary
    read_intrinsics_text = colmap_loader.read_intrinsics_text

    sparse = Path(scene_dir) / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))

    images_dir = Path(scene_dir) / images_name
    cameras = []
    for image_id in sorted(extrinsics):
        extr = extrinsics[image_id]
        image_path = images_dir / os.path.basename(extr.name)
        if not image_path.is_file() or image_path.suffix not in IMAGE_EXTENSIONS:
            continue
        rotation_w2c = qvec2rotmat(extr.qvec).astype(np.float32)
        translation = np.asarray(extr.tvec, dtype=np.float32)
        center = (-rotation_w2c.T @ translation).astype(np.float32)
        cameras.append({
            "name": image_path.stem,
            "path": image_path,
            "rotation_w2c": rotation_w2c,
            "translation": translation,
            "center": center,
            "camera": intrinsics[extr.camera_id],
        })
    cameras.sort(key=lambda item: item["name"])
    if len(cameras) < 2:
        raise RuntimeError(f"Scene has too few registered images: {scene_dir}")
    return cameras


def _normalize_scene_poses(cameras):
    centers = np.stack([item["center"] for item in cameras], axis=0)
    center_mean = centers.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(centers - center_mean, axis=1), 90))
    radius = max(radius, 1e-6)
    for item in cameras:
        normalized_center = (item["center"] - center_mean) / radius
        rotation = item["rotation_w2c"]
        translation = (-rotation @ normalized_center).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        item["pose"] = pose
    return {
        "center": center_mean.astype(np.float32),
        "radius": radius,
    }


def _discover_scenes(root_dir, exclude_scenes):
    root = Path(root_dir).expanduser().resolve()
    excluded = {name.strip() for name in exclude_scenes.split(",") if name.strip()}
    scenes = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in excluded:
            continue
        if (path / "sparse" / "0").is_dir() and (path / "images").is_dir():
            scenes.append(path)
    if not scenes:
        raise RuntimeError(f"No COLMAP scenes found in {root}")
    return scenes


class ColmapSceneData(Dataset):
    def __init__(
        self,
        root_dir,
        image_transforms,
        scaffold_root,
        images_name="images",
        exclude_scenes="",
        validation=False,
        T_in=3,
        T_out=3,
        fix_sample=False,
        window_radius=8,
        samples_per_scene=512,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.tform = image_transforms
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.fix_sample = fix_sample
        self.window_radius = int(window_radius)
        self.samples_per_scene = int(samples_per_scene)

        scene_paths = _discover_scenes(self.root_dir, exclude_scenes)
        if len(scene_paths) > 1:
            split = max(1, int(round(0.9 * len(scene_paths))))
            scene_paths = scene_paths[split:] if validation else scene_paths[:split]
            if not scene_paths:
                scene_paths = _discover_scenes(self.root_dir, exclude_scenes)

        self.scenes = []
        for scene_path in scene_paths:
            cameras = _load_colmap(scene_path, images_name, scaffold_root)
            normalization = _normalize_scene_poses(cameras)
            self.scenes.append({
                "name": scene_path.name,
                "path": scene_path,
                "cameras": cameras,
                "normalization": normalization,
            })
        print(
            f"Loaded {len(self.scenes)} COLMAP scenes for "
            f"{'validation' if validation else 'training'}: "
            + ", ".join(scene["name"] for scene in self.scenes)
        )

    def __len__(self):
        return max(1, len(self.scenes) * self.samples_per_scene)

    def _rng(self, index):
        if self.fix_sample:
            return np.random.default_rng(index)
        return np.random.default_rng()

    def _sample_indices(self, scene, index):
        rng = self._rng(index)
        n = len(scene["cameras"])
        total = self.T_in + self.T_out
        center_index = int(rng.integers(0, n))
        lo = max(0, center_index - self.window_radius)
        hi = min(n, center_index + self.window_radius + 1)
        pool = np.arange(lo, hi)
        replace = len(pool) < total
        chosen = rng.choice(pool, size=total, replace=replace)
        chosen = np.sort(chosen)
        input_indices = chosen[:self.T_in]
        target_indices = chosen[self.T_in:]
        return input_indices, target_indices

    def _load_image(self, camera):
        image = Image.open(camera["path"]).convert("RGB")
        return self.tform(image)

    def __getitem__(self, index):
        scene = self.scenes[index % len(self.scenes)]
        input_indices, target_indices = self._sample_indices(scene, index)
        cameras = scene["cameras"]

        input_ims = [self._load_image(cameras[i]) for i in input_indices]
        target_ims = [self._load_image(cameras[i]) for i in target_indices]
        cond_poses = np.stack([cameras[i]["pose"] for i in input_indices])
        target_poses = np.stack([cameras[i]["pose"] for i in target_indices])

        return {
            "image_input": torch.stack(input_ims, dim=0),
            "image_target": torch.stack(target_ims, dim=0),
            "pose_out": target_poses.astype(np.float32),
            "pose_out_inv": np.linalg.inv(target_poses).transpose([0, 2, 1]).astype(np.float32),
            "pose_in": cond_poses.astype(np.float32),
            "pose_in_inv": np.linalg.inv(cond_poses).transpose([0, 2, 1]).astype(np.float32),
        }
