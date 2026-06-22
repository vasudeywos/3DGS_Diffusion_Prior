"""Shared ViewCrafter job and teacher-cache utilities.

The ViewCrafter process runs in its own environment.  This module defines the
disk contract used by both that process and Scaffold-GS training.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from novel_view_sampler import _fit_ellipse_to_cameras, _focus_point_fn
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix


SCHEMA_VERSION = 2


def _camera_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(camera.R, dtype=np.float32)
    c2w[:3, 3] = camera.camera_center.detach().cpu().numpy()
    return c2w


def order_cameras_on_ellipse(cameras):
    """Order sparse cameras around their fitted trajectory manifold."""
    positions = np.stack([
        camera.camera_center.detach().cpu().numpy() for camera in cameras
    ])
    rotations = np.stack([
        np.asarray(camera.R, dtype=np.float32) for camera in cameras
    ])
    directions = rotations[:, :, 2]
    ups = -rotations[:, :, 1]
    focus = _focus_point_fn(positions, directions)
    reference_up = ups.mean(axis=0)
    if np.linalg.norm(reference_up) < 1e-6:
        reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    reference_up /= np.linalg.norm(reference_up)
    ellipse = _fit_ellipse_to_cameras(
        positions, focus, reference_up
    )

    relative = positions - ellipse["centre"]
    x = relative @ (ellipse["axis_a"] / (ellipse["semi_a"] + 1e-8))
    y = relative @ (ellipse["axis_b"] / (ellipse["semi_b"] + 1e-8))
    angles = np.mod(np.arctan2(y, x), 2 * np.pi)
    order = np.argsort(angles)
    # ViewCrafter interpolation is an open chain, not a closed loop. Start just
    # after the largest cyclic gap so the one omitted pair is the least
    # plausible bridge.
    ordered_angles = angles[order]
    cyclic_gaps = np.diff(np.r_[ordered_angles, ordered_angles[0] + 2 * np.pi])
    cut = (int(np.argmax(cyclic_gaps)) + 1) % len(order)
    order = np.roll(order, -cut)
    return [cameras[index] for index in order], focus, ellipse


def camera_pair_metrics(camera_a, camera_b, focus, ellipse):
    """Measure whether an ordered pair is safe for open-scene interpolation."""
    position_a = camera_a.camera_center.detach().cpu().numpy()
    position_b = camera_b.camera_center.detach().cpu().numpy()
    direction_a = np.asarray(camera_a.R, dtype=np.float32)[:, 2]
    direction_b = np.asarray(camera_b.R, dtype=np.float32)[:, 2]
    direction_cosine = float(np.dot(direction_a, direction_b))

    scene_radius = max(
        float(ellipse["semi_a"]), float(ellipse["semi_b"]), 1e-8
    )
    normalized_baseline = float(
        np.linalg.norm(position_b - position_a) / scene_radius
    )
    radius_a = float(np.linalg.norm(position_a - focus))
    radius_b = float(np.linalg.norm(position_b - focus))
    radial_difference = abs(radius_a - radius_b) / max(
        0.5 * (radius_a + radius_b), 1e-8
    )

    axis_a = ellipse["axis_a"] / (ellipse["semi_a"] + 1e-8)
    axis_b = ellipse["axis_b"] / (ellipse["semi_b"] + 1e-8)
    angles = []
    for position in (position_a, position_b):
        relative = position - ellipse["centre"]
        angles.append(np.arctan2(relative @ axis_b, relative @ axis_a))
    angular_gap = abs(np.angle(np.exp(1j * (angles[1] - angles[0]))))

    return {
        "angular_gap_degrees": float(np.degrees(angular_gap)),
        "view_direction_cosine": direction_cosine,
        "normalized_baseline": normalized_baseline,
        "relative_radial_difference": float(radial_difference),
    }


def _project_to_rotation(matrix):
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation.astype(np.float32)


def interpolate_camera(camera_a, camera_b, t, uid, name):
    """Interpolate a short, in-manifold pose between two sparse cameras."""
    c2w_a = _camera_c2w(camera_a)
    c2w_b = _camera_c2w(camera_b)
    rotation = _project_to_rotation(
        (1.0 - t) * c2w_a[:3, :3] + t * c2w_b[:3, :3]
    )
    position = (
        (1.0 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    ).astype(np.float32)
    translation = -rotation.T @ position
    fov_x = (1.0 - t) * camera_a.FoVx + t * camera_b.FoVx
    fov_y = (1.0 - t) * camera_a.FoVy + t * camera_b.FoVy
    height = camera_a.image_height
    width = camera_a.image_width

    camera = Camera(
        colmap_id=uid,
        R=rotation,
        T=translation,
        FoVx=float(fov_x),
        FoVy=float(fov_y),
        image=torch.zeros(3, height, width),
        gt_alpha_mask=None,
        image_name=name,
        uid=uid,
        data_device="cuda",
    )
    camera.projection_matrix = getProjectionMatrix(
        znear=camera.znear,
        zfar=camera.zfar,
        fovX=camera.FoVx,
        fovY=camera.FoVy,
    ).transpose(0, 1).to(camera.world_view_transform.device)
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return camera


def selected_frame_indices(video_length, interior_start=4, interior_end=21):
    """Return candidate interior frames; quality filtering happens after generation."""
    if video_length < 3:
        raise ValueError("ViewCrafter video length must be at least 3.")
    start = max(1, int(interior_start))
    end = min(video_length - 2, int(interior_end))
    if start > end:
        raise ValueError(
            f"Invalid interior frame range [{start}, {end}] for "
            f"video_length={video_length}."
        )
    return list(range(start, end + 1))


def viewcrafter_frame_t(frame_index, video_length):
    # This mirrors ViewCrafter's interp_traj: the first video_length-1 poses
    # use u=[0, ..., (video_length-2)/video_length], followed by the endpoint.
    if frame_index == video_length - 1:
        return 1.0
    return frame_index / float(video_length)


def camera_to_record(camera, clip_index, frame_index):
    return {
        "uid": int(camera.uid),
        "image_name": camera.image_name,
        "R": np.asarray(camera.R, dtype=np.float32).tolist(),
        "T": np.asarray(camera.T, dtype=np.float32).tolist(),
        "FoVx": float(camera.FoVx),
        "FoVy": float(camera.FoVy),
        "height": int(camera.image_height),
        "width": int(camera.image_width),
        "clip_index": int(clip_index),
        "frame_index": int(frame_index),
    }


def compute_job_signature(job):
    payload = dict(job)
    payload.pop("signature", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_is_complete(cache_dir):
    cache_dir = Path(cache_dir)
    job_path = cache_dir / "viewcrafter_job.json"
    complete_path = cache_dir / "generation_complete.json"
    if not job_path.is_file() or not complete_path.is_file():
        return False
    try:
        job = json.loads(job_path.read_text())
        complete = json.loads(complete_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    signature = compute_job_signature(job)
    expected = int(complete.get("teacher_count", 0))
    return (
        job.get("signature") == signature
        and complete.get("signature") == signature
        and expected >= int(job["frame_filter"]["minimum_total_teachers"])
        and len(list((cache_dir / "teacher_images").glob("*.png"))) == expected
        and len(list((cache_dir / "metadata").glob("*.json"))) == expected
    )


def record_to_camera(record, device="cuda"):
    camera = Camera(
        colmap_id=record["uid"],
        R=np.asarray(record["R"], dtype=np.float32),
        T=np.asarray(record["T"], dtype=np.float32),
        FoVx=float(record["FoVx"]),
        FoVy=float(record["FoVy"]),
        image=torch.zeros(3, record["height"], record["width"]),
        gt_alpha_mask=None,
        image_name=record["image_name"],
        uid=record["uid"],
        data_device=device,
    )
    return camera


def pil_to_tensor(image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


class ViewCrafterTeacherDataset:
    def __init__(self, cache_dir, device="cuda"):
        cache_dir = Path(cache_dir)
        if not cache_is_complete(cache_dir):
            raise RuntimeError(
                f"Incomplete or stale ViewCrafter cache: {cache_dir}. "
                "Run viewcrafter_bridge.py before distillation."
            )
        self.device = device
        self.pairs = []
        self.clips = {}
        for path in sorted((cache_dir / "metadata").glob("*.json")):
            record = json.loads(path.read_text())
            teacher_path = Path(record["teacher_path"])
            if not teacher_path.is_absolute():
                teacher_path = cache_dir / teacher_path
            item = (record_to_camera(record, device), teacher_path, record)
            self.pairs.append(item)
            self.clips.setdefault(record["clip_index"], []).append(item)
        for items in self.clips.values():
            items.sort(key=lambda item: item[2]["frame_index"])
        self.adjacent_pairs = [
            (items[index], items[index + 1])
            for items in self.clips.values()
            for index in range(len(items) - 1)
        ]
        if not self.adjacent_pairs:
            raise RuntimeError("ViewCrafter cache contains no adjacent teacher pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        camera, path, _ = self.pairs[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (camera.image_width, camera.image_height),
                Image.Resampling.LANCZOS,
            )
            tensor = pil_to_tensor(image, self.device)
        return camera, tensor

    def sample(self):
        index = torch.randint(0, len(self.pairs), (1,)).item()
        return self[index]

    def _load_item(self, item):
        camera, path, _ = item
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (camera.image_width, camera.image_height),
                Image.Resampling.LANCZOS,
            )
            tensor = pil_to_tensor(image, self.device)
        return camera, tensor

    def sample_adjacent(self):
        index = torch.randint(0, len(self.adjacent_pairs), (1,)).item()
        first, second = self.adjacent_pairs[index]
        return self._load_item(first), self._load_item(second)
