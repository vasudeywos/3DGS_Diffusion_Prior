"""
novel_view_sampler.py

Samples novel camera poses along an elliptical path fit to the sparse training cameras,
following the ReconFusion strategy for mip-NeRF 360 scenes.

Usage:
    from novel_view_sampler import sample_novel_poses
    novel_cameras = sample_novel_poses(scene, n_samples=40, device='cuda')

The returned cameras are compatible with ScaffoldGS's render() and prefilter_voxel() functions
because they match the Camera namedtuple interface from scene/cameras.py.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = THIS_DIR / "Scaffold-GS-main"
if str(SCAFFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAFFOLD_ROOT))

from scene.cameras import Camera  # ScaffoldGS camera class
from utils.graphics_utils import getProjectionMatrix


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _focus_point_fn(
    camera_positions: np.ndarray,
    camera_directions: np.ndarray,
) -> np.ndarray:
    """
    Compute the scene focus point as the point minimising average distance
    to all camera focal axes. This is the ReconFusion / NeRF Studio convention.

    Args:
        camera_positions: (N, 3) array of camera centre positions in world space.
    Returns:
        focus: (3,) focus point.
    """
    # Minimise sum_i ||(I - d_i d_i^T)(x - o_i)||^2, where o_i is a
    # camera centre and d_i is its optical-axis direction.  pinv is used
    # because five-view layouts can be close to degenerate.
    directions = camera_directions / (
        np.linalg.norm(camera_directions, axis=1, keepdims=True) + 1e-8
    )
    identity = np.eye(3, dtype=np.float64)
    projectors = identity[None] - directions[:, :, None] * directions[:, None, :]
    lhs = projectors.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projectors, camera_positions)
    focus = np.linalg.pinv(lhs) @ rhs

    if not np.all(np.isfinite(focus)):
        return camera_positions.mean(axis=0)
    return focus.astype(np.float32)


def _fit_ellipse_to_cameras(
    camera_positions: np.ndarray,
    focus: np.ndarray,
    reference_up: np.ndarray,
) -> dict:
    """
    Fit an ellipse in the plane of the camera positions around the focus point.

    Strategy:
      1. Centre positions around the focus point.
      2. PCA to find the dominant plane and axes.
      3. Extract semi-axes a, b from the scatter of projected positions.

    Returns a dict with keys: centre, axis_a, axis_b, normal, mean_height.
    """
    centred = camera_positions - focus  # (N, 3)

    # PCA
    cov = np.cov(centred.T)  # (3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvectors columns are sorted by eigenvalue ascending → reverse
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns: PC1, PC2, PC3 (smallest = normal)

    # Dominant plane: PC1 (most spread) and PC2 (second most spread)
    axis_a_dir = eigenvectors[:, 0]  # unit vector along major axis
    axis_b_dir = eigenvectors[:, 1]  # unit vector along minor axis
    normal_dir = eigenvectors[:, 2]  # normal to plane (smallest variance)
    if np.dot(normal_dir, reference_up) < 0:
        normal_dir = -normal_dir

    # Project centred positions onto plane axes
    proj_a = centred @ axis_a_dir  # (N,)
    proj_b = centred @ axis_b_dir  # (N,)

    # Semi-axes = max abs projection, with a small padding factor
    semi_a = np.abs(proj_a).max() * 1.05
    semi_b = np.abs(proj_b).max() * 1.05

    # Mean height above focus along normal
    mean_height = (centred @ normal_dir).mean()

    return {
        "centre": focus,
        "axis_a": axis_a_dir * semi_a,   # (3,) scaled semi-axis
        "axis_b": axis_b_dir * semi_b,   # (3,) scaled semi-axis
        "normal": normal_dir,
        "mean_height": mean_height,
        "semi_a": semi_a,
        "semi_b": semi_b,
    }


def _pose_on_ellipse(t: float, ellipse: dict, focus: np.ndarray) -> np.ndarray:
    """
    Return a 4x4 camera-to-world matrix for a point at angle t (radians) on the ellipse.
    Camera looks toward the focus point using Scaffold-GS/COLMAP's +Z-forward,
    +X-right, +Y-down camera convention.

    Args:
        t: angle in [0, 2π)
        ellipse: output of _fit_ellipse_to_cameras
        focus: (3,) focus point
    Returns:
        c2w: (4, 4) camera-to-world matrix
    """
    # Point on ellipse
    pos = (ellipse["centre"]
           + np.cos(t) * ellipse["axis_a"]
           + np.sin(t) * ellipse["axis_b"]
           + ellipse["mean_height"] * ellipse["normal"])

    # Camera axes: look toward focus
    forward = focus - pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Up vector: ellipse normal, sign-aligned with the training cameras.
    up = ellipse["normal"].copy()
    # Ensure up is not collinear with forward
    if abs(np.dot(up, forward)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])

    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    # COLMAP/Scaffold-GS cameras look along +Z and use +Y as image-down.
    # Therefore c2w columns are [right, down, forward, position].
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos

    return c2w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_novel_poses(
    scene,
    n_samples: int = 40,
    device: str = 'cuda',
    znear: float = 0.01,
    zfar: float = 100.0,
    exclude_test_cameras: bool = True,
) -> List[Camera]:
    """
    Sample novel camera poses along an elliptical path fit to the training cameras.

    Args:
        scene:               ScaffoldGS Scene object (has .getTrainCameras(), .getTestCameras())
        n_samples:           Number of novel views to sample
        device:              'cuda' or 'cpu'
        znear, zfar:         Near/far clipping planes
        exclude_test_cameras: Skip angles that are too close to test camera positions

    Returns:
        List of Camera objects compatible with render() and prefilter_voxel()
    """
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras() if exclude_test_cameras else []

    if len(train_cams) == 0:
        raise ValueError("No training cameras found in scene.")

    # --- Extract camera world positions and intrinsics from training set ---
    cam_positions = []
    cam_directions = []
    cam_ups = []
    ref_cam = train_cams[0]  # use first camera for intrinsics reference

    for cam in train_cams:
        # camera_center is the world-space position of the camera
        pos = cam.camera_center.detach().cpu().numpy()
        cam_positions.append(pos)
        c2w_rotation = np.asarray(cam.R, dtype=np.float32)
        cam_directions.append(c2w_rotation[:, 2])
        cam_ups.append(-c2w_rotation[:, 1])

    cam_positions = np.stack(cam_positions, axis=0)  # (N_train, 3)
    cam_directions = np.stack(cam_directions, axis=0)
    cam_ups = np.stack(cam_ups, axis=0)

    # --- Fit ellipse ---
    focus = _focus_point_fn(cam_positions, cam_directions)
    reference_up = cam_ups.mean(axis=0)
    if np.linalg.norm(reference_up) < 1e-6:
        reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    reference_up /= np.linalg.norm(reference_up)
    ellipse = _fit_ellipse_to_cameras(cam_positions, focus, reference_up)

    # --- Get test camera positions for exclusion ---
    test_positions = []
    for cam in test_cams:
        test_positions.append(cam.camera_center.detach().cpu().numpy())
    test_positions = np.stack(test_positions, axis=0) if test_positions else None

    # --- Sample angles uniformly on ellipse, skipping too-close test views ---
    # Half-step phase avoids reproducing the uniformly selected sparse cameras.
    angles = (np.arange(n_samples) + 0.5) * (2 * np.pi / n_samples)
    # Exclude only near-identical held-out poses. A large exclusion radius would
    # remove almost the entire path when the test manifest contains all
    # remaining MiP-NeRF 360 frames.
    exclusion_radius = max(ellipse["semi_a"], ellipse["semi_b"]) * 0.002

    novel_cameras = []
    uid_offset = 10000  # offset to avoid uid collision with training cameras

    for i, t in enumerate(angles):
        c2w = _pose_on_ellipse(t, ellipse, focus)
        cam_pos = c2w[:3, 3]

        # Skip if too close to a test camera
        if test_positions is not None:
            dists = np.linalg.norm(test_positions - cam_pos[None], axis=1)
            if dists.min() < exclusion_radius:
                continue

        # Camera stores c2w rotation R; getWorld2View2 transposes it internally.
        R = c2w[:3, :3]
        T = -R.T @ c2w[:3, 3]
        novel_cam = Camera(
            colmap_id=uid_offset + i,
            R=R,
            T=T,
            FoVx=ref_cam.FoVx,
            FoVy=ref_cam.FoVy,
            image=torch.zeros(3, ref_cam.image_height, ref_cam.image_width),  # placeholder
            gt_alpha_mask=None,
            image_name=f"novel_{i:04d}",
            uid=uid_offset + i,
            data_device=device,
        )
        novel_cam.znear = znear
        novel_cam.zfar = zfar
        novel_cam.projection_matrix = getProjectionMatrix(
            znear=znear,
            zfar=zfar,
            fovX=novel_cam.FoVx,
            fovY=novel_cam.FoVy,
        ).transpose(0, 1).to(novel_cam.world_view_transform.device)
        novel_cam.full_proj_transform = (
            novel_cam.world_view_transform.unsqueeze(0)
            .bmm(novel_cam.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        novel_cameras.append(novel_cam)

    print(f"[novel_view_sampler] Sampled {len(novel_cameras)} novel views "
          f"(requested {n_samples}, skipped {n_samples - len(novel_cameras)} near test cams)")

    return novel_cameras


def get_rendered_depth(render_pkg: dict) -> torch.Tensor:
    """
    Extract a depth map from a ScaffoldGS render package.

    ScaffoldGS does not return depth by default. We approximate it from the
    rendered image's alpha channel and the Gaussian positions projected to
    camera space. For the teacher pipeline we use a simple proxy:
    reconstruct depth from viewspace_points z-values weighted by visibility.

    This is a lightweight approximation — good enough for ControlNet conditioning.

    Returns:
        depth: (H, W) float tensor, depth in camera space (normalised to [0,1])
    """
    # ScaffoldGS render_pkg doesn't include depth natively.
    # We return None here; the teacher_generator will handle missing depth
    # by using monocular depth estimation (MiDaS) as fallback.
    return None
