"""
train_distill.py

Stage 4: Distillation fine-tuning of ScaffoldGS with diffusion teacher supervision.

This is the core experiment:
  - A ScaffoldGS checkpoint trained on 5 views (Checkpoint_A) is loaded.
  - Novel camera views are sampled along an elliptical path.
  - Teacher images are generated (or loaded from cache) using SD1.5 + ControlNet depth.
  - ScaffoldGS is fine-tuned with:
      L_total = L_real + λ_teacher * L_teacher + λ_anchor_reg * L_anchor_reg

The diffusion model is an offline pseudo-view teacher; no gradients pass
through diffusion. Parameter modes isolate whether supervision is absorbed by
shared MLPs, per-anchor features, geometry, or all Scaffold-GS parameters.

Usage:
    python train_distill.py \
        --source_path data/bicycle \
        --model_path output/bicycle/stage1 \
        --distill_output output/bicycle/stage2 \
        --teacher_cache_dir teacher_cache/bicycle \
        --distill_iterations 10000 \
        --n_novel_views 40 \
        --lambda_teacher 0.2 \
        --lambda_lpips 0.1 \
        --lambda_anchor_reg 0.01 \
        --teacher_strength 0.55 \
        --round 1

For round 2 (iterative refinement):
    python train_distill.py ... --start_checkpoint output/bicycle/stage2/chkpnt10000.pth --round 2
"""

import os
import sys

# CUDA must be selected before importing torch or modules that may initialise it.
if "--gpu" in sys.argv:
    _gpu_index = sys.argv.index("--gpu")
    if _gpu_index + 1 < len(sys.argv) and sys.argv[_gpu_index + 1] != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_gpu_index + 1]

import torch
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

THIS_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = THIS_DIR / "Scaffold-GS-main"
if str(SCAFFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAFFOLD_ROOT))

# ScaffoldGS imports
from scene import Scene, GaussianModel
from gaussian_renderer import prefilter_voxel, render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams

# Project imports
from novel_view_sampler import sample_novel_poses
from teacher_generator import (
    TeacherGenerator,
    TeacherDataset,
    teacher_cache_is_valid,
    teacher_generation_settings,
    unload_teacher_models,
)

lpips_fn = None


def get_lpips_fn():
    global lpips_fn
    if lpips_fn is None:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').eval().to('cuda')
        for parameter in lpips_fn.parameters():
            parameter.requires_grad_(False)
    return lpips_fn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ---------------------------------------------------------------------------
# Anchor regularisation loss
# ---------------------------------------------------------------------------

def anchor_regularisation_loss(
    gaussians: GaussianModel,
    anchor_init: torch.Tensor,
    offset_init: torch.Tensor,
) -> torch.Tensor:
    """
    Soft stiffness penalty on anchor positions and offsets.
    Penalises large deviations from the Stage 1 geometry.

    This is the key regulariser that lets geometry adapt while absorbing
    diffusion gradients smoothly through the MLP rather than shattering
    individual Gaussian positions.

    L_anchor_reg = mean(|| _anchor - anchor_init ||^2)
                 + 0.1 * mean(|| _offset - offset_init ||^2)
    """
    # Anchors may have grown via densification — only penalise existing ones
    n_orig = min(anchor_init.shape[0], gaussians._anchor.shape[0])

    anchor_diff = gaussians._anchor[:n_orig] - anchor_init[:n_orig]
    anchor_reg = (anchor_diff ** 2).mean()

    offset_diff = gaussians._offset[:n_orig] - offset_init[:n_orig]
    offset_reg = (offset_diff ** 2).mean()

    return anchor_reg + 0.1 * offset_reg


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def teacher_distillation_loss(
    rendered: torch.Tensor,
    teacher: torch.Tensor,
    lambda_lpips: float = 0.1,
) -> torch.Tensor:
    """
    L_teacher = L1(render, teacher) + λ_lpips * LPIPS(render, teacher)

    Args:
        rendered: (3, H, W) float32 tensor, clamped to [0,1]
        teacher:  (3, H, W) float32 tensor, [0,1]
        lambda_lpips: weight for perceptual loss

    Returns:
        Scalar distillation loss.
    """
    l1 = l1_loss(rendered, teacher)

    # LPIPS expects (B, 3, H, W) in [-1, 1]
    rendered_lpips = rendered.unsqueeze(0) * 2.0 - 1.0
    teacher_lpips = teacher.unsqueeze(0) * 2.0 - 1.0
    perceptual = get_lpips_fn()(rendered_lpips, teacher_lpips).mean()

    return l1 + lambda_lpips * perceptual


# ---------------------------------------------------------------------------
# Training setup helpers
# ---------------------------------------------------------------------------

def prepare_distill_output(output_path: str):
    os.makedirs(output_path, exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)

    return tb_writer


def get_logger(path: str):
    import logging
    logger = logging.getLogger("distill")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path, "distill.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _load_stage1_cfg(model_path: str):
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        return eval(f.read(), {"Namespace": Namespace})


def _explicit_cli_args(argv):
    explicit = set()
    for arg in argv:
        if arg.startswith("--"):
            explicit.add(arg[2:].split("=", 1)[0])
    return explicit


def _merge_stage1_model_config(args, explicit):
    """
    Keep this script outside ScaffoldGS while still matching the Stage 1 model
    architecture. Argparse defaults would otherwise override cfg_args values
    such as appearance_dim=0.
    """
    stage1_cfg = _load_stage1_cfg(args.model_path)
    if stage1_cfg is None:
        return args

    model_keys = [
        "sh_degree", "feat_dim", "n_offsets", "voxel_size", "update_depth",
        "update_init_factor", "update_hierachy_factor", "use_feat_bank",
        "images", "train_views_file", "test_views_file",
        "resolution", "white_background", "data_device", "eval",
        "lod", "appearance_dim", "lowpoly", "ds", "ratio", "undistorted",
        "add_opacity_dist", "add_cov_dist", "add_color_dist",
    ]
    for key in model_keys:
        if key not in explicit and hasattr(stage1_cfg, key):
            setattr(args, key, getattr(stage1_cfg, key))
    return args


def _point_cloud_iteration_dir(model_path: str, iteration: int, round_id=None):
    candidates = []
    if round_id is not None:
        candidates.append(os.path.join(model_path, "point_cloud", f"iteration_{iteration}_round{round_id}"))
    candidates.append(os.path.join(model_path, "point_cloud", f"iteration_{iteration}"))
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "point_cloud.ply")):
            return candidate
    return None


def _checkpoint_cache_tag(point_cloud_dir: str) -> str:
    entries = []
    for filename in (
        "point_cloud.ply",
        "opacity_mlp.pt",
        "cov_mlp.pt",
        "color_mlp.pt",
    ):
        path = os.path.join(point_cloud_dir, filename)
        stat = os.stat(path)
        entries.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(entries)


def configure_trainable_parameters(gaussians, mode: str, logger=None):
    """Select which Scaffold-GS parameter families may absorb teacher errors."""
    allowed_by_mode = {
        "all": None,
        "shared_mlp": {"mlp_opacity", "mlp_cov", "mlp_color", "mlp_featurebank"},
        "shared_mlp_features": {
            "mlp_opacity", "mlp_cov", "mlp_color", "mlp_featurebank",
            "anchor_feat",
        },
        "shared_mlp_geometry": {
            "mlp_opacity", "mlp_cov", "mlp_color", "mlp_featurebank",
            "anchor", "offset",
        },
    }
    if mode not in allowed_by_mode:
        raise ValueError(f"Unknown parameter mode: {mode}")

    allowed = allowed_by_mode[mode]
    trainable_groups = []
    frozen_groups = []
    for group in gaussians.optimizer.param_groups:
        is_trainable = allowed is None or group["name"] in allowed
        for parameter in group["params"]:
            parameter.requires_grad_(is_trainable)
        (trainable_groups if is_trainable else frozen_groups).append(group["name"])

    if logger:
        logger.info(
            f"Parameter mode '{mode}': trainable={sorted(set(trainable_groups))}; "
            f"frozen={sorted(set(frozen_groups))}"
        )


# ---------------------------------------------------------------------------
# Main distillation training function
# ---------------------------------------------------------------------------

def train_distill(
    dataset_args,
    opt_args,
    pipe_args,
    distill_args,
    logger=None,
    tb_writer=None,
):
    """
    Full Stage 4 distillation training loop.

    Loads Checkpoint_A, generates teacher views (or loads from cache),
    then fine-tunes ScaffoldGS with combined real + teacher supervision.
    """
    # -----------------------------------------------------------------------
    # 1. Load Stage 1 checkpoint
    # -----------------------------------------------------------------------
    gaussians = GaussianModel(
        dataset_args.feat_dim,
        dataset_args.n_offsets,
        dataset_args.voxel_size,
        dataset_args.update_depth,
        dataset_args.update_init_factor,
        dataset_args.update_hierachy_factor,
        dataset_args.use_feat_bank,
        dataset_args.appearance_dim,
        dataset_args.ratio,
        dataset_args.add_opacity_dist,
        dataset_args.add_cov_dist,
        dataset_args.add_color_dist,
    )

    loaded_state_dir = None
    if distill_args.start_checkpoint:
        scene = Scene(dataset_args, gaussians, shuffle=False)
        logger.info(f"Loading checkpoint: {distill_args.start_checkpoint}")
        ckpt = torch.load(distill_args.start_checkpoint)
        point_cloud_dir = None
        if isinstance(ckpt, dict):
            point_cloud_dir = ckpt.get("point_cloud_dir")
            if point_cloud_dir and not os.path.isabs(point_cloud_dir):
                point_cloud_dir = os.path.join(os.path.dirname(distill_args.start_checkpoint), point_cloud_dir)
        if point_cloud_dir is None:
            ckpt_name = os.path.basename(distill_args.start_checkpoint)
            ckpt_iter = "".join(ch for ch in ckpt_name.split("_", 1)[0] if ch.isdigit())
            if ckpt_iter:
                point_cloud_dir = _point_cloud_iteration_dir(
                    os.path.dirname(distill_args.start_checkpoint),
                    int(ckpt_iter),
                )
        if point_cloud_dir is None:
            raise FileNotFoundError("Could not infer point_cloud/iteration_* directory from start_checkpoint.")
        logger.info(f"Loading distillation state from {point_cloud_dir}")
        gaussians.load_ply_sparse_gaussian(os.path.join(point_cloud_dir, "point_cloud.ply"))
        gaussians.load_mlp_checkpoints(point_cloud_dir, mode='split')
        loaded_state_dir = point_cloud_dir
    else:
        stage1_dir = _point_cloud_iteration_dir(
            dataset_args.model_path,
            distill_args.stage1_iteration,
        )
        if stage1_dir is None:
            raise FileNotFoundError(
                f"Could not find Stage 1 point cloud at "
                f"{dataset_args.model_path}/point_cloud/iteration_{distill_args.stage1_iteration}"
            )
        logger.info(f"Loading Stage 1 ScaffoldGS state from {stage1_dir}")
        scene = Scene(dataset_args, gaussians, load_iteration=distill_args.stage1_iteration, shuffle=False)
        loaded_state_dir = stage1_dir

    # Loaded PLY checkpoints do not restore this value, but anchor and offset
    # learning-rate schedules depend on it.
    if dataset_args.appearance_dim > 0:
        raise ValueError(
            "Distillation novel cameras do not have trained appearance IDs. "
            "Train Stage 1 with --appearance_dim 0 (the pipeline default)."
        )
    gaussians.spatial_lr_scale = scene.cameras_extent
    opt_args.position_lr_init = distill_args.position_lr_init
    opt_args.position_lr_final = distill_args.position_lr_final
    opt_args.position_lr_delay_mult = 1.0
    opt_args.position_lr_max_steps = distill_args.distill_iterations
    opt_args.offset_lr_init = distill_args.offset_lr_init
    opt_args.offset_lr_final = distill_args.offset_lr_final
    opt_args.offset_lr_delay_mult = 1.0
    opt_args.offset_lr_max_steps = distill_args.distill_iterations
    gaussians.training_setup(opt_args)
    configure_trainable_parameters(
        gaussians, distill_args.parameter_mode, logger=logger
    )

    if distill_args.lambda_anchor_reg > 0 and distill_args.parameter_mode not in {
        "all", "shared_mlp_geometry"
    }:
        logger.warning(
            "lambda_anchor_reg is non-zero, but the selected parameter mode freezes "
            "anchors and offsets; the regularizer will have no effect."
        )
    if distill_args.lambda_anchor_reg > 0 and distill_args.distill_densify_until > 0:
        raise ValueError(
            "Anchor regularisation requires stable anchor identities. Set "
            "--distill_densify_until 0, or set --lambda_anchor_reg 0 for a "
            "separate densification experiment."
        )

    # -----------------------------------------------------------------------
    # 2. Snapshot initial anchor geometry for regularisation
    # -----------------------------------------------------------------------
    anchor_init = gaussians._anchor.detach().clone()
    offset_init = gaussians._offset.detach().clone()
    logger.info(f"Snapshotted {anchor_init.shape[0]} anchors for regularisation.")

    # -----------------------------------------------------------------------
    # 3. Sample novel views and generate/load teacher images
    # -----------------------------------------------------------------------
    logger.info("Sampling novel camera poses...")
    novel_cameras = sample_novel_poses(
        scene,
        n_samples=distill_args.n_novel_views,
        device='cuda',
        exclude_test_cameras=True,
    )

    teacher_cache = os.path.join(distill_args.teacher_cache_dir, f"round{distill_args.round}")

    # Reuse teachers only when poses, generation settings, and source
    # checkpoint identity all match.
    teacher_settings = teacher_generation_settings(
        distill_args.teacher_strength,
        distill_args.teacher_guidance_scale,
        distill_args.teacher_steps,
        distill_args.teacher_prompt,
        distill_args.teacher_negative_prompt,
        distill_args.controlnet_scale,
        True,
        distill_args.depth_consistency_threshold,
        distill_args.teacher_seed,
        _checkpoint_cache_tag(loaded_state_dir),
    )
    need_generation = not teacher_cache_is_valid(
        teacher_cache, novel_cameras, teacher_settings
    )

    if need_generation:
        logger.info("Generating teacher images (this may take 30-60 min)...")
        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        teacher_gen = TeacherGenerator(device='cuda', use_sdxl=distill_args.use_sdxl)
        try:
            teacher_gen.generate_teachers(
                gaussians=gaussians,
                pipe_args=pipe_args,
                novel_cameras=novel_cameras,
                output_dir=teacher_cache,
                background=background,
                strength=distill_args.teacher_strength,
                guidance_scale=distill_args.teacher_guidance_scale,
                num_inference_steps=distill_args.teacher_steps,
                prompt=distill_args.teacher_prompt,
                negative_prompt=distill_args.teacher_negative_prompt,
                controlnet_conditioning_scale=distill_args.controlnet_scale,
                depth_consistency_filter=True,
                depth_consistency_threshold=distill_args.depth_consistency_threshold,
                seed=distill_args.teacher_seed,
                cache_tag=_checkpoint_cache_tag(loaded_state_dir),
            )
        finally:
            teacher_gen.unload()
    else:
        logger.info(f"Using cached teacher images from {teacher_cache}")
    unload_teacher_models()

    # Load teacher dataset
    teacher_dataset = TeacherDataset(novel_cameras, teacher_cache, device='cuda')
    minimum_teachers = min(
        distill_args.min_teacher_views, len(novel_cameras)
    )
    if len(teacher_dataset) < minimum_teachers:
        raise RuntimeError(
            f"Only {len(teacher_dataset)} teacher views passed filtering; "
            f"at least {minimum_teachers} are required. Inspect rendered_rgb, "
            "rendered_depth, and teacher_images, then adjust the pose path or "
            "--depth_consistency_threshold."
        )

    logger.info(f"Teacher dataset: {len(teacher_dataset)} pairs.")

    # -----------------------------------------------------------------------
    # 4. Distillation training loop
    # -----------------------------------------------------------------------
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras().copy()

    iterations = distill_args.distill_iterations
    log_interval = 100
    save_interval = iterations  # save at end by default

    ema_loss = 0.0
    progress_bar = tqdm(range(1, iterations + 1), desc="Distillation")

    # Counters for logging
    loss_accum = {"real": 0.0, "teacher": 0.0, "anchor_reg": 0.0, "total": 0.0}
    count = 0

    gaussians.train()

    for iteration in range(1, iterations + 1):

        gaussians.update_learning_rate(iteration)

        # ------------------------------------------------------------------
        # 4a. Real view loss (photometric on 5 training views)
        # ------------------------------------------------------------------
        # Sample a real training camera
        real_cam = train_cameras[iteration % len(train_cameras)]

        voxel_visible_mask = prefilter_voxel(real_cam, gaussians, pipe_args, background)
        retain_grad = iteration < distill_args.distill_densify_until

        render_pkg_real = render(
            real_cam, gaussians, pipe_args, background,
            visible_mask=voxel_visible_mask, retain_grad=retain_grad
        )
        rendered_real = render_pkg_real["render"]
        gt_real = real_cam.original_image.cuda()

        Ll1_real = l1_loss(rendered_real, gt_real)
        ssim_real = 1.0 - ssim(rendered_real, gt_real)
        scaling_reg = render_pkg_real["scaling"].prod(dim=1).mean()

        L_real = (
            (1.0 - opt_args.lambda_dssim) * Ll1_real
            + opt_args.lambda_dssim * ssim_real
            + 0.01 * scaling_reg
        )

        # ------------------------------------------------------------------
        # 4b. Teacher distillation loss (novel views)
        # ------------------------------------------------------------------
        teacher_cam, teacher_img = teacher_dataset.sample()

        voxel_visible_mask_t = prefilter_voxel(teacher_cam, gaussians, pipe_args, background)
        render_pkg_teacher = render(
            teacher_cam, gaussians, pipe_args, background,
            visible_mask=voxel_visible_mask_t, retain_grad=False
        )
        rendered_teacher = render_pkg_teacher["render"]

        L_teacher = teacher_distillation_loss(
            rendered_teacher,
            teacher_img,
            lambda_lpips=distill_args.lambda_lpips,
        )

        # ------------------------------------------------------------------
        # 4c. Anchor regularisation loss (soft geometry stiffness)
        # ------------------------------------------------------------------
        L_anchor_reg = anchor_regularisation_loss(gaussians, anchor_init, offset_init)

        # ------------------------------------------------------------------
        # 4d. Total loss
        # ------------------------------------------------------------------
        L_total = (
            L_real
            + distill_args.lambda_teacher * L_teacher
            + distill_args.lambda_anchor_reg * L_anchor_reg
        )

        L_total.backward()

        # ------------------------------------------------------------------
        # 4e. Densification (only in early distillation iterations)
        # ------------------------------------------------------------------
        with torch.no_grad():
            if (iteration < distill_args.distill_densify_until
                    and iteration > opt_args.start_stat):
                viewspace_pts = render_pkg_real.get("viewspace_points")
                visibility = render_pkg_real.get("visibility_filter")
                offset_mask = render_pkg_real.get("selection_mask")
                voxel_mask = voxel_visible_mask
                opacity = render_pkg_real.get("neural_opacity")

                if viewspace_pts is not None and viewspace_pts.grad is not None:
                    gaussians.training_statis(
                        viewspace_pts, opacity, visibility, offset_mask, voxel_mask
                    )
                    if (iteration > opt_args.update_from
                            and iteration % opt_args.update_interval == 0):
                        gaussians.adjust_anchor(
                            check_interval=opt_args.update_interval,
                            success_threshold=opt_args.success_threshold,
                            grad_threshold=opt_args.densify_grad_threshold,
                            min_opacity=opt_args.min_opacity,
                        )
                        # Extend anchor_init to match new anchors (new anchors get zero reg penalty)
                        if gaussians._anchor.shape[0] > anchor_init.shape[0]:
                            new_anchor_pad = gaussians._anchor[anchor_init.shape[0]:].detach().clone()
                            new_offset_pad = gaussians._offset[anchor_init.shape[0]:].detach().clone()
                            anchor_init = torch.cat([anchor_init, new_anchor_pad], dim=0)
                            offset_init = torch.cat([offset_init, new_offset_pad], dim=0)

        # ------------------------------------------------------------------
        # 4f. Optimizer step
        # ------------------------------------------------------------------
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------
        # 4g. Logging
        # ------------------------------------------------------------------
        ema_loss = 0.4 * L_total.item() + 0.6 * ema_loss
        loss_accum["real"] += L_real.item()
        loss_accum["teacher"] += L_teacher.item()
        loss_accum["anchor_reg"] += L_anchor_reg.item()
        loss_accum["total"] += L_total.item()
        count += 1

        if iteration % log_interval == 0:
            avg = {k: v / count for k, v in loss_accum.items()}
            progress_bar.set_postfix({
                "total": f"{avg['total']:.5f}",
                "real": f"{avg['real']:.5f}",
                "teacher": f"{avg['teacher']:.5f}",
                "reg": f"{avg['anchor_reg']:.6f}",
                "anchors": gaussians._anchor.shape[0],
            })
            if tb_writer:
                for k, v in avg.items():
                    tb_writer.add_scalar(f"distill/{k}_loss", v, iteration)
                tb_writer.add_scalar("distill/n_anchors", gaussians._anchor.shape[0], iteration)
            loss_accum = {k: 0.0 for k in loss_accum}
            count = 0

        progress_bar.update(1)

        # ------------------------------------------------------------------
        # 4h. Save checkpoint
        # ------------------------------------------------------------------
        if iteration == iterations or iteration % save_interval == 0:
            os.makedirs(distill_args.distill_output, exist_ok=True)

            # Also save the point cloud + MLPs in ScaffoldGS's standard layout.
            scene.model_path = distill_args.distill_output
            scene.save(iteration)
            save_path = os.path.join(
                distill_args.distill_output,
                f"chkpnt{iteration}_round{distill_args.round}.pth"
            )
            torch.save({
                "iteration": iteration,
                "round": distill_args.round,
                "point_cloud_dir": os.path.join("point_cloud", f"iteration_{iteration}"),
            }, save_path)
            logger.info(f"[ITER {iteration}] Saved checkpoint: {save_path}")

    progress_bar.close()

    # -----------------------------------------------------------------------
    # 5. Final evaluation on test cameras
    # -----------------------------------------------------------------------
    logger.info("\n--- Final Distillation Evaluation ---")
    gaussians.eval()
    test_cameras = scene.getTestCameras()

    l1_total, psnr_total, ssim_total = 0.0, 0.0, 0.0

    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Evaluating"):
            voxel_mask = prefilter_voxel(cam, gaussians, pipe_args, background)
            render_pkg = render(cam, gaussians, pipe_args, background, visible_mask=voxel_mask)
            rendered = render_pkg["render"].clamp(0.0, 1.0)
            gt = cam.original_image.cuda().clamp(0.0, 1.0)

            l1_total += l1_loss(rendered, gt).item()
            psnr_total += psnr(rendered, gt).mean().item()
            ssim_total += ssim(rendered, gt).item()

    n = len(test_cameras)
    if n == 0:
        logger.warning("No test cameras found; skipping final held-out evaluation.")
        return {
            "L1": None,
            "PSNR": None,
            "SSIM": None,
            "round": distill_args.round,
            "n_anchors_final": gaussians._anchor.shape[0],
        }

    logger.info(f"  L1:   {l1_total / n:.5f}")
    logger.info(f"  PSNR: {psnr_total / n:.4f} dB")
    logger.info(f"  SSIM: {ssim_total / n:.5f}")

    results = {
        "L1": l1_total / n,
        "PSNR": psnr_total / n,
        "SSIM": ssim_total / n,
        "round": distill_args.round,
        "n_anchors_final": gaussians._anchor.shape[0],
    }
    results_path = os.path.join(distill_args.distill_output, f"results_round{distill_args.round}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    gaussians.train()

    return results


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

class DistillationParams:
    """Additional arguments for the distillation stage."""

    def __init__(self):
        # Paths
        self.distill_output = "./output/distill"
        self.teacher_cache_dir = "./teacher_cache"
        self.start_checkpoint = None
        self.stage1_iteration = 30000

        # Training
        self.distill_iterations = 10000
        self.distill_densify_until = 0
        self.parameter_mode = "all"
        self.position_lr_init = 1e-4
        self.position_lr_final = 1e-6
        self.offset_lr_init = 1e-3
        self.offset_lr_final = 5e-5

        # Novel views
        self.n_novel_views = 40
        self.min_teacher_views = 8

        # Loss weights
        self.lambda_teacher = 0.2
        self.lambda_lpips = 0.1
        self.lambda_anchor_reg = 0.01

        # Teacher generation
        self.use_sdxl = False
        self.teacher_strength = 0.55
        self.teacher_guidance_scale = 7.5
        self.teacher_steps = 20
        self.controlnet_scale = 0.8
        self.depth_consistency_threshold = 0.25
        self.teacher_seed = 42
        self.teacher_prompt = "a high quality photograph of an outdoor scene, sharp details, no artifacts"
        self.teacher_negative_prompt = "blurry, floaters, artifacts, low quality, distorted geometry"

        # Iterative refinement round
        self.round = 1


def add_distillation_args(parser: ArgumentParser):
    g = parser.add_argument_group("Distillation Parameters")
    g.add_argument("--distill_output", type=str, default="./output/distill")
    g.add_argument("--teacher_cache_dir", type=str, default="./teacher_cache")
    g.add_argument("--start_checkpoint", type=str, default=None)
    g.add_argument("--stage1_iteration", type=int, default=30000)
    g.add_argument("--distill_iterations", type=int, default=10000)
    g.add_argument("--distill_densify_until", type=int, default=0)
    g.add_argument(
        "--parameter_mode",
        choices=["all", "shared_mlp", "shared_mlp_features", "shared_mlp_geometry"],
        default="all",
    )
    g.add_argument("--position_lr_init", type=float, default=1e-4)
    g.add_argument("--position_lr_final", type=float, default=1e-6)
    g.add_argument("--offset_lr_init", type=float, default=1e-3)
    g.add_argument("--offset_lr_final", type=float, default=5e-5)
    g.add_argument("--n_novel_views", type=int, default=40)
    g.add_argument("--min_teacher_views", type=int, default=8)
    g.add_argument("--lambda_teacher", type=float, default=0.2)
    g.add_argument("--lambda_lpips", type=float, default=0.1)
    g.add_argument("--lambda_anchor_reg", type=float, default=0.01)
    g.add_argument("--use_sdxl", action="store_true", default=False)
    g.add_argument("--teacher_strength", type=float, default=0.55)
    g.add_argument("--teacher_guidance_scale", type=float, default=7.5)
    g.add_argument("--teacher_steps", type=int, default=20)
    g.add_argument("--controlnet_scale", type=float, default=0.8)
    g.add_argument("--depth_consistency_threshold", type=float, default=0.25)
    g.add_argument("--teacher_seed", type=int, default=42)
    g.add_argument(
        "--teacher_prompt",
        default="a high quality photograph of an outdoor scene, sharp details, no artifacts",
    )
    g.add_argument(
        "--teacher_negative_prompt",
        default="blurry, floaters, artifacts, low quality, distorted geometry",
    )
    g.add_argument("--round", type=int, default=1)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="ScaffoldGS Distillation Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    add_distillation_args(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")

    cli_argv = sys.argv[1:]
    explicit = _explicit_cli_args(cli_argv)
    args = parser.parse_args(cli_argv)

    # GPU selection
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.model_path = os.path.abspath(args.model_path)
    args.distill_output = os.path.abspath(args.distill_output)
    args.teacher_cache_dir = os.path.abspath(args.teacher_cache_dir)
    if args.start_checkpoint:
        args.start_checkpoint = os.path.abspath(args.start_checkpoint)
    args = _merge_stage1_model_config(args, explicit)

    # Logger
    os.makedirs(args.distill_output, exist_ok=True)
    logger = get_logger(args.distill_output)
    logger.info(f"Distillation args: {args}")

    # TensorBoard
    tb_writer = prepare_distill_output(args.distill_output) if TENSORBOARD_FOUND else None

    # Extract param groups
    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    # Build distill_args namespace from flat args
    distill_args = DistillationParams()
    for field in vars(distill_args):
        if hasattr(args, field):
            setattr(distill_args, field, getattr(args, field))

    safe_state(args.quiet)

    train_distill(
        dataset_args=dataset_args,
        opt_args=opt_args,
        pipe_args=pipe_args,
        distill_args=distill_args,
        logger=logger,
        tb_writer=tb_writer,
    )

    logger.info("Distillation complete.")
