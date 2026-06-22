"""
run_pipeline.py

Convenience runner for the sparse-view distillation pipeline.

Stage 0: Select a reproducible sparse train split       → split manifests
Stage 1: Train ScaffoldGS on 5 sparse views             → Checkpoint_A
Stage 2: Prepare ellipse-ordered trajectory job          → ViewCrafter job
Stage 3: Generate ViewCrafter_25_512 trajectory teachers → teacher cache
Stage 4: Distillation fine-tuning                        → Checkpoint_B
Stage 4b (optional): Second round with Checkpoint_B  → Checkpoint_C

Usage:
    # Full pipeline from scratch
    python run_pipeline.py \
        --source_path data/mipnerf360/bicycle \
        --output_dir output/bicycle_sparse5 \
        --n_sparse_views 5 \
        --gpu 0

    # Skip Stage 1 if already trained
    python run_pipeline.py ... --skip_stage1

    # Second round refinement
    python run_pipeline.py ... --skip_stage1 --round 2 \
        --start_checkpoint output/bicycle_sparse5/distill_round1/chkpnt10000_round1.pth
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = SCRIPT_DIR / "Scaffold-GS-main"


def run_cmd(cmd: list, dry_run: bool = False, cwd: Path = SCAFFOLD_ROOT):
    printable = " ".join(str(part) for part in cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}>>> {printable}\n")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def find_metrics(payload):
    if isinstance(payload, dict):
        if all(key in payload for key in ("PSNR", "SSIM", "LPIPS")):
            return {
                key: payload[key]
                for key in ("PSNR", "SSIM", "LPIPS")
            }
        for value in payload.values():
            metrics = find_metrics(value)
            if metrics is not None:
                return metrics
    return None


def main():
    parser = ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/experiment")
    parser.add_argument("--n_sparse_views", type=int, default=5)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--train_views_file", type=str, default=None)
    parser.add_argument("--test_views_file", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--stage1_iterations", type=int, default=30000)
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--distill_iterations", type=int, default=10000)
    parser.add_argument("--viewcrafter_root", type=str, default=None)
    parser.add_argument("--viewcrafter_python", type=str, default=None)
    parser.add_argument("--viewcrafter_checkpoint", type=str, default=None)
    parser.add_argument("--dust3r_checkpoint", type=str, default=None)
    parser.add_argument("--viewcrafter_config", type=str, default=None)
    parser.add_argument(
        "--viewcrafter_profile",
        choices=["512", "sparse"],
        default="512",
        help="Use the 320x512 ablation checkpoint or sparse-view 576x1024 checkpoint.",
    )
    parser.add_argument("--viewcrafter_min_frames_per_clip", type=int, default=8)
    parser.add_argument("--viewcrafter_max_frames_per_clip", type=int, default=12)
    parser.add_argument("--viewcrafter_max_pair_angle", type=float, default=110.0)
    parser.add_argument("--viewcrafter_min_view_cosine", type=float, default=0.2)
    parser.add_argument(
        "--viewcrafter_max_normalized_baseline", type=float, default=1.5
    )
    parser.add_argument(
        "--viewcrafter_max_radial_difference", type=float, default=0.4
    )
    parser.add_argument("--viewcrafter_ddim_steps", type=int, default=50)
    parser.add_argument("--viewcrafter_bg_trd", type=float, default=0.2)
    parser.add_argument(
        "--viewcrafter_max_alignment_error", type=float, default=0.15
    )
    parser.add_argument("--viewcrafter_seed", type=int, default=123)
    parser.add_argument("--skip_viewcrafter", action="store_true")
    parser.add_argument("--lambda_teacher", type=float, default=0.2)
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--lambda_trajectory", type=float, default=0.03)
    parser.add_argument("--lambda_anchor_reg", type=float, default=0.01)
    parser.add_argument(
        "--parameter_mode",
        choices=["all", "shared_mlp", "shared_mlp_features", "shared_mlp_geometry"],
        default="all",
    )
    parser.add_argument(
        "--compare_parameter_modes",
        action="store_true",
        help="Run shared_mlp and all from the same Stage-1 checkpoint/cache.",
    )
    parser.add_argument("--position_lr_init", type=float, default=1e-4)
    parser.add_argument("--position_lr_final", type=float, default=1e-6)
    parser.add_argument("--offset_lr_init", type=float, default=1e-3)
    parser.add_argument("--offset_lr_final", type=float, default=5e-5)
    parser.add_argument("--enable_densification_phase", action="store_true")
    parser.add_argument("--densification_iterations", type=int, default=4000)
    parser.add_argument("--densification_teacher_weight", type=float, default=0.08)
    parser.add_argument(
        "--densification_trajectory_weight", type=float, default=0.01
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--sanity_200",
        action="store_true",
        help="Run a 200-iteration Stage-1 + shared_mlp/all sanity comparison.",
    )
    args = parser.parse_args()
    if args.sanity_200:
        args.stage1_iterations = 200
        args.distill_iterations = 200
        args.compare_parameter_modes = True
    if args.compare_parameter_modes and args.enable_densification_phase:
        raise ValueError(
            "--compare_parameter_modes cannot be combined with the optional "
            "densification phase."
        )

    args.source_path = os.path.abspath(args.source_path)
    args.output_dir = os.path.abspath(args.output_dir)
    stage1_output = os.path.join(args.output_dir, "stage1")
    distill_script = str(SCRIPT_DIR / "train_distill.py")
    viewcrafter_root = Path(
        args.viewcrafter_root or SCRIPT_DIR / "ViewCrafter"
    ).expanduser().resolve()
    viewcrafter_python = str(Path(
        args.viewcrafter_python
        or Path.home() / "miniconda3/envs/viewcrafter/bin/python"
    ).expanduser().resolve())
    profile = {
        "512": {
            "checkpoint_name": "ViewCrafter_25_512",
            "height": 320,
            "width": 512,
            "checkpoint": "model.ckpt",
            "config": "inference_pvd_512.yaml",
        },
        "sparse": {
            "checkpoint_name": "ViewCrafter_25_sparse",
            "height": 576,
            "width": 1024,
            "checkpoint": "model_sparse.ckpt",
            "config": "inference_pvd_1024.yaml",
        },
    }[args.viewcrafter_profile]
    distill_output = os.path.join(
        args.output_dir,
        f"distill_{args.viewcrafter_profile}_round{args.round}_stable",
    )
    densified_output = os.path.join(
        args.output_dir,
        f"distill_{args.viewcrafter_profile}_round{args.round}_densified",
    )
    teacher_cache = os.path.join(
        args.output_dir, "teacher_cache", args.viewcrafter_profile
    )
    round_teacher_cache = os.path.join(
        teacher_cache, f"round{args.round}"
    )
    viewcrafter_checkpoint = Path(
        args.viewcrafter_checkpoint
        or viewcrafter_root / "checkpoints" / profile["checkpoint"]
    ).expanduser().resolve()
    dust3r_checkpoint = Path(
        args.dust3r_checkpoint
        or viewcrafter_root
        / "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ).expanduser().resolve()
    viewcrafter_config = Path(
        args.viewcrafter_config
        or viewcrafter_root / "configs" / profile["config"]
    ).expanduser().resolve()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 0: Reproducible sparse-view split
    # ------------------------------------------------------------------
    if args.train_views_file:
        train_views_file = os.path.abspath(args.train_views_file)
        if not args.test_views_file:
            raise ValueError("--test_views_file is required with --train_views_file.")
        test_views_file = os.path.abspath(args.test_views_file)
    else:
        split_dir = os.path.join(args.output_dir, "splits")
        train_views_file = os.path.join(split_dir, "train_views.txt")
        test_views_file = os.path.join(split_dir, "test_views.txt")
        split_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "select_sparse_views.py"),
            "--source_path", args.source_path,
            "--images", args.images,
            "--count", str(args.n_sparse_views),
            "--train_output", train_views_file,
            "--test_output", test_views_file,
        ]
        run_cmd(split_cmd, dry_run=args.dry_run, cwd=SCRIPT_DIR)

    # ------------------------------------------------------------------
    # Stage 1: Sparse-view ScaffoldGS training
    # ------------------------------------------------------------------
    if not args.skip_stage1:
        print("=" * 60)
        print("STAGE 1: Sparse-view ScaffoldGS training")
        print("=" * 60)

        stage1_cmd = [
            sys.executable, "train.py",
            "--source_path", args.source_path,
            "--images", args.images,
            "--train_views_file", train_views_file,
            "--test_views_file", test_views_file,
            "--model_path", stage1_output,
            "--iterations", str(args.stage1_iterations),
            "--resolution", str(args.resolution),
            "--voxel_size", "0.001",
            "--update_init_factor", "16",
            "--appearance_dim", "0",
            "--data_device", "cpu",
            "--ratio", "1",
            "--gpu", args.gpu,
            "--save_iterations", str(args.stage1_iterations),
            "--test_iterations", str(args.stage1_iterations),
            "--eval",
        ]
        run_cmd(stage1_cmd, dry_run=args.dry_run)
    else:
        print("Skipping Stage 1 (--skip_stage1 set).")

    # ------------------------------------------------------------------
    # Stage 2: Prepare ViewCrafter job and matching Scaffold-GS cameras
    # ------------------------------------------------------------------
    prepare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_viewcrafter_job.py"),
        "--source_path", args.source_path,
        "--images", args.images,
        "--train_views_file", train_views_file,
        "--test_views_file", test_views_file,
        "--model_path", stage1_output,
        "--data_device", "cpu",
        "--stage1_iteration", str(args.stage1_iterations),
        "--output_dir", round_teacher_cache,
        "--video_length", "25",
        "--min_frames_per_clip", str(args.viewcrafter_min_frames_per_clip),
        "--max_frames_per_clip", str(args.viewcrafter_max_frames_per_clip),
        "--interior_frame_start", "4",
        "--interior_frame_end", "21",
        "--max_pair_angle", str(args.viewcrafter_max_pair_angle),
        "--min_view_cosine", str(args.viewcrafter_min_view_cosine),
        "--max_normalized_baseline",
        str(args.viewcrafter_max_normalized_baseline),
        "--max_radial_difference",
        str(args.viewcrafter_max_radial_difference),
        "--viewcrafter_height", str(profile["height"]),
        "--viewcrafter_width", str(profile["width"]),
        "--checkpoint_name", profile["checkpoint_name"],
        "--seed", str(args.viewcrafter_seed),
        "--eval",
    ]
    run_cmd(prepare_cmd, dry_run=args.dry_run, cwd=SCRIPT_DIR)

    # ------------------------------------------------------------------
    # Stage 3: ViewCrafter generation in its isolated environment
    # ------------------------------------------------------------------
    if not args.skip_viewcrafter:
        bridge_cmd = [
            viewcrafter_python,
            str(SCRIPT_DIR / "viewcrafter_bridge.py"),
            "--viewcrafter_root", str(viewcrafter_root),
            "--job_dir", round_teacher_cache,
            "--checkpoint", str(viewcrafter_checkpoint),
            "--dust3r_checkpoint", str(dust3r_checkpoint),
            "--config", str(viewcrafter_config),
            "--device", "cuda:0",
            "--ddim_steps", str(args.viewcrafter_ddim_steps),
            "--bg_trd", str(args.viewcrafter_bg_trd),
            "--max_alignment_error",
            str(args.viewcrafter_max_alignment_error),
        ]
        run_cmd(bridge_cmd, dry_run=args.dry_run, cwd=SCRIPT_DIR)
    else:
        print("Skipping Stage 3 (--skip_viewcrafter set).")

    validate_cache_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "validate_viewcrafter_cache.py"),
        "--cache_dir", round_teacher_cache,
    ]
    run_cmd(validate_cache_cmd, dry_run=args.dry_run, cwd=SCRIPT_DIR)

    # ------------------------------------------------------------------
    # Stage 4: Distillation fine-tuning
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STAGE 4: Distillation fine-tuning (round {args.round})")
    print("=" * 60)

    start_ckpt_args = []
    if args.start_checkpoint:
        start_ckpt_args = ["--start_checkpoint", os.path.abspath(args.start_checkpoint)]

    parameter_modes = (
        ["shared_mlp", "all"]
        if args.compare_parameter_modes
        else [args.parameter_mode]
    )
    distill_outputs = {}
    for parameter_mode in parameter_modes:
        mode_output = (
            os.path.join(
                args.output_dir,
                f"distill_{args.viewcrafter_profile}_round"
                f"{args.round}_{parameter_mode}",
            )
            if args.compare_parameter_modes
            else distill_output
        )
        distill_outputs[parameter_mode] = mode_output
        distill_cmd = [
            sys.executable, distill_script,
            "--source_path", args.source_path,
            "--images", args.images,
            "--train_views_file", train_views_file,
            "--test_views_file", test_views_file,
            "--model_path", stage1_output,
            "--data_device", "cpu",
            "--resolution", str(args.resolution),
            "--distill_output", mode_output,
            "--teacher_cache_dir", teacher_cache,
            "--distill_iterations", str(args.distill_iterations),
            "--lambda_teacher", str(args.lambda_teacher),
            "--lambda_lpips", str(args.lambda_lpips),
            "--lambda_trajectory", str(args.lambda_trajectory),
            "--lambda_anchor_reg", str(args.lambda_anchor_reg),
            "--parameter_mode", parameter_mode,
            "--position_lr_init", str(args.position_lr_init),
            "--position_lr_final", str(args.position_lr_final),
            "--offset_lr_init", str(args.offset_lr_init),
            "--offset_lr_final", str(args.offset_lr_final),
            "--stage1_iteration", str(args.stage1_iterations),
            "--round", str(args.round),
            "--gpu", args.gpu,
            "--eval",
        ] + start_ckpt_args
        run_cmd(distill_cmd, dry_run=args.dry_run)

    if args.enable_densification_phase:
        stable_checkpoint = os.path.join(
            distill_output,
            f"chkpnt{args.distill_iterations}_round{args.round}.pth",
        )
        densify_cmd = [
            sys.executable, distill_script,
            "--source_path", args.source_path,
            "--images", args.images,
            "--train_views_file", train_views_file,
            "--test_views_file", test_views_file,
            "--model_path", stage1_output,
            "--data_device", "cpu",
            "--distill_output", densified_output,
            "--teacher_cache_dir", teacher_cache,
            "--distill_iterations", str(args.densification_iterations),
            "--distill_densify_until", str(args.densification_iterations),
            "--lambda_teacher", str(args.densification_teacher_weight),
            "--lambda_trajectory",
            str(args.densification_trajectory_weight),
            "--lambda_anchor_reg", "0.0",
            "--parameter_mode", "all",
            "--position_lr_init", str(args.position_lr_final),
            "--position_lr_final", str(args.position_lr_final * 0.1),
            "--offset_lr_init", str(args.offset_lr_final),
            "--offset_lr_final", str(args.offset_lr_final * 0.2),
            "--start_checkpoint", stable_checkpoint,
            "--stage1_iteration", str(args.stage1_iterations),
            "--round", str(args.round),
            "--gpu", args.gpu,
            "--eval",
        ]
        run_cmd(densify_cmd, dry_run=args.dry_run)

    if not args.dry_run:
        summary = {
            "stage1_iterations": args.stage1_iterations,
            "distill_iterations": args.distill_iterations,
            "viewcrafter_profile": args.viewcrafter_profile,
            "stage1": None,
            "distillation": {},
        }
        stage1_results = Path(stage1_output) / "results.json"
        if stage1_results.is_file():
            summary["stage1"] = find_metrics(
                json.loads(stage1_results.read_text())
            )
        for mode, output in distill_outputs.items():
            result_path = Path(output) / f"results_round{args.round}.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                metrics = find_metrics(result)
                summary["distillation"][mode] = {
                    "metrics": metrics,
                    "delta_vs_stage1": (
                        {
                            key: metrics[key] - summary["stage1"][key]
                            for key in ("PSNR", "SSIM", "LPIPS")
                        }
                        if metrics is not None and summary["stage1"] is not None
                        else None
                    ),
                    "details": result,
                }
        summary_path = (
            Path(args.output_dir)
            / f"comparison_summary_{args.viewcrafter_profile}.json"
        )
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Comparison summary: {summary_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Stage 1 output:     {stage1_output}")
    for mode, output in distill_outputs.items():
        print(f"Distill ({mode}): {output}")
    if args.enable_densification_phase:
        print(f"Densified output:   {densified_output}")
    print(f"Teacher cache:      {teacher_cache}")
    print("=" * 60)


if __name__ == "__main__":
    main()
