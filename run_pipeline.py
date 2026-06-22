"""
run_pipeline.py

Convenience runner for the sparse-view distillation pipeline.

Stage 0: Select a reproducible sparse train split       → split manifests
Stage 1: Train ScaffoldGS on 5 sparse views          → Checkpoint_A
Stage 2: Sample novel poses (elliptical path)        → novel_cameras
Stage 3: Generate teacher images (SD1.5+ControlNet)  → teacher_cache/round1/
Stage 4: Distillation fine-tuning                    → Checkpoint_B
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
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = SCRIPT_DIR / "Scaffold-GS-main"


def run_cmd(cmd: list, dry_run: bool = False, cwd: Path = SCAFFOLD_ROOT):
    printable = " ".join(str(part) for part in cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}>>> {printable}\n")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/experiment")
    parser.add_argument("--n_sparse_views", type=int, default=5)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--train_views_file", type=str, default=None)
    parser.add_argument("--test_views_file", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--n_novel_views", type=int, default=40)
    parser.add_argument("--distill_iterations", type=int, default=10000)
    parser.add_argument("--lambda_teacher", type=float, default=0.2)
    parser.add_argument("--lambda_anchor_reg", type=float, default=0.01)
    parser.add_argument(
        "--parameter_mode",
        choices=["all", "shared_mlp", "shared_mlp_features", "shared_mlp_geometry"],
        default="all",
    )
    parser.add_argument("--position_lr_init", type=float, default=1e-4)
    parser.add_argument("--position_lr_final", type=float, default=1e-6)
    parser.add_argument("--offset_lr_init", type=float, default=1e-3)
    parser.add_argument("--offset_lr_final", type=float, default=5e-5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.source_path = os.path.abspath(args.source_path)
    args.output_dir = os.path.abspath(args.output_dir)
    stage1_output = os.path.join(args.output_dir, "stage1")
    distill_output = os.path.join(args.output_dir, f"distill_round{args.round}")
    teacher_cache = os.path.join(args.output_dir, "teacher_cache")
    distill_script = str(SCRIPT_DIR / "train_distill.py")

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
            "--iterations", "30000",
            "--voxel_size", "0.001",
            "--update_init_factor", "16",
            "--appearance_dim", "0",
            "--ratio", "1",
            "--gpu", args.gpu,
            "--save_iterations", "30000",
            "--test_iterations", "30000",
            "--eval",
        ]
        run_cmd(stage1_cmd, dry_run=args.dry_run)
    else:
        print("Skipping Stage 1 (--skip_stage1 set).")

    # ------------------------------------------------------------------
    # Stage 4: Distillation fine-tuning
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STAGE 4: Distillation fine-tuning (round {args.round})")
    print("=" * 60)

    start_ckpt_args = []
    if args.start_checkpoint:
        start_ckpt_args = ["--start_checkpoint", os.path.abspath(args.start_checkpoint)]

    distill_cmd = [
        sys.executable, distill_script,
        "--source_path", args.source_path,
        "--images", args.images,
        "--train_views_file", train_views_file,
        "--test_views_file", test_views_file,
        "--model_path", stage1_output,
        "--distill_output", distill_output,
        "--teacher_cache_dir", teacher_cache,
        "--distill_iterations", str(args.distill_iterations),
        "--n_novel_views", str(args.n_novel_views),
        "--lambda_teacher", str(args.lambda_teacher),
        "--lambda_anchor_reg", str(args.lambda_anchor_reg),
        "--parameter_mode", args.parameter_mode,
        "--position_lr_init", str(args.position_lr_init),
        "--position_lr_final", str(args.position_lr_final),
        "--offset_lr_init", str(args.offset_lr_init),
        "--offset_lr_final", str(args.offset_lr_final),
        "--stage1_iteration", "30000",
        "--round", str(args.round),
        "--gpu", args.gpu,
        "--eval",
    ] + start_ckpt_args
    run_cmd(distill_cmd, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Stage 1 output:     {stage1_output}")
    print(f"Distill output:     {distill_output}")
    print(f"Teacher cache:      {teacher_cache}")
    print("=" * 60)


if __name__ == "__main__":
    main()
