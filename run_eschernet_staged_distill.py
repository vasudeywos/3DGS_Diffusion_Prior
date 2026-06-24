"""Run staged EscherNet distillation controls from one Stage-1 checkpoint."""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd, dry_run=False):
    print("\n>>> " + " ".join(str(part) for part in cmd) + "\n")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_cache_dir", required=True)
    parser.add_argument("--stage1_model_path", required=True)
    parser.add_argument("--train_views_file", required=True)
    parser.add_argument("--test_views_file", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--stage1_iteration", type=int, default=10000)
    parser.add_argument("--anchorfeat_iterations", type=int, default=3000)
    parser.add_argument("--geometry_iterations", type=int, default=3000)
    parser.add_argument("--densify_iterations", type=int, default=1500)
    parser.add_argument("--run_teacher_densify", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = SCRIPT_DIR / "train_distill.py"
    base = [
        sys.executable, str(train_script),
        "--source_path", os.path.abspath(args.source_path),
        "--model_path", os.path.abspath(args.stage1_model_path),
        "--images", args.images,
        "--train_views_file", os.path.abspath(args.train_views_file),
        "--test_views_file", os.path.abspath(args.test_views_file),
        "--teacher_cache_dir", os.path.abspath(args.teacher_cache_dir),
        "--stage1_iteration", str(args.stage1_iteration),
        "--resolution", str(args.resolution),
        "--data_device", "cpu",
        "--gpu", args.gpu,
        "--eval",
    ]

    real_anchorfeat = output_dir / "stage2_real_only_mlp_anchorfeat"
    escher_anchorfeat = output_dir / "stage2_eschernet_mlp_anchorfeat"
    escher_geometry = output_dir / "stage2_eschernet_geometry_light"
    escher_densify = output_dir / "stage2_eschernet_teacher_densify"

    run(base + [
        "--distill_output", str(real_anchorfeat),
        "--distill_iterations", str(args.anchorfeat_iterations),
        "--distill_save_iterations", "1000", "2000", str(args.anchorfeat_iterations),
        "--parameter_mode", "mlp_anchorfeat",
        "--lambda_teacher", "0",
        "--lambda_teacher_l1", "0",
        "--lambda_lpips", "0",
        "--lambda_trajectory", "0",
        "--lambda_anchor_reg", "0",
    ], dry_run=args.dry_run)

    run(base + [
        "--distill_output", str(escher_anchorfeat),
        "--distill_iterations", str(args.anchorfeat_iterations),
        "--distill_save_iterations", "1000", "2000", str(args.anchorfeat_iterations),
        "--parameter_mode", "mlp_anchorfeat",
        "--min_teacher_views", "24",
        "--lambda_teacher", "0.05",
        "--lambda_teacher_l1", "0.2",
        "--lambda_lpips", "1.0",
        "--teacher_supervision_scale", "1.0",
        "--teacher_mask_mode", "error",
        "--teacher_mask_gamma", "1.0",
        "--min_teacher_quality", "0.25",
        "--max_runtime_teacher_lpips", "0.85",
        "--runtime_teacher_tau", "0.6",
        "--teacher_start_iteration", "300",
        "--teacher_ramp_iterations", "700",
        "--lambda_trajectory", "0.005",
        "--lambda_anchor_reg", "0",
    ], dry_run=args.dry_run)

    checkpoint = (
        escher_anchorfeat
        / f"chkpnt{args.anchorfeat_iterations}_round1.pth"
    )
    run(base + [
        "--distill_output", str(escher_geometry),
        "--start_checkpoint", str(checkpoint),
        "--distill_iterations", str(args.geometry_iterations),
        "--distill_save_iterations", "1000", "2000", str(args.geometry_iterations),
        "--parameter_mode", "geometry_light",
        "--min_teacher_views", "24",
        "--lambda_teacher", "0.02",
        "--lambda_teacher_l1", "0.1",
        "--lambda_lpips", "1.0",
        "--teacher_supervision_scale", "1.0",
        "--teacher_mask_mode", "error",
        "--teacher_mask_gamma", "1.25",
        "--min_teacher_quality", "0.30",
        "--max_runtime_teacher_lpips", "0.80",
        "--runtime_teacher_tau", "0.5",
        "--teacher_start_iteration", "0",
        "--teacher_ramp_iterations", "500",
        "--teacher_recovery_start", str(max(1, int(args.geometry_iterations * 0.75))),
        "--teacher_recovery_scale", "0.5",
        "--lambda_trajectory", "0.005",
        "--lambda_anchor_reg", "0.01",
        "--position_lr_init", "0.0",
        "--position_lr_final", "0.0",
        "--offset_lr_init", "1e-4",
        "--offset_lr_final", "1e-5",
    ], dry_run=args.dry_run)

    if args.run_teacher_densify:
        geometry_checkpoint = (
            escher_geometry
            / f"chkpnt{args.geometry_iterations}_round1.pth"
        )
        run(base + [
            "--distill_output", str(escher_densify),
            "--start_checkpoint", str(geometry_checkpoint),
            "--distill_iterations", str(args.densify_iterations),
            "--distill_save_iterations", "500", "1000", str(args.densify_iterations),
            "--parameter_mode", "densify_light",
            "--min_teacher_views", "24",
            "--lambda_teacher", "0.01",
            "--lambda_teacher_l1", "0.05",
            "--lambda_lpips", "1.0",
            "--teacher_supervision_scale", "1.0",
            "--teacher_mask_mode", "error",
            "--teacher_mask_gamma", "1.5",
            "--min_teacher_quality", "0.35",
            "--max_runtime_teacher_lpips", "0.75",
            "--runtime_teacher_tau", "0.5",
            "--teacher_start_iteration", "0",
            "--teacher_ramp_iterations", "300",
            "--lambda_trajectory", "0.002",
            "--lambda_anchor_reg", "0",
            "--teacher_densify_from", "200",
            "--teacher_densify_until", str(args.densify_iterations),
            "--teacher_densify_interval", "100",
            "--teacher_densify_grad_threshold", "0.00025",
            "--teacher_densify_success_threshold", "0.85",
            "--teacher_densify_min_opacity", "0.005",
            "--teacher_densify_min_weight", "0.45",
            "--position_lr_init", "2e-6",
            "--position_lr_final", "5e-7",
            "--offset_lr_init", "5e-5",
            "--offset_lr_final", "1e-5",
        ], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
