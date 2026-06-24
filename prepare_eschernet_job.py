"""Prepare camera-controlled EscherNet teachers for Scaffold-GS distillation."""

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = THIS_DIR / "Scaffold-GS-main"
if str(SCAFFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAFFOLD_ROOT))

from arguments import ModelParams
from scene import GaussianModel, Scene
from viewcrafter_teacher import (
    SCHEMA_VERSION,
    camera_pair_metrics,
    camera_to_record,
    compute_job_signature,
    interpolate_camera,
    order_cameras_on_ellipse,
)


def tensor_to_image(tensor):
    array = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array)


def load_stage_config(model_path):
    config_path = Path(model_path) / "cfg_args"
    if not config_path.is_file():
        return None
    return eval(config_path.read_text(), {"Namespace": Namespace})


def merge_stage_config(args):
    saved = load_stage_config(args.model_path)
    if saved is None:
        return args
    for key in (
        "feat_dim", "n_offsets", "voxel_size", "update_depth",
        "update_init_factor", "update_hierachy_factor", "use_feat_bank",
        "appearance_dim", "ratio", "add_opacity_dist", "add_cov_dist",
        "add_color_dist", "resolution", "white_background", "data_device",
    ):
        if hasattr(saved, key):
            setattr(args, key, getattr(saved, key))
    return args


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    parser.add_argument("--stage1_iteration", type=int, default=10000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_resolution", type=int, default=256)
    parser.add_argument("--targets_per_pair", type=int, default=6)
    parser.add_argument("--max_total_teachers", type=int, default=12)
    parser.add_argument("--max_pair_angle", type=float, default=110.0)
    parser.add_argument("--min_view_cosine", type=float, default=0.2)
    parser.add_argument("--max_normalized_baseline", type=float, default=1.5)
    parser.add_argument("--max_radial_difference", type=float, default=0.4)
    parser.add_argument("--min_compatible_pairs", type=int, default=2)
    parser.add_argument("--checkpoint_name", default="eschernet-6dof")
    parser.add_argument("--seed", type=int, default=123)
    args = merge_stage_config(parser.parse_args())

    if args.teacher_resolution % 8 != 0:
        raise ValueError("--teacher_resolution must be divisible by 8.")
    if args.targets_per_pair < 2:
        raise ValueError("--targets_per_pair must be at least 2.")
    if args.max_total_teachers < 2:
        raise ValueError("--max_total_teachers must be at least 2.")

    dataset = model.extract(args)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
        dataset.update_depth, dataset.update_init_factor,
        dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio,
        dataset.add_opacity_dist, dataset.add_cov_dist,
        dataset.add_color_dist,
    )
    scene = Scene(
        dataset, gaussians,
        load_iteration=args.stage1_iteration,
        shuffle=False,
    )
    cameras, focus, ellipse = order_cameras_on_ellipse(scene.getTrainCameras())
    if len(cameras) < 2:
        raise ValueError("EscherNet needs at least two sparse training views.")

    output_dir = Path(args.output_dir).resolve()
    input_dir = output_dir / "input_images"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old_path in input_dir.glob("*"):
        if old_path.is_file():
            old_path.unlink()

    inputs = []
    for index, camera in enumerate(cameras):
        path = input_dir / f"{index}.png"
        tensor_to_image(camera.original_image[:3]).save(path)
        inputs.append({
            "index": index,
            "path": str(path.relative_to(output_dir)),
            "source_image_name": camera.image_name,
            "camera": camera_to_record(camera, -1, -1),
        })

    clips = []
    rejected_pairs = []
    uid = 30000
    per_pair_t = np.linspace(
        1.0 / (args.targets_per_pair + 1.0),
        args.targets_per_pair / (args.targets_per_pair + 1.0),
        args.targets_per_pair,
    )
    for source_segment_index, (start, end) in enumerate(
        zip(cameras[:-1], cameras[1:])
    ):
        metrics = camera_pair_metrics(start, end, focus, ellipse)
        compatible = (
            metrics["angular_gap_degrees"] <= args.max_pair_angle
            and metrics["view_direction_cosine"] >= args.min_view_cosine
            and metrics["normalized_baseline"] <= args.max_normalized_baseline
            and metrics["relative_radial_difference"]
            <= args.max_radial_difference
        )
        if not compatible:
            rejected_pairs.append({
                "source_segment_index": source_segment_index,
                "start_image": start.image_name,
                "end_image": end.image_name,
                "metrics": metrics,
            })
            continue

        clip_index = len(clips)
        teachers = []
        for local_index, t in enumerate(per_pair_t):
            name = f"es_clip{clip_index:02d}_frame{local_index:02d}"
            camera = interpolate_camera(start, end, float(t), uid, name)
            teachers.append(camera_to_record(camera, clip_index, local_index))
            uid += 1
        clips.append({
            "clip_index": clip_index,
            "source_segment_index": source_segment_index,
            "start_input": source_segment_index,
            "end_input": source_segment_index + 1,
            "pair_metrics": metrics,
            "teachers": teachers,
        })

    if len(clips) < args.min_compatible_pairs:
        raise RuntimeError(
            f"Only {len(clips)} camera pairs passed compatibility filtering; "
            f"at least {args.min_compatible_pairs} are required. Rejected: "
            f"{json.dumps(rejected_pairs, indent=2)}"
        )

    # Keep complete adjacent pairs so train_distill.sample_adjacent() remains valid.
    selected = {}
    total = 0
    for clip in clips:
        chosen = []
        for first, second in zip(clip["teachers"][:-1], clip["teachers"][1:]):
            additions = [
                teacher for teacher in (first, second)
                if teacher["frame_index"] not in {
                    item["frame_index"] for item in chosen
                }
            ]
            if total + len(additions) > args.max_total_teachers:
                continue
            chosen.extend(additions)
            total += len(additions)
            if total >= args.max_total_teachers:
                break
        selected[clip["clip_index"]] = {
            item["frame_index"] for item in chosen
        }
        if total >= args.max_total_teachers:
            break

    for clip in clips:
        keep = selected.get(clip["clip_index"], set())
        clip["teachers"] = [
            teacher for teacher in clip["teachers"]
            if teacher["frame_index"] in keep
        ]
    clips = [clip for clip in clips if clip["teachers"]]

    job = {
        "schema_version": SCHEMA_VERSION,
        "teacher_backend": "eschernet",
        "checkpoint_name": args.checkpoint_name,
        "resolution": [args.teacher_resolution, args.teacher_resolution],
        "seed": args.seed,
        "inputs": inputs,
        "clips": clips,
        "rejected_pairs": rejected_pairs,
        "pair_filter": {
            "max_angle_degrees": args.max_pair_angle,
            "min_view_direction_cosine": args.min_view_cosine,
            "max_normalized_baseline": args.max_normalized_baseline,
            "max_relative_radial_difference": args.max_radial_difference,
        },
        "frame_filter": {
            "minimum_total_teachers": total,
            "max_total_teachers": args.max_total_teachers,
        },
        "trajectory": {
            "type": "ellipse_ordered_pairwise_interpolation",
            "focus": np.asarray(focus).tolist(),
            "ellipse_semi_axes": [
                float(ellipse["semi_a"]), float(ellipse["semi_b"])
            ],
        },
    }
    job["signature"] = compute_job_signature(job)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "viewcrafter_job.json"
    previous_signature = None
    if job_path.is_file():
        try:
            previous_signature = json.loads(job_path.read_text()).get("signature")
        except json.JSONDecodeError:
            pass
    job_path.write_text(json.dumps(job, indent=2))
    complete = output_dir / "generation_complete.json"
    if previous_signature != job["signature"] and complete.exists():
        complete.unlink()
    print(
        f"Prepared {len(inputs)} EscherNet inputs, {len(clips)} compatible "
        f"clips, and {total} target teachers at "
        f"{args.teacher_resolution}x{args.teacher_resolution}."
    )


if __name__ == "__main__":
    main()
