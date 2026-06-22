"""Prepare ordered sparse inputs and Scaffold-GS camera metadata for ViewCrafter."""

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
    camera_to_record,
    compute_job_signature,
    interpolate_camera,
    order_cameras_on_ellipse,
    selected_frame_indices,
    viewcrafter_frame_t,
)


def tensor_to_image(tensor):
    array = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
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
    parser.add_argument("--stage1_iteration", type=int, default=30000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_length", type=int, default=25)
    parser.add_argument("--frames_per_clip", type=int, default=12)
    parser.add_argument("--viewcrafter_height", type=int, default=320)
    parser.add_argument("--viewcrafter_width", type=int, default=512)
    parser.add_argument("--checkpoint_name", default="ViewCrafter_25_512")
    parser.add_argument("--seed", type=int, default=123)
    args = merge_stage_config(parser.parse_args())

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
    cameras, focus, ellipse = order_cameras_on_ellipse(
        scene.getTrainCameras()
    )
    if len(cameras) < 2:
        raise ValueError("ViewCrafter needs at least two sparse training views.")

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
        })

    frame_indices = selected_frame_indices(
        args.video_length, args.frames_per_clip
    )
    clips = []
    uid = 20000
    for clip_index, (start, end) in enumerate(zip(cameras[:-1], cameras[1:])):
        teachers = []
        for frame_index in frame_indices:
            t = viewcrafter_frame_t(frame_index, args.video_length)
            name = f"vc_clip{clip_index:02d}_frame{frame_index:02d}"
            camera = interpolate_camera(start, end, t, uid, name)
            teachers.append(camera_to_record(
                camera, clip_index, frame_index
            ))
            uid += 1
        clips.append({
            "clip_index": clip_index,
            "start_input": clip_index,
            "end_input": clip_index + 1,
            "teachers": teachers,
        })

    job = {
        "schema_version": SCHEMA_VERSION,
        "teacher_backend": "viewcrafter",
        "checkpoint_name": args.checkpoint_name,
        "resolution": [args.viewcrafter_height, args.viewcrafter_width],
        "video_length": args.video_length,
        "seed": args.seed,
        "inputs": inputs,
        "clips": clips,
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
    (output_dir / "viewcrafter_job.json").write_text(
        json.dumps(job, indent=2)
    )
    complete = output_dir / "generation_complete.json"
    if complete.exists():
        complete.unlink()
    print(
        f"Prepared {len(inputs)} ordered inputs, {len(clips)} clips, and "
        f"{sum(len(clip['teachers']) for clip in clips)} teacher cameras."
    )


if __name__ == "__main__":
    main()
