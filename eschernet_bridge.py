"""Generate camera-controlled EscherNet teachers for Scaffold-GS distillation."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eschernet_root", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cape_type", choices=["6DoF"], default="6DoF")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--pose_convention",
        choices=["scaffold_w2c", "scaffold_c2w"],
        default="scaffold_w2c",
        help=(
            "Pose matrix passed to EscherNet before its internal inverse-pose "
            "construction. scaffold_w2c matches EscherNet's NeRF eval path."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def camera_w2c(record):
    rotation = np.asarray(record["R"], dtype=np.float32)
    translation = np.asarray(record["T"], dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = translation
    return matrix


def camera_c2w(record):
    return np.linalg.inv(camera_w2c(record)).astype(np.float32)


def teacher_intrinsics(record, width, height):
    fov_x = float(record["FoVx"])
    fov_y = float(record["FoVy"])
    fx = width / (2.0 * np.tan(fov_x / 2.0))
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    return float(fx), float(fy), width / 2.0, height / 2.0


def image_quality(image):
    gray = (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    )
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    sharpness = float(0.5 * (np.var(dx) + np.var(dy)))
    dark_fraction = float(np.mean(image < 0.03))
    clipped_fraction = float(np.mean((image < 0.01) | (image > 0.99)))
    contrast = float(np.std(gray))
    # Conservative heuristic: reject degenerate outputs, do not over-rank texture.
    sharp_score = min(1.0, sharpness / 0.01)
    contrast_score = min(1.0, contrast / 0.20)
    exposure_score = max(0.0, 1.0 - max(dark_fraction, clipped_fraction))
    confidence = float(np.clip(
        0.35 * sharp_score + 0.35 * contrast_score + 0.30 * exposure_score,
        0.0,
        1.0,
    ))
    return {
        "confidence": confidence,
        "quality_score": confidence,
        "sharpness": sharpness,
        "contrast": contrast,
        "dark_fraction": dark_fraction,
        "clipped_fraction": clipped_fraction,
    }


def main():
    args = parse_args()
    root = Path(args.eschernet_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    job_path = job_dir / "viewcrafter_job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"Missing EscherNet job: {job_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing EscherNet checkpoint: {checkpoint}")
    job = json.loads(job_path.read_text())
    if job.get("teacher_backend") != "eschernet":
        raise ValueError(
            f"Expected teacher_backend='eschernet', got "
            f"{job.get('teacher_backend')!r}."
        )

    height, width = [int(value) for value in job["resolution"]]
    if args.resolution is not None:
        if args.resolution != width or args.resolution != height:
            raise ValueError(
                f"Job expects {width}x{height}; got --resolution "
                f"{args.resolution}."
            )
    if width != height:
        raise ValueError("EscherNet bridge currently expects square outputs.")
    if args.dry_run:
        target_count = sum(len(clip["teachers"]) for clip in job["clips"])
        print(
            f"Would generate {target_count} EscherNet teachers from "
            f"{len(job['inputs'])} inputs at {width}x{height}."
        )
        return

    os.chdir(root)
    sys.path.insert(0, str(root / args.cape_type))
    sys.path.insert(0, str(root))

    import einops
    import torch
    import torchvision
    from accelerate.utils import set_seed
    from diffusers import DDIMScheduler
    from PIL import Image
    from torchvision import transforms

    from CN_encoder import CN_encoder
    from dataset import get_pose
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

    seed = int(job.get("seed", 123) if args.seed is None else args.seed)
    set_seed(seed)
    weight_dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    scheduler = DDIMScheduler.from_pretrained(str(checkpoint), subfolder="scheduler")
    image_encoder = CN_encoder.from_pretrained(
        str(checkpoint), subfolder="image_encoder"
    )
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        str(checkpoint),
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
    )
    pipeline.image_encoder = image_encoder
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception as error:
        print(f"WARNING: xformers attention unavailable: {error}")
    pipeline.enable_vae_slicing()

    generator = torch.Generator(device=device).manual_seed(seed)

    input_images_all = []
    pose_in_all = []
    for item in job["inputs"]:
        image_path = job_dir / item["path"]
        image = Image.open(image_path).convert("RGB")
        input_images_all.append(image_transforms(image))
        pose_matrix = (
            camera_w2c(item["camera"])
            if args.pose_convention == "scaffold_w2c"
            else camera_c2w(item["camera"])
        )
        pose_in_all.append(get_pose(pose_matrix))

    input_images_all = (
        torch.stack(input_images_all, dim=0)
        .to(device)
        .to(weight_dtype)
    )
    pose_in_all = np.stack(pose_in_all)

    teacher_dir = job_dir / "teacher_images"
    metadata_dir = job_dir / "metadata"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for path in list(teacher_dir.glob("*.png")) + list(metadata_dir.glob("*.json")):
        path.unlink()

    teacher_count = 0
    for clip in job["clips"]:
        teachers = clip["teachers"]
        if not teachers:
            continue
        reference_indices = teachers[0].get("reference_indices")
        if reference_indices is None:
            reference_indices = list(range(len(job["inputs"])))
        reference_indices = [int(index) for index in reference_indices]
        input_images = input_images_all[reference_indices]
        pose_in = pose_in_all[reference_indices]
        pose_in_inv = np.linalg.inv(pose_in).transpose([0, 2, 1])
        pose_in_t = torch.from_numpy(pose_in).to(device).to(weight_dtype).unsqueeze(0)
        pose_in_inv_t = (
            torch.from_numpy(pose_in_inv).to(device).to(weight_dtype).unsqueeze(0)
        )
        pose_out = []
        for teacher in teachers:
            pose_matrix = (
                camera_w2c(teacher)
                if args.pose_convention == "scaffold_w2c"
                else camera_c2w(teacher)
            )
            pose_out.append(get_pose(pose_matrix))
        pose_out = np.stack(pose_out)
        pose_out_inv = np.linalg.inv(pose_out).transpose([0, 2, 1])
        pose_out_t = (
            torch.from_numpy(pose_out).to(device).to(weight_dtype).unsqueeze(0)
        )
        pose_out_inv_t = (
            torch.from_numpy(pose_out_inv)
            .to(device).to(weight_dtype).unsqueeze(0)
        )

        with torch.autocast("cuda"):
            result = pipeline(
                input_imgs=input_images,
                prompt_imgs=input_images,
                poses=[[pose_out_t, pose_out_inv_t], [pose_in_t, pose_in_inv_t]],
                height=height,
                width=width,
                T_in=input_images.shape[0],
                T_out=len(teachers),
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                output_type="numpy",
            ).images

        temporal_jumps = []
        if len(result) > 1:
            for first, second in zip(result[:-1], result[1:]):
                temporal_jumps.append(float(np.mean(np.abs(second - first))))
        median_jump = float(np.median(temporal_jumps)) if temporal_jumps else 0.0

        for local_result_index, (image_array, teacher) in enumerate(zip(result, teachers)):
            filename = (
                f"clip_{teacher['clip_index']:02d}_"
                f"frame_{teacher['frame_index']:02d}.png"
            )
            image_float = np.clip(image_array, 0.0, 1.0)
            quality = image_quality(image_float)
            if temporal_jumps:
                local_jump = temporal_jumps[
                    min(local_result_index, len(temporal_jumps) - 1)
                ]
                temporal_score = float(np.clip(
                    1.0 - local_jump / max(0.20, 2.5 * median_jump, 1e-6),
                    0.0,
                    1.0,
                ))
                quality["temporal_jump"] = local_jump
                quality["temporal_score"] = temporal_score
                quality["confidence"] = float(
                    0.75 * quality["confidence"] + 0.25 * temporal_score
                )
                quality["quality_score"] = quality["confidence"]
            image = (image_float * 255.0).round().astype(np.uint8)
            Image.fromarray(image).save(teacher_dir / filename)
            fx, fy, cx, cy = teacher_intrinsics(teacher, width, height)
            record = dict(teacher)
            record["height"] = height
            record["width"] = width
            record["fx"] = fx
            record["fy"] = fy
            record["cx"] = cx
            record["cy"] = cy
            record["FoVx"] = float(2.0 * np.arctan(width / (2.0 * fx)))
            record["FoVy"] = float(2.0 * np.arctan(height / (2.0 * fy)))
            record["teacher_backend"] = "eschernet"
            record["pose_convention"] = args.pose_convention
            record["quality"] = quality
            record["teacher_path"] = str(Path("teacher_images") / filename)
            (metadata_dir / filename.replace(".png", ".json")).write_text(
                json.dumps(record, indent=2)
            )
            teacher_count += 1

    complete = {
        "signature": job["signature"],
        "teacher_count": teacher_count,
        "teacher_backend": "eschernet",
        "checkpoint": str(checkpoint),
        "cape_type": args.cape_type,
        "resolution": [height, width],
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": seed,
        "max_total_teachers": int(job["frame_filter"]["max_total_teachers"]),
        "pose_convention": args.pose_convention,
        "intrinsics_source": "scaffold_fov_scaled_to_eschernet_resolution",
    }
    (job_dir / "generation_complete.json").write_text(
        json.dumps(complete, indent=2)
    )
    print(
        f"Exported {teacher_count} EscherNet teacher frames to {job_dir} "
        f"at {width}x{height}."
    )


if __name__ == "__main__":
    main()
