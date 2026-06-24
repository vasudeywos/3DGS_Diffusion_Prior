"""Validate EscherNet teacher cache before Scaffold-GS distillation."""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


REQUIRED_INTRINSICS = ("fx", "fy", "cx", "cy", "height", "width")


def job_signature(job):
    payload = dict(job)
    payload.pop("signature", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    job = json.loads((cache_dir / "viewcrafter_job.json").read_text())
    complete = json.loads((cache_dir / "generation_complete.json").read_text())
    if job.get("teacher_backend") != "eschernet":
        raise RuntimeError("Cache job is not an EscherNet job.")
    signature = job_signature(job)
    if job.get("signature") != signature:
        raise RuntimeError("EscherNet job signature is stale or corrupted.")
    if complete.get("signature") != signature:
        raise RuntimeError("Generation result does not match the current job.")
    if complete.get("teacher_backend") != "eschernet":
        raise RuntimeError("Generation result is not tagged as EscherNet.")

    expected = int(complete["teacher_count"])
    metadata_paths = sorted((cache_dir / "metadata").glob("*.json"))
    image_paths = sorted((cache_dir / "teacher_images").glob("*.png"))
    if len(metadata_paths) != expected or len(image_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} teachers, found {len(image_paths)} images "
            f"and {len(metadata_paths)} metadata files."
        )

    adjacent_pairs = 0
    by_clip = {}
    for metadata_path in metadata_paths:
        record = json.loads(metadata_path.read_text())
        missing = [key for key in REQUIRED_INTRINSICS if key not in record]
        if missing:
            raise RuntimeError(f"{metadata_path.name} missing {missing}.")
        width, height = int(record["width"]), int(record["height"])
        fx, fy = float(record["fx"]), float(record["fy"])
        cx, cy = float(record["cx"]), float(record["cy"])
        if min(width, height, fx, fy) <= 0:
            raise RuntimeError(f"Invalid intrinsics in {metadata_path.name}.")
        if not (0 <= cx <= width and 0 <= cy <= height):
            raise RuntimeError(f"Invalid principal point in {metadata_path.name}.")
        teacher_path = Path(record["teacher_path"])
        if not teacher_path.is_absolute():
            teacher_path = cache_dir / teacher_path
        with Image.open(teacher_path) as image:
            if image.size != (width, height):
                raise RuntimeError(
                    f"{teacher_path.name} has size {image.size}; metadata "
                    f"expects {(width, height)}."
                )
        by_clip.setdefault(int(record["clip_index"]), []).append(
            int(record["frame_index"])
        )

    for frame_indices in by_clip.values():
        frame_indices = sorted(frame_indices)
        adjacent_pairs += sum(
            1 for first, second in zip(frame_indices[:-1], frame_indices[1:])
            if second == first + 1
        )
    if adjacent_pairs == 0:
        raise RuntimeError(
            "Cache has no consecutive frame pairs; train_distill cannot sample "
            "adjacent teacher pairs."
        )

    print(
        f"Validated {expected} EscherNet teachers at "
        f"{job['resolution'][1]}x{job['resolution'][0]} with "
        f"{adjacent_pairs} adjacent pairs. "
        f"Pose convention: {complete.get('pose_convention')}."
    )


if __name__ == "__main__":
    main()
