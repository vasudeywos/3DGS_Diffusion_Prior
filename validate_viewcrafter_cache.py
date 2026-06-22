"""Validate calibrated ViewCrafter teachers before Scaffold-GS distillation."""

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

    signature = job_signature(job)
    if job.get("signature") != signature:
        raise RuntimeError("ViewCrafter job signature is stale or corrupted.")
    if complete.get("signature") != signature:
        raise RuntimeError("Generation result does not match the current job.")
    if complete.get("intrinsics_source") != "viewcrafter_pytorch3d_trajectory":
        raise RuntimeError("Teacher cache does not contain calibrated trajectory intrinsics.")

    metadata_paths = sorted((cache_dir / "metadata").glob("*.json"))
    expected = int(complete["teacher_count"])
    if len(metadata_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} metadata files, found {len(metadata_paths)}."
        )

    maximum_principal_offset = 0.0
    maximum_principal_offset_pixels = 0.0
    for metadata_path in metadata_paths:
        record = json.loads(metadata_path.read_text())
        missing = [key for key in REQUIRED_INTRINSICS if key not in record]
        if missing:
            raise RuntimeError(f"{metadata_path.name} is missing {missing}.")

        width, height = int(record["width"]), int(record["height"])
        fx, fy = float(record["fx"]), float(record["fy"])
        cx, cy = float(record["cx"]), float(record["cy"])
        if min(width, height, fx, fy) <= 0:
            raise RuntimeError(f"Invalid focal length or size in {metadata_path.name}.")
        if not (0 <= cx <= width and 0 <= cy <= height):
            raise RuntimeError(f"Invalid principal point in {metadata_path.name}.")

        teacher_path = Path(record["teacher_path"])
        if not teacher_path.is_absolute():
            teacher_path = cache_dir / teacher_path
        with Image.open(teacher_path) as image:
            if image.size != (width, height):
                raise RuntimeError(
                    f"{teacher_path.name} is {image.size}, metadata expects "
                    f"{(width, height)}."
                )

        offset = max(
            abs(cx - width / 2.0) / width,
            abs(cy - height / 2.0) / height,
        )
        offset_pixels = max(
            abs(cx - width / 2.0),
            abs(cy - height / 2.0),
        )
        maximum_principal_offset = max(maximum_principal_offset, offset)
        maximum_principal_offset_pixels = max(
            maximum_principal_offset_pixels, offset_pixels
        )

    minimum = int(job["frame_filter"]["minimum_total_teachers"])
    if expected < minimum:
        raise RuntimeError(f"Only {expected} teachers were exported; need {minimum}.")

    print(
        f"Validated {expected} calibrated teachers at "
        f"{job['resolution'][1]}x{job['resolution'][0]}. "
        f"Maximum principal-point offset: "
        f"{maximum_principal_offset_pixels:.3f}px "
        f"({100.0 * maximum_principal_offset:.3f}% of image dimension)."
    )
    if maximum_principal_offset > 0.01:
        print(
            "WARNING: principal-point offset exceeds 1%. The calibrated "
            "off-center projection is enabled; inspect teacher/render overlays."
        )


if __name__ == "__main__":
    main()
