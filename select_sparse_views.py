"""Create reproducible sparse-view train/test manifests for COLMAP datasets."""

from argparse import ArgumentParser
from math import floor
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def choose_evenly_spaced(images, count, offset_fraction=0.0):
    if count < 1:
        raise ValueError("--count must be at least 1")
    if len(images) < count:
        raise ValueError(f"Found only {len(images)} images; cannot select {count}.")

    step = len(images) / count
    indices = [
        floor((index + offset_fraction) * step) % len(images)
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("Sparse-view selection produced duplicate indices.")
    return [images[index] for index in indices]


def write_manifest(path, images):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image.name}\n" for image in images))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--test_output", required=True)
    parser.add_argument(
        "--offset_fraction",
        type=float,
        default=0.5,
        help="Fractional position within each equal trajectory segment.",
    )
    args = parser.parse_args()

    image_dir = Path(args.source_path).expanduser().resolve() / args.images
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )
    train_images = choose_evenly_spaced(images, args.count, args.offset_fraction)
    train_names = {image.name for image in train_images}
    test_images = [image for image in images if image.name not in train_names]

    if not test_images:
        raise ValueError(
            "No held-out images remain. Use the complete dataset, not a five-image copy."
        )

    write_manifest(Path(args.train_output).resolve(), train_images)
    write_manifest(Path(args.test_output).resolve(), test_images)
    print(f"Selected {len(train_images)} training and {len(test_images)} test views.")
    print("Training views:")
    for image in train_images:
        print(f"  {image.name}")


if __name__ == "__main__":
    main()
