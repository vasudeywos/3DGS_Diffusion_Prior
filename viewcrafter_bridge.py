import argparse
import json
import os
import sys
from pathlib import Path

import sys
print("PYTHON =", sys.executable)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewcrafter_root", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dust3r_checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--bg_trd", type=float, default=0.2)
    parser.add_argument(
        "--render_chunk_size",
        type=int,
        default=4,
        help=(
            "Number of PyTorch3D trajectory views rendered together. "
            "Automatically halves on CUDA OOM down to one view."
        ),
    )
    parser.add_argument(
        "--max_alignment_error",
        type=float,
        default=0.15,
        help="Maximum DUSt3R/Scaffold camera RMSE as a fraction of camera radius.",
    )
    parser.add_argument("--prompt", default="Rotating view of a scene")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.viewcrafter_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    dust3r_checkpoint = Path(args.dust3r_checkpoint).resolve()
    job_path = job_dir / "viewcrafter_job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"Missing ViewCrafter job: {job_path}")
    job = json.loads(job_path.read_text())
    profiles = {
        "ViewCrafter_25_512": {
            "resolution": [320, 512],
            "load_size": 512,
            "force_1024": False,
            "config": "inference_pvd_512.yaml",
        },
        "ViewCrafter_25_sparse": {
            "resolution": [576, 1024],
            "load_size": 1024,
            "force_1024": True,
            "config": "inference_pvd_1024.yaml",
        },
    }
    expected_name = job.get("checkpoint_name")
    if expected_name not in profiles:
        raise ValueError(
            f"Unsupported ViewCrafter profile {expected_name!r}. "
            f"Expected one of {sorted(profiles)}."
        )
    profile = profiles[expected_name]
    config = (
        Path(args.config).resolve()
        if args.config
        else root / "configs" / profile["config"]
    )
    required = {
        "ViewCrafter root": root / "viewcrafter.py",
        "ViewCrafter checkpoint": checkpoint,
        "DUSt3R checkpoint": dust3r_checkpoint,
        "ViewCrafter config": config,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing ViewCrafter inputs:\n" + "\n".join(missing)
        )
    height, width = job["resolution"]
    if [height, width] != profile["resolution"]:
        raise ValueError(
            f"{expected_name} requires {profile['resolution']}, "
            f"got {[height, width]}."
        )

    minimum_teachers = int(job["frame_filter"]["minimum_total_teachers"])
    checkpoint_stat = checkpoint.stat()
    checkpoint_fingerprint = (
        f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}"
    )
    config_stat = config.stat()
    config_fingerprint = f"{config_stat.st_size}:{config_stat.st_mtime_ns}"
    complete_path = job_dir / "generation_complete.json"
    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text())
        except json.JSONDecodeError:
            complete = {}
        teacher_count = len(list((job_dir / "teacher_images").glob("*.png")))
        metadata_count = len(list((job_dir / "metadata").glob("*.json")))
        if (
            complete.get("signature") == job.get("signature")
            and complete.get("teacher_count", 0) >= minimum_teachers
            and complete.get("checkpoint") == str(checkpoint)
            and complete.get("checkpoint_fingerprint") == checkpoint_fingerprint
            and complete.get("config") == str(config)
            and complete.get("config_fingerprint") == config_fingerprint
            and complete.get("ddim_steps") == args.ddim_steps
            and complete.get("bg_trd") == args.bg_trd
            and complete.get("prompt") == args.prompt
            and teacher_count == complete.get("teacher_count")
            and metadata_count == complete.get("teacher_count")
        ):
            print(f"Reusing complete ViewCrafter cache at {job_dir}.")
            return

    if args.dry_run:
        print(
            f"Would run ViewCrafter on {len(job['inputs'])} sparse inputs, "
            f"{len(job['clips'])} clips, checkpoint={checkpoint}, config={config}"
        )
        return

    os.chdir(root)
    sys.path.insert(0, str(root))

    import glob
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from configs.infer_config import get_parser
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images
    from utils.pvd_utils import (
        generate_traj_interp,
        interp_traj,
        setup_renderer,
    )
    from viewcrafter import ViewCrafter

    class ScaffoldViewCrafter(ViewCrafter):
        def load_initial_dir(self, image_dir):
            image_files = sorted(
                glob.glob(os.path.join(image_dir, "*")),
                key=lambda path: int(Path(path).stem),
            )
            if len(image_files) < 2:
                raise ValueError("ViewCrafter needs at least two sparse views.")
            # Upstream forces the 1024 path. The 512 checkpoint instead needs
            # DUSt3R inputs at its native 512 preprocessing resolution.
            images = load_images(
                image_files,
                size=profile["load_size"],
                force_1024=profile["force_1024"],
            )
            originals = []
            for image in images:
                tensor = (image["img_ori"] + 1.0) / 2.0
                tensor = F.interpolate(
                    tensor,
                    size=(self.opts.height, self.opts.width),
                    mode="bilinear",
                    align_corners=False,
                )
                originals.append(tensor.squeeze(0).permute(1, 2, 0))
            return images, originals

        def run_render(
            self,
            pcd,
            imgs,
            masks,
            render_h,
            render_w,
            camera_traj,
            num_views,
            nbv=False,
        ):
            """Render the trajectory in bounded view batches.

            Upstream extends the complete point cloud to every trajectory
            camera at once. With five inputs and 97 interpolated poses this can
            exceed 20 GiB before diffusion starts.
            """
            total_views = int(num_views)
            chunk_size = min(max(1, args.render_chunk_size), total_views)

            while True:
                rendered_chunks = []
                mask_chunks = []
                try:
                    for start in range(0, total_views, chunk_size):
                        end = min(start + chunk_size, total_views)
                        # PyTorch3D 0.7.5 PerspectiveCameras does not support
                        # Python slice objects; it accepts explicit index lists.
                        chunk_cameras = camera_traj[list(range(start, end))]
                        renderer = setup_renderer(
                            chunk_cameras,
                            image_size=(render_h, render_w),
                        )["renderer"]
                        rendered, viewmask = self.render_pcd(
                            pcd,
                            imgs,
                            masks,
                            end - start,
                            renderer,
                            self.device,
                            nbv=nbv,
                        )
                        rendered_chunks.append(rendered.detach().cpu())
                        if viewmask is not None:
                            mask_chunks.append(viewmask.detach().cpu())
                        del renderer, rendered, viewmask
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    render_results = torch.cat(rendered_chunks, dim=0)
                    view_masks = (
                        torch.cat(mask_chunks, dim=0)
                        if mask_chunks
                        else None
                    )
                    print(
                        f"Rendered {total_views} trajectory views in chunks "
                        f"of {chunk_size}."
                    )
                    return render_results, view_masks
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    rendered_chunks.clear()
                    mask_chunks.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if chunk_size == 1:
                        raise RuntimeError(
                            "ViewCrafter point-cloud rendering still OOMs with "
                            "one camera at a time. Reduce point-cloud density "
                            "or use a larger-memory GPU."
                        )
                    new_chunk_size = max(1, chunk_size // 2)
                    print(
                        f"CUDA OOM with render_chunk_size={chunk_size}; "
                        f"retrying from the beginning with {new_chunk_size}."
                    )
                    chunk_size = new_chunk_size

        def generate_sparse_clips(self):
            c2ws = self.scene.get_im_poses().detach()
            principal_points = self.scene.get_principal_points().detach()
            focals = self.scene.get_focals().detach()
            shape = self.images[0]["true_shape"]
            render_h, render_w = int(shape[0][0]), int(shape[0][1])
            points = [
                point.detach()
                for point in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)
            ]

            if len(self.images) == 2:
                masks = None
            else:
                self.scene.min_conf_thr = float(
                    self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr))
                )
                masks = self.scene.get_masks()
                depths = self.scene.get_depthmaps()
                background_masks = []
                for depth in depths:
                    interior = depth[40:-40, :] if depth.shape[0] > 80 else depth
                    threshold = self.opts.bg_trd * (
                        torch.max(interior) + torch.min(interior)
                    )
                    background_masks.append(depth > threshold)
                masks = to_numpy([
                    mask + background
                    for mask, background in zip(masks, background_masks)
                ])

            images = np.asarray(self.scene.imgs)
            interpolated_c2ws = interp_traj(
                c2ws,
                n_inserts=self.opts.video_length,
                device=self.device,
            )
            trajectory, num_views = generate_traj_interp(
                c2ws,
                render_h,
                render_w,
                focals,
                principal_points,
                self.opts.video_length,
                self.device,
            )
            renderings, _ = self.run_render(
                points,
                images,
                masks,
                render_h,
                render_w,
                trajectory,
                num_views,
            )
            renderings = F.interpolate(
                renderings.permute(0, 3, 1, 2),
                size=(self.opts.height, self.opts.width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            for index, original in enumerate(self.img_ori):
                renderings[index * (self.opts.video_length - 1)] = original

            clips = []
            for clip_index in range(len(self.img_ori) - 1):
                start = clip_index * (self.opts.video_length - 1)
                clip = renderings[start:start + self.opts.video_length]
                # Keep only the active diffusion clip on the GPU. This does
                # not reduce the model's peak VRAM, but prevents completed
                # sparse-profile clips from accumulating there.
                generated = self.run_diffusion(clip).detach().cpu()
                clips.append(generated)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                clips,
                c2ws,
                interpolated_c2ws,
                trajectory,
                render_h,
                render_w,
            )

    official_parser = get_parser()
    opts = official_parser.parse_args([])
    opts.image_dir = str(job_dir / "input_images")
    opts.out_dir = str(job_dir / "viewcrafter_work")
    opts.exp_name = "scaffold_teacher"
    opts.save_dir = str(Path(opts.out_dir) / opts.exp_name)
    opts.mode = "sparse_view_interp"
    opts.bg_trd = args.bg_trd
    opts.seed = int(job["seed"])
    opts.ckpt_path = str(checkpoint)
    opts.config = str(config)
    opts.ddim_steps = args.ddim_steps
    opts.video_length = int(job["video_length"])
    opts.device = args.device
    opts.height = int(height)
    opts.width = int(width)
    opts.model_path = str(dust3r_checkpoint)
    opts.prompt = args.prompt
    opts.perframe_ae = True
    if args.render_chunk_size < 1:
        raise ValueError("--render_chunk_size must be at least 1.")
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        total_vram_gb = (
            torch.cuda.get_device_properties(device_index).total_memory
            / (1024 ** 3)
        )
        print(
            f"ViewCrafter profile={expected_name}, output={width}x{height}, "
            f"GPU VRAM={total_vram_gb:.1f} GiB."
        )
        if expected_name == "ViewCrafter_25_sparse" and total_vram_gb < 24:
            print(
                "WARNING: sparse 1024-profile inference may exceed available "
                "VRAM. Clips are retained on CPU after generation, but peak "
                "model/active-clip memory is unchanged."
            )

    model = ScaffoldViewCrafter(opts)
    (
        clips,
        dust_input_c2ws,
        dust_trajectory_c2ws,
        dust_trajectory_cameras,
        dust_render_h,
        dust_render_w,
    ) = (
        model.generate_sparse_clips()
    )
    if len(dust_trajectory_c2ws) != len(dust_trajectory_cameras):
        raise RuntimeError(
            "ViewCrafter pose and calibrated-camera trajectories have "
            f"different lengths: {len(dust_trajectory_c2ws)} vs "
            f"{len(dust_trajectory_cameras)}."
        )
    expected_generated_clips = len(job["inputs"]) - 1
    if len(clips) != expected_generated_clips:
        raise RuntimeError(
            f"ViewCrafter returned {len(clips)} clips; expected "
            f"{expected_generated_clips} from the ordered sparse inputs."
        )

    teacher_dir = job_dir / "teacher_images"
    metadata_dir = job_dir / "metadata"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for path in list(teacher_dir.glob("*.png")) + list(metadata_dir.glob("*.json")):
        path.unlink()

    teacher_count = 0

    def frame_intrinsics(trajectory_index):
        focal = (
            dust_trajectory_cameras.focal_length[trajectory_index]
            .detach().cpu().numpy().reshape(-1)
        )
        principal = (
            dust_trajectory_cameras.principal_point[trajectory_index]
            .detach().cpu().numpy().reshape(-1)
        )
        if focal.size == 1:
            focal = np.repeat(focal, 2)
        if principal.size != 2 or focal.size != 2:
            raise RuntimeError(
                "Unexpected ViewCrafter trajectory intrinsics shape: "
                f"focal={focal.shape}, principal={principal.shape}."
            )
        scale_x = width / float(dust_render_w)
        scale_y = height / float(dust_render_h)
        fx, fy = float(focal[0] * scale_x), float(focal[1] * scale_y)
        cx = float(principal[0] * scale_x)
        cy = float(principal[1] * scale_y)
        if min(fx, fy) <= 0 or not (0 <= cx <= width and 0 <= cy <= height):
            raise RuntimeError(
                "Invalid calibrated ViewCrafter intrinsics after resize: "
                f"fx={fx}, fy={fy}, cx={cx}, cy={cy}, size={width}x{height}."
            )
        return fx, fy, cx, cy

    def camera_center(record):
        rotation = np.asarray(record["R"], dtype=np.float64)
        translation = np.asarray(record["T"], dtype=np.float64)
        return -rotation @ translation

    def similarity_alignment(source, target):
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean
        covariance = (
            target_centered.T @ source_centered / source.shape[0]
        )
        u, singular_values, vh = np.linalg.svd(covariance)
        correction = np.eye(3)
        if np.linalg.det(u @ vh) < 0:
            correction[-1, -1] = -1
        rotation = u @ correction @ vh
        variance = np.mean(np.sum(source_centered ** 2, axis=1))
        scale = np.sum(singular_values * np.diag(correction)) / max(
            variance, 1e-12
        )
        translation = target_mean - scale * rotation @ source_mean
        return scale, rotation, translation

    source_centers = dust_input_c2ws[:, :3, 3].detach().cpu().numpy()
    target_centers = np.stack([
        camera_center(input_record["camera"])
        for input_record in job["inputs"]
    ])
    if source_centers.shape != target_centers.shape:
        raise RuntimeError(
            "DUSt3R input camera count does not match the Scaffold-GS job."
        )
    align_scale, align_rotation, align_translation = similarity_alignment(
        source_centers, target_centers
    )
    aligned_input_centers = (
        align_scale * (align_rotation @ source_centers.T).T
        + align_translation
    )
    alignment_rmse = float(np.sqrt(np.mean(np.sum(
        (aligned_input_centers - target_centers) ** 2, axis=1
    ))))
    target_radius = float(np.max(np.linalg.norm(
        target_centers - target_centers.mean(axis=0), axis=1
    )))
    normalized_alignment_error = alignment_rmse / max(target_radius, 1e-8)
    if normalized_alignment_error > args.max_alignment_error:
        raise RuntimeError(
            "DUSt3R camera alignment is unreliable: normalized RMSE "
            f"{normalized_alignment_error:.4f} exceeds "
            f"{args.max_alignment_error:.4f}. Inspect the five ordered inputs "
            "or adjust DUSt3R confidence/background thresholds."
        )

    quality_report = []
    exported_intrinsics = []

    def frame_quality_records(clip_tensor, candidate_indices):
        frames = clip_tensor.detach().float()
        gray = (
            0.299 * frames[..., 0]
            + 0.587 * frames[..., 1]
            + 0.114 * frames[..., 2]
        )
        endpoint_a = frames[0]
        endpoint_b = frames[-1]
        all_jumps = torch.mean(
            torch.abs(frames[1:] - frames[:-1]), dim=(1, 2, 3)
        )
        median_jump = float(torch.median(all_jumps).item())
        records = []
        for frame_index in candidate_indices:
            image = frames[frame_index]
            image_gray = gray[frame_index]
            dx = image_gray[:, 1:] - image_gray[:, :-1]
            dy = image_gray[1:, :] - image_gray[:-1, :]
            sharpness = float(
                0.5 * (torch.var(dx) + torch.var(dy))
            )
            clipped_fraction = float(
                ((image < 0.01) | (image > 0.99)).float().mean()
            )
            endpoint_novelty = float(min(
                torch.mean(torch.abs(image - endpoint_a)).item(),
                torch.mean(torch.abs(image - endpoint_b)).item(),
            ))
            local_jump = float(max(
                all_jumps[frame_index - 1].item(),
                all_jumps[min(frame_index, len(all_jumps) - 1)].item(),
            ))
            records.append({
                "frame_index": int(frame_index),
                "sharpness": sharpness,
                "clipped_fraction": clipped_fraction,
                "endpoint_novelty": endpoint_novelty,
                "local_jump": local_jump,
                "median_clip_jump": median_jump,
            })

        median_sharpness = float(np.median([
            record["sharpness"] for record in records
        ]))
        maximum_jump = max(0.20, 2.5 * median_jump)
        for record in records:
            record["accepted_by_filter"] = (
                record["sharpness"] >= 0.25 * median_sharpness
                and record["clipped_fraction"] <= 0.60
                and record["endpoint_novelty"] >= 0.01
                and record["local_jump"] <= maximum_jump
            )
            record["quality_score"] = (
                record["sharpness"] / max(median_sharpness, 1e-8)
                + 2.0 * record["endpoint_novelty"]
                - record["local_jump"] / max(maximum_jump, 1e-8)
                - record["clipped_fraction"]
            )
        return records

    def choose_filtered_frames(records, minimum, maximum):
        accepted = [
            record for record in records if record["accepted_by_filter"]
        ]
        if len(accepted) < minimum:
            accepted = sorted(
                records,
                key=lambda record: record["quality_score"],
                reverse=True,
            )[:minimum]
        accepted.sort(key=lambda record: record["frame_index"])
        if len(accepted) > maximum:
            positions = np.linspace(0, len(accepted) - 1, maximum)
            accepted = [accepted[round(position)] for position in positions]
        return {record["frame_index"] for record in accepted}

    for clip_record in job["clips"]:
        clip_index = clip_record["clip_index"]
        source_segment_index = clip_record["source_segment_index"]
        clip_tensor = clips[source_segment_index]
        if clip_tensor.shape[0] != job["video_length"]:
            raise RuntimeError(
                f"Clip {clip_index} has {clip_tensor.shape[0]} frames; "
                f"expected {job['video_length']}."
            )
        clip_tensor = ((clip_tensor + 1.0) / 2.0).clamp(0, 1)
        candidates = [
            teacher["frame_index"] for teacher in clip_record["teachers"]
        ]
        records = frame_quality_records(clip_tensor, candidates)
        selected = choose_filtered_frames(
            records,
            int(job["frame_filter"]["minimum_per_clip"]),
            int(job["frame_filter"]["maximum_per_clip"]),
        )
        quality_report.append({
            "clip_index": clip_index,
            "source_segment_index": source_segment_index,
            "selected_frames": sorted(selected),
            "frames": records,
        })
        for teacher in clip_record["teachers"]:
            frame_index = teacher["frame_index"]
            if frame_index not in selected:
                continue
            trajectory_index = (
                source_segment_index * (job["video_length"] - 1)
                + frame_index
            )
            dust_pose = (
                dust_trajectory_c2ws[trajectory_index]
                .detach().cpu().numpy()
            )
            aligned_rotation = (
                align_rotation @ dust_pose[:3, :3]
            ).astype(np.float32)
            aligned_position = (
                align_scale * align_rotation @ dust_pose[:3, 3]
                + align_translation
            ).astype(np.float32)
            aligned_translation = (
                -aligned_rotation.T @ aligned_position
            ).astype(np.float32)
            fx, fy, cx, cy = frame_intrinsics(trajectory_index)
            filename = (
                f"clip_{clip_index:02d}_frame_{frame_index:02d}.png"
            )
            array = (
                clip_tensor[frame_index].detach().cpu().numpy() * 255.0
            ).round().astype(np.uint8)
            Image.fromarray(array).save(teacher_dir / filename)

            record = dict(teacher)
            record["R"] = aligned_rotation.tolist()
            record["T"] = aligned_translation.tolist()
            record["height"] = int(height)
            record["width"] = int(width)
            record["fx"] = fx
            record["fy"] = fy
            record["cx"] = cx
            record["cy"] = cy
            record["FoVx"] = float(2.0 * np.arctan(width / (2.0 * fx)))
            record["FoVy"] = float(2.0 * np.arctan(height / (2.0 * fy)))
            exported_intrinsics.append({
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            })
            record["quality"] = next(
                item for item in records
                if item["frame_index"] == frame_index
            )
            record["teacher_path"] = str(
                Path("teacher_images") / filename
            )
            (metadata_dir / filename.replace(".png", ".json")).write_text(
                json.dumps(record, indent=2)
            )
            teacher_count += 1

    intrinsics_summary = {}
    if exported_intrinsics:
        for key in ("fx", "fy", "cx", "cy"):
            values = [item[key] for item in exported_intrinsics]
            intrinsics_summary[key] = {
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "mean": float(np.mean(values)),
            }
    (job_dir / "generation_complete.json").write_text(json.dumps({
        "signature": job["signature"],
        "teacher_count": teacher_count,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "config": str(config),
        "config_fingerprint": config_fingerprint,
        "ddim_steps": args.ddim_steps,
        "bg_trd": args.bg_trd,
        "prompt": args.prompt,
        "camera_alignment_rmse": alignment_rmse,
        "normalized_camera_alignment_error": normalized_alignment_error,
        "intrinsics_source": "viewcrafter_pytorch3d_trajectory",
        "intrinsics_summary": intrinsics_summary,
        "quality_report": quality_report,
    }, indent=2))
    print(
        f"Exported {teacher_count} ViewCrafter teacher frames to {job_dir}. "
        "DUSt3R-to-Scaffold normalized camera RMSE: "
        f"{normalized_alignment_error:.6f}"
    )


if __name__ == "__main__":
    main()
