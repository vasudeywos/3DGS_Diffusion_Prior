"""
teacher_generator.py

Generates teacher images from ScaffoldGS renders using:
  - SD 1.5 img2img (base)
  - ControlNet depth conditioning (geometry preservation)
  - MiDaS monocular depth as fallback when ScaffoldGS depth is unreliable

Pipeline per novel view:
  1. Render RGB from ScaffoldGS
  2. Estimate/extract depth map
  3. Run SDXL/SD1.5 + ControlNet(depth) → corrected teacher image
  4. Optionally filter inconsistent teachers via monocular depth check
  5. Cache to disk

Usage:
    generator = TeacherGenerator(device='cuda', use_sdxl=False)
    generator.generate_teachers(
        scene=scene,
        gaussians=gaussians,
        pipe_args=pipe_args,
        novel_cameras=novel_cameras,
        output_dir='teacher_cache/',
        strength=0.6,
        guidance_scale=7.5,
    )

Requirements:
    pip install diffusers transformers controlnet-aux timm
    # SD1.5 weights downloaded automatically from HuggingFace
    # ControlNet depth: lllyasviel/sd-controlnet-depth
"""

import os
import sys
import gc
import hashlib
import json
import shutil
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
SCAFFOLD_ROOT = THIS_DIR / "Scaffold-GS-main"
if str(SCAFFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCAFFOLD_ROOT))

# Deferred imports (only load when actually generating, to avoid import-time GPU memory)
_diffusers_loaded = False
_pipeline = None
_depth_estimator = None
_midas = None
_midas_transform = None


def _teacher_cache_signature(novel_cameras, settings: dict) -> str:
    camera_payload = []
    for camera in novel_cameras:
        camera_payload.append({
            "uid": int(camera.uid),
            "name": camera.image_name,
            "R": np.asarray(camera.R).round(7).tolist(),
            "T": np.asarray(camera.T).round(7).tolist(),
            "FoVx": float(camera.FoVx),
            "FoVy": float(camera.FoVy),
            "width": int(camera.image_width),
            "height": int(camera.image_height),
        })
    payload = json.dumps(
        {"cameras": camera_payload, "settings": settings},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def teacher_cache_is_valid(output_dir, novel_cameras, settings: dict) -> bool:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "cache_manifest.json"
    completion_path = output_dir / "generation_complete.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        completion = json.loads(completion_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = _teacher_cache_signature(novel_cameras, settings)
    kept = int(completion.get("kept", 0))
    teacher_count = len(list((output_dir / "teacher_images").glob("*.png")))
    metadata_count = len(list((output_dir / "metadata").glob("*.json")))
    return (
        manifest.get("signature") == expected
        and completion.get("signature") == expected
        and kept > 0
        and teacher_count >= kept
        and metadata_count >= kept
    )


def teacher_generation_settings(
    strength,
    guidance_scale,
    num_inference_steps,
    prompt,
    negative_prompt,
    controlnet_conditioning_scale,
    depth_consistency_filter,
    depth_consistency_threshold,
    seed,
    cache_tag="",
):
    return {
        "model": "runwayml/stable-diffusion-v1-5",
        "controlnet": "lllyasviel/sd-controlnet-depth",
        "strength": float(strength),
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": int(num_inference_steps),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "depth_consistency_filter": bool(depth_consistency_filter),
        "depth_consistency_threshold": float(depth_consistency_threshold),
        "seed": int(seed),
        "cache_tag": str(cache_tag),
    }


def _load_pipeline(device: str, use_sdxl: bool = False):
    """Lazy-load the diffusion pipeline. Called once."""
    global _diffusers_loaded, _pipeline, _depth_estimator

    if _diffusers_loaded:
        return
    if use_sdxl:
        raise NotImplementedError("use_sdxl=True is not implemented; use the SD1.5 ControlNet path.")

    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
    from transformers import pipeline as hf_pipeline

    print("[teacher_generator] Loading ControlNet depth model...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    )

    print("[teacher_generator] Loading SD 1.5 pipeline...")
    _pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,          # disable for speed
    )
    _pipeline.scheduler = UniPCMultistepScheduler.from_config(_pipeline.scheduler.config)
    if device == "cuda":
        try:
            _pipeline.enable_model_cpu_offload()
            print("[teacher_generator] Model CPU offload enabled.")
        except Exception:
            _pipeline = _pipeline.to(device)
            print("[teacher_generator] CPU offload unavailable; pipeline moved to CUDA.")
    else:
        _pipeline = _pipeline.to(device)
    _pipeline.enable_attention_slicing()  # reduce VRAM

    # Try to enable xformers if available
    try:
        _pipeline.enable_xformers_memory_efficient_attention()
        print("[teacher_generator] xformers enabled.")
    except Exception:
        pass

    # Depth estimator for fallback (DPT-Large / MiDaS via transformers)
    print("[teacher_generator] Loading depth estimator (MiDaS)...")
    _depth_estimator = hf_pipeline(
        "depth-estimation",
        model="Intel/dpt-large",
        # Keep DPT on CPU so it does not compete with Scaffold-GS and
        # ControlNet for VRAM.
        device=-1,
    )

    _diffusers_loaded = True
    print("[teacher_generator] Pipeline ready.")


def unload_teacher_models():
    """Release diffusion and depth models before Scaffold-GS optimisation."""
    global _diffusers_loaded, _pipeline, _depth_estimator
    _pipeline = None
    _depth_estimator = None
    _diffusers_loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Depth extraction helpers
# ---------------------------------------------------------------------------

def _render_and_get_depth(
    viewpoint_cam,
    gaussians,
    pipe_args,
    background: torch.Tensor,
) -> tuple:
    """
    Render a view and extract RGB + depth.
    Returns:
        rgb_tensor:   (3, H, W) float32 cuda tensor in [0, 1]
        depth_image:  PIL Image (H, W) grayscale, normalised, for ControlNet
    """
    from gaussian_renderer import prefilter_voxel, render

    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe_args, background)

    with torch.no_grad():
        render_pkg = render(
            viewpoint_cam, gaussians, pipe_args, background,
            visible_mask=voxel_visible_mask, retain_grad=False
        )

    rgb = render_pkg["render"].clamp(0.0, 1.0)  # (3, H, W)

    # ScaffoldGS doesn't output depth natively.
    # Use MiDaS on the rendered RGB as a proxy depth map.
    rgb_pil = _tensor_to_pil(rgb)
    depth_pil = _estimate_depth_midas(rgb_pil)

    return rgb, depth_pil


def _estimate_depth_midas(rgb_pil: Image.Image) -> Image.Image:
    """Run MiDaS depth estimation on a PIL image. Returns grayscale depth PIL."""
    global _depth_estimator
    result = _depth_estimator(rgb_pil)
    depth_np = np.array(result["depth"])  # (H, W) uint8 or float

    # Normalise to [0, 255] uint8
    depth_np = depth_np.astype(np.float32)
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max > depth_min:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_np = np.zeros_like(depth_np)

    # ControlNet depth expects an RGB image (3-channel grayscale)
    depth_uint8 = depth_np.astype(np.uint8)
    depth_rgb = Image.fromarray(depth_uint8).convert("RGB")
    return depth_rgb


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) float32 tensor [0,1] to PIL RGB image."""
    arr = (t.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image, device: str = 'cuda') -> torch.Tensor:
    """Convert PIL RGB image to (3, H, W) float32 tensor [0,1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).to(device)


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _depth_consistency_check(
    rendered_depth_pil: Image.Image,
    teacher_pil: Image.Image,
    threshold: float = 0.4,
) -> bool:
    """
    Check if the teacher image is geometrically consistent with the render.
    We re-estimate depth on the teacher image and compare to rendered depth.

    Returns True if consistent (teacher should be kept), False if discard.
    """
    global _depth_estimator
    teacher_depth_pil = _estimate_depth_midas(teacher_pil)

    render_d = np.array(rendered_depth_pil.convert("L")).astype(np.float32) / 255.0
    teacher_d = np.array(teacher_depth_pil.convert("L")).astype(np.float32) / 255.0

    # Resize to same shape if needed
    if render_d.shape != teacher_d.shape:
        h, w = render_d.shape
        teacher_d = np.array(
            Image.fromarray((teacher_d * 255).astype(np.uint8)).resize((w, h))
        ).astype(np.float32) / 255.0

    # Mean absolute error in depth
    mae = np.abs(render_d - teacher_d).mean()
    return mae < threshold


# ---------------------------------------------------------------------------
# Main teacher generation function
# ---------------------------------------------------------------------------

class TeacherGenerator:
    """
    Wraps the SD1.5 + ControlNet pipeline for batch teacher image generation.

    Args:
        device:     'cuda' or 'cpu'
        use_sdxl:   Use SDXL instead of SD1.5 (requires ~16GB VRAM)
    """

    def __init__(self, device: str = 'cuda', use_sdxl: bool = False):
        self.device = device
        self.use_sdxl = use_sdxl

    def _ensure_loaded(self):
        _load_pipeline(self.device, self.use_sdxl)

    def unload(self):
        unload_teacher_models()

    def generate_teachers(
        self,
        gaussians,
        pipe_args,
        novel_cameras: list,
        output_dir: str,
        background: Optional[torch.Tensor] = None,
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        prompt: str = "a high quality photograph of an outdoor scene, sharp details, no artifacts",
        negative_prompt: str = "blurry, floaters, artifacts, low quality, distorted geometry",
        controlnet_conditioning_scale: float = 0.8,
        depth_consistency_filter: bool = True,
        depth_consistency_threshold: float = 0.45,
        seed: int = 42,
        cache_tag: str = "",
    ) -> List[str]:
        """
        For each novel camera:
          1. Render RGB + estimate depth from current ScaffoldGS model
          2. Run SD1.5 + ControlNet(depth) → teacher image
          3. Optional depth consistency filter
          4. Save to output_dir

        Args:
            gaussians:           ScaffoldGS GaussianModel
            pipe_args:           PipelineParams from arguments/
            novel_cameras:       List of Camera objects from novel_view_sampler
            output_dir:          Directory to cache teacher images + renders
            background:          Background tensor (cuda). If None, uses black.
            strength:            img2img strength [0,1]. Lower = stays closer to render.
            guidance_scale:      CFG scale. 7.5 is standard.
            num_inference_steps: Denoising steps (20 is enough for img2img).
            prompt:              Text prompt (generic outdoor scene description).
            negative_prompt:     Negative prompt to suppress artifacts.
            controlnet_conditioning_scale: ControlNet influence weight.
            depth_consistency_filter: Whether to filter teachers by depth check.
            depth_consistency_threshold: MAE threshold for depth filter.
            seed:                RNG seed for reproducibility.

        Returns:
            List of paths to saved teacher images.
        """
        settings = teacher_generation_settings(
            strength,
            guidance_scale,
            num_inference_steps,
            prompt,
            negative_prompt,
            controlnet_conditioning_scale,
            depth_consistency_filter,
            depth_consistency_threshold,
            seed,
            cache_tag,
        )
        signature = _teacher_cache_signature(novel_cameras, settings)

        output_dir = Path(output_dir)
        manifest_path = output_dir / "cache_manifest.json"
        old_signature = None
        if manifest_path.is_file():
            try:
                old_signature = json.loads(manifest_path.read_text()).get("signature")
            except (OSError, json.JSONDecodeError):
                pass
        if old_signature != signature:
            for dirname in ("rendered_rgb", "rendered_depth", "teacher_images", "metadata"):
                shutil.rmtree(output_dir / dirname, ignore_errors=True)
            completion_path = output_dir / "generation_complete.json"
            if completion_path.exists():
                completion_path.unlink()

        rgb_dir = output_dir / "rendered_rgb"
        depth_dir = output_dir / "rendered_depth"
        teacher_dir = output_dir / "teacher_images"
        meta_dir = output_dir / "metadata"

        for d in [rgb_dir, depth_dir, teacher_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)
        (output_dir / "cache_manifest.json").write_text(json.dumps({
            "signature": signature,
            "settings": settings,
            "camera_count": len(novel_cameras),
        }, indent=2))
        self._ensure_loaded()

        if background is None:
            background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        teacher_paths = []
        skipped = 0

        gaussians.eval()

        print(f"[teacher_generator] Generating {len(novel_cameras)} teacher images...")

        for idx, cam in enumerate(tqdm(novel_cameras, desc="Teacher generation")):
            save_name = f"{idx:04d}_{cam.image_name}"
            teacher_path = teacher_dir / f"{save_name}.png"
            meta_path = meta_dir / f"{save_name}.json"

            # Skip if already cached
            if teacher_path.exists() and meta_path.exists():
                teacher_paths.append(str(teacher_path))
                continue

            # --- Step 1: Render RGB + depth from current model ---
            try:
                rgb_tensor, depth_pil = _render_and_get_depth(
                    cam, gaussians, pipe_args, background
                )
            except Exception as e:
                print(f"[teacher_generator] Render failed for cam {idx}: {e}")
                continue

            rgb_pil = _tensor_to_pil(rgb_tensor)

            # Save intermediate renders
            rgb_pil.save(rgb_dir / f"{save_name}.png")
            depth_pil.save(depth_dir / f"{save_name}.png")

            # --- Step 2: Run SD1.5 + ControlNet ---
            try:
                result = _pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=rgb_pil,
                    control_image=depth_pil,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator,
                )
                teacher_pil = result.images[0]
            except Exception as e:
                print(f"[teacher_generator] Diffusion failed for cam {idx}: {e}")
                continue

            # --- Step 3: Optional depth consistency filter ---
            if depth_consistency_filter:
                is_consistent = _depth_consistency_check(
                    depth_pil, teacher_pil, threshold=depth_consistency_threshold
                )
                if not is_consistent:
                    skipped += 1
                    print(f"[teacher_generator] Cam {idx} failed depth consistency — skipping.")
                    continue

            # --- Step 4: Save ---
            teacher_pil.save(teacher_path)
            teacher_paths.append(str(teacher_path))

            # Save metadata (camera uid, angle) for training lookup
            meta = {
                "uid": cam.uid,
                "image_name": cam.image_name,
                "teacher_path": str(teacher_path),
                "rgb_path": str(rgb_dir / f"{save_name}.png"),
                "depth_path": str(depth_dir / f"{save_name}.png"),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        gaussians.train()
        (output_dir / "generation_complete.json").write_text(json.dumps({
            "signature": signature,
            "requested": len(novel_cameras),
            "kept": len(teacher_paths),
            "skipped": skipped,
        }, indent=2))

        print(f"[teacher_generator] Done. Kept {len(teacher_paths)}, "
              f"skipped {skipped} (depth inconsistent).")

        return teacher_paths


# ---------------------------------------------------------------------------
# Teacher dataset helper for training loop
# ---------------------------------------------------------------------------

class TeacherDataset:
    """
    Lightweight dataset wrapping cached teacher images + their cameras.
    Used in train_distill.py to iterate over (camera, teacher_image) pairs.

    Args:
        novel_cameras:  List of Camera objects from novel_view_sampler
        teacher_dir:    Path to directory containing teacher .png files
        device:         'cuda' or 'cpu'
    """

    def __init__(self, novel_cameras: list, teacher_dir: str, device: str = 'cuda'):
        import json

        self.device = device
        self.pairs = []  # list of (Camera, teacher_path); images load lazily

        teacher_dir = Path(teacher_dir)
        meta_dir = teacher_dir / "metadata"

        if not meta_dir.exists():
            raise FileNotFoundError(f"Teacher metadata not found at {meta_dir}. "
                                    f"Run TeacherGenerator.generate_teachers() first.")

        # Build uid → camera map
        uid_to_cam = {cam.uid: cam for cam in novel_cameras}

        for meta_file in sorted(meta_dir.glob("*.json")):
            with open(meta_file) as f:
                meta = json.load(f)

            uid = meta["uid"]
            teacher_path = meta["teacher_path"]

            if uid not in uid_to_cam:
                continue
            if not Path(teacher_path).exists():
                continue

            cam = uid_to_cam[uid]
            self.pairs.append((cam, str(teacher_path)))

        print(f"[TeacherDataset] Loaded {len(self.pairs)} teacher pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cam, teacher_path = self.pairs[idx]
        with Image.open(teacher_path) as image:
            teacher_img = image.convert("RGB")
            H, W = cam.image_height, cam.image_width
            teacher_img = teacher_img.resize((W, H), Image.LANCZOS)
            teacher_tensor = _pil_to_tensor(teacher_img, device=self.device)
        return cam, teacher_tensor

    def sample(self):
        """Return a random (camera, teacher_tensor) pair."""
        idx = torch.randint(0, len(self.pairs), (1,)).item()
        return self[idx]
