# ViewCrafter Trajectory Distillation into Scaffold-GS

## Research question

This project tests whether a pose-controlled video diffusion prior can improve
five-view Scaffold-GS when its trajectory-consistent novel-view sequences are
distilled into the trainable Gaussian representation. Improvement over
five-view 3DGS or Scaffold-GS is an experimental hypothesis, not a guaranteed
outcome; it must be established on held-out views.

The fixed teacher is `ViewCrafter_25_512` at 320x512 and 25 generated frames.
It runs in a separate Python 3.9 / PyTorch 1.13 environment. No gradient passes
through ViewCrafter: PNG trajectory frames and camera metadata form the process
boundary with Scaffold-GS.

## Training regime

1. Train Scaffold-GS on exactly five real Bicycle views.
2. Fit an ellipse to those camera positions and order the views around it.
3. Retain only neighboring pairs whose angular gap, viewing direction,
   normalized baseline, and focus-radius difference are compatible.
4. Give the ordered real images to ViewCrafter sparse interpolation, while
   using only compatible pair clips as supervision.
5. Generate all 25 frames, consider interior frames 4–21, then retain a
   quality-filtered subset of typically 8–12 frames per compatible clip.
6. Associate each selected frame with the corresponding interpolated
   Scaffold-GS camera.
7. Fine-tune all Scaffold-GS parameters using real-view and trajectory losses.

ViewCrafter uses its DUSt3R point-based coarse representation internally. The
integration does not pretend that ViewCrafter consumes Scaffold-GS camera
matrices directly; instead, both systems use matching interpolation positions
between the same ordered sparse endpoint images.

## Proposed configuration

The proposed run trains all Scaffold-GS parameters: shared opacity,
covariance, and color MLPs; anchor positions and offsets; per-anchor features;
base scaling, rotation, and opacity. Anchor positions and offsets receive a
soft penalty against their Stage-1 values, allowing geometry to adapt while
discouraging large jumps caused by generative teacher noise.

The default distillation loss is:

```text
L = L_real + lambda_teacher * (L1_teacher + lambda_lpips * LPIPS_teacher)
    + lambda_trajectory * L1((render_b - render_a),
                             (teacher_b - teacher_a))
    + lambda_anchor_reg * (MSE(anchor, anchor_initial)
                           + 0.1 * MSE(offset, offset_initial))
```

The trajectory term matches the change between neighboring camera views. It
does not describe scene dynamics and does not force neighboring views to
become identical.

Anchor densification is disabled during the stable distillation phase because
pruning or growing anchors changes their identities and invalidates pointwise
Stage-1 geometry regularisation. An optional second phase can enable
conservative densification with reduced teacher/trajectory weights and zero
pointwise anchor regularisation:

```bash
--enable_densification_phase
```

Current experiment: five-view appearance training and ViewCrafter distillation
using the standard MiP-NeRF 360 COLMAP camera/point initialization.

Strict variant: rerun COLMAP using only the selected five images and use that
reconstruction consistently for every baseline. The current protocol must not
be described as five-image reconstruction from scratch.

## Required ablations

Use the same sparse split, novel cameras, cached teachers, and random seeds:

1. `shared_mlp`: shared decoders only.
2. `shared_mlp_features`: shared decoders plus per-anchor features.
3. `shared_mlp_geometry --lambda_anchor_reg 0`: geometry without stiffness.
4. `shared_mlp_geometry --lambda_anchor_reg 0.01`: restricted representation.
5. `all --lambda_anchor_reg 0`: unconstrained all-parameter fine-tuning.
6. `all --lambda_anchor_reg 0.01`: proposed configuration.
7. Real-view-only controls with
   `--lambda_teacher 0 --lambda_trajectory 0`.

Comparing items 5 and 6 isolates anchor regularisation while all parameters
remain trainable. Comparing items 1 through 4 studies where the pseudo-view
signal can be absorbed.

The contribution is a study of neural parameter sharing under noisy pseudo-view
supervision, with trajectory-level ViewCrafter consistency rather than
independent image inpainting.

## Weakly aligned exploratory branch

When DUSt3R and COLMAP endpoint cameras do not align reliably, use
`--weak_prior_exploratory`. This branch treats ViewCrafter as perceptual
guidance rather than pixel-accurate geometric truth. It:

- keeps at most 12 high-quality frames selected as adjacent pairs;
- runs a matched real-only `shared_mlp` continuation;
- runs a weak-prior `shared_mlp` continuation from the same Stage-1 state;
- uses `lambda_teacher=0.01`, teacher L1 weight `0.0`, LPIPS weight `1.0`,
  teacher supervision scale `0.25`, and trajectory weight `0.0`;
- leaves anchor regularisation at zero because anchors are frozen in
  `shared_mlp`.

```bash
python run_pipeline.py \
  --source_path /path/to/bicycle \
  --output_dir /path/to/output/bicycle_weak_prior \
  --viewcrafter_root /path/to/ViewCrafter \
  --viewcrafter_python /path/to/viewcrafter/bin/python \
  --weak_prior_exploratory \
  --gpu 0
```

The resulting comparison tests whether weak diffusion guidance helps beyond
the same amount of real-only continued optimization. It must be described as
weakly aligned diffusion-prior distillation, not exact novel-view supervision.

## ViewCrafter setup

Clone the official repository and create its upstream-pinned environment:

```bash
git clone https://github.com/Drexubery/ViewCrafter.git Gauss_Code/ViewCrafter
conda create -n viewcrafter python=3.9.16
conda activate viewcrafter
pip install -r Gauss_Code/ViewCrafter/requirements.txt
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d/0.7.5-py39_cu117_pyt1131.tar.bz2
```

Place:

- `ViewCrafter_25_512/model.ckpt` at
  `Gauss_Code/ViewCrafter/checkpoints/model.ckpt`.
- `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` in the same checkpoints folder.

The bridge uses `configs/inference_pvd_512.yaml`, `height=320`, `width=512`,
`video_length=25`, and `perframe_ae=True`.

Each exported teacher now carries the exact focal length and principal point
from ViewCrafter's PyTorch3D trajectory, scaled through the final output
resize. Scaffold-GS renders the teacher supervision with that calibrated
projection instead of stretching the teacher to an inherited COLMAP FOV.

Run the complete pipeline from the Scaffold-GS environment:

```bash
python Gauss_Code/run_pipeline.py \
  --source_path /path/to/mipnerf360/bicycle \
  --output_dir /path/to/output/bicycle_viewcrafter \
  --viewcrafter_root Gauss_Code/ViewCrafter \
  --viewcrafter_python ~/miniconda3/envs/viewcrafter/bin/python \
  --enable_densification_phase \
  --gpu 0
```

For the required 200-iteration plumbing and parameter-sharing comparison:

```bash
python Gauss_Code/run_pipeline.py \
  --source_path /path/to/mipnerf360/bicycle \
  --output_dir /path/to/output/bicycle_sanity_200 \
  --viewcrafter_root Gauss_Code/ViewCrafter \
  --viewcrafter_python ~/miniconda3/envs/viewcrafter/bin/python \
  --sanity_200 \
  --gpu 0
```

This produces `comparison_summary_512.json` containing Stage-1 and distillation
PSNR, SSIM, and LPIPS for `shared_mlp` and `all`.

The 200-iteration experiment is a plumbing and early-trend test only. It is
not evidence of converged reconstruction quality. Use it to catch divergence,
miscalibration, or a consistently harmful teacher signal before committing to
full-length runs.

To evaluate ViewCrafter's sparse-view-specific 576x1024 checkpoint, place it
at `checkpoints/model_sparse.ckpt` and rerun into a separate output directory:

```bash
python Gauss_Code/run_pipeline.py \
  --source_path /path/to/mipnerf360/bicycle \
  --output_dir /path/to/output/bicycle_sanity_200 \
  --viewcrafter_root Gauss_Code/ViewCrafter \
  --viewcrafter_python ~/miniconda3/envs/viewcrafter/bin/python \
  --viewcrafter_profile sparse \
  --skip_stage1 \
  --sanity_200 \
  --gpu 0
```

The second command reuses the exact Stage-1 checkpoint while keeping the
teacher caches, distillation outputs, and `comparison_summary_sparse.json`
separate by ViewCrafter profile.

The cache validator prints the maximum principal-point deviation in pixels and
as a percentage of image size. Teacher images must exactly match the calibrated
render resolution; distillation refuses to resize them.

The sparse profile is likely too large for some 20 GiB GPUs. Generated clips
are moved to CPU immediately so multiple clips do not accumulate in GPU
memory, but this cannot reduce the peak memory of the model plus one active
25-frame clip. Do not reduce `video_length` without first confirming that the
chosen checkpoint and configuration support a different temporal length.

PyTorch3D coarse rendering is chunked independently of diffusion using
`--viewcrafter_render_chunk_size 4`. If a chunk still OOMs, the bridge retries
with progressively smaller chunks down to fully sequential one-camera
rendering. This preserves the complete 25-frame diffusion sequence and camera
ordering.
