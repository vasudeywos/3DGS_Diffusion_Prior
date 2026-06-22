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
