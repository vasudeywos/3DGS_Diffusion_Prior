# Neural Gaussian Distillation

## Research question

This project tests whether noisy diffusion-generated pseudo-view supervision is
absorbed more robustly by Scaffold-GS's shared neural attribute decoders than
by unconstrained per-anchor parameters.

The diffusion model is an offline teacher. It does not backpropagate into
Scaffold-GS. Corrected novel-view images instead provide L1 and LPIPS
supervision through the differentiable Scaffold-GS renderer.

## Proposed configuration

The proposed default trains all Scaffold-GS parameters: shared opacity,
covariance, and color MLPs; anchor positions and offsets; per-anchor features;
base scaling, rotation, and opacity. Anchor positions and offsets receive a
soft penalty against their Stage-1 values, allowing geometry to adapt while
discouraging large jumps caused by noisy pseudo-view supervision.

The default distillation loss is:

```text
L = L_real + lambda_teacher * (L1_teacher + lambda_lpips * LPIPS_teacher)
    + lambda_anchor_reg * (MSE(anchor, anchor_initial)
                           + 0.1 * MSE(offset, offset_initial))
```

Anchor densification is disabled because pruning or growing anchors changes
their identities and invalidates a pointwise Stage-1 geometry regulariser.

The supplied MiP-NeRF 360 COLMAP model provides camera poses and the initial
point cloud. This is a five-image photometric-training protocol, not a
five-image structure-from-motion protocol. For a strict image-only claim,
reconstruct `sparse/0` with COLMAP using only the five manifest images and use
that reconstruction consistently for every baseline.

## Required ablations

Use the same sparse split, novel cameras, cached teachers, and random seeds:

1. `shared_mlp`: shared decoders only.
2. `shared_mlp_features`: shared decoders plus per-anchor features.
3. `shared_mlp_geometry --lambda_anchor_reg 0`: geometry without stiffness.
4. `shared_mlp_geometry --lambda_anchor_reg 0.01`: restricted representation.
5. `all --lambda_anchor_reg 0`: unconstrained all-parameter fine-tuning.
6. `all --lambda_anchor_reg 0.01`: proposed configuration.
7. Real-view-only controls with `--lambda_teacher 0`.

Comparing items 5 and 6 isolates anchor regularisation while all parameters
remain trainable. Comparing items 1 through 4 studies where the pseudo-view
signal can be absorbed.

The contribution is a study of neural parameter sharing under noisy pseudo-view
supervision, not a fundamentally different diffusion integration paradigm.
