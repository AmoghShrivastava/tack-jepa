# URDF assets

## allegro_hand/

Allegro Hand (right, 16 DoF) vendored from **dexsuite/dex-urdf**
(https://github.com/dexsuite/dex-urdf), commit `f5e7132f22108164577fea4c25ef99b5cc0e1900`,
fetched 2026-07-04. Chosen over the raw `allegro_hand_ros` description because dex-urdf's
models are simulation-optimized (primitive collision geometry + clean per-link visual
meshes, curated inertials).

Licenses (both permit redistribution with attribution; copies committed here):

- `allegro_hand/LICENSE` — BSD (SimLab, 2016): the original Allegro Hand description
  dex-urdf derives from.
- `DEX_URDF_LICENSE` — MIT: the dex-urdf repository itself.

Contents kept: `allegro_hand_right.urdf`, collision meshes, visual `.obj`/`.mtl` meshes
(used in Phase 2 for taxel FPS placement on link surfaces). Omitted: left-hand variant,
`.glb` duplicates, `variation/` URDFs.
