# Literature Grounding — TacK-JEPA

Living document. Seeded from [PRD.md](../PRD.md) §3 (as of July 2026); update as new
relevant work appears or as we actually read/verify papers during the build.

| Work | What it does | Why this project is different |
|---|---|---|
| SPARSH (Meta, ~2024-25) | Family of ViT touch representations self-supervised via MAE, DINO, *and* I-JEPA on optical tactile images across sensor types | Already includes a JEPA variant — the gap isn't "JEPA for touch," it's that SPARSH is still image-patch based (GelSight/DIGIT), with no explicit geometry and no action-conditioned world-model/prediction objective (masked-patch prediction within a single static image, not future-state prediction) |
| AnyTouch / AnyTouch 2 | Unified static+dynamic optical tactile representation, masked modeling + cross-sensor matching | Cross-sensor generalization goal (not ours); still image-patch based |
| T3 / UniT / UniTac-NV | Cross-sensor / cross-embodiment transferable tactile representations | Same image-patch limitation; morphology-aware tokens, not kinematic grounding |
| FTP-1 | Generalist tactile policy across sensors via Morphology-Aware Tactile Token Space | Closest prior art for heterogeneous sensor geometry, but operates on sensor *images*; targets policy-generalist breadth, not force-native single-embodiment depth |
| TacForeSight | Force-guided tactile world model, JEPA-adjacent latent dynamics, conditioned on real wrist force/torque | Requires real instrumented dual-finger hardware; not kinematically grounded; 1–2 sensing points, not a distributed multi-taxel field |
| Dream-Tac / DreamTacVLA (2026) | Action-conditioned world models predicting future tactile (and visual) latents, DreamTacVLA built on a frozen V-JEPA2 backbone | Same category as TacForeSight: real 1–2 point force/torque or optical-patch sensors, fused with vision in DreamTacVLA's case; not kinematically grounded, not a distributed taxel field |
| Visuo-Tactile World Models (VT-WM) | Multi-task world model combining vision+touch | Vision+image-tactile fusion, not touch-only, not kinematically grounded |
| Tactile-WAM | Touch-aware world-action model; identifies "tactile pollution" (naive tactile fusion degrades video/action prediction) | Cautionary finding we design around (PRD §5.10); still image-tactile, fused with video |
| ART-Glove (CMU) | Hardware: 2048-taxel + 22-DoF articulated glove for human demonstration capture | Data-capture device, not a model; TacK-JEPA is the world model built for this hardware class |
| OSMO | Open-source tactile glove for human-to-robot skill transfer | Same category as ART-Glove; potential future real-data source (PRD §7.5) |
| VibeAct, Sound of Touch | Vibration/acoustic tactile sensing | Different modality; possible future extension, not v1 |
| DIFFTACTILE, TacEx, FOTS | Optical/differentiable tactile *simulators* | Not used in v1 — this project avoids optical simulation entirely by being force-native |

**The gap this project fills:** kinematically-grounded (exact FK geometry) + force-native
(no optical rendering) + JEPA-style (latent, action-conditioned, predictive) world model
for a distributed multi-taxel articulated hand. This combination does not appear in the
literature as of the PRD's writing (July 2026).

## Reading / verification log

*(Add entries here as papers are actually read or claims re-verified during the build.)*
