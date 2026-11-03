# Kinetic 3D benchmark: draft design choices (2026-11-03, for the user's approval)

Basis: the external generator `hlillemark/Miniworld`, `miniworld/envs/moving_blockworld.py` (read 2026-11-03, not yet
cloned here); the released Blockworld validation files on the NAS; the 2D lessons in PLAN_v4_KWM.md and REVIEW_pr2_opus.md.

Unit: one cell = one Blockworld grid unit (room is 15 × 15; the agent's forward step is one cell in grid mode).

## Facts that drive the choices

- The generator already has a continuous mode (`grid_mode=False`): direction uniform in (−π, π), speed
  `forward_step × box_speed_scale` (Miniworld default forward_step 0.15). The released data uses `grid_mode=True`, which
  snaps the agent **and** the blocks to integer cells and draws integer velocities.
- Wall collisions test only the end point of each step (`intersect_circle_segs`, distance from the new centre to the wall
  segments < radius). Block radius = size·√2/2 ≈ 0.42–0.60. Two consequences off the integer grid:
  1. a block faster than about 2·radius (0.85–1.2 cells/step) can pass through a wall;
  2. on a hit the velocity component is reversed and the block steps back from its old position, so it turns before
     reaching the wall. Its position differs from a true reflection by 2(x − r), anywhere in [0, 2|v|), depending on its
     sub-cell position.
  Neither happens with integer positions and speed 1, which is why the released data is clean.
- Released per-clip files: rgb.mp4 0.51 MB, map_2d.mp4 0.57 MB, actions.pt 20 kB (agent state only, no block state),
  depth.pt 32.8 MB (97% of the 32 GB validation snapshot). Episodes are 500 frames. Released dynamic train = 40 shards vs
  16 for the 1000-clip validation; from shard metadata sizes, about 9–10k training clips (estimate, to confirm).
- 2D lesson: order-2 detection failed in 5-digit scenes because per-step node velocity noise (σ ≈ 0.14 px/frame) swamped
  accelerations of 2e-3–2e-2 px/frame². The estimator itself works on a single clean digit (a = 4e-3 detected 94–100%).

## The five choices

| # | Decision | Recommendation |
|---|---|---|
| 1 | Block velocity law | Continuous. Direction uniform in (−π, π); speed uniform in [0.25, 1.0] cells/step per block, constant except at bounces. Report by speed bin (0.25–0.5, 0.5–0.75, 0.75–1.0). |
| 2 | Acceleration | Yes, but as a separate secondary split, not the primary. Per-block constant planar acceleration, direction uniform, \|a\| log-uniform in [2e-3, 1e-2] cells/step². CPU estimator gate on this split before any GPU training. |
| 3 | Bounce rule | Exact mirror reflection against the room's inner bounds [r, 15 − r], computed analytically per axis (reflect the overshoot). No block–block or block–agent collisions, as released. |
| 4 | Camera | Keep the released discrete actions (turn ±90°, forward one cell, no-op) and the released action law. The agent and its heading stay grid-snapped; only the blocks go continuous. |
| 5 | Size and length | As released: 500-frame episodes, 6–10 blocks, same room. Train ≈ released count; 1000 validation (selection); 1000 held-out test on a fresh seed. Evaluate at 70 + 70 and 70 + 210. No depth. Store per-block ground truth every frame. |

### Why

1. **Velocity.** Uniform direction is what the generator's continuous mode already draws, and it is invariant under 90°
   turns, so the C4 structure stays exact. The 1.0 cap equals the released speed, so magnitudes stay inside what
   FloWM-3D's five integer channels (v_range 1) span. What FloWM-3D cannot represent is fractional and diagonal
   velocity, which is exactly what the benchmark tests. Fixed speed classes would let a FloWM variant add channels for
   them and close the gap without continuous dynamics. The 0.25 floor keeps blocks visibly moving, since the released
   dynamic law also has no static blocks.
2. **Acceleration.** The range is chosen to be visible within the context window: Δv over 70 context frames is
   0.14–0.7 cells/step, and the path bends 0.4–2 cells over 20 frames. With mirror bounces each block's energy is
   conserved (a ball on a tilted table), so speed stays bounded: v² ≤ v₀² + 2|a|·L ≈ 1.4 at worst, i.e. below
   ~1.2 cells/step. Rejection-sample anything above that. It stays out of the primary because the 2D failure was
   measurement noise, and 3D adds occlusion and out-of-view gaps. Per-block acceleration matches the 2D K-accel law,
   so the hypothesis is the same. If the gate fails, the fallback is one shared field per episode (all blocks
   accelerate together), which lets evidence pool across blocks.
3. **Bounce.** Fixes both problems above. Trajectories become an exact function of continuous state, which is what a
   Galilean model should be able to learn.
4. **Camera.** FloWM-3D stays a valid baseline: it trains on the new data unchanged. 90° turns keep the pr6 C4 work
   exact. Integer ego-motion means any gain comes from object dynamics, not camera handling. Continuous yaw breaks
   C4 and FloWM's map rotation, so it belongs in a later SO(2) stage.
5. **Size.** Matching the released protocol keeps the anchor comparable. Dropping depth cuts storage ~30×
   (~1.1 MB/clip, ~13 GB for 12k clips per split). The NAS has 16 TB free. Per-block ground truth (position, velocity,
   acceleration, bounce events) is needed for oracle ceilings, realised-law statistics and speed-bin breakdowns, which
   plan v4 already requires for K-accel.

## Splits

- **K3D-int**: the released integer law through our pipeline. Check that the released FloWM-3D checkpoint scores ≈ the
  anchor (0.000603 MSE @70) on it. This separates pipeline effects from law effects, like the bit-exact MNIST World
  generator check in 2D.
- **K3D-v**: continuous velocity. Primary.
- **K3D-a**: continuous velocity + per-block acceleration. Secondary, gated.

## Generator changes needed

1. A hybrid mode: grid-snap the agent and heading, not the blocks (today `grid_mode` snaps both).
2. Continuous velocity law with the speed range above (today: speed from `forward_step`).
3. Analytic mirror bounce against room bounds, replacing the end-point segment test for blocks.
4. Optional per-block acceleration with rejection sampling on max speed.
5. Per-frame block state written to the per-clip `.pt`; depth off.
6. Headless rendering (xvfb-run is installed; Miniworld needs OpenGL). CPU only, so it does not compete with GPU 1.

## Not on the original list: which model runs on 3D

KWM is a 2D image-plane model; Blockworld is first-person 128 × 128. A kinetic model needs a lift from the egocentric
view into a top-down map. Lean: put KWM's continuous transport and node estimator inside FloWM-3D's top-down latent map
(replacing its integer channel rolls). FloWM-3D already solves the lift, and its turn handling is the C4 structure.
This is the largest piece of work and sets the timeline, so it deserves its own design step before data is generated at
scale. FloWM-3D must also be retrained on K3D-v at the same budget (≥3 seeds); the released checkpoint is evaluated
zero-shot as a reference only.

## Corrections after reading the generator source and the released data (2026-11-03)

Checked against `hlillemark/Miniworld` commit b28988d (`scripts/generate_videos.py`, `moving_blockworld.py`),
the released validation clips, and the paper's Appendix E.1. The five decisions stand; these facts change details.

1. **Room geometry and the bounce bounds.** The released command uses `--room-size 16`: walls at 0 and 16, and
   the agent and the blocks occupy the integer cells 1..15 (the paper's "15 × 15" is the number of occupiable
   coordinates). The released integer rule is already an exact mirror about the cell lines 1 and 15: a block at 1
   moving −1 has its candidate 0 inside the wall, so it steps back to 2, and 2, 1, 2 is a reflection about 1.
   So the continuous law should reflect block **centres** at the bounds [1, 15], not at [r, 15 − r]. With these
   bounds the released integer law is the integer special case of our law (same code path, bit-identical
   trajectories, which becomes a unit test), and K3D-int is a true control. Blocks are a fixed 0.7 × 1.5 × 0.7
   (`--block-size-xy 0.7 --block-height 1.5`), radius 0.495, not the random 0.6–0.85 of the fork's default; the
   end-point tunnelling threshold is therefore 2r = 0.99 cells/step. Choice 3 is amended accordingly.
2. **Released velocities are cardinal unit vectors only** (`--grid-cardinal-only`): (±1, 0) or (0, ±1), never
   diagonal and never zero (paper: "−1 to +1, no diagonals"). FloWM-3D's five channels cover exactly this set.
3. **Released counts** are 10,000 training and 1,000 validation clips per subset (Appendix E.1), not 9–10k.
4. **Rendering.** The fork's own headless EGL path fails on this machine ("No standard config is available");
   `xvfb-run` with Mesa llvmpipe works, CPU only (no GPU touched), about 8 s per 60 frames including start-up,
   so roughly one minute per 500-frame clip and about 35 minutes per 1000 clips with 32 processes. Under llvmpipe
   the multisample buffers fall back from 8/16 to 4 samples; frames look identical to the released ones by eye,
   and the K3D-int anchor check quantifies any effect.
5. **Agent policy** as released: `biased_walk_v2`, forward probability 0.92, five no-op "look" steps at each
   wall, spawn heading 0 (`--heading-zero`), field of view 60°. Measured action mix in the released validation
   split: forward 58.8 %, no-op 31.5 %, left 6.3 %, right 3.4 %. Kept unchanged (choice 4).
6. **Per-frame block state** already exists in the fork as a separate `_block_info.pt` (`--store-block-info`:
   world and agent-frame positions and velocities, colours), which the FloWM loader never reads. Keep it separate:
   the loader returns every key of `_actions.pt` except four as collated metadata, so variable-size block arrays in
   `_actions.pt` would break batching. We extend `_block_info.pt` with acceleration, speed and bounce events.
7. **Seeding.** The fork draws the block count with Python's unseeded `random` in `build_env`, so `--seed` does not
   reproduce it; the k3d generator derives it from the per-item seed.
8. **Acceleration bound.** With centre bounds [1, 15] the path length is L = 14, so v² ≤ v₀² + 2·0.01·14 = 1.28,
   i.e. v ≤ 1.13 cells/step at the worst corner of the ranges; the draft's 1.4 / 1.2 were slightly high. The
   conclusion (speed stays bounded, cap at 1.2 by rejection) is unchanged.
9. **K3D-int generation.** The released command is run verbatim (grid mode) on our stack, with depth disabled and
   block info stored; a unit test checks that the new continuous code path reproduces grid-mode trajectories
   exactly from integer initial states.
