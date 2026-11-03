# Review: KINETIC3D_model_design.md (option A: KWM transport and estimator inside FloWM-3D's map)

**Verdict: NOT READY for the user.** The direction (reuse FloWM-3D's lift, change the transport) is sound, but four load-bearing claims are wrong:
1. the measurement pair;
2. turns with adaptive nodes "permute nothing";
3. the mixing conv is "identity-initialised";
4. memory and compute "do not depend on V".

Corrected, the compute makes the proposed matrix (3 arms × 3 seeds at 30k × bs8, A at V = 17) roughly an order of magnitude too large for GPU 1. Risk check 1 as written tests the wrong pair, at the wrong spatial scale. Fix the points below and reorder the sequence before showing it.

Measurements: CPU only, `CUDA_VISIBLE_DEVICES=""`, scripts in `scratchpad/review/k3d_mem.py`, `k3d_mem2.py`, `k3d_v17.py`.

## FACTUAL ERRORS

**F1. Mixing conv: wrong weight shape, and it is not identity-initialised.**
- The doc (§1, table) says it is depthwise with weight (256·V, 1, 3, 3) and re-initialises "to its identity (Dirac) start".
- The code (`flowm_models_3d.py:486-496`): `Conv2d(256V, 256V, groups=embed_dim=256)` on channels ordered `d*V+v` (`:579-580`). Measured weight `[1280, 5, 3, 3]`. Each group mixes **all V velocity channels** of one feature over 3×3. So FloWM-3D *does* have learned cross-node coupling, with weights indexed by node.
- `nn.init.dirac_(w)` is called without `groups=` (`:496`), so only 5 weights are non-zero. Measured: the mixing output is 99.6 % zeros at init, which erases the map at every step.
- Consequences:
  - "Re-initialised to identity" for a new V actually means "memory wiped".
  - Option D's "load everything except the mixing conv" starts with a memory-erasing operator.
  - The correct identity is `dirac_(w, groups=256)`.

**F2. Measurement pair (§2 A: "transported FOV content vs newly written FOV content, the same pair KWM compares in 2D").**
- In lock mode KWM compares the **previous observation snapshot** `P·conv_u(u_{t-1})` (K = 8 fixed random channels, one per sample, shared by all nodes; `cells.py:427-433`, stored as the snapshot at `cells.py:442,446`), transported by each node's `nu_i·dt − a` (`lock_nodes.py:232-234`), against the **current observation imprint** `P·conv_u(u_t)`.
- Memory features never enter the estimator.
- FloWM-3D has no observation-only top-down tensor at all. Its only top-down quantity is the gated blend `(1−g)·h_cv + g·h_cv_new` (`flowm_models_3d.py:332-334`), whose attention also reads the transported memory tokens (`:318`).

**F3. "Nothing in lock_nodes.py assumes a full rectangle except the boundary exclusion."**
Measured FOV: Ncv = 177 cells, bounding box rows 7–25 × cols 16–32 (19 × 17, not square, not at the origin; `compute_fov_mask` `:207-219`). The code assumes a **square window at the top-left corner** throughout:
- `pad_world` (`lock_nodes.py:25-26`);
- `shifted_window_mask` builds a win×win block of ones at the origin (`:127-136`);
- `_candidate_geometry` / `LocalSearch` take `win = current.shape[-1]` and build win×win masks (`:141-147, 158, 168`);
- `lock_measurement` slices `[..., :h, :w]` and passes only `shape[-1]` (`:234, 237, 243`);
- `LockState.snapshot` is `[B,K,win,win]` (`:18`).

Unmasked window means sit in four places:
- the texture threshold (`:70, :240`);
- the coherence sigma (`:72`);
- the LK damping reference (`transport.py:154-155`);
- the LK gradients, which wrap with `torch.roll` inside the window (`transport.py:145-146`).

With a cone that fills 55 % of its bounding box, each of these shifts.

**F4. Risk 3: "a turn rotates the velocity vectors (rot_vec) and the map (rot_map) and permutes nothing" is wrong.**
- Adaptive velocity is `lattice_i + rho_i` (`cells.py:274, 425`), with rho bounded ±0.75 around the lattice (`cells.py:439`). `two_modes` restricts candidates to each node's lattice box (`lock_nodes.py:95`).
- `rot_state` **permutes** the nodes (`perms[(-k)%4]`) *and* applies `rot_vec` to rho (`rot.py:326-332`). `rotation_perms` asserts that the lattice is closed under C4 (`rot.py:312-321`).
- Rotating rho without permuting gives `lattice_i + R·rho_i ≠ R(lattice_i + rho_i)`.
- PR6 rotates features with `rot_regular` (orientation-channel roll, `rot.py:304-309`). FloWM features have no such channels, so in 3D use `rot_map` only.
- `tests/test_rot.py` tests a symmetry (rotating whole episodes; PR6 doc l.380-385), not a turn action. A new test is needed.

**F5. Compute does depend on V.** The weights don't, but the attention sequence is V·177 + 64 tokens: 949 / 1657 / 3073 / 4489 for V = 5 / 9 / 17 / 25.

| V | Linear part, counted (GFLOP/step, fwd, B = 1) | + attention 4N²D × 6 blocks (hand) | Total | Relative to C |
|---|---|---|---|---|
| 5 | 9.8 | | ≈ 15 | 1× |
| 9 | 16.9 | | ≈ 34 | 2.3× |
| 17 | 31.7 | | ≈ 90 | ≈ 6× |
| 25 | 47.1 | | ≈ 171 | ≈ 11× |

Measured CPU forward times scale 1 : 1.5 : 3.3 : 6.5. "Same budget for all arms" in steps is therefore 6–11× more compute for A and B than for C.

**F6. Memory (§3 risk 2): the maps are a small part, not the dominant term.**
- Training does full BPTT over all 140 steps:
  - `max_frames 140`, `context_frames 50` (`train/dynamic.yaml:25,39`);
  - the loss covers the 90 predicted frames (`flowm_video.py:115-117`);
  - each step's decoded frame is fed back as the next observation (`flowm_models_3d.py:656-662`);
  - teacher forcing 0.125 (`dynamic.yaml:13`);
  - no checkpointing and no truncation anywhere (the only `checkpoint(` calls in `algorithms/` are in cogv).
- Measured saved activations: **≈ 140 MB per sample per context step at V = 5** (158 MB per predicted step). The map is 5.6 MB of that; the encoder blocks over 949 tokens dominate (MLP hidden 949×1024 twice, q/k/v, LayerNorm inputs).
- 140 steps × batch 8 ≈ **150 GB**. Arm C as configured cannot run at batch 8 on a 49 GB A6000. The released config does not seem to fit an 80 GB H100 either, so check how the authors trained (possibly something outside the config).
- Under per-step checkpointing the doc's stored-map numbers are right (V = 17: 21 GB), plus a transient of one step's activations: ≈ 1 GB at V = 5, ≈ 3–4 GB at V = 17.

**F7. "KWM moves the window over a torus."** `cells.py:161-162`: the window is fixed at the top-left and content moves (stream subtracts `a`, `transport.py:74`). Both models fix the window, so this difference does not exist.

**F8. `velocity_weights`** does not exist in `kwm/` (grep).

**F9. Lattice sizes.**
- `lattice_nodes` (`collision.py:19-23`) yields only (2n+1)² sets that contain (0,0).
- "13 nodes (vmax 1, spacing 0.5)" is not produced. The 13-node L1 diamond excludes diagonal speeds near 1: (0.71, 0.71) has L1 = 1.41.
- "16 adaptive + 1 static" is not produced either. It needs new lattice code (e.g. {±0.25, ±0.75}², which is C4-closed) and a pinned static node.

**F10. "Turns are exact permutations" is true only of the grid operations** (`rot90` about (16,16) on the odd 33 grid; `roll` `:260-274`):
- The cross-V mixing conv applied after every roll (`:574-587`) has unconstrained node-indexed weights, so FloWM-3D's turn handling is not C4-exact as a model.
- With continuous block positions, sub-cell offsets coded inside 256-d features are not rotated by `rot90`; PR6 needed `rot_regular` for exactly this.
- The cone reaches the map edge (column 32; `max_range=None`, `:214`). The wrap-around of the forward roll (`:274`) and spectral ringing therefore land inside the FOV. This is consistent with the baseline, but not harmless for spectral transport of 1-cell blobs.

**Verified correct:**
- Map (B, 5, 33, 33, 256), agent at the centre looking +x, v_list (`:167`).
- FOV cone fixed in the map frame (`:209-218`, cached `:521-525`).
- Joint attention + sigmoid gate (`:318-334`); max over V in the decoder (`dynamic.yaml:19`, `:646-648`).
- `v_channel_identity_emb=false` in the released configuration.
- Encoder, gate and decoder weights are independent of V, so everything except the mixing conv loads.
- `stream` accepts `[B,V,2]` per-sample nodes (`transport.py:67-75`) on any H×W torus; 33 is odd, so there is no Nyquist bin and `c4` is moot.
- `rot_map` works for any square size and pivots at (window−1)/2, i.e. 16 with window = 33 (`rot.py:277-301`).
- The grad_ckpt flag capture is fixed on the PR6 branch (`model.py:255-262`; PR6 doc l.138-145).
- Arithmetic:
  - 1.115 MB per node-map;
  - 6.2 GB and 21 GB;
  - 27.5 EFLOPs / 30 TFLOPS = 10.6 days;
  - 30k×8 / (150k×16) = 1/10;
  - 2.75 EFLOPs / 30 TFLOPS = 25 h.

## MAJOR

**M1. Risk check 1 would test a circular pair, so a pass would mean nothing.**
- Failure scenario:
  - Where the gate g ≈ 0, the "written" content equals the transported content. The residual is then 0, which reads as a confident lock at whatever velocity the node already has.
  - Where g ≈ 1, the attention, which sees the memory tokens, decides which node receives the block, so the estimator measures the write's routing, not physics.
  - Either way the check can "pass" on noise.
- Fix: define an **observation-only lift**: run `map_processor` on a zero map (exactly what the encoder sees at t = 0 of every clip, so it is in-distribution for the released weights). Then:
  - snapshot = fixed projection P of that output (detached);
  - compare the transported previous snapshot with the current one on cone ∩ shifted cone, as KWM does.
- Cost: one extra encoder pass on 177 + 64 tokens per context step. Specify it in the doc and run risk check 1 on this pair.

**M2. The lift may not carry sub-cell position at all.**
- One cell = one grid unit, and blocks are 0.7 wide, so a block is about one cell.
- The released encoder was trained on integer positions only, and it uses absolute sin-cos positions (`:296-299, 314`), so nothing makes it shift-equivariant below one cell.
- If the BEV features quantise position to cells, then LK, SSD-with-parabola and continuous transport have no signal, whatever the estimator.
- Test first (CPU, cheap): on K3D-v with the released weights, a ridge probe from the observation-only BEV features at the cell containing each block (from `_block_info.pt`) to its sub-cell offset. Report the probe's R² against a shuffled control.
- If R² is low, option A needs a retrained lift (and possibly a finer map, e.g. 2 sub-cells per unit = 65×65, which is 4× tokens) before the estimator question is meaningful.

**M3. The estimator's spatial scales are about 3× the object size.**
- `LocalSearch` aggregates SSD with a Gaussian σ = 2, radius 6 (`lock_nodes.py:50, 195-196`): a 13×13-cell support inside a 19×17 box holding 6–10 blocks.
- That is effectively window-level pooling: the 2D gate's failure mode (window correlation, 76–91 % misses).
- The LK window is a 5×5 Gaussian (`transport.py:115-118`).
- Fix: rescale σ to about 0.5–1 cell (or run on an upsampled BEV). Make the scale a parameter in risk check 1 and report error versus σ.

**M4. The static background will capture nodes.**
- In 2D the background is black. In the BEV, floor and walls are textured everywhere and move exactly by −a, so after the action is subtracted they are a massive zero-velocity mode.
- `two_modes` keeps the heavier mode (`lock_nodes.py:116-123`). Every node whose ±0.75 box contains 0 locks onto the background. With {±0.25, ±0.75}² that is at least the four inner nodes.
- Fix: measure on the residual after the static node's prediction (snapshot minus the transported static snapshot), or mask cells explained by the static channel.
- Include a "background-only clip" null in risk check 1: blocks removed, or K3D-int.

**M5. Compute makes the plan infeasible as written.**
- Empirical anchor: the released eval did 1000 clips × 140 frames forward in ~18 min on GPU 1 (`FEAS3D_anchor.md:6`). With the paper-implied 2.7e10 forward FLOPs per frame (27.5e18 / (150k·16·140) / 3), that is ≈ 3.5 TFLOPS effective (inference, shared GPU, includes I/O), against the doc's assumed 30 sustained.
- Recompute for checkpointing adds about 1/3.
- At 5–15 TFLOPS sustained:

| Arm | Per run at 1/10 budget |
|---|---|
| C | ≈ 3–8 days |
| A, V = 17 (× 6) | ≈ 3–7 weeks |
| B, V = 25 (× 11) | ≈ 1.5–3 months |

- Fixes:
  - (a) **V-independent write** for A and B: pool V before the attention (max, as the decoder already does), attend over 177 + 64 tokens, then a per-node gated write `g_i = σ(W[h_i, h_new])` of the shared output. This is KWM's own lift semantics: one encoding written into all nodes (`cells.py:221-225`). A/B per-step cost then ≈ C's.
  - (b) Otherwise V = 9 (3×3, spacing 1, bound 0.75 → covers ±1.75; C4-closed; centre node pinned as static).
  - (c) Shorter training rollouts (e.g. 50 + 40) for all arms.
  - (d) Measure s/step on the GPU for C, with and without checkpointing, before any budget is written down.
  - Budget by equal FLOPs or equal wall-clock, not equal steps, or report both.

**M6. Memory plan.**
- (F6) Arm C needs checkpointing on the A6000 too. Implement it identically for all arms. With checkpointing, keep the teacher-forcing coin `torch.rand(1).item()` (`:656`) outside the checkpointed function, or rely on RNG replay and test that the two gradients are identical.
- The "128-channel moving nodes" fallback is incompatible with shared 256-d attention tokens and the released weights (it would need per-width projections). Drop it, or use bf16 autocast for the attention, keeping the estimator in fp32 as KWM does.
- Per the 2026-10-26 session notes, another user held about 40 GB on GPU 1. "Fits beside the two ~7 GB KWM jobs" must be checked against actual free memory.

**M7. Turn handling (F4, F10).**
- Implement turns as `rot_state` semantics: permute + `rot_vec`(rho) + `rot_map`; no `rot_regular`; a C4-closed lattice.
- Snapshot after a turn: the previous 60° cone rotated by 90° overlaps the new cone only at the apex, so turn steps (9.7 % of actions) give no evidence. That is fine, but say so.
- Decide what to do with the cross-V mixing conv in A/B. It has fixed node-indexed weights, so it is incoherent once node velocities are per-sample, and it leaks content between populations moving at different velocities, destroying "exact transport". Use KWM-style per-node shared depthwise 3×3 (`cells.py:227-230`) in A/B, and keep FloWM's in C. State the asymmetry.
- Report error on turn steps separately.

**M8. The init bug (F1) confounds every new-V arm.**
- Reproducing FloWM-3D faithfully (C from scratch) means keeping the near-zero init.
- If A/B get a correct identity init and C does not, part of any gain is the init. Pick one rule for all arms and report it.
- For D (fine-tuning), the A/B mixing conv must be built as an identity per group, not re-run through `dirac_`.

**M9. Fairness and readability of the comparison.**
- A 1/10 budget favours the arm with the stronger prior; FloWM-3D will be far from converged.
- The released checkpoint is trained on the integer law, so it is not a "ceiling" on K3D-v; call it a zero-shot reference.
- Add: learning curves at matched steps and FLOPs; C at 2× budget as a convergence check; fine-tuning from the released weights for all arms as a second, cheaper protocol (with the M8 fix).
- Seeds: the user's standing rule (2026-10-25) is 3 seeds for screening and 5 for primaries. Three seeds for the headline violates it unless the doc says this is screening.
- Global MSE on 128² frames is dominated by walls and floor. Velocity errors on small blocks may be invisible at 1000 clips. Have the generator store per-frame block masks (or project boxes from `_block_info.pt`) and report block-region MSE, per speed bin, with paired per-clip tests across arms.

**M10. Missing arm: oracle nodes.**
- Per-sample nodes set from ground-truth agent-frame block velocities (`_block_info.pt`) + static. KWM already has this path (`model.py:103-107`, `train.py:114`, `oracle_nodes_from`).
- It is the ceiling of option A, isolates "continuous transport with correct velocities" from the estimator, and is cheap (V = blocks + 1 ≤ 11, no estimator).
- If A-oracle does not beat C on block-region MSE, stop before building the estimator.

**M11. Sequence order.** The current order puts packaging before the only kill tests and implements the full backbone before cost is known. Proposed order:
1. Define the observation-only lift. Run the sub-cell probe (M2) and the static-background null (M4) on K3D-int / K3D-v (CPU).
2. Risk check 1 with the correct pair, the cone mask, σ sweep and background handling.
3. GPU micro-benchmark (≤ 50 steps): C with and without checkpointing, and the V-pooled write at V = 9/17. Decide V and the write design from the numbers.
4. Oracle-node pilot against C.
5. Then implement the estimator in the backbone, and package KWM.

## MINOR

1. The FloWM map is channels-last (B,V,H,W,D); `stream` needs (…,H,W). That is two full-map permute copies per step: fuse them, or keep the map channels-first in the new backbone.
2. "C (V = 5) bit-identical to FloWM-3D" holds only if C keeps `torch.roll`. Through `stream` with integer nodes it is ~1e-6 (spectral). Say which.
3. Training uses 50 context frames (`dynamic.yaml:25`) and evaluation uses 70. One A6000 at batch 8 is half the released global batch 16 at the same lr 1e-4. State both.
4. Prediction phase: FloWM-3D writes its own decoded frames back (`:659`). Freeze nodes and the snapshot in prediction, per PR3; the doc should say this explicitly.
5. Max over 17–25 nodes raises the decoder's input floor (max of noisy or ringing nodes). Log per-node norms; `latent_maps` currently reduces over V (`:630`).
6. Hungarian matching against `_block_info.pt` needs a tested axis and sign convention (map column = +x forward, agent at (16,16)).
7. Make the risk-check pass criterion numeric, e.g. matched median ≤ 0.1 cell/step and misses ≤ 30 % over the 0.25–1.0 speeds. "Clearly better than ±1 lattice" is untestable, since nearest-integer error is up to ≈ 0.5.
8. CPU cost of risk check 1 is fine: about 0.15 s per step at B = 1 for V = 5 (measured), so 200 × 140 steps ≈ 1.2 h plus the estimator.
9. The FOV cone extends to the torus edge, so wrapped content and sinc tails enter the far field of view. Consider a padded torus for A/B (the positional encoding is computed from H, W, so a bigger map changes token positions and breaks direct weight reuse; note the trade-off).

## Change the doc before it goes to the user

1. Correct F1–F10:
   - mixing conv shape and init;
   - measurement pair;
   - window assumptions (list them);
   - turns need permutation plus rot_vec;
   - V-dependent compute and memory, with the measured numbers;
   - KWM window fixed; no `velocity_weights`; lattice sizes.
2. Specify the observation-only lift and make it the estimator's input (M1). Add the sub-cell probe and the background null as gates before risk check 1 (M2, M4). Add the σ rescale (M3).
3. Replace §4 with FLOP-based, measured numbers:
   - include the V multiplier, checkpoint recompute and the ~3.5 TFLOPS anchor;
   - propose the V-pooled write or V = 9;
   - equal-FLOP budgets;
   - checkpointing for all arms;
   - recheck GPU 1 free memory, including other users.
4. Add the oracle-node arm and the C-at-2×-budget and fine-tune protocols. Seeds per the user's 3-screen / 5-primary rule. Block-region metrics.
5. Reorder §5 as in M11.
6. Decide and state the mixing-conv policy per arm (init, and cross-node versus per-node).
