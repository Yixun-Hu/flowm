# Kinetic 3D: which model runs on the benchmark (design step v2, 2026-11-03, for the user's decisions)

Basis: `algorithms/mem_wm/backbones/flowm/flowm_models_3d.py` (FloWM-3D, `ViTBlockworldModel`), `algorithms/flowm_video.py`,
the KWM fork (`kwm/cells.py`, `kwm/transport.py`, `kwm/lock_nodes.py`, `kwm/rot.py`, PR1 lock nodes, PR6 C4), paper
Tables 12–13, the approved benchmark choices (KINETIC3D_choices_draft.md), and the adversarial Opus review of v1
(`scratchpad/review/opus_k3d_design_review.md`: 10 factual errors and 11 major issues, all folded in below; its CPU
measurements are quoted where used).

## 1. What FloWM-3D is, in KWM terms (verified against the code)

FloWM-3D keeps an egocentric top-down map `h ∈ (B, V, 33, 33, 256)`, agent at the centre looking +x, V = 5 fixed
integer velocities {(0,0), (±1,0), (0,±1)}. One step:

| FloWM-3D stage | Code | KWM counterpart |
|---|---|---|
| transport each channel by its velocity | `shift_v_channels`: `torch.roll` by (vx, vy) | `transport.stream`: spectral shift by continuous per-sample node velocities (works on the odd 33×33 torus; verified) |
| ego-motion | `roll_map`: forward = roll one cell; turn = `rot90` of the map and a cyclic shift of the four moving channels | camera term in `stream`; PR6 `rot_map` + node permutation + `rot_vec` on the adaptive offsets (`rot_state`) |
| cross-channel mixing | `v_channel_mixing`: `Conv2d(256V, 256V, 3×3, groups=256)`, i.e. each feature is mixed **across all V velocity channels** with node-indexed weights; `nn.init.dirac_` is called without `groups`, so only 5 of 1280 output channels are non-zero at init (the map is almost erased each step until training fixes it) | `conv_h`: per-node shared 3×3; no cross-node coupling |
| write the observation | joint self-attention over the FOV-cone tokens of all V channels (177 cells × V) and 64 image patches, then a sigmoid gate blends old and new tokens in the cone | `lift` + `feature_update`: one encoding `conv_u(frame)` written into every node |
| node update | none | `AdaptiveNodeCell`: lock-then-refine compares the **previous observation imprint** `P·conv_u(u_{t−1})`, transported by each node's velocity, with the **current imprint** `P·conv_u(u_t)` (K = 8 fixed random channels). Memory features never enter the estimator. |
| decode | cross-attention decoder from the cone tokens, max over V | `decode` |

Facts that shape the design (all verified): the cone is fixed in the map frame (177 cells, bounding box rows 7–25 ×
cols 16–32, reaching the map edge, so forward-roll wrap-around lands inside the far field of view, as in the
baseline); the encoder, gate and decoder weights are independent of V, only the mixing conv is not; the attention
sequence is V·177 + 64 tokens, so **cost grows with V** (forward FLOPs ≈ 1 / 2.3 / 6 / 11 × for V = 5 / 9 / 17 / 25;
measured CPU step times scale 1 : 1.5 : 3.3 : 6.5); training is full backpropagation through all 140 steps with the
model's own decoded frames fed back (teacher forcing 0.125), no truncation and no activation checkpointing
(`flowm_video.py:115`); the reviewer measured ≈ 140 MB of saved activations per sample per step at V = 5, of which the
map is 5.6 MB, so even the unchanged baseline needs per-step checkpointing on the 49 GB A6000 (≈ 150 GB at batch 8
otherwise; how the authors fit batch 8 on an 80 GB H100 is to be checked by the GPU micro-benchmark, not assumed).

KWM facts that do not transfer unchanged: `lock_nodes.py` assumes a square window at the map origin (padding,
candidate masks, snapshot shape, and four unmasked window means: texture threshold, coherence sigma, LK damping
reference, LK gradients with wrap-around); its SSD aggregation uses a Gaussian of σ = 2 cells, radius 6, which here is
a 13×13 support in a 19×17 box holding 6–10 one-cell blocks, i.e. the window-pooling failure mode of the 2D gate;
adaptive velocities are `lattice_i + rho_i` with |rho| ≤ 0.75, so a turn must permute the nodes **and** rotate rho
(`rot_state`), which requires a C4-closed lattice; `lattice_nodes` produces only (2n+1)² sets.

## 2. Candidate arms

**C. FloWM-3D unchanged, retrained on K3D-v** (required baseline). The released checkpoint is a zero-shot reference,
not a ceiling: it was trained on the integer law.

**A-oracle. FloWM-3D's lift and decoder with KWM transport and ground-truth nodes.** Per-sample node velocities set
from the agent-frame block velocities in `_block_info.pt` (V = blocks + 1 static ≤ 11), spectral transport, no
estimator. KWM already has this path (`oracle_nodes_from`). It is the ceiling of everything below and isolates
"continuous transport with the right velocities" from "estimating them". **If A-oracle does not beat C on
block-region error, the estimator is pointless on this lift and we stop.**

**A. KWM estimator inside FloWM-3D's map** (the target). As A-oracle, with the node velocities estimated by
lock-then-refine on an **observation-only lift**: the map processor run on a zero map (exactly what the encoder sees at
t = 0 of every clip, so in-distribution for released weights), projected by the fixed P to 8 channels, giving the
imprint pair KWM actually uses. The cone is a mask: the estimator runs on the bounding box with the mask as texture
support and the four window means restricted to the mask; σ becomes a parameter (0.5–1 cell expected). The static
background (floor and walls, textured everywhere, moving exactly by the ego-motion) would capture every node whose box
contains zero velocity under the heavier-mode rule, so the measurement is taken on the residual after the static
node's transported imprint, with a background-only null (K3D-int clips or blocks removed) as a control.

**B. Fixed finer lattice, spectral transport, no estimator** (ablation): V = 9, the 3×3 lattice at spacing 1 pinned at
the static node, which is C4-closed and covers the benchmark speeds with |rho| ≤ 0.75. Tells whether continuous
transport alone, without adaptive nodes, closes the gap.

**Write design for A/A-oracle/B (cost control).** Per-node attention tokens make A/B 2–11 × C's cost. Proposed
V-independent write: max-pool over V before the attention (the decoder already does this), attend over 177 + 64
tokens, then a per-node gated write of the shared output, `g_i = σ(W[h_i, h_new])`. This is KWM's own lift semantics
(one encoding written into all nodes) and brings every arm to about C's per-step cost; it is also what makes V = 17
affordable. The alternative (keep FloWM's per-node tokens) is kept as an option only at V = 9.

**Mixing conv policy.** One rule for all arms, stated in the results: C keeps FloWM's node-indexed cross-V conv and its
original (near-zero) init, reproduced faithfully; A/B use KWM's per-node shared 3×3 (node-indexed weights are
meaningless for per-sample nodes and leak content between velocity populations). If any arm is initialised from the
released weights, its conv is built as a per-group identity, not via `dirac_`.

**Turns.** Implemented as `rot_state`: `rot_map` of the map and snapshot, node permutation, `rot_vec` on rho; no
orientation channels (FloWM features have none). After a turn the previous cone overlaps the new one only at the apex,
so turn steps (≈ 10 % of actions) carry no velocity evidence; errors on turn steps are reported separately. Exactness
under turns gets its own test (`tests/test_rot.py` tests episode symmetry, not a turn action).

**Prediction phase.** Nodes and snapshot frozen during prediction (PR3 result), decoded frames fed back as in FloWM.

**D. Initialising from the released checkpoint** stays a pilot/second protocol: cheaper, but the lift was trained on
integer positions and the mixing conv must be rebuilt (above).

## 3. Kill tests before any GPU training (all CPU, released weights, K3D-int and K3D-v validation clips)

1. **Sub-cell position probe.** Does the lift carry position below one cell at all? Blocks are 0.7 cells wide, the
   encoder saw only integer positions and uses absolute sin-cos positions. Ridge probe from the observation-only cone
   features at the block's cell to its sub-cell offset (from `_block_info.pt`), R² against a shuffled control. Low
   R² means option A needs a retrained or finer lift (65×65 map, 4× tokens) before the estimator question exists.
2. **Background null.** Estimator on K3D-int / block-free clips: must report no moving nodes.
3. **Estimator gate (risk check 1, corrected).** Lock-then-refine on the observation-only imprint pair with the cone
   mask and the static residual, σ ∈ {0.5, 1, 2}, Hungarian match to the agent-frame block velocities as in the 2D
   gate, 200 clips. Pass: matched median ≤ 0.1 cell/step and misses ≤ 30 % over speeds 0.25–1.0 (nearest-integer
   assignment errs by up to 0.5, so the ±1 lattice is not a usable reference). Cost ≈ 0.15 s per step at B = 1 for the
   lift, ≈ 1.5 h plus the estimator.

## 4. Compute and memory (measured anchors, not assumptions)

- Released: 150k steps, batch 16, 2 × H100, 27.5 EFLOPs (Table 13). The released evaluation on GPU 1 (1000 clips ×
  140 frames forward in ≈ 18 min) implies ≈ 3.5 TFLOPS effective on the shared A6000; checkpoint recompute adds
  about a third. At 5–15 TFLOPS sustained, a 1/10 budget for C alone is 3–8 days per run; A at V = 17 with per-node
  tokens would be 3–7 weeks, which is why the V-pooled write is proposed.
- Budgets are set by equal FLOPs (or equal wall-clock, both reported), not equal steps.
- Per-step activation checkpointing for every arm, identical code path; the teacher-forcing coin kept outside the
  checkpointed function, with a test that gradients match the unchecked version.
- GPU 1 is shared with the H1/C4 chains (≈ 14 GB) and at times another user's ≈ 40 GB; free memory is checked, not
  assumed. The chains hold the GPU for roughly eight more days.
- Before any budget is written down: a ≤ 50-step GPU micro-benchmark (gated on free memory) of C with and without
  checkpointing, and of the V-pooled write at V = 9 and 17: s/step and peak memory at batch 8.
- Seeds: 3 for screening arms, 5 for the primary comparison (standing rule). Learning curves at matched FLOPs; C at
  2× budget as a convergence check.

## 5. Metrics

Released protocol (70 + 70 and 70 + 210; MSE/PSNR/SSIM on 1000 clips) stays primary for comparability, but global MSE
on 128² frames is dominated by walls and floor. Secondary, pre-registered: block-region MSE (per-frame block masks
projected from `_block_info.pt`, added to the generator), per speed bin, per-clip paired tests across arms; errors on
turn steps separately; node-velocity error against oracle for A. Selection on validation, reporting on a held-out test
split with a fresh seed.

## 6. Proposed sequence

1. Generator (done, branch k3d): K3D-int control and anchor check; K3D-v validation (1000) now; test and training
   splits after this document is approved; add block masks to `_block_info.pt`.
2. Observation-only lift + kill tests 1–3 (CPU, released weights).
3. GPU micro-benchmark (≤ 50 steps): decides V, the write design and the budget.
4. A-oracle vs C pilot at a small budget (block-region metrics). Stop if A-oracle does not beat C.
5. Only then: estimator in the backbone (`algorithms/mem_wm/backbones/kwm3d/`), KWM as a pinned editable dependency,
   C bit-identical reference test (C keeps `torch.roll`; through `stream` it is ≈ 1e-6, not bit-exact), the turn test;
   Codex implements, Opus reviews, as before.
6. Campaign plan with measured numbers, for approval before training.

## 7. Decisions requested

| # | Decision | Recommendation |
|---|---|---|
| D1 | Arms | C (baseline), A-oracle (ceiling and kill test), A (target), B (ablation); D only as a cheaper second protocol |
| D2 | Write and V | V-pooled write for A/A-oracle/B; V = 9 fixed lattice for B, V = blocks + 1 for A-oracle, V decided for A by the micro-benchmark (9 or 17) |
| D3 | Budget | Equal FLOPs across arms, size fixed after the micro-benchmark; 3 seeds screening, 5 primary; checkpointing everywhere |
| D4 | Metrics | Released protocol primary; block-region, speed-bin and turn-step metrics secondary, pre-registered |
| D5 | Kill tests first | Sub-cell probe, background null and the corrected estimator gate before any GPU training; A-oracle before the estimator is built |
| D6 | KWM dependency | Pinned editable install of the fork, no copied code |
| D7 | Scheduling | 3D GPU work only after the micro-benchmark and when free memory allows; otherwise after the C4 chain |

## Addendum (2026-11-03): answers to the user's four questions before D1–D7

**Q1. No arm isolates the write design.** Correct as the v2 table stands: A, A-oracle and B change both the
transport and the write relative to C. The cheap fix is a fifth arm, **C-pw**: FloWM-3D's integer `torch.roll`
transport with the V-pooled write at V = 5. It costs the same as C (the pooled attention is slightly cheaper than
the per-node one) and completes a two-factor design: write effect = C-pw vs C; transport effect = A-oracle or B vs
C-pw; estimator effect = A vs A-oracle. B with FloWM's per-node tokens at V = 9 would isolate the write differently
but costs 2.3 × C and only at one V, so C-pw is the better control. Worth the compute: it is one more run per seed at
baseline cost, and without it any gain of the kinetic arms is unattributable. The budget plan below counts it.

**Q2. The V = 17 node set.** Intended set: the static node (0, 0) plus the 16 points {±0.25, ±0.75}², i.e. four
C4 orbits of four: (±0.25, ±0.25), (±0.75, ±0.75), and the two orbits generated by (0.25, 0.75) and (0.75, 0.25).
It is closed under 90° rotations and under the axis reflections (`reflection_perms` closure holds too), spacing 0.5
between neighbours, and with the adaptive offset bound |rho| ≤ 0.75 the union of the node boxes covers
[−1.5, 1.5]², which contains every benchmark velocity (≤ 1.0, ≤ 1.13 with acceleration). `rot_state` supports any
C4-closed set: `rotation_perms` (`kwm/rot.py:59-66`) computes the permutation from the node coordinates and only
asserts closure and bijectivity. `two_modes` (`kwm/lock_nodes.py:87-95`) restricts candidates to each node's box
with the same bound and does not assume a regular grid. What has to change is only the constructor:
`lattice_nodes` (`kwm/collision.py:19-23`) builds (2n+1)² grids containing (0, 0), so a `--lattice` option with
this named set (and the V = 9 set {−1, 0, 1}² for B) is a small addition; the static node is pinned by zeroing its
rho and excluding it from the update, as the zero node already is in the 2D model. The alternative 5 × 5 grid at
spacing 0.5 (V = 25) is also C4-closed; 17 is preferred because the pooled write makes attention cost V-independent
but the transport, per-node conv and estimator still scale linearly in V.

**Q3. Per-node map conv: p4 group conv or plain 3 × 3?** Neither as PR6 does it. PR6's p4 convolutions act on
regular-representation features (C = 4F orientation channels, rolled under rotation), and FloWM's 256-d map
features are produced by a ViT with absolute sin-cos positions in the cone; they carry no orientation channels, so
a regular p4 conv cannot be applied to them, and the lift and decoder are not turn-equivariant in any case (FloWM's
own mixing conv is unconstrained too). What can be exact at zero extra cost is the map-side update on *scalar*
fields: a shared depthwise 3 × 3 whose kernel is constrained to be C4-symmetric (three free values per channel:
centre, edge, corner; built by averaging a free kernel over the four rotations, as PR6's invariant-output layer does
for scalar inputs). With it, transport + per-node conv + turn are exactly equivariant on the map, and the only
non-equivariant parts are the write and the decoder, which is the honest statement for 3D. Cost: identical to a
plain depthwise conv (the tying happens once per forward on a 256 × 3 × 3 weight). Trade-off: a plain conv has
about 3 × more parameters per channel; in 2D the C4 branch showed no accuracy loss from tying, so the symmetric
kernel is the recommendation, with the plain conv as an ablation only if time allows.

**Q4. Realistic GPU time, and a cheaper screening plan.** Per-run cost at the 1/10 budget (30k steps, batch 8,
140-frame clips, checkpointing): C ≈ 3–8 days on GPU 1 (the reviewer's range at 5–15 TFLOPS sustained with
recompute; the released evaluation implies only ≈ 3.5 TFLOPS effective, so the upper end is more likely). With the
pooled write, A-oracle, B and A cost ≈ 1.1–1.3 × C (transport FFTs over V × 256 × 33² and the estimator are small
next to the ViT), C-pw ≈ 1 × C. Five arms × 3 seeds ≈ 15 runs × 4–9 days ≈ **60–135 GPU-days**, sequential on a
shared GPU that the H1/C4 chains hold for another week. That is not realistic, and running jobs side by side on one
A6000 does not shorten it.

Cheaper plan, three stages, each with a kill criterion:

1. **CPU, no GPU time:** the three kill tests of §3 (sub-cell probe, background null, estimator gate).
2. **Pilot, ≈ 2–4 GPU-days:** micro-benchmark (≤ 50 steps each arm) for measured s/step and memory; then A-oracle
   vs C, one seed each, at a 1/50 budget with shorter training clips (50 context + 40 predicted instead of 50 + 90,
   which cuts the backpropagation length by a third; evaluation stays at 70 + 70 / 70 + 210). Kill if A-oracle does
   not beat C on block-region error.
3. **Screening, ≈ 6–10 GPU-days:** all five arms × 3 seeds, **fine-tuned from the released checkpoint** (the
   reviewer's second protocol; the mixing conv rebuilt as a per-group identity for every arm, including C, so the
   init is common), 5k steps at batch 8 with the shorter clips, ≈ 0.4–0.7 day per run. This ranks the arms with the
   same lift initialisation and leaves from-scratch training to the confirmation.
4. **Confirmation, ≈ 20–50 GPU-days:** only the winning kinetic arm against C (and C-pw if the write effect
   matters), from scratch at the 1/10 budget, 3 seeds (5 would double it; the standing 5-seed rule for primaries
   would then need either a second GPU or dropping the from-scratch confirmation to the pair that matters).

Total with the cheap plan: about 1–2 GPU-weeks before any from-scratch training, and 3–7 further GPU-weeks for the
confirmation, versus 2–4.5 GPU-months for the v2 plan. If a second GPU (cloud or lab) can be found for the
confirmation stage, the wall-clock halves; otherwise the confirmation starts only after the C4 chain finishes.
