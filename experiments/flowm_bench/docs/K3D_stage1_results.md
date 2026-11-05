# Kinetic 3D stage 1: released-weight CPU kill tests


Measured 2026-11-04. No training and no GPU execution. Results below concern the released observation-only lift; they do not select a subsequent model arm.


| Test | Pre-registered criterion | Measurement | Verdict |
| --- | --- | --- | --- |
| sub-cell probe | R² ≥ .25 on both axes and ≥ 5 × shuffled R² | forward R²=0.29013, shuffled=-0.02277; right R²=0.39626, shuffled=-0.01708 | PASS |
| background null | Alarm ≤ 2% and evidence-weighted speed ≤ .05 on both null splits | σ=0.5; k3d_null_noblocks: 4.43%, 1.01117; k3d_null_static: 4.08%, 1.03680 | FAIL |
| estimator gate (K=8) | Matched median ≤ .1 and misses ≤ 30%, speeds .25–1, best σ | σ=0.5, median=1.21972, misses=98.32% | FAIL |


## Protocol, conventions and deviations


- Checkpoint loaded strictly: 240 `main_model.` tensors, V=5, 256 channels, six map and decoder layers, eight heads, 16-pixel patches, max-pool decode, flattened 2D velocity mixing, no velocity identity embedding.
- Numeric shard/index clip order; frame zero start. Read-only dataset adapter calls the existing dataset's `__getitem__`, `/255` decoder, resize, `action_int` loader and `t-1->t` alignment; it bypasses writable NAS metadata indexing. First action is discarded by the model. No NAS data or KWM package files were edited.
- Actions in stored data: 0 left, 1 right, 2 forward, 4 no-op. Aligned action at t is stored action t−1. Forward shifts the map by −1 column; left rotates clockwise, right counterclockwise.
- The requested range t=1..139 contains **139**, not 138, measurement pairs. All clips retain 140 frames. Delay diagnostics use t=1..138, so truth t+1 remains inside those 140 frames; a delayed pair is labelled turn if either its estimate or target transition turns.
- **Velocity metadata mismatch:** generator `block_vel_agent[t]` is outgoing block displacement minus outgoing camera displacement in heading t. KWM returns intrinsic velocity because camera displacement is already subtracted in transport. The literal requested comparison remains primary. `intrinsic_incoming` is a separately labelled diagnostic: `block_vel_world[t−1]` rotated into heading t, without camera subtraction. Its speed bins use incoming nominal `block_speed`; the literal comparison bins by its literal target norm. Reflections can make finite-step displacement differ from instantaneous pre-bounce speed. No labels enter the estimator.
- Turns provide no cone-pair measurement. Rotate and permute rho, accumulated evidence and lock metadata using the named lattice's C4 permutation; carry the rotated node estimates into the separately reported turn diagnostic. Both-time visibility is retained, so turn truth counts can be very small. Null turns are skipped and counted.
- `measure_nodes` is imported from KWM, as are named nodes, fixed projections and rotation primitives. Hungarian matching uses SciPy, following the 2D harness's Euclidean assignment and lower-median convention. Its fixed five-digit final-frame metric cannot be imported unchanged. The 0.5 miss cutoff and speed bins are this task's overrides, not the 2D gate's 0.1 cutoff.
- Lock evidence means normalized LK agreement ≥0.1, as implemented in `lock_nodes.py` and explained in `PR1_lock_nodes.md`, the method linked by `GATE_estimator.md`. The API's raw selected evidence is **not** thresholded at 0.1. A scoped observational wrapper reads the existing coherence calculation to recover agreement; numerical outputs are unchanged. Full lock also requires pooled LK residual norm ≤0.3. The API does not expose agreement: `.scratch/kwm_patch.diff` proposes an optional return field, left unapplied; the local wrapper is the workaround.
- Gate candidates are current locked nodes with accumulated selected evidence >1e−6; the pinned static node is excluded. Scores use updated recurrent velocities. Update is exactly s=1, W+=e, gain=e/max(W,1e−6), step=clip(gain·residual,±0.3), rho=clip(rho+step,±0.75). Each clip resets independently. Gaussian aggregation uses radius=ceil(3σ), via the primitive's radius=None option.
- Null v̂ is the immediate `measure_nodes` result under that same recurrent tracker. Alarm uses agreement ≥0.1 and |v̂|>0.15. Primary denominator is all 17 node-steps; the stricter 16 moving-node denominator is also shown. Evidence-weighted mean includes selected evidence even on unlocked nodes; the pinned node contributes zero.
- The gate uses the same rounded-cell cone and rel_x≥1 visibility definition as the probe. Hungarian assignment is performed once against every jointly visible block; speed bins then filter assigned truths. An unassigned truth is a miss, as is an assigned distance >0.5. Matched statistics include every assigned pair, including >0.5 errors, and exclude unassigned pairs. False alarms count active nodes with no visible truth within 0.5, independently of uniqueness; overall and criterion-band FA denominators include all active nodes; the three narrow speed-bin FA denominators use node speed. A missing denominator is shown as —.
- Full 256-channel tokens, gate and pre-gate h_new are cached in fp16; projection and all measurements are fp32. B=32 zero-map writes retain all five velocity copies during attention and assert equality before selecting one. CPU SDPA replaces the materialized attention fast path; model mathematics and weights are unchanged.
- Probe: 14 frames per clip (0,10,…,130), visible rounded cells with rel_x≥1, no occlusion filtering, 3×3×256 zero-padded features. Half-open residual convention is p−floor(p+.5). Five outer clip groups and a disjoint inner clip split; StandardScaler and ridge fit only training data; ridge solves use float64 after the initial fp32 fit emitted conditioning warnings. α∈{1,100,10000} is selected separately for sub-cell, integer, colour and shuffled tasks. Colour is one-hot ridge classification. Shuffled training/validation/test labels are independently permuted inside their respective folds. Pooled held-out R² is primary. Negative shuffled R² is retained, not converted into a positive ratio. Distance bins are [1,5), [5,9), [9,13).



## Released-model sanity


Eight-clip mean MSE **0.000305736**, 70 context + 70 autoregressive frames, 105.4 seconds. The 1000-clip reference is 0.000625; these eight deterministic first clips are a loading/alignment check, not an estimate of the full-split mean. Predictions are clamped to [0,1] for the released metric; raw MSE is also recorded.


| Clip | MSE | Raw MSE | Turn MSE | Nonturn MSE | Turn frames |
| --- | --- | --- | --- | --- | --- |
| 0/0000 | 0.00010 | 0.00010 | 0.00043 | 0.00007 | 6 |
| 0/0001 | 0.00026 | 0.00026 | 0.00019 | 0.00027 | 8 |
| 0/0002 | 0.00021 | 0.00021 | 0.00085 | 0.00013 | 8 |
| 0/0003 | 0.00023 | 0.00023 | 0.00018 | 0.00024 | 8 |
| 0/0004 | 0.00018 | 0.00018 | 0.00059 | 0.00013 | 8 |
| 0/0005 | 0.00031 | 0.00031 | 0.00046 | 0.00030 | 7 |
| 0/0006 | 0.00010 | 0.00010 | 0.00020 | 0.00009 | 8 |
| 0/0007 | 0.00105 | 0.00105 | 0.00874 | 0.00033 | 6 |


## Map geometry evidence


`compute_fov_mask` agrees with the independently called helper: 177 cells, rows 7..25, columns 16..32. Agent at (16,16), forward is +column. Empirical activity supports **row=16+rel_z**, column=16+rel_x. Compare each off-axis block's candidate cell to its mirrored cell over 20 integer clips; cell-centered scores subtract each cell's temporal mean.


The stored velocity identity was checked directly: `block_vel_agent + outgoing_camera_agent = rotate(block_vel_world, heading_t)`. Counts below are the 139 incoming transitions per clip. No unknown action codes were found.


| Split | Clips | Left | Right | Forward | No-op | Max velocity identity error | Max forward-distance error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k3d_int_validation | 20 | 187 | 77 | 1616 | 900 | 6e-08 | 0 |
| k3d_null_noblocks | 50 | 486 | 205 | 3939 | 2320 | 0 | 0 |
| k3d_null_static | 50 | 466 | 196 | 4090 | 2198 | 0 | 0 |
| k3d_v_validation | 200 | 1910 | 782 | 16158 | 8950 | 6e-08 | 0 |

| Signal | Normalization | Block observations | Plus sign | Minus sign |
| --- | --- | --- | --- | --- |
| tokens | raw | 4012 | 16.129267 | 13.555372 |
| tokens | cell_centered | 4012 | 2.668253 | 0.059255 |
| g | raw | 4012 | 0.705274 | 0.531865 |
| g | cell_centered | 4012 | 0.180846 | 0.005872 |


## Test 1: sub-cell probe


| Lift | Steps | Distance | Axis | N | Held-out R² | Shuffled R² | Integer R² | Colour accuracy | Balanced colour accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | all | all | forward | 5333 | 0.29013 | -0.02277 | 0.99275 | 75.15% | 75.20% |
| tokens | all | all | right | 5333 | 0.39626 | -0.01708 | 0.97956 | 75.15% | 75.20% |
| tokens | nonturn | all | forward | 4739 | 0.29047 | -0.02031 | 0.99276 | 75.44% | 75.50% |
| tokens | nonturn | all | right | 4739 | 0.39074 | -0.01670 | 0.97973 | 75.44% | 75.50% |
| tokens | turn | all | forward | 594 | 0.28590 | -0.04337 | 0.99246 | 72.90% | 72.75% |
| tokens | turn | all | right | 594 | 0.43499 | -0.02189 | 0.97781 | 72.90% | 72.75% |
| tokens | all | 1-4 | forward | 1305 | 0.28870 | -0.03553 | 0.95050 | 85.52% | 85.65% |
| tokens | all | 1-4 | right | 1305 | 0.48960 | -0.02220 | 0.94381 | 85.52% | 85.65% |
| tokens | nonturn | 1-4 | forward | 1180 | 0.30162 | -0.03580 | 0.95037 | 85.68% | 85.80% |
| tokens | nonturn | 1-4 | right | 1180 | 0.48220 | -0.01892 | 0.94405 | 85.68% | 85.80% |
| tokens | turn | 1-4 | forward | 125 | 0.16301 | -0.03440 | 0.95179 | 84.00% | 83.93% |
| tokens | turn | 1-4 | right | 125 | 0.53301 | -0.05164 | 0.93843 | 84.00% | 83.93% |
| tokens | all | 5-8 | forward | 2038 | 0.27427 | -0.02525 | 0.95462 | 74.93% | 74.99% |
| tokens | all | 5-8 | right | 2038 | 0.39108 | -0.01613 | 0.98330 | 74.93% | 74.99% |
| tokens | nonturn | 5-8 | forward | 1841 | 0.27023 | -0.02224 | 0.95570 | 75.34% | 75.40% |
| tokens | nonturn | 5-8 | right | 1841 | 0.38191 | -0.01357 | 0.98337 | 75.34% | 75.40% |
| tokens | turn | 5-8 | forward | 197 | 0.30401 | -0.05140 | 0.94426 | 71.07% | 70.83% |
| tokens | turn | 5-8 | right | 197 | 0.47390 | -0.04480 | 0.98200 | 71.07% | 70.83% |
| tokens | all | 9-12 | forward | 1681 | 0.31515 | -0.01054 | 0.93186 | 68.00% | 68.04% |
| tokens | all | 9-12 | right | 1681 | 0.35148 | -0.01412 | 0.98247 | 68.00% | 68.04% |
| tokens | nonturn | 9-12 | forward | 1463 | 0.31539 | -0.00807 | 0.93288 | 68.15% | 68.24% |
| tokens | nonturn | 9-12 | right | 1463 | 0.34681 | -0.01735 | 0.98256 | 68.15% | 68.24% |
| tokens | turn | 9-12 | forward | 218 | 0.31182 | -0.03509 | 0.92500 | 66.97% | 65.88% |
| tokens | turn | 9-12 | right | 218 | 0.37722 | -0.00716 | 0.98146 | 66.97% | 65.88% |
| h_new | all | all | forward | 5333 | 0.28234 | -0.02061 | 0.99491 | 76.34% | 76.38% |
| h_new | all | all | right | 5333 | 0.39582 | -0.01235 | 0.98853 | 76.34% | 76.38% |
| h_new | nonturn | all | forward | 4739 | 0.28681 | -0.01801 | 0.99492 | 76.66% | 76.71% |
| h_new | nonturn | all | right | 4739 | 0.38997 | -0.01154 | 0.98842 | 76.66% | 76.71% |
| h_new | turn | all | forward | 594 | 0.24458 | -0.04224 | 0.99467 | 73.74% | 73.55% |
| h_new | turn | all | right | 594 | 0.43702 | -0.02050 | 0.98901 | 73.74% | 73.55% |
| h_new | all | 1-4 | forward | 1305 | 0.29201 | -0.03294 | 0.96458 | 86.51% | 86.61% |
| h_new | all | 1-4 | right | 1305 | 0.49469 | -0.01487 | 0.96891 | 86.51% | 86.61% |
| h_new | nonturn | 1-4 | forward | 1180 | 0.30793 | -0.03203 | 0.96502 | 86.61% | 86.69% |
| h_new | nonturn | 1-4 | right | 1180 | 0.48652 | -0.01350 | 0.96932 | 86.61% | 86.69% |
| h_new | turn | 1-4 | forward | 125 | 0.13728 | -0.04165 | 0.95995 | 85.60% | 85.56% |
| h_new | turn | 1-4 | right | 125 | 0.54523 | -0.02718 | 0.96282 | 85.60% | 85.56% |
| h_new | all | 5-8 | forward | 2038 | 0.26770 | -0.02175 | 0.97046 | 76.15% | 76.21% |
| h_new | all | 5-8 | right | 2038 | 0.38938 | -0.01261 | 0.99052 | 76.15% | 76.21% |
| h_new | nonturn | 5-8 | forward | 1841 | 0.26737 | -0.01922 | 0.97071 | 76.70% | 76.76% |
| h_new | nonturn | 5-8 | right | 1841 | 0.38111 | -0.01051 | 0.99052 | 76.70% | 76.76% |
| h_new | turn | 5-8 | forward | 197 | 0.26477 | -0.04368 | 0.96811 | 71.07% | 70.92% |
| h_new | turn | 5-8 | right | 197 | 0.46398 | -0.03652 | 0.99020 | 71.07% | 70.92% |
| h_new | all | 9-12 | forward | 1681 | 0.28878 | -0.00911 | 0.95333 | 69.84% | 69.88% |
| h_new | all | 9-12 | right | 1681 | 0.34513 | -0.01206 | 0.99050 | 69.84% | 69.88% |
| h_new | nonturn | 9-12 | forward | 1463 | 0.29178 | -0.00715 | 0.95375 | 69.99% | 70.09% |
| h_new | nonturn | 9-12 | right | 1463 | 0.33938 | -0.01326 | 0.99047 | 69.99% | 70.09% |
| h_new | turn | 9-12 | forward | 218 | 0.26440 | -0.03008 | 0.95046 | 68.81% | 67.84% |
| h_new | turn | 9-12 | right | 218 | 0.37734 | -0.01719 | 0.99050 | 68.81% | 67.84% |


## Test 2: background null


Criterion evaluated at σ=0.5, selected from the primary K=8 gate by passing status, then matched median, then miss rate. All widths are shown. K3D-int is a moving-block contrast and is not scored as a null. Null split source comments identify CPU rendering, while K3D-v/int validation uses NVIDIA-rendered source videos; all computation here is CPU.


| Split | Lift | σ | Clips | Node-steps | Skipped turns | Alarm /17 | Alarm /16 | Evidence-weighted speed | Null threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k3d_null_noblocks | tokens | 0.5 | 50 | 106403 | 691 | 4.43% | 4.71% | 1.01117 | False |
| k3d_null_noblocks | tokens | 1.0 | 50 | 106403 | 691 | 4.75% | 5.05% | 1.02528 | False |
| k3d_null_noblocks | tokens | 2.0 | 50 | 106403 | 691 | 5.12% | 5.44% | 1.04892 | False |
| k3d_null_noblocks | h_new | 0.5 | 50 | 106403 | 691 | 5.58% | 5.93% | 1.00393 | False |
| k3d_null_noblocks | h_new | 1.0 | 50 | 106403 | 691 | 6.08% | 6.46% | 1.02654 | False |
| k3d_null_noblocks | h_new | 2.0 | 50 | 106403 | 691 | 6.05% | 6.43% | 1.03259 | False |
| k3d_null_static | tokens | 0.5 | 50 | 106896 | 662 | 4.08% | 4.33% | 1.03680 | False |
| k3d_null_static | tokens | 1.0 | 50 | 106896 | 662 | 4.26% | 4.53% | 1.07295 | False |
| k3d_null_static | tokens | 2.0 | 50 | 106896 | 662 | 4.39% | 4.67% | 1.10960 | False |
| k3d_null_static | h_new | 0.5 | 50 | 106896 | 662 | 4.06% | 4.32% | 1.02069 | False |
| k3d_null_static | h_new | 1.0 | 50 | 106896 | 662 | 4.34% | 4.61% | 1.05888 | False |
| k3d_null_static | h_new | 2.0 | 50 | 106896 | 662 | 4.28% | 4.54% | 1.09139 | False |
| k3d_int_validation | tokens | 0.5 | 20 | 42772 | 264 | 6.01% | 6.38% | 1.07536 | not a null |
| k3d_int_validation | tokens | 1.0 | 20 | 42772 | 264 | 5.93% | 6.30% | 1.11992 | not a null |
| k3d_int_validation | tokens | 2.0 | 20 | 42772 | 264 | 5.60% | 5.95% | 1.18364 | not a null |
| k3d_int_validation | h_new | 0.5 | 20 | 42772 | 264 | 5.67% | 6.02% | 1.06933 | not a null |
| k3d_int_validation | h_new | 1.0 | 20 | 42772 | 264 | 5.82% | 6.18% | 1.12331 | not a null |
| k3d_int_validation | h_new | 2.0 | 20 | 42772 | 264 | 5.40% | 5.74% | 1.18036 | not a null |


### Static-block raw velocity distribution


Raw CSVs and quantiles cover the 16 moving nodes; the omitted pinned node is identically zero. Unfiltered estimates include unobserved nodes at their current lattice/offset values. `lock_evidence` selects normalized agreement≥0.1. The full empirical per-node samples are in `null_static_raw_velocities.csv`; quantiles for all three splits are in the distribution CSVs.


| Lift | σ | Variable | Subset | N | Min | p5 | Median | p95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 0.5 | vx | all | 100608 | -1.49987 | -1.11351 | -0.00500 | 1.03372 | 1.48136 |
| tokens | 0.5 | vx | lock_evidence | 4359 | -0.26860 | 0.69926 | 0.99448 | 1.17127 | 1.34498 |
| tokens | 0.5 | vy | all | 100608 | -1.49969 | -1.06168 | -0.00294 | 1.06222 | 1.49538 |
| tokens | 0.5 | vy | lock_evidence | 4359 | -1.37954 | -0.75818 | 0.00492 | 0.84522 | 1.37038 |
| tokens | 0.5 | speed | all | 100608 | 0.01405 | 0.54770 | 1.01147 | 1.42019 | 2.07010 |
| tokens | 0.5 | speed | lock_evidence | 4359 | 0.50165 | 0.80336 | 1.06917 | 1.32513 | 1.70533 |
| tokens | 1.0 | vx | all | 100608 | -1.49984 | -1.16186 | -0.01456 | 1.05700 | 1.49365 |
| tokens | 1.0 | vx | lock_evidence | 4556 | -0.49613 | 0.73347 | 0.99481 | 1.17828 | 1.37301 |
| tokens | 1.0 | vy | all | 100608 | -1.49943 | -1.08672 | 0.00429 | 1.09465 | 1.49985 |
| tokens | 1.0 | vy | lock_evidence | 4556 | -1.34943 | -0.79049 | 0.00324 | 0.84534 | 1.38734 |
| tokens | 1.0 | speed | all | 100608 | 0.00056 | 0.62373 | 1.04761 | 1.46878 | 2.09281 |
| tokens | 1.0 | speed | lock_evidence | 4556 | 0.32220 | 0.83150 | 1.07274 | 1.36463 | 1.84959 |
| tokens | 2.0 | vx | all | 100608 | -1.49995 | -1.16820 | -0.02240 | 1.10966 | 1.49990 |
| tokens | 2.0 | vx | lock_evidence | 4697 | -0.42184 | 0.76479 | 1.00869 | 1.20132 | 1.37893 |
| tokens | 2.0 | vy | all | 100608 | -1.49978 | -1.14705 | 0.00560 | 1.14981 | 1.49989 |
| tokens | 2.0 | vy | lock_evidence | 4697 | -1.38686 | -0.78905 | 0.00180 | 0.85091 | 1.35758 |
| tokens | 2.0 | speed | all | 100608 | 0.04631 | 0.72912 | 1.09913 | 1.50998 | 2.07349 |
| tokens | 2.0 | speed | lock_evidence | 4697 | 0.59099 | 0.87054 | 1.10134 | 1.38813 | 1.78936 |
| h_new | 0.5 | vx | all | 100608 | -1.49991 | -1.14626 | -0.00728 | 1.04022 | 1.46931 |
| h_new | 0.5 | vx | lock_evidence | 4344 | -0.08443 | 0.68335 | 0.96789 | 1.19051 | 1.37582 |
| h_new | 0.5 | vy | all | 100608 | -1.49926 | -1.06053 | -0.00271 | 1.06021 | 1.49795 |
| h_new | 0.5 | vy | lock_evidence | 4344 | -1.24982 | -0.72518 | -0.01021 | 0.71834 | 1.20330 |
| h_new | 0.5 | speed | all | 100608 | 0.00901 | 0.51175 | 0.99481 | 1.42686 | 2.04061 |
| h_new | 0.5 | speed | lock_evidence | 4344 | 0.30655 | 0.76268 | 1.03964 | 1.30640 | 1.65578 |
| h_new | 1.0 | vx | all | 100608 | -1.49999 | -1.19660 | -0.01516 | 1.07982 | 1.49954 |
| h_new | 1.0 | vx | lock_evidence | 4641 | 0.17489 | 0.71676 | 0.96834 | 1.20457 | 1.43153 |
| h_new | 1.0 | vy | all | 100608 | -1.49904 | -1.11811 | 0.00522 | 1.12384 | 1.49980 |
| h_new | 1.0 | vy | lock_evidence | 4641 | -1.31887 | -0.75474 | -0.00752 | 0.74933 | 1.31847 |
| h_new | 1.0 | speed | all | 100608 | 0.01688 | 0.65750 | 1.05225 | 1.48920 | 2.08827 |
| h_new | 1.0 | speed | lock_evidence | 4641 | 0.43688 | 0.80883 | 1.05413 | 1.32564 | 1.71188 |
| h_new | 2.0 | vx | all | 100608 | -1.49999 | -1.18734 | -0.02063 | 1.12760 | 1.49949 |
| h_new | 2.0 | vx | lock_evidence | 4570 | 0.16068 | 0.73435 | 0.98586 | 1.22855 | 1.38893 |
| h_new | 2.0 | vy | all | 100608 | -1.49997 | -1.16788 | 0.00880 | 1.17054 | 1.49962 |
| h_new | 2.0 | vy | lock_evidence | 4570 | -1.32337 | -0.77468 | -0.03265 | 0.75873 | 1.41159 |
| h_new | 2.0 | speed | all | 100608 | 0.01788 | 0.72694 | 1.10568 | 1.52104 | 2.09972 |
| h_new | 2.0 | speed | lock_evidence | 4570 | 0.56767 | 0.84549 | 1.07923 | 1.35120 | 1.66919 |


## Test 3: estimator gate


Primary criterion: matched median≤0.1 and misses≤30% for speed 0.25–1.0, nonturn steps, best σ. K=8 is the default; K=32 and h_new are reported as variants. Nearest-integer is an independent per-truth assignment to the five FloWM velocities, without uniqueness or evidence restrictions.


| Lift | K | σ | Median | p90 | Miss | False alarm | Criterion met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | 1.21972 | 1.78974 | 98.32% | 95.91% | False |
| tokens | 8 | 1.0 | 1.23093 | 1.82163 | 98.47% | 96.28% | False |
| tokens | 8 | 2.0 | 1.27773 | 1.86643 | 98.80% | 96.96% | False |
| tokens | 32 | 0.5 | 1.16769 | 1.77494 | 97.88% | 95.56% | False |
| tokens | 32 | 1.0 | 1.21891 | 1.81422 | 98.33% | 96.14% | False |
| tokens | 32 | 2.0 | 1.22311 | 1.84509 | 98.58% | 96.49% | False |
| h_new | 8 | 0.5 | 1.31744 | 1.79967 | 98.69% | 97.00% | False |
| h_new | 8 | 1.0 | 1.37520 | 1.83784 | 98.91% | 97.50% | False |
| h_new | 8 | 2.0 | 1.42116 | 1.89201 | 99.22% | 98.00% | False |
| h_new | 32 | 0.5 | 1.29704 | 1.79303 | 98.35% | 96.31% | False |
| h_new | 32 | 1.0 | 1.31345 | 1.80435 | 98.57% | 96.67% | False |
| h_new | 32 | 2.0 | 1.35194 | 1.85216 | 98.82% | 96.96% | False |


### literal; delay 0; nonturn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 25108 | 44990 | 4749 | 1.47691 | 2.41513 | 98.76% | 17355 | 95.91% |
| nearest_integer | 0 | 0 | all | 25108 | 44990 | 44990 | 0.46223 | 0.76245 | 41.96% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 25108 | 30097 | 2943 | 1.21972 | 1.78974 | 98.32% | 17355 | 95.91% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 25108 | 30097 | 30097 | 0.42849 | 0.61407 | 29.88% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 25108 | 9012 | 712 | 1.14899 | 1.48015 | 99.56% | 16 | 81.25% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 25108 | 9012 | 9012 | 0.36734 | 0.47218 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 25108 | 10256 | 1063 | 1.31687 | 1.71800 | 98.65% | 570 | 93.51% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 25108 | 10256 | 10256 | 0.49491 | 0.61681 | 49.13% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 25108 | 10829 | 1168 | 1.19840 | 1.89955 | 96.97% | 3801 | 96.45% |
| nearest_integer | 0 | 0 | 0.75-1 | 25108 | 10829 | 10829 | 0.40485 | 0.66452 | 36.52% | 0 | — |
| tokens | 8 | 1.0 | all | 25108 | 44990 | 4815 | 1.50992 | 2.42690 | 98.86% | 18161 | 96.28% |
| tokens | 8 | 1.0 | 0.25-1.0 | 25108 | 30097 | 2930 | 1.23093 | 1.82163 | 98.47% | 18161 | 96.28% |
| tokens | 8 | 1.0 | 0.25-0.5 | 25108 | 9012 | 725 | 1.17322 | 1.51851 | 99.74% | 13 | 76.92% |
| tokens | 8 | 1.0 | 0.5-0.75 | 25108 | 10256 | 1022 | 1.33802 | 1.74328 | 98.90% | 275 | 94.55% |
| tokens | 8 | 1.0 | 0.75-1 | 25108 | 10829 | 1183 | 1.16821 | 1.92947 | 97.00% | 3885 | 96.22% |
| tokens | 8 | 2.0 | all | 25108 | 44990 | 4449 | 1.55343 | 2.48824 | 99.07% | 17950 | 96.96% |
| tokens | 8 | 2.0 | 0.25-1.0 | 25108 | 30097 | 2668 | 1.27773 | 1.86643 | 98.80% | 17950 | 96.96% |
| tokens | 8 | 2.0 | 0.25-0.5 | 25108 | 9012 | 643 | 1.22581 | 1.59435 | 99.88% | 2 | 50.00% |
| tokens | 8 | 2.0 | 0.5-0.75 | 25108 | 10256 | 934 | 1.38238 | 1.78553 | 99.10% | 54 | 85.19% |
| tokens | 8 | 2.0 | 0.75-1 | 25108 | 10829 | 1091 | 1.23581 | 1.97073 | 97.63% | 3189 | 96.05% |
| tokens | 32 | 0.5 | all | 25108 | 44990 | 5248 | 1.43923 | 2.36339 | 98.49% | 19301 | 95.56% |
| tokens | 32 | 0.5 | 0.25-1.0 | 25108 | 30097 | 3368 | 1.16769 | 1.77494 | 97.88% | 19301 | 95.56% |
| tokens | 32 | 0.5 | 0.25-0.5 | 25108 | 9012 | 848 | 1.16128 | 1.50936 | 99.57% | 18 | 38.89% |
| tokens | 32 | 0.5 | 0.5-0.75 | 25108 | 10256 | 1201 | 1.20481 | 1.70956 | 97.95% | 367 | 92.37% |
| tokens | 32 | 0.5 | 0.75-1 | 25108 | 10829 | 1319 | 1.12933 | 1.88848 | 96.41% | 4663 | 95.97% |
| tokens | 32 | 1.0 | all | 25108 | 44990 | 5167 | 1.46806 | 2.37721 | 98.75% | 19756 | 96.14% |
| tokens | 32 | 1.0 | 0.25-1.0 | 25108 | 30097 | 3297 | 1.21891 | 1.81422 | 98.33% | 19756 | 96.14% |
| tokens | 32 | 1.0 | 0.25-0.5 | 25108 | 9012 | 804 | 1.16705 | 1.50824 | 99.83% | 2 | 100.00% |
| tokens | 32 | 1.0 | 0.5-0.75 | 25108 | 10256 | 1169 | 1.29745 | 1.73416 | 98.58% | 57 | 92.98% |
| tokens | 32 | 1.0 | 0.75-1 | 25108 | 10829 | 1324 | 1.19635 | 1.92811 | 96.83% | 4198 | 95.71% |
| tokens | 32 | 2.0 | all | 25108 | 44990 | 4752 | 1.49555 | 2.43007 | 98.90% | 18903 | 96.49% |
| tokens | 32 | 2.0 | 0.25-1.0 | 25108 | 30097 | 3026 | 1.22311 | 1.84509 | 98.58% | 18903 | 96.49% |
| tokens | 32 | 2.0 | 0.25-0.5 | 25108 | 9012 | 731 | 1.20998 | 1.56265 | 99.89% | 2 | 100.00% |
| tokens | 32 | 2.0 | 0.5-0.75 | 25108 | 10256 | 1082 | 1.31426 | 1.77962 | 98.78% | 20 | 90.00% |
| tokens | 32 | 2.0 | 0.75-1 | 25108 | 10829 | 1213 | 1.15995 | 1.96929 | 97.30% | 3179 | 93.87% |
| h_new | 8 | 0.5 | all | 25108 | 44990 | 3906 | 1.59686 | 2.45868 | 99.09% | 16513 | 97.00% |
| h_new | 8 | 0.5 | 0.25-1.0 | 25108 | 30097 | 2337 | 1.31744 | 1.79967 | 98.69% | 16513 | 97.00% |
| h_new | 8 | 0.5 | 0.25-0.5 | 25108 | 9012 | 593 | 1.20194 | 1.48771 | 99.52% | 25 | 60.00% |
| h_new | 8 | 0.5 | 0.5-0.75 | 25108 | 10256 | 833 | 1.38569 | 1.72277 | 98.92% | 748 | 94.12% |
| h_new | 8 | 0.5 | 0.75-1 | 25108 | 10829 | 911 | 1.50210 | 1.92504 | 97.79% | 4874 | 96.92% |
| h_new | 8 | 1.0 | all | 25108 | 44990 | 3834 | 1.62889 | 2.45927 | 99.23% | 17413 | 97.50% |
| h_new | 8 | 1.0 | 0.25-1.0 | 25108 | 30097 | 2231 | 1.37520 | 1.83784 | 98.91% | 17413 | 97.50% |
| h_new | 8 | 1.0 | 0.25-0.5 | 25108 | 9012 | 536 | 1.22133 | 1.53324 | 99.77% | 14 | 64.29% |
| h_new | 8 | 1.0 | 0.5-0.75 | 25108 | 10256 | 818 | 1.41305 | 1.76330 | 99.07% | 285 | 94.74% |
| h_new | 8 | 1.0 | 0.75-1 | 25108 | 10829 | 877 | 1.55588 | 1.93515 | 98.04% | 4587 | 97.21% |
| h_new | 8 | 2.0 | all | 25108 | 44990 | 3330 | 1.66412 | 2.51243 | 99.44% | 16431 | 98.00% |
| h_new | 8 | 2.0 | 0.25-1.0 | 25108 | 30097 | 1954 | 1.42116 | 1.89201 | 99.22% | 16431 | 98.00% |
| h_new | 8 | 2.0 | 0.25-0.5 | 25108 | 9012 | 480 | 1.29133 | 1.59720 | 99.90% | 7 | 71.43% |
| h_new | 8 | 2.0 | 0.5-0.75 | 25108 | 10256 | 693 | 1.46660 | 1.79580 | 99.40% | 88 | 87.50% |
| h_new | 8 | 2.0 | 0.75-1 | 25108 | 10829 | 781 | 1.61804 | 2.01840 | 98.48% | 3430 | 96.91% |
| h_new | 32 | 0.5 | all | 25108 | 44990 | 5059 | 1.54647 | 2.41193 | 98.83% | 18737 | 96.31% |
| h_new | 32 | 0.5 | 0.25-1.0 | 25108 | 30097 | 3126 | 1.29704 | 1.79303 | 98.35% | 18737 | 96.31% |
| h_new | 32 | 0.5 | 0.25-0.5 | 25108 | 9012 | 807 | 1.20788 | 1.50563 | 99.51% | 8 | 50.00% |
| h_new | 32 | 0.5 | 0.5-0.75 | 25108 | 10256 | 1110 | 1.31951 | 1.69862 | 98.57% | 246 | 87.80% |
| h_new | 32 | 0.5 | 0.75-1 | 25108 | 10829 | 1209 | 1.51416 | 1.91633 | 97.18% | 5003 | 95.92% |
| h_new | 32 | 1.0 | all | 25108 | 44990 | 5069 | 1.57638 | 2.44661 | 99.00% | 19004 | 96.67% |
| h_new | 32 | 1.0 | 0.25-1.0 | 25108 | 30097 | 3087 | 1.31345 | 1.80435 | 98.57% | 19004 | 96.67% |
| h_new | 32 | 1.0 | 0.25-0.5 | 25108 | 9012 | 786 | 1.21652 | 1.52496 | 99.78% | 2 | 50.00% |
| h_new | 32 | 1.0 | 0.5-0.75 | 25108 | 10256 | 1103 | 1.37505 | 1.74090 | 98.78% | 82 | 80.49% |
| h_new | 32 | 1.0 | 0.75-1 | 25108 | 10829 | 1198 | 1.47181 | 1.92393 | 97.36% | 4834 | 95.97% |
| h_new | 32 | 2.0 | all | 25108 | 44990 | 4545 | 1.59888 | 2.46288 | 99.14% | 17522 | 96.96% |
| h_new | 32 | 2.0 | 0.25-1.0 | 25108 | 30097 | 2806 | 1.35194 | 1.85216 | 98.82% | 17522 | 96.96% |
| h_new | 32 | 2.0 | 0.25-0.5 | 25108 | 9012 | 710 | 1.24801 | 1.57573 | 99.87% | 8 | 25.00% |
| h_new | 32 | 2.0 | 0.5-0.75 | 25108 | 10256 | 998 | 1.40956 | 1.77669 | 98.97% | 32 | 50.00% |
| h_new | 32 | 2.0 | 0.75-1 | 25108 | 10829 | 1098 | 1.48210 | 1.97164 | 97.82% | 3643 | 95.14% |


### literal; delay 0; turn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1470 | 100.00% |
| nearest_integer | 0 | 0 | all | 2692 | 13 | 13 | 0.28006 | 0.56744 | 23.08% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1470 | 100.00% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 2692 | 13 | 13 | 0.28006 | 0.56744 | 23.08% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 3 | 100.00% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 2692 | 3 | 3 | 0.42267 | 0.43987 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 72 | 100.00% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 2692 | 3 | 3 | 0.55018 | 0.56744 | 66.67% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 403 | 100.00% |
| nearest_integer | 0 | 0 | 0.75-1 | 2692 | 7 | 7 | 0.21157 | 0.44894 | 14.29% | 0 | — |
| tokens | 8 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1500 | 100.00% |
| tokens | 8 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1500 | 100.00% |
| tokens | 8 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 3 | 100.00% |
| tokens | 8 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 14 | 100.00% |
| tokens | 8 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 415 | 100.00% |
| tokens | 8 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1474 | 100.00% |
| tokens | 8 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1474 | 100.00% |
| tokens | 8 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 8 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 4 | 100.00% |
| tokens | 8 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 285 | 100.00% |
| tokens | 32 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1508 | 100.00% |
| tokens | 32 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1508 | 100.00% |
| tokens | 32 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 29 | 100.00% |
| tokens | 32 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 419 | 100.00% |
| tokens | 32 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1574 | 100.00% |
| tokens | 32 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1574 | 100.00% |
| tokens | 32 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 376 | 100.00% |
| tokens | 32 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1530 | 100.00% |
| tokens | 32 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1530 | 100.00% |
| tokens | 32 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 245 | 100.00% |
| h_new | 8 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1436 | 100.00% |
| h_new | 8 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1436 | 100.00% |
| h_new | 8 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 79 | 100.00% |
| h_new | 8 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 461 | 100.00% |
| h_new | 8 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1477 | 100.00% |
| h_new | 8 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1477 | 100.00% |
| h_new | 8 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 7 | 100.00% |
| h_new | 8 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 418 | 100.00% |
| h_new | 8 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1426 | 100.00% |
| h_new | 8 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1426 | 100.00% |
| h_new | 8 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 1 | 100.00% |
| h_new | 8 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 319 | 100.00% |
| h_new | 32 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1396 | 100.00% |
| h_new | 32 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1396 | 100.00% |
| h_new | 32 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 9 | 100.00% |
| h_new | 32 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 414 | 100.00% |
| h_new | 32 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1406 | 100.00% |
| h_new | 32 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1406 | 100.00% |
| h_new | 32 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 351 | 100.00% |
| h_new | 32 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1311 | 100.00% |
| h_new | 32 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1311 | 100.00% |
| h_new | 32 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 1 | 100.00% |
| h_new | 32 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 189 | 100.00% |


### literal; delay 1; nonturn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 22963 | 39652 | 4104 | 1.40571 | 2.39517 | 98.60% | 15871 | 95.54% |
| nearest_integer | 0 | 0 | all | 22963 | 39652 | 39652 | 0.46243 | 0.76245 | 42.05% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 22963 | 26560 | 2612 | 1.14256 | 1.78262 | 98.18% | 15871 | 95.54% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 22963 | 26560 | 26560 | 0.42856 | 0.61407 | 29.91% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 22963 | 7921 | 663 | 1.12084 | 1.47869 | 99.44% | 14 | 71.43% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 22963 | 7921 | 7921 | 0.36763 | 0.47218 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 22963 | 9069 | 922 | 1.19545 | 1.71047 | 98.46% | 501 | 92.61% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 22963 | 9069 | 9069 | 0.49491 | 0.61680 | 49.05% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 22963 | 9570 | 1027 | 1.10680 | 1.89837 | 96.87% | 3409 | 95.57% |
| nearest_integer | 0 | 0 | 0.75-1 | 22963 | 9570 | 9570 | 0.40485 | 0.66438 | 36.53% | 0 | — |
| tokens | 8 | 1.0 | all | 22963 | 39652 | 4196 | 1.42047 | 2.40375 | 98.73% | 16654 | 95.93% |
| tokens | 8 | 1.0 | 0.25-1.0 | 22963 | 26560 | 2634 | 1.14789 | 1.80459 | 98.40% | 16654 | 95.93% |
| tokens | 8 | 1.0 | 0.25-0.5 | 22963 | 7921 | 667 | 1.13874 | 1.50616 | 99.70% | 10 | 70.00% |
| tokens | 8 | 1.0 | 0.5-0.75 | 22963 | 9069 | 915 | 1.20482 | 1.72053 | 98.84% | 261 | 92.72% |
| tokens | 8 | 1.0 | 0.75-1 | 22963 | 9570 | 1052 | 1.08415 | 1.91371 | 96.91% | 3488 | 95.70% |
| tokens | 8 | 2.0 | all | 22963 | 39652 | 3855 | 1.46624 | 2.45672 | 98.94% | 16476 | 96.64% |
| tokens | 8 | 2.0 | 0.25-1.0 | 22963 | 26560 | 2381 | 1.20152 | 1.85918 | 98.70% | 16476 | 96.64% |
| tokens | 8 | 2.0 | 0.25-0.5 | 22963 | 7921 | 582 | 1.18519 | 1.59235 | 99.82% | 2 | 50.00% |
| tokens | 8 | 2.0 | 0.5-0.75 | 22963 | 9069 | 840 | 1.25735 | 1.77539 | 99.00% | 51 | 82.35% |
| tokens | 8 | 2.0 | 0.75-1 | 22963 | 9570 | 959 | 1.16388 | 1.97421 | 97.50% | 2913 | 95.57% |
| tokens | 32 | 0.5 | all | 22963 | 39652 | 4588 | 1.33660 | 2.33847 | 98.35% | 17757 | 95.30% |
| tokens | 32 | 0.5 | 0.25-1.0 | 22963 | 26560 | 3025 | 1.07310 | 1.75915 | 97.82% | 17757 | 95.30% |
| tokens | 32 | 0.5 | 0.25-0.5 | 22963 | 7921 | 799 | 1.13080 | 1.49543 | 99.50% | 18 | 44.44% |
| tokens | 32 | 0.5 | 0.5-0.75 | 22963 | 9069 | 1079 | 1.05321 | 1.68926 | 97.84% | 337 | 92.28% |
| tokens | 32 | 0.5 | 0.75-1 | 22963 | 9570 | 1147 | 0.98869 | 1.88388 | 96.43% | 4235 | 95.49% |
| tokens | 32 | 1.0 | all | 22963 | 39652 | 4506 | 1.38023 | 2.34662 | 98.60% | 18161 | 95.89% |
| tokens | 32 | 1.0 | 0.25-1.0 | 22963 | 26560 | 2948 | 1.13188 | 1.78295 | 98.21% | 18161 | 95.89% |
| tokens | 32 | 1.0 | 0.25-0.5 | 22963 | 7921 | 763 | 1.13900 | 1.50004 | 99.75% | 2 | 100.00% |
| tokens | 32 | 1.0 | 0.5-0.75 | 22963 | 9069 | 1045 | 1.15999 | 1.70789 | 98.48% | 57 | 91.23% |
| tokens | 32 | 1.0 | 0.75-1 | 22963 | 9570 | 1140 | 1.03502 | 1.90565 | 96.69% | 3829 | 95.25% |
| tokens | 32 | 2.0 | all | 22963 | 39652 | 4140 | 1.40307 | 2.39891 | 98.84% | 17360 | 96.33% |
| tokens | 32 | 2.0 | 0.25-1.0 | 22963 | 26560 | 2723 | 1.14985 | 1.82471 | 98.57% | 17360 | 96.33% |
| tokens | 32 | 2.0 | 0.25-0.5 | 22963 | 7921 | 687 | 1.17870 | 1.56757 | 99.87% | 2 | 100.00% |
| tokens | 32 | 2.0 | 0.5-0.75 | 22963 | 9069 | 974 | 1.20577 | 1.76196 | 98.80% | 20 | 95.00% |
| tokens | 32 | 2.0 | 0.75-1 | 22963 | 9570 | 1062 | 1.01963 | 1.96974 | 97.28% | 2938 | 93.60% |
| h_new | 8 | 0.5 | all | 22963 | 39652 | 3357 | 1.50993 | 2.44934 | 98.98% | 15071 | 96.71% |
| h_new | 8 | 0.5 | 0.25-1.0 | 22963 | 26560 | 2085 | 1.23216 | 1.78847 | 98.58% | 15071 | 96.71% |
| h_new | 8 | 0.5 | 0.25-0.5 | 22963 | 7921 | 547 | 1.13777 | 1.47214 | 99.41% | 25 | 68.00% |
| h_new | 8 | 0.5 | 0.5-0.75 | 22963 | 9069 | 733 | 1.32742 | 1.70982 | 98.71% | 669 | 92.97% |
| h_new | 8 | 0.5 | 0.75-1 | 22963 | 9570 | 805 | 1.38407 | 1.91461 | 97.78% | 4432 | 96.62% |
| h_new | 8 | 1.0 | all | 22963 | 39652 | 3293 | 1.55137 | 2.45735 | 99.12% | 15929 | 97.14% |
| h_new | 8 | 1.0 | 0.25-1.0 | 22963 | 26560 | 1993 | 1.28774 | 1.79967 | 98.78% | 15929 | 97.14% |
| h_new | 8 | 1.0 | 0.25-0.5 | 22963 | 7921 | 529 | 1.18244 | 1.52131 | 99.66% | 14 | 64.29% |
| h_new | 8 | 1.0 | 0.5-0.75 | 22963 | 9069 | 701 | 1.35946 | 1.73251 | 98.85% | 279 | 93.55% |
| h_new | 8 | 1.0 | 0.75-1 | 22963 | 9570 | 763 | 1.47218 | 1.91087 | 97.98% | 4174 | 96.69% |
| h_new | 8 | 2.0 | all | 22963 | 39652 | 2837 | 1.60968 | 2.50703 | 99.34% | 14992 | 97.68% |
| h_new | 8 | 2.0 | 0.25-1.0 | 22963 | 26560 | 1729 | 1.36497 | 1.87354 | 99.09% | 14992 | 97.68% |
| h_new | 8 | 2.0 | 0.25-0.5 | 22963 | 7921 | 436 | 1.26371 | 1.58725 | 99.86% | 7 | 71.43% |
| h_new | 8 | 2.0 | 0.5-0.75 | 22963 | 9069 | 614 | 1.41842 | 1.78132 | 99.25% | 87 | 86.21% |
| h_new | 8 | 2.0 | 0.75-1 | 22963 | 9570 | 679 | 1.51976 | 1.99779 | 98.30% | 3125 | 96.45% |
| h_new | 32 | 0.5 | all | 22963 | 39652 | 4387 | 1.47414 | 2.39620 | 98.69% | 17290 | 95.99% |
| h_new | 32 | 0.5 | 0.25-1.0 | 22963 | 26560 | 2814 | 1.23282 | 1.78247 | 98.21% | 17290 | 95.99% |
| h_new | 32 | 0.5 | 0.25-0.5 | 22963 | 7921 | 738 | 1.15099 | 1.49008 | 99.50% | 8 | 50.00% |
| h_new | 32 | 0.5 | 0.5-0.75 | 22963 | 9069 | 1007 | 1.23913 | 1.69275 | 98.30% | 236 | 87.71% |
| h_new | 32 | 0.5 | 0.75-1 | 22963 | 9570 | 1069 | 1.41290 | 1.91114 | 97.05% | 4589 | 95.51% |
| h_new | 32 | 1.0 | all | 22963 | 39652 | 4361 | 1.48180 | 2.42198 | 98.86% | 17568 | 96.40% |
| h_new | 32 | 1.0 | 0.25-1.0 | 22963 | 26560 | 2793 | 1.23117 | 1.78721 | 98.42% | 17568 | 96.40% |
| h_new | 32 | 1.0 | 0.25-0.5 | 22963 | 7921 | 752 | 1.15637 | 1.51706 | 99.75% | 2 | 50.00% |
| h_new | 32 | 1.0 | 0.5-0.75 | 22963 | 9069 | 983 | 1.29184 | 1.71847 | 98.57% | 82 | 80.49% |
| h_new | 32 | 1.0 | 0.75-1 | 22963 | 9570 | 1058 | 1.34852 | 1.92393 | 97.19% | 4488 | 95.43% |
| h_new | 32 | 2.0 | all | 22963 | 39652 | 3935 | 1.51520 | 2.44815 | 99.01% | 16167 | 96.61% |
| h_new | 32 | 2.0 | 0.25-1.0 | 22963 | 26560 | 2529 | 1.26269 | 1.83670 | 98.69% | 16167 | 96.61% |
| h_new | 32 | 2.0 | 0.25-0.5 | 22963 | 7921 | 674 | 1.19018 | 1.56482 | 99.79% | 8 | 25.00% |
| h_new | 32 | 2.0 | 0.5-0.75 | 22963 | 9069 | 903 | 1.33943 | 1.77489 | 98.77% | 31 | 45.16% |
| h_new | 32 | 2.0 | 0.75-1 | 22963 | 9570 | 952 | 1.37398 | 1.95507 | 97.70% | 3447 | 94.31% |


### literal; delay 1; turn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 4637 | 5114 | 481 | 1.15312 | 2.00456 | 98.73% | 2791 | 95.99% |
| nearest_integer | 0 | 0 | all | 4637 | 5114 | 5114 | 0.45751 | 0.75055 | 40.50% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 4637 | 3465 | 198 | 1.02262 | 1.63074 | 99.13% | 2791 | 95.99% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 4637 | 3465 | 3465 | 0.42472 | 0.61469 | 29.44% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 4637 | 1070 | 45 | 0.97705 | 1.41460 | 99.91% | 5 | 80.00% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 4637 | 1070 | 1070 | 0.36536 | 0.47029 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 4637 | 1164 | 69 | 0.95213 | 1.54940 | 98.80% | 136 | 92.65% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 4637 | 1164 | 1164 | 0.49744 | 0.61860 | 49.74% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 4637 | 1231 | 84 | 1.08282 | 1.76738 | 98.78% | 766 | 95.69% |
| nearest_integer | 0 | 0 | 0.75-1 | 4637 | 1231 | 1231 | 0.39987 | 0.66724 | 35.82% | 0 | — |
| tokens | 8 | 1.0 | all | 4637 | 5114 | 482 | 1.15350 | 1.94569 | 98.47% | 2827 | 95.61% |
| tokens | 8 | 1.0 | 0.25-1.0 | 4637 | 3465 | 205 | 1.06556 | 1.66937 | 98.93% | 2827 | 95.61% |
| tokens | 8 | 1.0 | 0.25-0.5 | 4637 | 1070 | 50 | 1.03372 | 1.43393 | 99.81% | 6 | 83.33% |
| tokens | 8 | 1.0 | 0.5-0.75 | 4637 | 1164 | 69 | 1.03865 | 1.50095 | 98.97% | 26 | 88.46% |
| tokens | 8 | 1.0 | 0.75-1 | 4637 | 1231 | 86 | 1.08806 | 1.84297 | 98.13% | 775 | 94.58% |
| tokens | 8 | 2.0 | all | 4637 | 5114 | 486 | 1.16687 | 1.95528 | 98.55% | 2782 | 95.90% |
| tokens | 8 | 2.0 | 0.25-1.0 | 4637 | 3465 | 199 | 1.12765 | 1.75027 | 99.13% | 2782 | 95.90% |
| tokens | 8 | 2.0 | 0.25-0.5 | 4637 | 1070 | 48 | 1.13676 | 1.48673 | 99.81% | 0 | — |
| tokens | 8 | 2.0 | 0.5-0.75 | 4637 | 1164 | 65 | 1.05526 | 1.57725 | 99.23% | 7 | 85.71% |
| tokens | 8 | 2.0 | 0.75-1 | 4637 | 1231 | 86 | 1.15049 | 1.84948 | 98.46% | 534 | 95.69% |
| tokens | 32 | 0.5 | all | 4637 | 5114 | 492 | 1.11483 | 1.97972 | 98.42% | 2856 | 95.73% |
| tokens | 32 | 0.5 | 0.25-1.0 | 4637 | 3465 | 204 | 0.98332 | 1.72144 | 98.76% | 2856 | 95.73% |
| tokens | 32 | 0.5 | 0.25-0.5 | 4637 | 1070 | 48 | 1.00028 | 1.37753 | 99.72% | 0 | — |
| tokens | 32 | 0.5 | 0.5-0.75 | 4637 | 1164 | 68 | 0.92058 | 1.55037 | 98.63% | 56 | 92.86% |
| tokens | 32 | 0.5 | 0.75-1 | 4637 | 1231 | 88 | 0.96203 | 1.90211 | 98.05% | 796 | 94.47% |
| tokens | 32 | 1.0 | all | 4637 | 5114 | 489 | 1.17250 | 1.93789 | 98.32% | 2966 | 95.85% |
| tokens | 32 | 1.0 | 0.25-1.0 | 4637 | 3465 | 202 | 1.00685 | 1.73999 | 98.85% | 2966 | 95.85% |
| tokens | 32 | 1.0 | 0.25-0.5 | 4637 | 1070 | 48 | 1.01778 | 1.37735 | 99.72% | 0 | — |
| tokens | 32 | 1.0 | 0.5-0.75 | 4637 | 1164 | 67 | 0.99882 | 1.59491 | 99.05% | 0 | — |
| tokens | 32 | 1.0 | 0.75-1 | 4637 | 1231 | 87 | 1.09749 | 1.85325 | 97.89% | 709 | 95.49% |
| tokens | 32 | 2.0 | all | 4637 | 5114 | 495 | 1.21241 | 2.01177 | 98.65% | 2883 | 95.91% |
| tokens | 32 | 2.0 | 0.25-1.0 | 4637 | 3465 | 208 | 1.18385 | 1.70739 | 99.31% | 2883 | 95.91% |
| tokens | 32 | 2.0 | 0.25-0.5 | 4637 | 1070 | 49 | 1.18385 | 1.43408 | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.5-0.75 | 4637 | 1164 | 68 | 1.22633 | 1.63595 | 99.57% | 0 | — |
| tokens | 32 | 2.0 | 0.75-1 | 4637 | 1231 | 91 | 1.02966 | 1.93469 | 98.46% | 462 | 95.67% |
| h_new | 8 | 0.5 | all | 4637 | 5114 | 471 | 1.13770 | 1.87139 | 98.53% | 2720 | 95.74% |
| h_new | 8 | 0.5 | 0.25-1.0 | 4637 | 3465 | 199 | 0.93995 | 1.61191 | 98.76% | 2720 | 95.74% |
| h_new | 8 | 0.5 | 0.25-0.5 | 4637 | 1070 | 47 | 0.92870 | 1.28802 | 99.35% | 0 | — |
| h_new | 8 | 0.5 | 0.5-0.75 | 4637 | 1164 | 67 | 0.91576 | 1.51514 | 98.63% | 152 | 93.42% |
| h_new | 8 | 0.5 | 0.75-1 | 4637 | 1231 | 85 | 0.95453 | 1.79120 | 98.38% | 870 | 95.40% |
| h_new | 8 | 1.0 | all | 4637 | 5114 | 471 | 1.15827 | 2.02258 | 98.49% | 2790 | 95.41% |
| h_new | 8 | 1.0 | 0.25-1.0 | 4637 | 3465 | 195 | 1.01093 | 1.67761 | 98.96% | 2790 | 95.41% |
| h_new | 8 | 1.0 | 0.25-0.5 | 4637 | 1070 | 46 | 1.01093 | 1.41437 | 99.72% | 0 | — |
| h_new | 8 | 1.0 | 0.5-0.75 | 4637 | 1164 | 66 | 1.03391 | 1.52533 | 98.97% | 12 | 83.33% |
| h_new | 8 | 1.0 | 0.75-1 | 4637 | 1231 | 83 | 0.92635 | 1.83934 | 98.29% | 788 | 93.65% |
| h_new | 8 | 2.0 | all | 4637 | 5114 | 470 | 1.18157 | 1.99979 | 98.67% | 2694 | 95.95% |
| h_new | 8 | 2.0 | 0.25-1.0 | 4637 | 3465 | 191 | 1.08201 | 1.73236 | 99.13% | 2694 | 95.95% |
| h_new | 8 | 2.0 | 0.25-0.5 | 4637 | 1070 | 46 | 1.03384 | 1.38421 | 99.81% | 0 | — |
| h_new | 8 | 2.0 | 0.5-0.75 | 4637 | 1164 | 66 | 1.11636 | 1.64049 | 99.14% | 2 | 100.00% |
| h_new | 8 | 2.0 | 0.75-1 | 4637 | 1231 | 79 | 1.05732 | 1.87047 | 98.54% | 596 | 94.30% |
| h_new | 32 | 0.5 | all | 4637 | 5114 | 450 | 1.11800 | 1.92697 | 98.65% | 2651 | 96.15% |
| h_new | 32 | 0.5 | 0.25-1.0 | 4637 | 3465 | 185 | 1.02435 | 1.65944 | 99.05% | 2651 | 96.15% |
| h_new | 32 | 0.5 | 0.25-0.5 | 4637 | 1070 | 44 | 0.99755 | 1.36334 | 99.72% | 0 | — |
| h_new | 32 | 0.5 | 0.5-0.75 | 4637 | 1164 | 63 | 0.97477 | 1.53820 | 99.05% | 18 | 94.44% |
| h_new | 32 | 0.5 | 0.75-1 | 4637 | 1231 | 78 | 1.04601 | 1.76651 | 98.46% | 786 | 96.31% |
| h_new | 32 | 1.0 | all | 4637 | 5114 | 445 | 1.15700 | 1.98194 | 98.61% | 2657 | 96.27% |
| h_new | 32 | 1.0 | 0.25-1.0 | 4637 | 3465 | 183 | 1.09671 | 1.69489 | 99.16% | 2657 | 96.27% |
| h_new | 32 | 1.0 | 0.25-0.5 | 4637 | 1070 | 42 | 1.02134 | 1.41517 | 99.81% | 0 | — |
| h_new | 32 | 1.0 | 0.5-0.75 | 4637 | 1164 | 61 | 1.09471 | 1.59885 | 99.05% | 0 | — |
| h_new | 32 | 1.0 | 0.75-1 | 4637 | 1231 | 80 | 1.12868 | 1.74396 | 98.70% | 659 | 94.84% |
| h_new | 32 | 2.0 | all | 4637 | 5114 | 450 | 1.20855 | 2.02347 | 98.79% | 2489 | 95.78% |
| h_new | 32 | 2.0 | 0.25-1.0 | 4637 | 3465 | 186 | 1.11159 | 1.69750 | 99.39% | 2489 | 95.78% |
| h_new | 32 | 2.0 | 0.25-0.5 | 4637 | 1070 | 45 | 1.11159 | 1.45227 | 100.00% | 0 | — |
| h_new | 32 | 2.0 | 0.5-0.75 | 4637 | 1164 | 64 | 1.23109 | 1.66816 | 99.31% | 2 | 100.00% |
| h_new | 32 | 2.0 | 0.75-1 | 4637 | 1231 | 77 | 0.97167 | 1.86590 | 98.94% | 355 | 95.21% |


### intrinsic_incoming; delay 0; nonturn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 25108 | 44990 | 4749 | 0.71452 | 1.43506 | 96.93% | 17355 | 89.36% |
| nearest_integer | 0 | 0 | all | 25108 | 44990 | 44990 | 0.41527 | 0.60726 | 28.06% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 25108 | 44990 | 4749 | 0.71452 | 1.43506 | 96.93% | 17355 | 89.36% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 25108 | 44990 | 44990 | 0.41527 | 0.60726 | 28.06% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 25108 | 14738 | 1292 | 0.89419 | 1.32090 | 99.34% | 16 | 75.00% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 25108 | 14738 | 14738 | 0.35536 | 0.46841 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 25108 | 14612 | 1599 | 0.67829 | 1.47959 | 97.13% | 570 | 79.12% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 25108 | 14612 | 14612 | 0.49744 | 0.61662 | 49.47% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 25108 | 15640 | 1858 | 0.54490 | 1.55406 | 94.48% | 3801 | 88.21% |
| nearest_integer | 0 | 0 | 0.75-1 | 25108 | 15640 | 15640 | 0.38496 | 0.65607 | 34.51% | 0 | — |
| tokens | 8 | 1.0 | all | 25108 | 44990 | 4815 | 0.74607 | 1.45051 | 97.21% | 18161 | 90.22% |
| tokens | 8 | 1.0 | 0.25-1.0 | 25108 | 44990 | 4815 | 0.74607 | 1.45051 | 97.21% | 18161 | 90.22% |
| tokens | 8 | 1.0 | 0.25-0.5 | 25108 | 14738 | 1331 | 0.91885 | 1.33194 | 99.63% | 13 | 69.23% |
| tokens | 8 | 1.0 | 0.5-0.75 | 25108 | 14612 | 1589 | 0.73000 | 1.46605 | 97.50% | 275 | 80.73% |
| tokens | 8 | 1.0 | 0.75-1 | 25108 | 15640 | 1895 | 0.55563 | 1.59220 | 94.65% | 3885 | 86.69% |
| tokens | 8 | 2.0 | all | 25108 | 44990 | 4449 | 0.79418 | 1.50580 | 97.80% | 17950 | 92.45% |
| tokens | 8 | 2.0 | 0.25-1.0 | 25108 | 44990 | 4449 | 0.79418 | 1.50580 | 97.80% | 17950 | 92.45% |
| tokens | 8 | 2.0 | 0.25-0.5 | 25108 | 14738 | 1204 | 0.97965 | 1.40087 | 99.73% | 2 | 50.00% |
| tokens | 8 | 2.0 | 0.5-0.75 | 25108 | 14612 | 1498 | 0.77940 | 1.52346 | 98.11% | 54 | 74.07% |
| tokens | 8 | 2.0 | 0.75-1 | 25108 | 15640 | 1747 | 0.60270 | 1.65051 | 95.70% | 3189 | 85.54% |
| tokens | 32 | 0.5 | all | 25108 | 44990 | 5248 | 0.70734 | 1.41923 | 96.65% | 19301 | 89.73% |
| tokens | 32 | 0.5 | 0.25-1.0 | 25108 | 44990 | 5248 | 0.70734 | 1.41923 | 96.65% | 19301 | 89.73% |
| tokens | 32 | 0.5 | 0.25-0.5 | 25108 | 14738 | 1476 | 0.87706 | 1.32756 | 99.36% | 18 | 38.89% |
| tokens | 32 | 0.5 | 0.5-0.75 | 25108 | 14612 | 1760 | 0.67358 | 1.43413 | 96.74% | 367 | 82.56% |
| tokens | 32 | 0.5 | 0.75-1 | 25108 | 15640 | 2012 | 0.53372 | 1.57064 | 94.02% | 4663 | 88.81% |
| tokens | 32 | 1.0 | all | 25108 | 44990 | 5167 | 0.75044 | 1.44733 | 97.09% | 19756 | 90.64% |
| tokens | 32 | 1.0 | 0.25-1.0 | 25108 | 44990 | 5167 | 0.75044 | 1.44733 | 97.09% | 19756 | 90.64% |
| tokens | 32 | 1.0 | 0.25-0.5 | 25108 | 14738 | 1462 | 0.92617 | 1.34787 | 99.67% | 2 | 100.00% |
| tokens | 32 | 1.0 | 0.5-0.75 | 25108 | 14612 | 1732 | 0.71974 | 1.47769 | 97.32% | 57 | 75.44% |
| tokens | 32 | 1.0 | 0.75-1 | 25108 | 15640 | 1973 | 0.56245 | 1.55407 | 94.45% | 4198 | 86.18% |
| tokens | 32 | 2.0 | all | 25108 | 44990 | 4752 | 0.78932 | 1.50454 | 97.77% | 18903 | 92.69% |
| tokens | 32 | 2.0 | 0.25-1.0 | 25108 | 44990 | 4752 | 0.78932 | 1.50454 | 97.77% | 18903 | 92.69% |
| tokens | 32 | 2.0 | 0.25-0.5 | 25108 | 14738 | 1347 | 0.96098 | 1.41019 | 99.80% | 2 | 100.00% |
| tokens | 32 | 2.0 | 0.5-0.75 | 25108 | 14612 | 1595 | 0.76548 | 1.49876 | 97.99% | 20 | 80.00% |
| tokens | 32 | 2.0 | 0.75-1 | 25108 | 15640 | 1810 | 0.61303 | 1.64462 | 95.65% | 3179 | 85.09% |
| h_new | 8 | 0.5 | all | 25108 | 44990 | 3906 | 0.72811 | 1.45063 | 97.43% | 16513 | 90.66% |
| h_new | 8 | 0.5 | 0.25-1.0 | 25108 | 44990 | 3906 | 0.72811 | 1.45063 | 97.43% | 16513 | 90.66% |
| h_new | 8 | 0.5 | 0.25-0.5 | 25108 | 14738 | 1086 | 0.85697 | 1.28484 | 99.38% | 25 | 48.00% |
| h_new | 8 | 0.5 | 0.5-0.75 | 25108 | 14612 | 1287 | 0.70846 | 1.45821 | 97.52% | 748 | 83.16% |
| h_new | 8 | 0.5 | 0.75-1 | 25108 | 15640 | 1533 | 0.53725 | 1.63442 | 95.51% | 4874 | 88.90% |
| h_new | 8 | 1.0 | all | 25108 | 44990 | 3834 | 0.75373 | 1.47380 | 97.66% | 17413 | 91.44% |
| h_new | 8 | 1.0 | 0.25-1.0 | 25108 | 44990 | 3834 | 0.75373 | 1.47380 | 97.66% | 17413 | 91.44% |
| h_new | 8 | 1.0 | 0.25-0.5 | 25108 | 14738 | 1088 | 0.90372 | 1.31530 | 99.46% | 14 | 57.14% |
| h_new | 8 | 1.0 | 0.5-0.75 | 25108 | 14612 | 1256 | 0.72334 | 1.48859 | 97.84% | 285 | 82.81% |
| h_new | 8 | 1.0 | 0.75-1 | 25108 | 15640 | 1490 | 0.57763 | 1.62274 | 95.81% | 4587 | 87.42% |
| h_new | 8 | 2.0 | all | 25108 | 44990 | 3330 | 0.79142 | 1.50569 | 98.27% | 16431 | 93.61% |
| h_new | 8 | 2.0 | 0.25-1.0 | 25108 | 44990 | 3330 | 0.79142 | 1.50569 | 98.27% | 16431 | 93.61% |
| h_new | 8 | 2.0 | 0.25-0.5 | 25108 | 14738 | 919 | 0.96952 | 1.37335 | 99.65% | 7 | 71.43% |
| h_new | 8 | 2.0 | 0.5-0.75 | 25108 | 14612 | 1083 | 0.78605 | 1.52741 | 98.67% | 88 | 80.68% |
| h_new | 8 | 2.0 | 0.75-1 | 25108 | 15640 | 1328 | 0.62396 | 1.65725 | 96.60% | 3430 | 87.96% |
| h_new | 32 | 0.5 | all | 25108 | 44990 | 5059 | 0.73391 | 1.44149 | 96.89% | 18737 | 89.19% |
| h_new | 32 | 0.5 | 0.25-1.0 | 25108 | 44990 | 5059 | 0.73391 | 1.44149 | 96.89% | 18737 | 89.19% |
| h_new | 32 | 0.5 | 0.25-0.5 | 25108 | 14738 | 1457 | 0.88465 | 1.28435 | 99.35% | 8 | 25.00% |
| h_new | 32 | 0.5 | 0.5-0.75 | 25108 | 14612 | 1722 | 0.69640 | 1.47072 | 96.95% | 246 | 69.51% |
| h_new | 32 | 0.5 | 0.75-1 | 25108 | 15640 | 1880 | 0.54851 | 1.59884 | 94.53% | 5003 | 86.07% |
| h_new | 32 | 1.0 | all | 25108 | 44990 | 5069 | 0.77420 | 1.48179 | 97.19% | 19004 | 89.95% |
| h_new | 32 | 1.0 | 0.25-1.0 | 25108 | 44990 | 5069 | 0.77420 | 1.48179 | 97.19% | 19004 | 89.95% |
| h_new | 32 | 1.0 | 0.25-0.5 | 25108 | 14738 | 1457 | 0.91000 | 1.33789 | 99.49% | 2 | 50.00% |
| h_new | 32 | 1.0 | 0.5-0.75 | 25108 | 14612 | 1724 | 0.73537 | 1.48447 | 97.26% | 82 | 68.29% |
| h_new | 32 | 1.0 | 0.75-1 | 25108 | 15640 | 1888 | 0.61024 | 1.63726 | 94.97% | 4834 | 85.35% |
| h_new | 32 | 2.0 | all | 25108 | 44990 | 4545 | 0.80652 | 1.51863 | 97.71% | 17522 | 91.45% |
| h_new | 32 | 2.0 | 0.25-1.0 | 25108 | 44990 | 4545 | 0.80652 | 1.51863 | 97.71% | 17522 | 91.45% |
| h_new | 32 | 2.0 | 0.25-0.5 | 25108 | 14738 | 1286 | 0.96026 | 1.39604 | 99.73% | 8 | 37.50% |
| h_new | 32 | 2.0 | 0.5-0.75 | 25108 | 14612 | 1509 | 0.77145 | 1.51664 | 97.83% | 32 | 40.62% |
| h_new | 32 | 2.0 | 0.75-1 | 25108 | 15640 | 1750 | 0.62971 | 1.64509 | 95.70% | 3643 | 84.13% |


### intrinsic_incoming; delay 0; turn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1470 | 100.00% |
| nearest_integer | 0 | 0 | all | 2692 | 13 | 13 | 0.28006 | 0.56744 | 23.08% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1470 | 100.00% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 2692 | 13 | 13 | 0.28006 | 0.56744 | 23.08% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 3 | 100.00% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 2692 | 3 | 3 | 0.42267 | 0.43987 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 72 | 100.00% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 2692 | 3 | 3 | 0.55018 | 0.56744 | 66.67% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 403 | 100.00% |
| nearest_integer | 0 | 0 | 0.75-1 | 2692 | 7 | 7 | 0.21157 | 0.44894 | 14.29% | 0 | — |
| tokens | 8 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1500 | 100.00% |
| tokens | 8 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1500 | 100.00% |
| tokens | 8 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 3 | 100.00% |
| tokens | 8 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 14 | 100.00% |
| tokens | 8 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 415 | 100.00% |
| tokens | 8 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1474 | 100.00% |
| tokens | 8 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1474 | 100.00% |
| tokens | 8 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 8 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 4 | 100.00% |
| tokens | 8 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 285 | 100.00% |
| tokens | 32 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1508 | 100.00% |
| tokens | 32 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1508 | 100.00% |
| tokens | 32 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 29 | 100.00% |
| tokens | 32 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 419 | 100.00% |
| tokens | 32 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1574 | 100.00% |
| tokens | 32 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1574 | 100.00% |
| tokens | 32 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 376 | 100.00% |
| tokens | 32 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1530 | 100.00% |
| tokens | 32 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1530 | 100.00% |
| tokens | 32 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 245 | 100.00% |
| h_new | 8 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1436 | 100.00% |
| h_new | 8 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1436 | 100.00% |
| h_new | 8 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 79 | 100.00% |
| h_new | 8 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 461 | 100.00% |
| h_new | 8 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1477 | 100.00% |
| h_new | 8 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1477 | 100.00% |
| h_new | 8 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 7 | 100.00% |
| h_new | 8 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 418 | 100.00% |
| h_new | 8 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1426 | 100.00% |
| h_new | 8 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1426 | 100.00% |
| h_new | 8 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 8 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 1 | 100.00% |
| h_new | 8 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 319 | 100.00% |
| h_new | 32 | 0.5 | all | 2692 | 13 | 0 | — | — | 100.00% | 1396 | 100.00% |
| h_new | 32 | 0.5 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1396 | 100.00% |
| h_new | 32 | 0.5 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 0.5 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 9 | 100.00% |
| h_new | 32 | 0.5 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 414 | 100.00% |
| h_new | 32 | 1.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1406 | 100.00% |
| h_new | 32 | 1.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1406 | 100.00% |
| h_new | 32 | 1.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 1.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 1.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 351 | 100.00% |
| h_new | 32 | 2.0 | all | 2692 | 13 | 0 | — | — | 100.00% | 1311 | 100.00% |
| h_new | 32 | 2.0 | 0.25-1.0 | 2692 | 13 | 0 | — | — | 100.00% | 1311 | 100.00% |
| h_new | 32 | 2.0 | 0.25-0.5 | 2692 | 3 | 0 | — | — | 100.00% | 0 | — |
| h_new | 32 | 2.0 | 0.5-0.75 | 2692 | 3 | 0 | — | — | 100.00% | 1 | 100.00% |
| h_new | 32 | 2.0 | 0.75-1 | 2692 | 7 | 0 | — | — | 100.00% | 189 | 100.00% |


### intrinsic_incoming; delay 1; nonturn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 22963 | 39652 | 4104 | 0.72759 | 1.44184 | 97.01% | 15871 | 90.01% |
| nearest_integer | 0 | 0 | all | 22963 | 39652 | 39652 | 0.41527 | 0.60761 | 28.07% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 22963 | 39652 | 4104 | 0.72759 | 1.44184 | 97.01% | 15871 | 90.01% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 22963 | 39652 | 39652 | 0.41527 | 0.60761 | 28.07% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 22963 | 12957 | 1111 | 0.89309 | 1.32478 | 99.36% | 14 | 64.29% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 22963 | 12957 | 12957 | 0.35536 | 0.46841 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 22963 | 12901 | 1386 | 0.68128 | 1.45162 | 97.17% | 501 | 79.44% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 22963 | 12901 | 12901 | 0.49690 | 0.61662 | 49.40% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 22963 | 13794 | 1607 | 0.55936 | 1.59242 | 94.65% | 3409 | 88.59% |
| nearest_integer | 0 | 0 | 0.75-1 | 22963 | 13794 | 13794 | 0.38562 | 0.65607 | 34.50% | 0 | — |
| tokens | 8 | 1.0 | all | 22963 | 39652 | 4196 | 0.75950 | 1.47228 | 97.34% | 16654 | 90.95% |
| tokens | 8 | 1.0 | 0.25-1.0 | 22963 | 39652 | 4196 | 0.75950 | 1.47228 | 97.34% | 16654 | 90.95% |
| tokens | 8 | 1.0 | 0.25-0.5 | 22963 | 12957 | 1163 | 0.91978 | 1.35219 | 99.61% | 10 | 80.00% |
| tokens | 8 | 1.0 | 0.5-0.75 | 22963 | 12901 | 1397 | 0.73000 | 1.46881 | 97.61% | 261 | 81.61% |
| tokens | 8 | 1.0 | 0.75-1 | 22963 | 13794 | 1636 | 0.56688 | 1.61440 | 94.95% | 3488 | 87.27% |
| tokens | 8 | 2.0 | all | 22963 | 39652 | 3855 | 0.80690 | 1.52383 | 97.85% | 16476 | 92.94% |
| tokens | 8 | 2.0 | 0.25-1.0 | 22963 | 39652 | 3855 | 0.80690 | 1.52383 | 97.85% | 16476 | 92.94% |
| tokens | 8 | 2.0 | 0.25-0.5 | 22963 | 12957 | 1016 | 0.98190 | 1.40066 | 99.69% | 2 | 50.00% |
| tokens | 8 | 2.0 | 0.5-0.75 | 22963 | 12901 | 1321 | 0.78830 | 1.52750 | 98.16% | 51 | 70.59% |
| tokens | 8 | 2.0 | 0.75-1 | 22963 | 13794 | 1518 | 0.61927 | 1.65345 | 95.82% | 2913 | 85.93% |
| tokens | 32 | 0.5 | all | 22963 | 39652 | 4588 | 0.72278 | 1.43691 | 96.81% | 17757 | 90.60% |
| tokens | 32 | 0.5 | 0.25-1.0 | 22963 | 39652 | 4588 | 0.72278 | 1.43691 | 96.81% | 17757 | 90.60% |
| tokens | 32 | 0.5 | 0.25-0.5 | 22963 | 12957 | 1280 | 0.87498 | 1.31020 | 99.35% | 18 | 33.33% |
| tokens | 32 | 0.5 | 0.5-0.75 | 22963 | 12901 | 1560 | 0.69304 | 1.46926 | 96.81% | 337 | 82.49% |
| tokens | 32 | 0.5 | 0.75-1 | 22963 | 13794 | 1748 | 0.55228 | 1.55637 | 94.43% | 4235 | 89.49% |
| tokens | 32 | 1.0 | all | 22963 | 39652 | 4506 | 0.76102 | 1.46667 | 97.27% | 18161 | 91.41% |
| tokens | 32 | 1.0 | 0.25-1.0 | 22963 | 39652 | 4506 | 0.76102 | 1.46667 | 97.27% | 18161 | 91.41% |
| tokens | 32 | 1.0 | 0.25-0.5 | 22963 | 12957 | 1273 | 0.93970 | 1.36137 | 99.68% | 2 | 100.00% |
| tokens | 32 | 1.0 | 0.5-0.75 | 22963 | 12901 | 1539 | 0.72859 | 1.47446 | 97.37% | 57 | 73.68% |
| tokens | 32 | 1.0 | 0.75-1 | 22963 | 13794 | 1694 | 0.58480 | 1.57547 | 94.90% | 3829 | 87.02% |
| tokens | 32 | 2.0 | all | 22963 | 39652 | 4140 | 0.80111 | 1.49747 | 97.88% | 17360 | 93.27% |
| tokens | 32 | 2.0 | 0.25-1.0 | 22963 | 39652 | 4140 | 0.80111 | 1.49747 | 97.88% | 17360 | 93.27% |
| tokens | 32 | 2.0 | 0.25-0.5 | 22963 | 12957 | 1182 | 0.96475 | 1.42037 | 99.81% | 2 | 100.00% |
| tokens | 32 | 2.0 | 0.5-0.75 | 22963 | 12901 | 1402 | 0.77298 | 1.50230 | 98.09% | 20 | 85.00% |
| tokens | 32 | 2.0 | 0.75-1 | 22963 | 13794 | 1556 | 0.62552 | 1.59297 | 95.89% | 2938 | 85.77% |
| h_new | 8 | 0.5 | all | 22963 | 39652 | 3357 | 0.73135 | 1.45198 | 97.49% | 15071 | 91.32% |
| h_new | 8 | 0.5 | 0.25-1.0 | 22963 | 39652 | 3357 | 0.73135 | 1.45198 | 97.49% | 15071 | 91.32% |
| h_new | 8 | 0.5 | 0.25-0.5 | 22963 | 12957 | 925 | 0.85980 | 1.28249 | 99.39% | 25 | 52.00% |
| h_new | 8 | 0.5 | 0.5-0.75 | 22963 | 12901 | 1130 | 0.70801 | 1.46945 | 97.46% | 669 | 83.41% |
| h_new | 8 | 0.5 | 0.75-1 | 22963 | 13794 | 1302 | 0.54460 | 1.63079 | 95.74% | 4432 | 89.53% |
| h_new | 8 | 1.0 | all | 22963 | 39652 | 3293 | 0.75920 | 1.47754 | 97.74% | 15929 | 92.12% |
| h_new | 8 | 1.0 | 0.25-1.0 | 22963 | 39652 | 3293 | 0.75920 | 1.47754 | 97.74% | 15929 | 92.12% |
| h_new | 8 | 1.0 | 0.25-0.5 | 22963 | 12957 | 913 | 0.90778 | 1.32662 | 99.46% | 14 | 57.14% |
| h_new | 8 | 1.0 | 0.5-0.75 | 22963 | 12901 | 1094 | 0.72354 | 1.48247 | 97.80% | 279 | 84.23% |
| h_new | 8 | 1.0 | 0.75-1 | 22963 | 13794 | 1286 | 0.59641 | 1.64632 | 96.06% | 4174 | 88.09% |
| h_new | 8 | 2.0 | all | 22963 | 39652 | 2837 | 0.79475 | 1.53903 | 98.35% | 14992 | 94.08% |
| h_new | 8 | 2.0 | 0.25-1.0 | 22963 | 39652 | 2837 | 0.79475 | 1.53903 | 98.35% | 14992 | 94.08% |
| h_new | 8 | 2.0 | 0.25-0.5 | 22963 | 12957 | 759 | 0.96913 | 1.38892 | 99.69% | 7 | 57.14% |
| h_new | 8 | 2.0 | 0.5-0.75 | 22963 | 12901 | 964 | 0.77806 | 1.54783 | 98.64% | 87 | 83.91% |
| h_new | 8 | 2.0 | 0.75-1 | 22963 | 13794 | 1114 | 0.62396 | 1.68626 | 96.82% | 3125 | 88.74% |
| h_new | 32 | 0.5 | all | 22963 | 39652 | 4387 | 0.74391 | 1.44842 | 97.00% | 17290 | 90.10% |
| h_new | 32 | 0.5 | 0.25-1.0 | 22963 | 39652 | 4387 | 0.74391 | 1.44842 | 97.00% | 17290 | 90.10% |
| h_new | 32 | 0.5 | 0.25-0.5 | 22963 | 12957 | 1258 | 0.88115 | 1.30125 | 99.34% | 8 | 37.50% |
| h_new | 32 | 0.5 | 0.5-0.75 | 22963 | 12901 | 1513 | 0.70622 | 1.47555 | 97.01% | 236 | 72.88% |
| h_new | 32 | 0.5 | 0.75-1 | 22963 | 13794 | 1616 | 0.55964 | 1.59900 | 94.81% | 4589 | 87.08% |
| h_new | 32 | 1.0 | all | 22963 | 39652 | 4361 | 0.78220 | 1.48178 | 97.27% | 17568 | 90.85% |
| h_new | 32 | 1.0 | 0.25-1.0 | 22963 | 39652 | 4361 | 0.78220 | 1.48178 | 97.27% | 17568 | 90.85% |
| h_new | 32 | 1.0 | 0.25-0.5 | 22963 | 12957 | 1251 | 0.91837 | 1.33994 | 99.49% | 2 | 50.00% |
| h_new | 32 | 1.0 | 0.5-0.75 | 22963 | 12901 | 1486 | 0.74103 | 1.48638 | 97.27% | 82 | 69.51% |
| h_new | 32 | 1.0 | 0.75-1 | 22963 | 13794 | 1624 | 0.61725 | 1.62867 | 95.19% | 4488 | 86.23% |
| h_new | 32 | 2.0 | all | 22963 | 39652 | 3935 | 0.80822 | 1.52694 | 97.75% | 16167 | 92.10% |
| h_new | 32 | 2.0 | 0.25-1.0 | 22963 | 39652 | 3935 | 0.80822 | 1.52694 | 97.75% | 16167 | 92.10% |
| h_new | 32 | 2.0 | 0.25-0.5 | 22963 | 12957 | 1091 | 0.95226 | 1.39616 | 99.68% | 8 | 25.00% |
| h_new | 32 | 2.0 | 0.5-0.75 | 22963 | 12901 | 1343 | 0.77585 | 1.54354 | 97.79% | 31 | 41.94% |
| h_new | 32 | 2.0 | 0.75-1 | 22963 | 13794 | 1501 | 0.64048 | 1.64847 | 95.89% | 3447 | 85.20% |


### intrinsic_incoming; delay 1; turn


| Lift/reference | K | σ | Speed bin | Steps | Truths | Assigned | Median | p90 | Miss | Nodes | False alarm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | all | 4637 | 5114 | 481 | 0.88802 | 1.55822 | 97.91% | 2791 | 93.69% |
| nearest_integer | 0 | 0 | all | 4637 | 5114 | 5114 | 0.41453 | 0.60544 | 27.92% | 0 | — |
| tokens | 8 | 0.5 | 0.25-1.0 | 4637 | 5114 | 481 | 0.88802 | 1.55822 | 97.91% | 2791 | 93.69% |
| nearest_integer | 0 | 0 | 0.25-1.0 | 4637 | 5114 | 5114 | 0.41453 | 0.60544 | 27.92% | 0 | — |
| tokens | 8 | 0.5 | 0.25-0.5 | 4637 | 1718 | 141 | 0.93090 | 1.32894 | 99.24% | 5 | 80.00% |
| nearest_integer | 0 | 0 | 0.25-0.5 | 4637 | 1718 | 1718 | 0.35536 | 0.46813 | 0.00% | 0 | — |
| tokens | 8 | 0.5 | 0.5-0.75 | 4637 | 1634 | 172 | 0.88132 | 1.55656 | 97.67% | 136 | 84.56% |
| nearest_integer | 0 | 0 | 0.5-0.75 | 4637 | 1634 | 1634 | 0.50026 | 0.61377 | 50.18% | 0 | — |
| tokens | 8 | 0.5 | 0.75-1 | 4637 | 1762 | 168 | 0.80288 | 1.70081 | 96.82% | 766 | 90.47% |
| nearest_integer | 0 | 0 | 0.75-1 | 4637 | 1762 | 1762 | 0.38200 | 0.65607 | 34.51% | 0 | — |
| tokens | 8 | 1.0 | all | 4637 | 5114 | 482 | 0.90492 | 1.58502 | 98.22% | 2827 | 94.52% |
| tokens | 8 | 1.0 | 0.25-1.0 | 4637 | 5114 | 482 | 0.90492 | 1.58502 | 98.22% | 2827 | 94.52% |
| tokens | 8 | 1.0 | 0.25-0.5 | 4637 | 1718 | 138 | 0.96040 | 1.31079 | 99.59% | 6 | 83.33% |
| tokens | 8 | 1.0 | 0.5-0.75 | 4637 | 1634 | 170 | 0.90408 | 1.50203 | 98.10% | 26 | 84.62% |
| tokens | 8 | 1.0 | 0.75-1 | 4637 | 1762 | 174 | 0.85536 | 1.80033 | 96.99% | 775 | 91.35% |
| tokens | 8 | 2.0 | all | 4637 | 5114 | 486 | 0.97402 | 1.64079 | 98.53% | 2782 | 95.90% |
| tokens | 8 | 2.0 | 0.25-1.0 | 4637 | 5114 | 486 | 0.97402 | 1.64079 | 98.53% | 2782 | 95.90% |
| tokens | 8 | 2.0 | 0.25-0.5 | 4637 | 1718 | 145 | 1.02632 | 1.37927 | 99.71% | 0 | — |
| tokens | 8 | 2.0 | 0.5-0.75 | 4637 | 1634 | 170 | 0.98743 | 1.53962 | 98.65% | 7 | 100.00% |
| tokens | 8 | 2.0 | 0.75-1 | 4637 | 1762 | 171 | 0.89213 | 1.84126 | 97.28% | 534 | 92.13% |
| tokens | 32 | 0.5 | all | 4637 | 5114 | 492 | 0.89960 | 1.56442 | 98.28% | 2856 | 94.64% |
| tokens | 32 | 0.5 | 0.25-1.0 | 4637 | 5114 | 492 | 0.89960 | 1.56442 | 98.28% | 2856 | 94.64% |
| tokens | 32 | 0.5 | 0.25-0.5 | 4637 | 1718 | 146 | 1.01185 | 1.33983 | 99.71% | 0 | — |
| tokens | 32 | 0.5 | 0.5-0.75 | 4637 | 1634 | 173 | 0.88981 | 1.47189 | 98.29% | 56 | 89.29% |
| tokens | 32 | 0.5 | 0.75-1 | 4637 | 1762 | 173 | 0.82352 | 1.78343 | 96.88% | 796 | 90.83% |
| tokens | 32 | 1.0 | all | 4637 | 5114 | 489 | 0.91849 | 1.59460 | 98.44% | 2966 | 95.38% |
| tokens | 32 | 1.0 | 0.25-1.0 | 4637 | 5114 | 489 | 0.91849 | 1.59460 | 98.44% | 2966 | 95.38% |
| tokens | 32 | 1.0 | 0.25-0.5 | 4637 | 1718 | 141 | 0.95473 | 1.34317 | 99.77% | 0 | — |
| tokens | 32 | 1.0 | 0.5-0.75 | 4637 | 1634 | 170 | 0.89575 | 1.51749 | 98.47% | 0 | — |
| tokens | 32 | 1.0 | 0.75-1 | 4637 | 1762 | 178 | 0.85845 | 1.75379 | 97.11% | 709 | 92.52% |
| tokens | 32 | 2.0 | all | 4637 | 5114 | 495 | 0.97786 | 1.64259 | 98.75% | 2883 | 96.18% |
| tokens | 32 | 2.0 | 0.25-1.0 | 4637 | 5114 | 495 | 0.97786 | 1.64259 | 98.75% | 2883 | 96.18% |
| tokens | 32 | 2.0 | 0.25-0.5 | 4637 | 1718 | 146 | 1.03486 | 1.40034 | 100.00% | 0 | — |
| tokens | 32 | 2.0 | 0.5-0.75 | 4637 | 1634 | 175 | 0.99041 | 1.57375 | 98.84% | 0 | — |
| tokens | 32 | 2.0 | 0.75-1 | 4637 | 1762 | 174 | 0.89280 | 1.80536 | 97.45% | 462 | 93.07% |
| h_new | 8 | 0.5 | all | 4637 | 5114 | 471 | 0.85771 | 1.49135 | 97.99% | 2720 | 94.01% |
| h_new | 8 | 0.5 | 0.25-1.0 | 4637 | 5114 | 471 | 0.85771 | 1.49135 | 97.99% | 2720 | 94.01% |
| h_new | 8 | 0.5 | 0.25-0.5 | 4637 | 1718 | 138 | 0.85769 | 1.27738 | 99.13% | 0 | — |
| h_new | 8 | 0.5 | 0.5-0.75 | 4637 | 1634 | 168 | 0.88842 | 1.44342 | 97.49% | 152 | 88.16% |
| h_new | 8 | 0.5 | 0.75-1 | 4637 | 1762 | 165 | 0.82977 | 1.68017 | 97.33% | 870 | 90.69% |
| h_new | 8 | 1.0 | all | 4637 | 5114 | 471 | 0.91377 | 1.56193 | 98.47% | 2790 | 94.98% |
| h_new | 8 | 1.0 | 0.25-1.0 | 4637 | 5114 | 471 | 0.91377 | 1.56193 | 98.47% | 2790 | 94.98% |
| h_new | 8 | 1.0 | 0.25-0.5 | 4637 | 1718 | 138 | 0.92733 | 1.31479 | 99.53% | 0 | — |
| h_new | 8 | 1.0 | 0.5-0.75 | 4637 | 1634 | 167 | 0.93582 | 1.50701 | 98.47% | 12 | 91.67% |
| h_new | 8 | 1.0 | 0.75-1 | 4637 | 1762 | 166 | 0.83572 | 1.74025 | 97.45% | 788 | 91.50% |
| h_new | 8 | 2.0 | all | 4637 | 5114 | 470 | 0.95824 | 1.60333 | 98.34% | 2694 | 95.10% |
| h_new | 8 | 2.0 | 0.25-1.0 | 4637 | 5114 | 470 | 0.95824 | 1.60333 | 98.34% | 2694 | 95.10% |
| h_new | 8 | 2.0 | 0.25-0.5 | 4637 | 1718 | 140 | 0.99785 | 1.38588 | 99.59% | 0 | — |
| h_new | 8 | 2.0 | 0.5-0.75 | 4637 | 1634 | 167 | 0.94955 | 1.57902 | 98.47% | 2 | 100.00% |
| h_new | 8 | 2.0 | 0.75-1 | 4637 | 1762 | 163 | 0.83603 | 1.81645 | 96.99% | 596 | 91.28% |
| h_new | 32 | 0.5 | all | 4637 | 5114 | 450 | 0.93698 | 1.56446 | 98.30% | 2651 | 94.42% |
| h_new | 32 | 0.5 | 0.25-1.0 | 4637 | 5114 | 450 | 0.93698 | 1.56446 | 98.30% | 2651 | 94.42% |
| h_new | 32 | 0.5 | 0.25-0.5 | 4637 | 1718 | 130 | 0.95544 | 1.32828 | 99.48% | 0 | — |
| h_new | 32 | 0.5 | 0.5-0.75 | 4637 | 1634 | 161 | 0.94905 | 1.51184 | 98.41% | 18 | 83.33% |
| h_new | 32 | 0.5 | 0.75-1 | 4637 | 1762 | 159 | 0.79202 | 1.76625 | 97.05% | 786 | 91.48% |
| h_new | 32 | 1.0 | all | 4637 | 5114 | 445 | 0.93559 | 1.59715 | 98.47% | 2657 | 95.18% |
| h_new | 32 | 1.0 | 0.25-1.0 | 4637 | 5114 | 445 | 0.93559 | 1.59715 | 98.47% | 2657 | 95.18% |
| h_new | 32 | 1.0 | 0.25-0.5 | 4637 | 1718 | 127 | 0.99275 | 1.34328 | 99.88% | 0 | — |
| h_new | 32 | 1.0 | 0.5-0.75 | 4637 | 1634 | 162 | 0.96005 | 1.57130 | 98.53% | 0 | — |
| h_new | 32 | 1.0 | 0.75-1 | 4637 | 1762 | 156 | 0.80553 | 1.76187 | 97.05% | 659 | 93.02% |
| h_new | 32 | 2.0 | all | 4637 | 5114 | 450 | 0.99443 | 1.61161 | 98.77% | 2489 | 95.42% |
| h_new | 32 | 2.0 | 0.25-1.0 | 4637 | 5114 | 450 | 0.99443 | 1.61161 | 98.77% | 2489 | 95.42% |
| h_new | 32 | 2.0 | 0.25-0.5 | 4637 | 1718 | 133 | 1.05710 | 1.43225 | 99.88% | 0 | — |
| h_new | 32 | 2.0 | 0.5-0.75 | 4637 | 1634 | 161 | 1.03171 | 1.56161 | 98.90% | 2 | 50.00% |
| h_new | 32 | 2.0 | 0.75-1 | 4637 | 1762 | 156 | 0.85831 | 1.78004 | 97.56% | 355 | 90.42% |


## Commands, runtime, artifacts and validation


Run from the repository root. `run_cpu.sh` fixes the supplied Python interpreter, CPU visibility, eight-thread limits, offline W&B, and scratch/cache paths before importing Python packages. Exact executed commands and their wall times are in `results/commands.csv`.


Original lift times below are sums of measured per-clip wall times, including decode and preprocessing. They remain recorded when a resumed command reuses caches; the short cached command times are not the original lift cost. Probe times include feature extraction and cross-validation. Campaign command times below include imports and CSV scoring.


| Stage | Clips | Measured seconds |
| --- | --- | --- |
| Original lift: k3d_int_validation | 20 | 152.48 |
| Original lift: k3d_v_validation | 200 | 1559.12 |
| Original lift: k3d_null_noblocks | 50 | 407.11 |
| Original lift: k3d_null_static | 50 | 387.09 |
| Nested probe: tokens | 200 | 23.59 |
| Nested probe: h_new | 200 | 23.36 |

| Command | Wall seconds | Exit status |
| --- | --- | --- |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh released_sanity | 117.219 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh geometry --clips 20 | 14.396 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_int_validation --clips 20 | 19.264 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_v_validation --clips 200 | 26.592 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh probe_subcell --clips 200 | 62.961 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_null_noblocks --clips 50 | 428.822 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_null_static --clips 50 | 413.045 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_null_noblocks --clips 50 | 361.450 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_null_static --clips 50 | 360.290 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_int_validation --clips 20 | 144.865 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh estimator_gate --clips 200 | 3821.520 | 0 |
| bash /home/yixunhu/codespace/flowm/experiments/flowm_bench/k3d/run_cpu.sh report | 11.068 | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh probe_subcell --clips 200 | 59.932 | 0 |

Initial runs, calibration and final checks are listed below with explicit timing bases. They are additional to the campaign commands; a sum of these values is not elapsed time because initial calibration briefly overlapped caching. Commands use the CPU/scratch environment shown in README.


| Command | Seconds | Timing basis | Exit status |
| --- | --- | --- | --- |
| /home/yixunhu/miniconda3/envs/flowm/bin/python -m experiments.flowm_bench.k3d.released_sanity | 95.1 | in-Python measured sanity time; initial run | 0 |
| /home/yixunhu/miniconda3/envs/flowm/bin/python -m experiments.flowm_bench.k3d.lift --split k3d_int_validation --clips 20 | 152.48 | sum of original per-clip times, excludes process imports | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_v_validation --clips 200 | 1559.12 | sum of original per-clip times, excludes process imports | 0 |
| /home/yixunhu/miniconda3/envs/flowm/bin/python -m experiments.flowm_bench.k3d.geometry --clips 20 | not timed | initial empirical sign check; timed repeat in commands.csv | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_int_validation --clips 1 --variants tokens --sigmas 1 | 4.52 | estimator batch only; calibration subsequently discarded | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_int_validation --clips 8 --variants tokens --sigmas 1 | 9.30 | seven new estimator clips; one cached; calibration subsequently discarded | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh estimator_gate --clips 8 --variants tokens --ks 32 --sigmas 1 | 18.17 | estimator batch only; calibration subsequently discarded | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh probe_subcell --clips 10 | not timed | smoke run; overwritten by full 200-clip run | 0 |
| bash experiments/flowm_bench/k3d/run_cpu.sh data_audit | not timed | 320-clip action/velocity identity audit | 0 |
| /home/yixunhu/miniconda3/envs/flowm/bin/python experiments/flowm_bench/k3d/.scratch/audit_kwm_revision.py | 3.240 | in-script dependency parity comparison; excludes imports | 0 |
| /home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/k3d/test_stage1.py -q -o cache_dir=experiments/flowm_bench/k3d/.scratch/pytest_cache | 11.53 | pytest-reported final duration | 0 |


### Dependency provenance audit


KWM changed externally from `0190d250f6d2c6a8bc0ebd77e92bb90ad3301083` to `2e73903d7ef5e6268cf6bca3a9703a3b9971ccf9` during this task. Its final estimator files were last written at 16:39:30 EST, before the first full null process launched at 16:41:41; the commit was recorded at 16:46:45. All final null and gate trajectories therefore used the latter file contents. `_fixed_projection` did not change. The added residual-mask mode was not used: `static_residual=True` remains subtraction. An independent comparison of the original and final measurement code on 12 real nonturn pairs (four each from moving, empty and static clips), random nonzero rho, both lift variants, both K and all σ produced bitwise-identical velocities, evidence and lock bits in all 144 pair/configuration cases. This does not establish equivalence for unused modes. Initial and final hashes are retained separately in `provenance.csv`. No KWM source was edited by this work.


| Lift | K | σ | Pairs | Bitwise equal | Max difference |
| --- | --- | --- | --- | --- | --- |
| tokens | 8 | 0.5 | 12 | True | 0.0 |
| tokens | 8 | 1.0 | 12 | True | 0.0 |
| tokens | 8 | 2.0 | 12 | True | 0.0 |
| tokens | 32 | 0.5 | 12 | True | 0.0 |
| tokens | 32 | 1.0 | 12 | True | 0.0 |
| tokens | 32 | 2.0 | 12 | True | 0.0 |
| h_new | 8 | 0.5 | 12 | True | 0.0 |
| h_new | 8 | 1.0 | 12 | True | 0.0 |
| h_new | 8 | 2.0 | 12 | True | 0.0 |
| h_new | 32 | 0.5 | 12 | True | 0.0 |
| h_new | 32 | 1.0 | 12 | True | 0.0 |
| h_new | 32 | 2.0 | 12 | True | 0.0 |


All per-clip measurements are CSV files under `k3d/results/`; binary lifts, trajectories, logs, probe predictions, and the proposed commit message are under `k3d/.scratch/`. README documents restart/resume and the meaning of every metric.



Additional execution notes are recorded in `k3d/results/execution_notes.csv`. Source/checkpoint hashes are in `k3d/results/provenance.csv`. The `tests/k3d/test_stage1.py` checks cover geometry and turn signs, empirical sign evidence, dataset alignment, five-channel lift identity, box round-trip and projections, Hungarian uniqueness/empty cases/cutoffs, static pinning, turn-state round trip and bitwise agreement-observer parity.


| Item | Detail |
| --- | --- |
| Initial calibration | Eight-clip sanity 95.1 s; lift B=32 about 7–8 s/clip; estimator K=8 B=7 9.30 s and K=32 B=8 18.17 s per full batch (139 pairs/clip). Brief calibration/test processes overlapped lift caching, with eight threads per process (up to 16 nominal Torch threads concurrently), to check throughput and correctness before the long campaign. The main campaign runs sequentially at eight threads. |
| Sampling | No planned clip or frame reduction: 200 K3D-v clips, 50 per null, 20 integer contrast, 140 frames each. Use 139 incoming measurement pairs because 1..139 is inclusive; delayed comparison uses 138 pairs. |
| Source field interpretation | Primary gate preserves the requested literal block_vel_agent target. Additional intrinsic_incoming diagnostic corrects both the outgoing/incoming index and subtraction of camera motion. |
| Temporary cache setup | Initial eager experiments-package imports briefly created automatically cleaned Matplotlib caches in /tmp. The run_cpu wrapper sets TMPDIR, XDG_CACHE_HOME and MPLCONFIGDIR before any Python imports; retained scratch files are under k3d/.scratch. |
| Working tree | Concurrent external commits changed FloWM HEAD from b4e31293 to 1b61c7f8. Released model/utils/base_video and checkpoint hashes stayed unchanged. Blockworld changes added opt-in load_block_info paths, unused by this adapter. External edits were neither made nor reverted by this work. |
| KWM | KWM changed externally from 0190d250 to 2e73903d during the run; all final estimator processes loaded the final files, written at 16:39:30 EST before their launches. No KWM source was edited by this work. All 144 original/final real pair/configuration comparisons were bitwise equal on the static_residual=True path. The API still lacks normalized LK agreement; the scoped observer was used. The refreshed, unapplied .scratch/kwm_patch.diff adds agreement to optional diagnostics. |
| Fast tests | Final rerun after dependency audit: 7 passed in 11.53 s; GPU hidden; empirical geometry evidence and bitwise agreement-observer parity included. |
| Ridge precision | The first full ridge run emitted float32 conditioning warnings. The final probe used float64 solves (59.932 s wall); cached lifts and KWM measurements remain fp32. Verdict unchanged. |
| Result retention | A scoped k3d/.gitignore overrides repository-wide results/ and *.csv exclusions for the requested CSV evidence. Scratch remains ignored. No git commit was attempted. |
| Elapsed time | First sanity launch 2026-11-04T15:51:44-05:00; final numerical/audit check 2026-11-04T18:07:54-05:00; 136.2 minutes elapsed. Core campaign ended at 18:00:00 EST. This elapsed interval includes implementation/reporting and overlaps; summed CPU-stage wall durations are separately recorded. |


<!-- followup:header -->
## Follow-up — 2026-11-04

CPU only, eight threads; all clips retain frames 0–139. Original sections above are unchanged.
Follow-up uses KWM `2e73903`, fixed K=8 unless stated, and cached fp16 lifts with fp32 measurements.
Primary gate truth is now **intrinsic_incoming**; literal outgoing camera-relative metadata is secondary.
Null alarm remains normalized LK agreement ≥.1 and immediate measured speed >.15, divided by all 17 node-steps.
Null PASS requires alarm ≤2% and evidence-weighted speed ≤.05 on both nulls at the gate-selected sigma.
Gate PASS requires Hungarian matched lower median ≤.1 and misses ≤30% in speeds .25–1 on nonturn pairs.
Evidence diagnostics retain every node, before and after mode selection, irrespective of locking or assignment.
Follow-up artifacts have the `followup_` prefix; trajectories are under `.scratch/followup_trajectories/`.
Results below are appended as each measurement completes. Fractions, including alarm/miss rates, are in [0,1].


<!-- followup:null-obs-k3d_null_noblocks-0.5 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_null_noblocks | 50 | 0.50000 | 6259 | 15904 | 0.14947 | 0.05135 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:null-obs-k3d_null_noblocks-1 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_null_noblocks | 50 | 1.00000 | 6259 | 15226 | 0.14310 | 0.01688 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:null-obs-k3d_null_static-0.5 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_null_static | 50 | 0.50000 | 6288 | 19367 | 0.18118 | 0.05158 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:null-obs-k3d_null_static-1 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_null_static | 50 | 1.00000 | 6288 | 20936 | 0.19585 | 0.01861 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:null-obs-k3d_int_validation-0.5 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_int_validation | 20 | 0.50000 | 2516 | 15935 | 0.37256 | 0.73875 | contrast |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:null-obs-k3d_int_validation-1 -->
### 1. Mask null / contrast — observation-only tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | k3d_int_validation | 20 | 1.00000 | 2516 | 15701 | 0.36709 | 0.43143 | contrast |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:gate-obs-0.5 -->
### 1. Mask estimator gate — observation-only tokens, σ=0.5

| source | truth_mode | speed_bin | truths | assigned | median | p90 | miss_rate | false_alarm_rate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | intrinsic_incoming | all | 44990 | 3536 | 0.41176 | 0.92604 | 0.95117 | 0.50106 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.25-1 | 44990 | 3536 | 0.41176 | 0.92604 | 0.95117 | 0.50106 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.25-0.5 | 14738 | 1162 | 0.45456 | 0.89903 | 0.95434 | 0.39541 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.5-0.75 | 14612 | 1213 | 0.40176 | 0.90723 | 0.94703 | 0.32870 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.75-1 | 15640 | 1161 | 0.37903 | 1.02851 | 0.95205 | 0.42242 | FAIL |
| observation-only tokens | literal | all | 44990 | 3536 | 0.66970 | 1.60480 | 0.97066 | 0.69602 | FAIL |
| observation-only tokens | literal | 0.25-1 | 30097 | 2579 | 0.51394 | 1.26530 | 0.95827 | 0.69602 | FAIL |
| observation-only tokens | literal | 0.25-0.5 | 9012 | 806 | 0.49080 | 1.10685 | 0.95417 | 0.56122 | FAIL |
| observation-only tokens | literal | 0.5-0.75 | 10256 | 915 | 0.49273 | 1.32055 | 0.95505 | 0.63272 | FAIL |
| observation-only tokens | literal | 0.75-1 | 10829 | 858 | 0.56609 | 1.43939 | 0.96472 | 0.67738 | FAIL |

Criterion applies to intrinsic_incoming, speed .25–1: median ≤.1 and miss ≤.30. Other rows are diagnostics.

All-node selectivity on nonturn steps with jointly visible blocks. `any_nearest_top` is the fraction of steps where at least one truth's nearest node has positive maximum evidence; `all_nearest_top` requires all truths' nearest nodes to tie that maximum. `object_nearest_top` counts object-step pairs. Nearest nodes use the fixed lattice or pre-update adaptive velocities as labelled. Wrong nodes lie >.5 from **every** visible truth; ratio is their largest evidence divided by the largest nearest-true-node evidence. Zero denominator with positive wrong mass is infinity; 0/0 is undefined and excluded. Both counts are explicit.

| source | truth_mode | node_basis | field | steps | any_nearest_top | all_nearest_top | object_nearest_top | mean_best_wrong_ratio | finite_mean_ratio | infinite_ratios | undefined_0_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | intrinsic_incoming | lattice | evidence | 15876 | 0.19394 | 0.02696 | 0.07599 | inf | 3504078.80492 | 2966 | 1571 |
| observation-only tokens | intrinsic_incoming | lattice | evidence_before_modes | 15876 | 0.18871 | 0.02708 | 0.07397 | inf | 573427.53170 | 2797 | 1326 |
| observation-only tokens | intrinsic_incoming | prior_adaptive | evidence | 15876 | 0.22443 | 0.03187 | 0.08884 | inf | 4130.02564 | 2744 | 1718 |
| observation-only tokens | intrinsic_incoming | prior_adaptive | evidence_before_modes | 15876 | 0.23803 | 0.03546 | 0.09447 | inf | 334.58790 | 2354 | 1430 |
| observation-only tokens | literal | lattice | evidence | 15876 | 0.15514 | 0.02154 | 0.06584 | inf | 3155676.41873 | 4423 | 1458 |
| observation-only tokens | literal | lattice | evidence_before_modes | 15876 | 0.15249 | 0.02192 | 0.06575 | inf | 693197.72029 | 3818 | 1252 |
| observation-only tokens | literal | prior_adaptive | evidence | 15876 | 0.17019 | 0.02948 | 0.07795 | inf | 77316.28070 | 4473 | 1508 |
| observation-only tokens | literal | prior_adaptive | evidence_before_modes | 15876 | 0.18355 | 0.03175 | 0.08435 | inf | 72309.77664 | 3838 | 1279 |

Mean evidence and lock fraction on **all** nonturn node-steps (no truth selection):

| source | node | vx | vy | evidence | evidence_before_modes | locked |
| --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| observation-only tokens | 1 | 0.25000 | 0.25000 | 0.74706 | 0.97132 | 0.01406 |
| observation-only tokens | 2 | -0.25000 | 0.25000 | 0.68542 | 0.89390 | 0.01338 |
| observation-only tokens | 3 | -0.25000 | -0.25000 | 0.69780 | 0.90456 | 0.01378 |
| observation-only tokens | 4 | 0.25000 | -0.25000 | 0.77019 | 0.99737 | 0.01545 |
| observation-only tokens | 5 | 0.75000 | 0.75000 | 0.48544 | 0.97538 | 0.01059 |
| observation-only tokens | 6 | -0.75000 | 0.75000 | 0.36483 | 0.78466 | 0.00908 |
| observation-only tokens | 7 | -0.75000 | -0.75000 | 0.36314 | 0.77165 | 0.00924 |
| observation-only tokens | 8 | 0.75000 | -0.75000 | 0.48541 | 0.96973 | 0.01008 |
| observation-only tokens | 9 | 0.25000 | 0.75000 | 0.56595 | 0.89336 | 0.01302 |
| observation-only tokens | 10 | -0.75000 | 0.25000 | 0.46483 | 0.82676 | 0.01123 |
| observation-only tokens | 11 | -0.25000 | -0.75000 | 0.59454 | 0.95371 | 0.01394 |
| observation-only tokens | 12 | 0.75000 | -0.25000 | 0.70491 | 1.14160 | 0.01888 |
| observation-only tokens | 13 | 0.75000 | 0.25000 | 0.72577 | 1.19391 | 0.01816 |
| observation-only tokens | 14 | -0.25000 | 0.75000 | 0.55583 | 0.89393 | 0.01211 |
| observation-only tokens | 15 | -0.75000 | -0.25000 | 0.46328 | 0.78752 | 0.00988 |
| observation-only tokens | 16 | 0.25000 | -0.75000 | 0.62974 | 0.98609 | 0.01426 |

Per-step evidence comparisons: `results/followup_gate_obs_s0.5_evidence_steps.csv`. All 17-node arrays and estimates at every step are retained in the versioned trajectory tensors.


<!-- followup:gate-obs-1 -->
### 1. Mask estimator gate — observation-only tokens, σ=1

| source | truth_mode | speed_bin | truths | assigned | median | p90 | miss_rate | false_alarm_rate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | intrinsic_incoming | all | 44990 | 676 | 0.40741 | 0.83619 | 0.99049 | 0.57658 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.25-1 | 44990 | 676 | 0.40741 | 0.83619 | 0.99049 | 0.57658 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.25-0.5 | 14738 | 178 | 0.41984 | 0.85642 | 0.99274 | 0.23529 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.5-0.75 | 14612 | 239 | 0.44937 | 0.76970 | 0.99021 | 0.19565 | FAIL |
| observation-only tokens | intrinsic_incoming | 0.75-1 | 15640 | 259 | 0.33919 | 0.90361 | 0.98862 | 0.26710 | FAIL |
| observation-only tokens | literal | all | 44990 | 676 | 0.61566 | 1.46428 | 0.99389 | 0.71797 | FAIL |
| observation-only tokens | literal | 0.25-1 | 30097 | 514 | 0.49071 | 1.25624 | 0.99129 | 0.71797 | FAIL |
| observation-only tokens | literal | 0.25-0.5 | 9012 | 151 | 0.44961 | 1.11470 | 0.99101 | 0.42647 | FAIL |
| observation-only tokens | literal | 0.5-0.75 | 10256 | 186 | 0.51309 | 1.40534 | 0.99142 | 0.51087 | FAIL |
| observation-only tokens | literal | 0.75-1 | 10829 | 177 | 0.45548 | 1.42650 | 0.99141 | 0.53094 | FAIL |

Criterion applies to intrinsic_incoming, speed .25–1: median ≤.1 and miss ≤.30. Other rows are diagnostics.

All-node selectivity on nonturn steps with jointly visible blocks. `any_nearest_top` is the fraction of steps where at least one truth's nearest node has positive maximum evidence; `all_nearest_top` requires all truths' nearest nodes to tie that maximum. `object_nearest_top` counts object-step pairs. Nearest nodes use the fixed lattice or pre-update adaptive velocities as labelled. Wrong nodes lie >.5 from **every** visible truth; ratio is their largest evidence divided by the largest nearest-true-node evidence. Zero denominator with positive wrong mass is infinity; 0/0 is undefined and excluded. Both counts are explicit.

| source | truth_mode | node_basis | field | steps | any_nearest_top | all_nearest_top | object_nearest_top | mean_best_wrong_ratio | finite_mean_ratio | infinite_ratios | undefined_0_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | intrinsic_incoming | lattice | evidence | 15876 | 0.06324 | 0.00787 | 0.02501 | inf | 13693.31016 | 1043 | 12367 |
| observation-only tokens | intrinsic_incoming | lattice | evidence_before_modes | 15876 | 0.06374 | 0.00781 | 0.02534 | inf | 14203.49375 | 1067 | 12228 |
| observation-only tokens | intrinsic_incoming | prior_adaptive | evidence | 15876 | 0.07300 | 0.00964 | 0.03021 | inf | 1187.01359 | 969 | 12371 |
| observation-only tokens | intrinsic_incoming | prior_adaptive | evidence_before_modes | 15876 | 0.07697 | 0.01014 | 0.03196 | inf | 2176.38690 | 931 | 12192 |
| observation-only tokens | literal | lattice | evidence | 15876 | 0.04995 | 0.00630 | 0.02158 | inf | 142.87126 | 1611 | 12293 |
| observation-only tokens | literal | lattice | evidence_before_modes | 15876 | 0.05329 | 0.00655 | 0.02294 | inf | 1468.94927 | 1559 | 12163 |
| observation-only tokens | literal | prior_adaptive | evidence | 15876 | 0.05732 | 0.00831 | 0.02638 | inf | 55.23811 | 1606 | 12270 |
| observation-only tokens | literal | prior_adaptive | evidence_before_modes | 15876 | 0.06230 | 0.00926 | 0.02847 | inf | 923.65415 | 1559 | 12128 |

Mean evidence and lock fraction on **all** nonturn node-steps (no truth selection):

| source | node | vx | vy | evidence | evidence_before_modes | locked |
| --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| observation-only tokens | 1 | 0.25000 | 0.25000 | 0.38581 | 0.43942 | 0.00546 |
| observation-only tokens | 2 | -0.25000 | 0.25000 | 0.36935 | 0.42255 | 0.00530 |
| observation-only tokens | 3 | -0.25000 | -0.25000 | 0.40216 | 0.46168 | 0.00585 |
| observation-only tokens | 4 | 0.25000 | -0.25000 | 0.38032 | 0.43720 | 0.00585 |
| observation-only tokens | 5 | 0.75000 | 0.75000 | 0.08246 | 0.18146 | 0.00219 |
| observation-only tokens | 6 | -0.75000 | 0.75000 | 0.05844 | 0.14487 | 0.00167 |
| observation-only tokens | 7 | -0.75000 | -0.75000 | 0.06629 | 0.15153 | 0.00203 |
| observation-only tokens | 8 | 0.75000 | -0.75000 | 0.06525 | 0.15055 | 0.00163 |
| observation-only tokens | 9 | 0.25000 | 0.75000 | 0.14677 | 0.25327 | 0.00343 |
| observation-only tokens | 10 | -0.75000 | 0.25000 | 0.10791 | 0.21320 | 0.00263 |
| observation-only tokens | 11 | -0.25000 | -0.75000 | 0.16016 | 0.25366 | 0.00327 |
| observation-only tokens | 12 | 0.75000 | -0.25000 | 0.14457 | 0.25726 | 0.00291 |
| observation-only tokens | 13 | 0.75000 | 0.25000 | 0.14114 | 0.25057 | 0.00243 |
| observation-only tokens | 14 | -0.25000 | 0.75000 | 0.13592 | 0.23489 | 0.00331 |
| observation-only tokens | 15 | -0.75000 | -0.25000 | 0.12496 | 0.22525 | 0.00235 |
| observation-only tokens | 16 | 0.25000 | -0.75000 | 0.20319 | 0.29630 | 0.00378 |

Per-step evidence comparisons: `results/followup_gate_obs_s1_evidence_steps.csv`. All 17-node arrays and estimates at every step are retained in the versioned trajectory tensors.


<!-- followup:breakdown-k3d_null_noblocks-subtract-all -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, subtract, all support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 5058 | 0.04754 | 0.26618 | 1.00000 | 0.95176 | 0.52887 | 0.00000 | 0.97454 | 0.00796 |
| observation-only tokens | forward | 3828 | 65076 | 4687 | 0.07202 | 0.40935 | 1.00000 | 0.95071 | 0.52336 | 0.00000 | 0.97686 | 0.00945 |
| observation-only tokens | no-op | 1460 | 24820 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | turn-adjacent | 971 | 16507 | 371 | 0.02248 | 0.10196 | 1.00000 | 0.96496 | 0.59838 | 0.00000 | 0.94525 | -0.01091 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 3 | 0.00059 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 18 | 0.00356 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 2732 | 0.54013 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 2305 | 0.45571 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 38 | 0.00751 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 448 | 0.08857 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 2062 | 0.40767 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 1965 | 0.38849 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 496 | 0.09806 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 49 | 0.00969 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_subtract_all_histogram.csv`; half-open bins include under/overflow.


<!-- followup:mask-verdict -->
### 1. Verdict — observation-only tokens

**FAIL** for both nulls and for the estimator gate at both widths. The original best-width ordering (passing status, then matched median, then miss rate) selects σ=1: intrinsic matched median 0.40741, p90 0.83619, miss 99.05%. Its empty/static null alarm rates are 14.31%/19.59%, versus the ≤2% limit; weighted speeds 0.01688/0.01861 satisfy ≤.05. At σ=.5 the gate misses fewer truths (95.12%) but its median remains 0.41176. No acceptance threshold changed. All-node evidence is not selective: fixed-lattice nearest-true nodes lead on 19.39%/6.32% of eligible steps at σ=.5/1; adaptive-prior comparisons give 22.44%/7.30%. Mean best-wrong ratios are infinite at both widths because some steps have positive wrong evidence and zero nearest-true evidence. The undefined 0/0 and finite-only ratios are reported explicitly above. Mask rerun time: 945.89 s, including scoring.


<!-- followup:breakdown-k3d_null_noblocks-subtract-near -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, subtract, near support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 1821 | 0.01711 | 0.16632 | 0.42943 | 0.42943 | 0.00769 | 0.03240 | 0.45307 | -0.01862 |
| observation-only tokens | forward | 3828 | 65076 | 1762 | 0.02708 | 0.25993 | 0.42225 | 0.42225 | 0.00738 | 0.03292 | 0.45178 | -0.01267 |
| observation-only tokens | no-op | 1460 | 24820 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | turn-adjacent | 971 | 16507 | 59 | 0.00357 | 0.04737 | 0.64407 | 0.64407 | 0.01695 | 0.01695 | 0.49160 | -0.19643 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 189 | 0.10379 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 9 | 0.00494 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 472 | 0.25920 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 1142 | 0.62713 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 9 | 0.00494 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 7 | 0.00384 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 467 | 0.25645 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 558 | 0.30643 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 360 | 0.19769 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 426 | 0.23394 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 3 | 0.00165 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_subtract_near_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_null_noblocks-subtract-far -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, subtract, far support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 5034 | 0.04731 | 0.27257 | 1.00000 | 0.92372 | 0.40286 | 0.00000 | 0.98722 | 0.01304 |
| observation-only tokens | forward | 3828 | 65076 | 4664 | 0.07167 | 0.42032 | 1.00000 | 0.92024 | 0.39880 | 0.00000 | 0.99070 | 0.01337 |
| observation-only tokens | no-op | 1460 | 24820 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | turn-adjacent | 971 | 16507 | 370 | 0.02241 | 0.09990 | 1.00000 | 0.96757 | 0.45405 | 0.00000 | 0.94335 | 0.00879 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 0 | 0.00000 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 21 | 0.00417 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 2387 | 0.47418 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 2626 | 0.52165 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 84 | 0.01669 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 618 | 0.12277 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 1842 | 0.36591 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 1677 | 0.33313 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 739 | 0.14680 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 74 | 0.01470 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_subtract_far_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_null_noblocks-mask-all -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, mask, all support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 15226 | 0.14310 | 0.53044 | 0.05307 | 0.00059 | 0.01222 | 0.12229 | 0.13538 | 0.03431 |
| observation-only tokens | forward | 3828 | 65076 | 10697 | 0.16438 | 0.69958 | 0.01169 | 0.00000 | 0.01365 | 0.10180 | 0.18688 | 0.05242 |
| observation-only tokens | no-op | 1460 | 24820 | 742 | 0.02990 | 0.07603 | 0.11051 | 0.00000 | 0.00809 | 0.18868 | -0.05036 | -0.05168 |
| observation-only tokens | turn-adjacent | 971 | 16507 | 3787 | 0.22942 | 0.54686 | 0.15870 | 0.00238 | 0.00898 | 0.16715 | 0.02630 | 0.00001 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 258 | 0.01694 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 2030 | 0.13332 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 3726 | 0.24471 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 4954 | 0.32536 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 3472 | 0.22803 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 786 | 0.05162 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 365 | 0.02397 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 2648 | 0.17391 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 4018 | 0.26389 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 4753 | 0.31216 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 3097 | 0.20340 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 345 | 0.02266 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_mask_all_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_null_noblocks-mask-near -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, mask, near support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 6954 | 0.06536 | 0.16840 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.08650 | -0.03394 |
| observation-only tokens | forward | 3828 | 65076 | 2218 | 0.03408 | 0.14368 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.30140 | -0.13593 |
| observation-only tokens | no-op | 1460 | 24820 | 800 | 0.03223 | 0.06438 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.01000 | 0.08438 |
| observation-only tokens | turn-adjacent | 971 | 16507 | 3936 | 0.23844 | 0.42225 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.01499 | -0.00051 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 1642 | 0.23612 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 1289 | 0.18536 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 1724 | 0.24791 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 2299 | 0.33060 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 1797 | 0.25841 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 1893 | 0.27222 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 1726 | 0.24820 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 1538 | 0.22117 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_mask_near_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_null_noblocks-mask-far -->
### 2. Alarm breakdown — observation-only tokens, k3d_null_noblocks, mask, far support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 6259 | 106403 | 14688 | 0.13804 | 0.49688 | 0.05590 | 0.00197 | 0.02090 | 0.13337 | 0.16009 | 0.04236 |
| observation-only tokens | forward | 3828 | 65076 | 10286 | 0.15806 | 0.64080 | 0.01342 | 0.00000 | 0.02421 | 0.11015 | 0.22172 | 0.06423 |
| observation-only tokens | no-op | 1460 | 24820 | 655 | 0.02639 | 0.07740 | 0.09924 | 0.00000 | 0.01832 | 0.20916 | -0.00581 | -0.04876 |
| observation-only tokens | turn-adjacent | 971 | 16507 | 3747 | 0.22699 | 0.56025 | 0.16493 | 0.00774 | 0.01228 | 0.18388 | 0.01992 | -0.00176 |
| observation-only tokens | turn | 691 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 369 | 0.02512 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 1541 | 0.10492 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 3547 | 0.24149 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 5110 | 0.34790 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 3017 | 0.20541 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 1104 | 0.07516 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 448 | 0.03050 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 2240 | 0.15251 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 4086 | 0.27819 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 4724 | 0.32162 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 2800 | 0.19063 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 390 | 0.02655 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_null_noblocks_mask_far_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-subtract-all -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, subtract, all support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 18161 | 0.04255 | 0.27095 | 1.00000 | 1.00000 | 0.35290 | 0.00066 | 0.86978 | 0.02773 |
| observation-only tokens | forward | 15554 | 264418 | 15150 | 0.05730 | 0.35843 | 1.00000 | 1.00000 | 0.37835 | 0.00000 | 0.95195 | 0.03311 |
| observation-only tokens | no-op | 5655 | 96135 | 1096 | 0.01140 | 0.10203 | 1.00000 | 1.00000 | 0.05292 | 0.00639 | -0.17346 | 0.00414 |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 1915 | 0.02889 | 0.16697 | 1.00000 | 1.00000 | 0.32324 | 0.00261 | 0.81674 | -0.00131 |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 278 | 0.01531 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 240 | 0.01322 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 545 | 0.03001 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 505 | 0.02781 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 7324 | 0.40328 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 9269 | 0.51038 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 651 | 0.03585 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 2647 | 0.14575 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 5775 | 0.31799 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 5014 | 0.27609 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 3260 | 0.17951 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 814 | 0.04482 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_subtract_all_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-subtract-near -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, subtract, near support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 7838 | 0.01836 | 0.22407 | 1.00000 | 1.00000 | 0.09556 | 0.03662 | 0.34545 | -0.00025 |
| observation-only tokens | forward | 15554 | 264418 | 5692 | 0.02153 | 0.26662 | 1.00000 | 1.00000 | 0.11033 | 0.04339 | 0.44021 | 0.00906 |
| observation-only tokens | no-op | 5655 | 96135 | 1410 | 0.01467 | 0.16658 | 1.00000 | 1.00000 | 0.04468 | 0.02199 | 0.05690 | -0.01909 |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 736 | 0.01110 | 0.13773 | 1.00000 | 1.00000 | 0.07880 | 0.01223 | 0.16538 | -0.03611 |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 90 | 0.01148 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 1051 | 0.13409 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 709 | 0.09046 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 1759 | 0.22442 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 3906 | 0.49834 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 323 | 0.04121 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 179 | 0.02284 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 1937 | 0.24713 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 1868 | 0.23833 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 1776 | 0.22659 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 1894 | 0.24164 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 184 | 0.02348 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_subtract_near_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-subtract-far -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, subtract, far support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 18805 | 0.04406 | 0.27923 | 1.00000 | 1.00000 | 0.31284 | 0.00096 | 0.89736 | 0.02132 |
| observation-only tokens | forward | 15554 | 264418 | 15933 | 0.06026 | 0.37682 | 1.00000 | 1.00000 | 0.33107 | 0.00000 | 0.97260 | 0.02399 |
| observation-only tokens | no-op | 5655 | 96135 | 948 | 0.00986 | 0.09001 | 1.00000 | 1.00000 | 0.04958 | 0.01477 | -0.24956 | -0.01548 |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 1924 | 0.02903 | 0.16440 | 1.00000 | 1.00000 | 0.29158 | 0.00208 | 0.83947 | 0.01726 |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 284 | 0.01510 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 219 | 0.01165 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 501 | 0.02664 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 319 | 0.01696 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 6946 | 0.36937 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 10536 | 0.56028 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 600 | 0.03191 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 3166 | 0.16836 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 5612 | 0.29843 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 4983 | 0.26498 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 3712 | 0.19739 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 732 | 0.03893 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_subtract_far_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-mask-all -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, mask, all support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 1358 | 0.00318 | 0.02704 | 0.99190 | 1.00000 | 0.04492 | 0.55228 | 0.01609 | 0.01663 |
| observation-only tokens | forward | 15554 | 264418 | 239 | 0.00090 | 0.01196 | 0.99582 | 1.00000 | 0.17573 | 0.03766 | 0.11732 | 0.10394 |
| observation-only tokens | no-op | 5655 | 96135 | 490 | 0.00510 | 0.04810 | 0.98367 | 1.00000 | 0.01837 | 0.44898 | -0.01149 | -0.02346 |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 629 | 0.00949 | 0.05668 | 0.99682 | 1.00000 | 0.01590 | 0.82830 | -0.00089 | 0.01469 |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 17 | 0.01252 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 90 | 0.06627 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 544 | 0.40059 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 576 | 0.42415 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 111 | 0.08174 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 20 | 0.01473 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 27 | 0.01988 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 136 | 0.10015 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 524 | 0.38586 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 495 | 0.36451 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 154 | 0.11340 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 22 | 0.01620 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_mask_all_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-mask-near -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, mask, near support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | forward | 15554 | 264418 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | no-op | 5655 | 96135 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 0 | 0.00000 | 0.00000 | nan | nan | nan | nan | nan | nan |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 0 | 0.00000 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 0 | 0.00000 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 0 | 0.00000 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 0 | 0.00000 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 0 | 0.00000 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 0 | 0.00000 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_mask_near_histogram.csv`; half-open bins include under/overflow.


<!-- followup:breakdown-k3d_v_validation-mask-far -->
### 2. Locked breakdown — observation-only tokens, k3d_v_validation, mask, far support

| source | action | steps | node_steps | detections | node_rate | step_rate | evidenced_fraction | locked_fraction | near_forward_fraction | near_zero_fraction | mean_vx | mean_vy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 25108 | 426836 | 2888 | 0.00677 | 0.04206 | 0.97645 | 1.00000 | 0.02078 | 0.73650 | 0.01125 | 0.00272 |
| observation-only tokens | forward | 15554 | 264418 | 301 | 0.00114 | 0.01279 | 1.00000 | 1.00000 | 0.11628 | 0.28571 | 0.16673 | 0.00522 |
| observation-only tokens | no-op | 5655 | 96135 | 1006 | 0.01046 | 0.07975 | 0.98608 | 1.00000 | 0.01690 | 0.64712 | -0.03078 | 0.00758 |
| observation-only tokens | turn-adjacent | 3899 | 66283 | 1581 | 0.02385 | 0.10413 | 0.96584 | 1.00000 | 0.00506 | 0.87919 | 0.00839 | -0.00085 |
| observation-only tokens | turn | 2692 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |

Nonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.

| source | axis | lo | hi | count | fraction |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | vx | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vx | -1.50000 | -1.00000 | 22 | 0.00762 |
| observation-only tokens | vx | -1.00000 | -0.50000 | 75 | 0.02597 |
| observation-only tokens | vx | -0.50000 | 0.00000 | 1370 | 0.47438 |
| observation-only tokens | vx | 0.00000 | 0.50000 | 1286 | 0.44529 |
| observation-only tokens | vx | 0.50000 | 1.00000 | 85 | 0.02943 |
| observation-only tokens | vx | 1.00000 | 1.50000 | 50 | 0.01731 |
| observation-only tokens | vx | 1.50000 | inf | 0 | 0.00000 |
| observation-only tokens | vy | -inf | -1.50000 | 0 | 0.00000 |
| observation-only tokens | vy | -1.50000 | -1.00000 | 31 | 0.01073 |
| observation-only tokens | vy | -1.00000 | -0.50000 | 110 | 0.03809 |
| observation-only tokens | vy | -0.50000 | 0.00000 | 1354 | 0.46884 |
| observation-only tokens | vy | 0.00000 | 0.50000 | 1245 | 0.43109 |
| observation-only tokens | vy | 0.50000 | 1.00000 | 113 | 0.03913 |
| observation-only tokens | vy | 1.00000 | 1.50000 | 35 | 0.01212 |
| observation-only tokens | vy | 1.50000 | inf | 0 | 0.00000 |

Full action-conditioned 2D velocity histogram: `results/followup_breakdown_k3d_v_validation_mask_far_histogram.csv`; half-open bins include under/overflow.


<!-- followup:stationary-content -->
### 2. Direct cone-stationarity check — observation-only tokens

| source | region | pairs | same_cell_ssd | ego_ssd | ratio_of_means | same_cell_better |
| --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all | 3939 | 0.96822 | 0.71979 | 1.34515 | 0.27215 |
| observation-only tokens | near | 3939 | 1.00040 | 1.19964 | 0.83392 | 0.84336 |
| observation-only tokens | far | 3939 | 0.94985 | 0.60574 | 1.56809 | 0.39731 |

K=8, empty-room forward pairs only. SSD sums channels then averages shared valid cells. Both hypotheses use identical overlap, excluding newly visible edge cells. Same-cell matches imply apparent intrinsic (+1,0) after adding forward ego motion. This identifies cone-stationary imprint structure; the test does not semantically separate walls from floor texture.


<!-- followup:patch-tokens-8 -->
### 3. Direct block-motion visibility — observation-only tokens, K=8

| source | k | group | bin | pairs | usable | median | p90 | within_025 | boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | 8 | all | all | 44990 | 44990 | 0.63751 | 1.73585 | 0.12383 | 0.15083 |
| observation-only tokens | 8 | speed | .25-.5 | 14738 | 14738 | 0.53849 | 1.55214 | 0.14860 | 0.12634 |
| observation-only tokens | 8 | speed | .5-.75 | 14612 | 14612 | 0.64991 | 1.70457 | 0.09088 | 0.14495 |
| observation-only tokens | 8 | speed | .75-1 | 15640 | 15640 | 0.71494 | 1.92561 | 0.13127 | 0.17941 |
| observation-only tokens | 8 | distance | [1,5) | 8764 | 8764 | 0.63813 | 1.59491 | 0.10714 | 0.14628 |
| observation-only tokens | 8 | distance | [5,9) | 17263 | 17263 | 0.63739 | 1.75952 | 0.13393 | 0.15664 |
| observation-only tokens | 8 | distance | [9,13) | 15683 | 15683 | 0.64001 | 1.79151 | 0.12294 | 0.14697 |
| observation-only tokens | 8 | distance | [13,17) | 3280 | 3280 | 0.62865 | 1.68311 | 0.11951 | 0.15091 |
| observation-only tokens | 8 | support_distance | rounded_x<=6 | 14834 | 14834 | 0.67921 | 1.70735 | 0.10314 | 0.16091 |
| observation-only tokens | 8 | support_distance | rounded_x>=7 | 30156 | 30156 | 0.61816 | 1.75156 | 0.13400 | 0.14587 |

All jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.


<!-- followup:patch-tokens-32 -->
### 3. Direct block-motion visibility — observation-only tokens, K=32

| source | k | group | bin | pairs | usable | median | p90 | within_025 | boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | 32 | all | all | 44990 | 44990 | 0.64170 | 1.77453 | 0.12152 | 0.15870 |
| observation-only tokens | 32 | speed | .25-.5 | 14738 | 14738 | 0.54077 | 1.59788 | 0.14324 | 0.13726 |
| observation-only tokens | 32 | speed | .5-.75 | 14612 | 14612 | 0.65214 | 1.73757 | 0.09095 | 0.15590 |
| observation-only tokens | 32 | speed | .75-1 | 15640 | 15640 | 0.71738 | 1.95220 | 0.12960 | 0.18152 |
| observation-only tokens | 32 | distance | [1,5) | 8764 | 8764 | 0.62167 | 1.49728 | 0.11285 | 0.12414 |
| observation-only tokens | 32 | distance | [5,9) | 17263 | 17263 | 0.64238 | 1.79644 | 0.12964 | 0.16614 |
| observation-only tokens | 32 | distance | [9,13) | 15683 | 15683 | 0.65277 | 1.86542 | 0.11898 | 0.16808 |
| observation-only tokens | 32 | distance | [13,17) | 3280 | 3280 | 0.64715 | 1.77868 | 0.11402 | 0.16707 |
| observation-only tokens | 32 | support_distance | rounded_x<=6 | 14834 | 14834 | 0.66636 | 1.66814 | 0.10584 | 0.15040 |
| observation-only tokens | 32 | support_distance | rounded_x>=7 | 30156 | 30156 | 0.62647 | 1.82022 | 0.12923 | 0.16279 |

All jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.


<!-- followup:patch-h_new-8 -->
### 3. Direct block-motion visibility — observation-only h_new, K=8

| source | k | group | bin | pairs | usable | median | p90 | within_025 | boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only h_new | 8 | all | all | 44990 | 44990 | 0.61373 | 1.74950 | 0.13732 | 0.16561 |
| observation-only h_new | 8 | speed | .25-.5 | 14738 | 14738 | 0.51728 | 1.59762 | 0.16271 | 0.13889 |
| observation-only h_new | 8 | speed | .5-.75 | 14612 | 14612 | 0.62975 | 1.72212 | 0.10881 | 0.16131 |
| observation-only h_new | 8 | speed | .75-1 | 15640 | 15640 | 0.69133 | 1.92384 | 0.14003 | 0.19482 |
| observation-only h_new | 8 | distance | [1,5) | 8764 | 8764 | 0.61452 | 1.51274 | 0.11445 | 0.13921 |
| observation-only h_new | 8 | distance | [5,9) | 17263 | 17263 | 0.60028 | 1.75049 | 0.15113 | 0.16596 |
| observation-only h_new | 8 | distance | [9,13) | 15683 | 15683 | 0.62371 | 1.85692 | 0.13849 | 0.17815 |
| observation-only h_new | 8 | distance | [13,17) | 3280 | 3280 | 0.63463 | 1.75222 | 0.12012 | 0.17439 |
| observation-only h_new | 8 | support_distance | rounded_x<=6 | 14834 | 14834 | 0.64861 | 1.64179 | 0.11217 | 0.15909 |
| observation-only h_new | 8 | support_distance | rounded_x>=7 | 30156 | 30156 | 0.59690 | 1.80173 | 0.14969 | 0.16882 |

All jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.


<!-- followup:patch-h_new-32 -->
### 3. Direct block-motion visibility — observation-only h_new, K=32

| source | k | group | bin | pairs | usable | median | p90 | within_025 | boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only h_new | 32 | all | all | 44990 | 44990 | 0.61611 | 1.73397 | 0.13385 | 0.15288 |
| observation-only h_new | 32 | speed | .25-.5 | 14738 | 14738 | 0.51264 | 1.55822 | 0.15864 | 0.13224 |
| observation-only h_new | 32 | speed | .5-.75 | 14612 | 14612 | 0.63481 | 1.68833 | 0.10553 | 0.15036 |
| observation-only h_new | 32 | speed | .75-1 | 15640 | 15640 | 0.69510 | 1.91625 | 0.13696 | 0.17468 |
| observation-only h_new | 32 | distance | [1,5) | 8764 | 8764 | 0.59623 | 1.43372 | 0.12209 | 0.11136 |
| observation-only h_new | 32 | distance | [5,9) | 17263 | 17263 | 0.61465 | 1.75781 | 0.14302 | 0.15472 |
| observation-only h_new | 32 | distance | [9,13) | 15683 | 15683 | 0.62672 | 1.85165 | 0.13422 | 0.17101 |
| observation-only h_new | 32 | distance | [13,17) | 3280 | 3280 | 0.64217 | 1.76124 | 0.11524 | 0.16738 |
| observation-only h_new | 32 | support_distance | rounded_x<=6 | 14834 | 14834 | 0.64040 | 1.59141 | 0.11366 | 0.13496 |
| observation-only h_new | 32 | support_distance | rounded_x>=7 | 30156 | 30156 | 0.60272 | 1.80531 | 0.14379 | 0.16169 |

All jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.


<!-- followup:patch-tokens-256 -->
### 3. Direct block-motion visibility — observation-only tokens, K=256

| source | k | group | bin | pairs | usable | median | p90 | within_025 | boundary_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | 256 | all | all | 44990 | 44990 | 0.63580 | 1.73459 | 0.12058 | 0.14368 |
| observation-only tokens | 256 | speed | .25-.5 | 14738 | 14738 | 0.53718 | 1.54527 | 0.14337 | 0.12437 |
| observation-only tokens | 256 | speed | .5-.75 | 14612 | 14612 | 0.64839 | 1.70292 | 0.08794 | 0.14105 |
| observation-only tokens | 256 | speed | .75-1 | 15640 | 15640 | 0.70824 | 1.91532 | 0.12960 | 0.16432 |
| observation-only tokens | 256 | distance | [1,5) | 8764 | 8764 | 0.62369 | 1.49443 | 0.10726 | 0.11547 |
| observation-only tokens | 256 | distance | [5,9) | 17263 | 17263 | 0.63766 | 1.76226 | 0.13063 | 0.15200 |
| observation-only tokens | 256 | distance | [9,13) | 15683 | 15683 | 0.64118 | 1.81366 | 0.11879 | 0.14787 |
| observation-only tokens | 256 | distance | [13,17) | 3280 | 3280 | 0.63896 | 1.74683 | 0.11189 | 0.15518 |
| observation-only tokens | 256 | support_distance | rounded_x<=6 | 14834 | 14834 | 0.66790 | 1.65522 | 0.10253 | 0.14049 |
| observation-only tokens | 256 | support_distance | rounded_x>=7 | 30156 | 30156 | 0.62007 | 1.77035 | 0.12946 | 0.14524 |

All jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.


<!-- followup:patch-verdict -->
### 3. Verdict — observation-only patches

| source | k | median | p90 | within_025 |
| --- | --- | --- | --- | --- |
| observation-only tokens | 8 | 0.63751 | 1.73585 | 0.12383 |
| observation-only tokens | 32 | 0.64170 | 1.77453 | 0.12152 |
| observation-only h_new | 8 | 0.61373 | 1.74950 | 0.13732 |
| observation-only h_new | 32 | 0.61611 | 1.73397 | 0.13385 |
| observation-only tokens | 256 | 0.63580 | 1.73459 | 0.12058 |

**Weak direct motion visibility under this patch test.** All five configurations use the same 44,990 jointly visible nonturn block pairs, and all pairs have usable support. Removing projection does not rescue tracking: raw-token median 0.63580, p90 1.73459, only 12.06% within .25. The full per-speed and per-distance tables above retain all pairs, including boundary minima. This result shows that oracle-localized 3×3 appearance does not reliably translate with intrinsic block motion; it does not prove that every learned or longer-context estimator must fail. No acceptance threshold was preregistered for this diagnostic. Runtime: 256.06 s in the module, 268.0 s including interpreter/import overhead.


<!-- followup:stationary-attribution -->
### 2. Evidence-localized cone-stationarity — observation-only tokens

| source | subset | region | pairs | evidenced_pairs | mass | same_ssd | ego_ssd | same_to_ego | same_better_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observation-only tokens | all_alarms | all | 3939 | 1666 | 32301.70946 | 0.02537 | 0.53040 | 0.04782 | 0.98972 |
| observation-only tokens | all_alarms | near | 3939 | 1499 | 2885.39227 | 0.03530 | 1.29743 | 0.02721 | 0.98895 |
| observation-only tokens | all_alarms | far | 3939 | 1666 | 29416.31711 | 0.02439 | 0.45516 | 0.05359 | 0.98979 |
| observation-only tokens | near_ego_forward | all | 3939 | 1357 | 22505.29756 | 0.01697 | 0.54182 | 0.03132 | 0.99416 |
| observation-only tokens | near_ego_forward | near | 3939 | 1144 | 2131.95466 | 0.02464 | 0.97634 | 0.02524 | 0.99479 |
| observation-only tokens | near_ego_forward | far | 3939 | 1357 | 20373.34281 | 0.01617 | 0.49634 | 0.03257 | 0.99409 |

Additional attribution check because whole-region SSD did not establish global far-field stationarity. All 3,939 empty-room forward pairs are re-evaluated at their cached subtraction priors, sigma=1/K=8. A local observer returns the production `two_modes` result unchanged and copies its selected pixel weights. We compare same-cell and ego-transported **plain** imprint SSD at those selected alarm-evidence locations, on common valid support. `near_ego_forward` restricts alarm velocities to radius .25 around (+1,0). This distinguishes selected nuisance cells from the rest of the far field; it does not distinguish walls, floor texture, or learned positional features. The maximum reconstructed/cached velocity component difference was 9e-06; runtime 31.48 s.


<!-- followup:breakdown-verdict -->
### 2. Verdict — observation-only tokens

**Yes: the original subtraction alarms are strongly explained by cone-stationary imprint content.** Empty-room forward/non-adjacent no-op alarm rates are 7.20%/0%. Alarm mean velocity is (0.97454, 0.00796), with 52.89% within .25 of (+1,0). Near-only support reduces the all-nonturn rate from 4.75% to 1.71%, while far-only retains 4.73%. Whole-field SSD alone was inconclusive (the far field is not globally stationary). The added evidence-localized replay is decisive for the selected nuisance: same-cell SSD is 4.78% of ego SSD at alarm-evidence locations, 98.97% of selected mass prefers the same-cell match, and 91.07% of that mass is in rel_x≥7. Among near-(+1,0) alarms, the SSD ratio is 3.13% and 99.42% of selected mass prefers same-cell matching. Thus ego transport makes the selected cone-stationary structure appear to move at approximately +1 cell/step. This identifies a representation-level nuisance; it does not uniquely separate far-wall/floor image texture from learned positional features.

**Mask changes the failure mode.** Only 808/15,226 (5.31%) nominal empty-room alarms retain selected evidence >1e-6, and only 9 are active locks. Its preregistered alarm failure therefore should not be read as 15,226 surviving detections. Near-only mask yields zero active locks in both empty and moving scenes. On K3D-v, subtraction has 18,161 full-support active locks, 35.29% near (+1,0); near-only lowers that fraction to 9.56%, while far-only retains 31.28%. Mask has only 1,358 full-support locks, 55.23% near zero; far-only has 2,888, 73.65% near zero. Removing the nuisance does not restore block-motion recall. No new acceptance criterion is introduced.

Breakdown runtime: 1,178.31 s inside the module; evidence-localized attribution added 31.48 s because the whole-region SSD did not establish the mechanism. All 3,939 empty-room forward pairs were replayed; maximum velocity-component difference from cached measurements was 9.0e-6.


<!-- followup:memory-lift-k3d_null_noblocks -->
### 4. Memory-consistent imprint cache

| source | split | clips | frames | seconds |
| --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | k3d_null_noblocks | 50 | 7000 | 380.72081 |

Released weights and original internal flow, ego transform, velocity mixing and gated write; all 140 frames are observed context. Imprint is the cone **after** `(1-g)*h_old + g*h_new`, max over all five velocity channels, before the decoder's shared positional encoding. No predicted frames or decoder calls. This uses model memory and is explicitly a diagnostic variant, not the design's observation-only lift. Cached fp16 under `.scratch/lifts_mem/`.


<!-- followup:null-mem-k3d_null_noblocks-1 -->
### 4. Mask null / contrast — memory post-gate max-V tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | k3d_null_noblocks | 50 | 1.00000 | 6259 | 17886 | 0.16810 | 0.10501 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:memory-lift-k3d_null_static -->
### 4. Memory-consistent imprint cache

| source | split | clips | frames | seconds |
| --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | k3d_null_static | 50 | 7000 | 359.62145 |

Released weights and original internal flow, ego transform, velocity mixing and gated write; all 140 frames are observed context. Imprint is the cone **after** `(1-g)*h_old + g*h_new`, max over all five velocity channels, before the decoder's shared positional encoding. No predicted frames or decoder calls. This uses model memory and is explicitly a diagnostic variant, not the design's observation-only lift. Cached fp16 under `.scratch/lifts_mem/`.


<!-- followup:null-mem-k3d_null_static-1 -->
### 4. Mask null / contrast — memory post-gate max-V tokens

| source | split | clips | sigma | steps | alarms | alarm_rate | evidence_weighted_speed | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | k3d_null_static | 50 | 1.00000 | 6288 | 41258 | 0.38596 | 0.15025 | FAIL |

Criterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.


<!-- followup:memory-lift-k3d_v_validation -->
### 4. Memory-consistent imprint cache

| source | split | clips | frames | seconds |
| --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | k3d_v_validation | 100 | 14000 | 732.72441 |

Released weights and original internal flow, ego transform, velocity mixing and gated write; all 140 frames are observed context. Imprint is the cone **after** `(1-g)*h_old + g*h_new`, max over all five velocity channels, before the decoder's shared positional encoding. No predicted frames or decoder calls. This uses model memory and is explicitly a diagnostic variant, not the design's observation-only lift. Cached fp16 under `.scratch/lifts_mem/`.


<!-- followup:gate-mem-1 -->
### 4. Mask estimator gate — memory post-gate max-V tokens, σ=1

| source | truth_mode | speed_bin | truths | assigned | median | p90 | miss_rate | false_alarm_rate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | intrinsic_incoming | all | 22412 | 3168 | 0.36410 | 0.71904 | 0.90104 | 0.32227 | FAIL |
| memory post-gate max-V tokens | intrinsic_incoming | 0.25-1 | 22412 | 3168 | 0.36410 | 0.71904 | 0.90104 | 0.32227 | FAIL |
| memory post-gate max-V tokens | intrinsic_incoming | 0.25-0.5 | 7105 | 1356 | 0.28888 | 0.55201 | 0.83420 | 0.31888 | FAIL |
| memory post-gate max-V tokens | intrinsic_incoming | 0.5-0.75 | 7449 | 1067 | 0.40232 | 0.70356 | 0.90777 | 0.23464 | FAIL |
| memory post-gate max-V tokens | intrinsic_incoming | 0.75-1 | 7858 | 745 | 0.52156 | 0.89417 | 0.95508 | 0.42188 | FAIL |
| memory post-gate max-V tokens | literal | all | 22412 | 3168 | 0.49448 | 1.23383 | 0.92848 | 0.50619 | FAIL |
| memory post-gate max-V tokens | literal | 0.25-1 | 14952 | 2525 | 0.41361 | 0.89354 | 0.89734 | 0.50619 | FAIL |
| memory post-gate max-V tokens | literal | 0.25-0.5 | 4348 | 1016 | 0.29367 | 0.59138 | 0.80313 | 0.50119 | FAIL |
| memory post-gate max-V tokens | literal | 0.5-0.75 | 5205 | 891 | 0.48361 | 0.82539 | 0.90913 | 0.49534 | FAIL |
| memory post-gate max-V tokens | literal | 0.75-1 | 5399 | 618 | 0.65486 | 1.06483 | 0.96184 | 0.58333 | FAIL |

Criterion applies to intrinsic_incoming, speed .25–1: median ≤.1 and miss ≤.30. Other rows are diagnostics.

All-node selectivity on nonturn steps with jointly visible blocks. `any_nearest_top` is the fraction of steps where at least one truth's nearest node has positive maximum evidence; `all_nearest_top` requires all truths' nearest nodes to tie that maximum. `object_nearest_top` counts object-step pairs. Nearest nodes use the fixed lattice or pre-update adaptive velocities as labelled. Wrong nodes lie >.5 from **every** visible truth; ratio is their largest evidence divided by the largest nearest-true-node evidence. Zero denominator with positive wrong mass is infinity; 0/0 is undefined and excluded. Both counts are explicit.

| source | truth_mode | node_basis | field | steps | any_nearest_top | all_nearest_top | object_nearest_top | mean_best_wrong_ratio | finite_mean_ratio | infinite_ratios | undefined_0_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | intrinsic_incoming | lattice | evidence | 7984 | 0.19088 | 0.01678 | 0.07844 | inf | 1.64978 | 551 | 2818 |
| memory post-gate max-V tokens | intrinsic_incoming | lattice | evidence_before_modes | 7984 | 0.16771 | 0.01503 | 0.06778 | inf | 1.85968 | 536 | 2699 |
| memory post-gate max-V tokens | intrinsic_incoming | prior_adaptive | evidence | 7984 | 0.17786 | 0.02117 | 0.07398 | inf | 1.33330 | 603 | 2935 |
| memory post-gate max-V tokens | intrinsic_incoming | prior_adaptive | evidence_before_modes | 7984 | 0.20679 | 0.02355 | 0.08736 | inf | 1.33582 | 430 | 2738 |
| memory post-gate max-V tokens | literal | lattice | evidence | 7984 | 0.14905 | 0.01453 | 0.06555 | inf | 2.48788 | 1157 | 2770 |
| memory post-gate max-V tokens | literal | lattice | evidence_before_modes | 7984 | 0.14466 | 0.01315 | 0.06358 | inf | 2.66714 | 824 | 2661 |
| memory post-gate max-V tokens | literal | prior_adaptive | evidence | 7984 | 0.12926 | 0.02142 | 0.05912 | inf | 2.46487 | 1328 | 2830 |
| memory post-gate max-V tokens | literal | prior_adaptive | evidence_before_modes | 7984 | 0.15857 | 0.02342 | 0.07438 | inf | 2.43972 | 888 | 2691 |

Mean evidence and lock fraction on **all** nonturn node-steps (no truth selection):

| source | node | vx | vy | evidence | evidence_before_modes | locked |
| --- | --- | --- | --- | --- | --- | --- |
| memory post-gate max-V tokens | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| memory post-gate max-V tokens | 1 | 0.25000 | 0.25000 | 2.64302 | 4.39732 | 0.03458 |
| memory post-gate max-V tokens | 2 | -0.25000 | 0.25000 | 2.65228 | 4.45824 | 0.03650 |
| memory post-gate max-V tokens | 3 | -0.25000 | -0.25000 | 2.58562 | 4.32746 | 0.03403 |
| memory post-gate max-V tokens | 4 | 0.25000 | -0.25000 | 2.60706 | 4.35145 | 0.03618 |
| memory post-gate max-V tokens | 5 | 0.75000 | 0.75000 | 0.84227 | 2.65130 | 0.01179 |
| memory post-gate max-V tokens | 6 | -0.75000 | 0.75000 | 0.93773 | 3.00593 | 0.01028 |
| memory post-gate max-V tokens | 7 | -0.75000 | -0.75000 | 0.91770 | 2.94791 | 0.01132 |
| memory post-gate max-V tokens | 8 | 0.75000 | -0.75000 | 0.83214 | 2.63559 | 0.01211 |
| memory post-gate max-V tokens | 9 | 0.25000 | 0.75000 | 1.48187 | 3.39491 | 0.01793 |
| memory post-gate max-V tokens | 10 | -0.75000 | 0.25000 | 1.65192 | 3.67701 | 0.02112 |
| memory post-gate max-V tokens | 11 | -0.25000 | -0.75000 | 1.56196 | 3.50040 | 0.02096 |
| memory post-gate max-V tokens | 12 | 0.75000 | -0.25000 | 1.52761 | 3.39986 | 0.02064 |
| memory post-gate max-V tokens | 13 | 0.75000 | 0.25000 | 1.52519 | 3.42137 | 0.01920 |
| memory post-gate max-V tokens | 14 | -0.25000 | 0.75000 | 1.50346 | 3.47095 | 0.01729 |
| memory post-gate max-V tokens | 15 | -0.75000 | -0.25000 | 1.60971 | 3.57872 | 0.01785 |
| memory post-gate max-V tokens | 16 | 0.25000 | -0.75000 | 1.47557 | 3.37312 | 0.01920 |

Per-step evidence comparisons: `results/followup_gate_mem_s1_evidence_steps.csv`. All 17-node arrays and estimates at every step are retained in the versioned trajectory tensors.


<!-- followup:dependency-audit -->
### Dependency update audit

The external KWM checkout changed from `2e73903` to `c7067d3` during execution. `lock_nodes.py` was updated at 18:43:18 EST (commit 18:44:03), after the mask, tests and breakdown processes had imported the original code, and before patch/attribution/memory processes started. The sole diff adds optional `support_mask_key=None` forwarding; this campaign never supplies that argument. No KWM source was edited by this work. Both revisions were loaded from git into local scratch snapshots and compared on 18 real observation/memory pairs, including adaptive priors, all three support masks, both residual modes and both sigma values. All three outputs and every diagnostic tensor were bitwise identical in all 216 pair/configuration cases. This establishes parity for the used path, not for the new optional feature.

| source | mode | region | sigma | pairs | bitwise_equal | max_difference |
| --- | --- | --- | --- | --- | --- | --- |
| observation-only + memory post-gate max-V tokens | subtract | all | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | subtract | all | 1.00000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | subtract | near | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | subtract | near | 1.00000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | subtract | far | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | subtract | far | 1.00000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | all | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | all | 1.00000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | near | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | near | 1.00000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | far | 0.50000 | 18 | True | 0.00000 |
| observation-only + memory post-gate max-V tokens | mask | far | 1.00000 | 18 | True | 0.00000 |

Audit runtime: 4.83 s. Source hashes and stage mapping are in `followup_kwm_revisions.csv`; initial provenance remains intact.


<!-- followup:memory-verdict -->
### 4. Verdict — memory post-gate max-V tokens

**FAIL** for both null splits and the 100-clip estimator gate. Empty/static alarm rates are 16.81%/38.60%, and evidence-weighted speeds are .10501/.15025, exceeding both limits (≤2%, ≤.05). Primary intrinsic gate median is .36410, p90 .71904, miss 90.10%, false-alarm rate 32.23%; required median ≤.1 and miss ≤30%. Literal secondary median is .41361, p90 .89354, miss 89.73%. The nearest true lattice node leads selected evidence on 19.09% of eligible steps; mean best-wrong ratio is infinite (551 positive-wrong/zero-true cases; 2,818 undefined 0/0 cases).

Using the released model's own memory does not rescue these null and motion-estimation criteria. This is a diagnostic memory source, not the design's observation-only lift. The gate uses 100 clips versus 200 in item 1; the null cohorts are exactly matched at 50 clips each. Memory stage runtime was 1,794.40 s including caching and scoring, excluding the separately reported attribution check and imports.


<!-- followup:tests-final -->
### 5. Fast tests and artifact validation

**PASS: 11 tests in 26.76 s** (29.37 s including process/import overhead), against the final external KWM revision. The first run passed in 19.81 s. Tests cover synthetic fractional patch motion with and without ego translation, exact integer Fourier samples, unusable support, alarm/action/histogram bookkeeping inputs, multiple-truth evidence ratios, and exact cached-precision parity of the memory adapter with released forward context writes, as well as the original seven checks. The two warnings are existing dependency deprecations. Logs: `.scratch/followup_tests.log` and `.scratch/followup_tests_final.log`.

Artifact checks passed: all 1,840 follow-up trajectories, 200 memory clips and 1,000 patch variant/clip caches exist; all five patch runs use the identical 44,990 unique pairs; all 12 breakdowns have correct action partitions and histogram totals; all node tables include 17 nodes with zero pinned static metadata. The original report is an unchanged byte-for-byte prefix (SHA-256 recorded in provenance). No original result CSV was overwritten. `followup_validation.csv` records these checks; final hashes are in `followup_provenance_final.csv`.


<!-- followup:execution-final -->
### Commands and runtime

All Python commands below run in the CPU-only, eight-thread environment configured by `run_cpu.sh`; W&B is offline, bytecode is disabled, and cache/temp directories are under `.scratch/`. The pipeline command was `bash experiments/flowm_bench/k3d/run_cpu.sh run_followup --after-mask`. Every completed stage returned zero.

| source | stage | command | seconds | basis | status |
| --- | --- | --- | --- | --- | --- |
| observation-only tokens | mask rerun | bash experiments/flowm_bench/k3d/run_cpu.sh followup mask | 956.223 | log creation to completed timestamp (approximate wall) | 0 |
| synthetic + observation/released context | tests | /home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/k3d/test_stage1.py -q -o cache_dir=experiments/flowm_bench/k3d/.scratch/pytest_cache | 22.256 | subprocess wall; includes imports | 0 |
| observation-only tokens | breakdown | /home/yixunhu/miniconda3/envs/flowm/bin/python -m k3d.alarm_breakdown | 1189.580 | subprocess wall; includes imports | 0 |
| observation-only tokens/h_new | patch | /home/yixunhu/miniconda3/envs/flowm/bin/python -m k3d.patch_motion | 267.988 | subprocess wall; includes imports | 0 |
| memory tokens + observation attribution | memory | /home/yixunhu/miniconda3/envs/flowm/bin/python -m k3d.memory_lift | 1836.935 | subprocess wall; includes imports | 0 |
| observation + memory tokens | dependency audit | bash experiments/flowm_bench/k3d/run_cpu.sh followup_dependency_audit | 16.290 | process wall; includes imports | 0 |
| synthetic + observation/released context | final tests | /home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/k3d/test_stage1.py -q -o cache_dir=experiments/flowm_bench/k3d/.scratch/pytest_cache | 29.370 | process wall; includes imports | 0 |

Measured command wall times total **71.98 minutes**. The memory command includes the extra 31.48 s attribution check; do not add it again. From the first measurement command (2026-11-04 18:25:58 EST) through this finalization (2026-11-04T19:41:37-05:00) is **75.65 minutes**, plus initial reading/implementation before the first command. The full task stayed well within the approximate three-hour budget. Model inference and measurement jobs ran sequentially at eight threads, with no GPU execution.


<!-- followup:summary-final -->
### Follow-up summary and deviations

1. **Mask rerun: FAIL.** Both widths fail both nulls and the primary gate. Best-width selection by the original ordering gives σ=1, median .40741 and misses 99.05%. The failure survives the corrected intrinsic-incoming target.
2. **Null diagnosis: cone-stationary nuisance supported.** Subtraction alarm evidence overwhelmingly selects same-cell content (98.97% of mass), mostly in the far field (91.07%). Forward transport turns this into approximately (+1,0) intrinsic motion. Mask rejects most such nominal alarms but leaves very poor true-motion recall.
3. **Direct visibility: weak for this oracle-location patch test.** Across tokens/h_new and K=8/32/raw-256, medians are .61373–.64170 and within-.25 fractions are only 12.06–13.73%. Removing projection does not solve the patch-motion mismatch. There is no preregistered pass threshold for this diagnostic.
4. **Memory diagnostic: FAIL.** Empty/static weighted speeds .10501/.15025 and alarm rates 16.81%/38.60%; 100-clip intrinsic gate median .36410, misses 90.10%. The model's own map does not rescue the requested criteria.
5. **Tests: PASS**, all 11 in under one minute. No sample or frame reductions, threshold changes, training, GPU computation, data writes, or KWM file edits were made by this work.

Deviations, implementation choices and reasons:

- The external KWM checkout changed to `c7067d3` while the campaign was running. The original mask/tests/breakdown processes retained `2e73903`; later imports used the new optional mask-key API. The campaign leaves that argument unset. All outputs and diagnostics were bitwise equal in a 216-case real-pair audit. Initial and final hashes, revision-stage mapping, and scratch source snapshots are retained. No dependency edit or commit was performed here.
- Added an evidence-localized attribution replay after whole-region SSD did **not** establish global far-field stationarity. This was necessary to distinguish the cells selected by alarms from the rest of the scene. It ran sequentially after the patch stage, which is why its additional item-2 tables appear after item 3. The result supports cone-stationary **imprint** content; it cannot semantically isolate far walls, floor texture, or positional features.
- The public released forward cannot return a zero-length prediction sequence. A local context-only adapter reproduces its original shift/roll/mix/write loop, avoiding prediction and decoder calls. Exact fp16 write parity was tested against hooks on the released forward; no model code or weights were changed.
- Direct search uses the permitted 3×3 SSD/parabolic approach with a uniform patch and exact half-cell Fourier interpolation. It searches intrinsic ±1.5 around the ego-transported previous centre, normalizes shared support, requires half the patch's valid mass, and retains and reports boundary minima. All 44,990 pairs were usable. These choices are documented rather than represented as the full production LocalSearch pooling/rejection policy. The test is an oracle-location diagnostic, not a proof of an information-theoretic ceiling for arbitrary learned estimators.
- Multiple visible blocks make a single nearest-true-node score ambiguous. Both fixed-lattice and adaptive-prior definitions, object/any/all-step success, and evidence before/after selection are reported. A wrong node must be farther than .5 from every visible true velocity. Positive-wrong/zero-true ratios are infinite; 0/0 is undefined, with counts and finite-only means reported separately. No epsilon was added to conceal nonselectivity.
- Action groups are disjoint: neighboring-turn pairs are turn-adjacent even if their current action is forward/no-op. Actual turns carry state but are excluded from fresh detection denominators. Restricted-support runs reset and evolve independently, so near/far alarm totals are not additive. All clips retain 140 frames and therefore 139 incoming pairs before turn exclusion. Visibility retains the original rounded-cone/no-occlusion-filter convention.
- The original normalized-agreement alarm criterion is retained even when mask mode's later search confirmation or mode selection removes evidence. Survivor fractions are shown explicitly so nominal alarms are not confused with final detections. The new memory gate uses the requested 100 clips; no claim of a matched-cohort gate improvement over the 200-clip observation run is made.

Code, CSVs and all scratch artifacts remain in the allowed experiment/test paths. Unrelated concurrent backbone/dataset/script edits were left alone. The original report sections are unchanged. Proposed commit message: `experiments/flowm_bench/k3d/.scratch/COMMIT_MSG2.txt`; no commit was attempted.
