# Opus review: PR3 prediction-input modes (commit d093e2e, branch `pr3-pred-input`, base 0c1198a)

Reviewer scratch (scripts and logs): `/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/pr3_review/`
(`bitid.py`, `repro.py`, `variants.py`, `mixonce.py`, `hybrid32.py`, `*.log`). All runs were CPU only, 4 torch threads, read-only `ckpt_step*.pt` archives, and the same selection cache (sha `1d3f58ce…`). Nothing was written to the worktree or to `kinetic-world-models`.

## VERDICT

**APPROVE WITH CHANGES. No blockers.** The implementation is correct. Defaults are bit-identical to the base (I checked this independently against the base commit's code, not only with the PR's own tests). `frozen` reproduces the reviewer's B1 numbers exactly, and the NO-GO for frozen variants holds up. The changes are about scope and claims:

- Do not expose `frozen_prelift` in the training CLI.
- Correct the ghosting explanation. At τ = 1 the gap comes entirely from the missing mix step, not from ghosts.
- Record the cheap variants a maintainer will ask about. I ran them below; none passes robustly.
- Change H3's primary horizon from MSE@150 to MSE@20.

## Verification performed

| Check | Result |
|---|---|
| Base vs PR, real checkpoints (lock 9000, lk 11000), 2 sequences: forward `flowm` and `zero` (8 prediction steps), plus training-mode gradients (`tf_ratio = 0.5`, grad_ckpt path, fixed seeds) | `torch.equal` True for all 6 comparisons |
| PR `--dec_input frozen`, 32 sequences, lock 9000 / lk 11000 | 0.0566/0.0843/0.1275 and 0.0694/0.1042/0.1532: **identical** to B1 at 4 decimals |
| PR `zero` on the same setup | 0.0218/0.0343/0.0689 and 0.0284/0.0475/0.0833: identical to B1 |
| My own reimplementation of frozen and prelift K=0 at the gate checkpoints (16 sequences) | 0.060444 and 0.048426 (lock), 0.084617 and 0.052015 (lk): identical to the gate JSONs |
| `tests/test_pred_input.py`, CPU, 1 thread | 73 passed in 8.6 s. No GPU use. Does not read `data/` (the CLI test monkeypatches `val_cache`) |
| Doc line references (cells.py 62–67, 84, 100, 232, 249, 267) | Correct |

## Q1. Default bit-identity, CLI, checkpoint format

- **Numbers.** Bit-identical, as shown above. `feature_update` (cells.py:62–67) keeps the original operation order: lift, then mix, then add, then activation.
- **File names.** `configure_prediction` (eval.py:75–89) returns `""` when no override is given or when the override equals the saved mode, so default file names are unchanged. The CLI test checks this byte for byte.
- **JSON content does change by default.** It gains `dec_input`, `frozen_k`, `node_update_pred_ignored` and `node_concentration`. The doc says so.
- **Evaluation cost increases by default.** The default evaluation now always computes concentration (eval.py:400–402, model.py:355–357).
- **Training CLI.** `kwm.train --dec_input` accepts a strict superset: `flowm` and `zero` behave as before, and `frozen_prelift` is new.
- **Checkpoint compatibility.** New checkpoints store `frozen_k`. Old checkpoints default to `g("frozen_k", 1)` (model.py:392), so they load.
- **Forward compatibility gap.** A checkpoint trained with `dec_input=frozen_prelift` and loaded by pre-PR3 code silently runs as `flowm`. The old `_forward` treats every mode other than `zero` as autoregressive and does not validate the value. This is minor, but it is one more reason not to ship training support.

## Q2. Frozen modes against the spec

- **`frozen`.** The anchor is the final post-lift, post-ReLU `h_T` (model.py:330). There is one spectral shift per τ from that anchor: `stream(feat, nu, Σ_{s≤τ} a_s, τ·dt)`, which gives displacement `ν τ dt − Σa` (transport.py:64). This sign matches the recurrence. `ν = lattice + offsets(h)`, which for lock is `LockState.nodes[..., :2]`, so the carried velocity is correct. Measurement metadata is carried unchanged. There is no recurrence, measurement or lift in prediction. This is correct, and it equals B1 bit for bit.
- **`frozen_prelift` with K = 0.** The frozen tensor is `self.mix(hs)` at cells.py:65, where `hs` is the observation-step transport of `h_{T−1}`. In LK, `hs` is the feature slice of the stream output (cells.py:212). In lock it is cells.py:249. In both, any collision has been applied first. The tensor is taken before the lift is added and before the ReLU. ReLU is applied after each transport (model.py:256–257). Metadata comes from `h_T`, so it includes the T-th measurement. This matches the spec.
- **`--frozen_k K > 0` does what the doc says, but the knob mixes two effects** (model.py:308–329):
  1. The anchor is the state after observation T−K.
  2. K zero-input steps follow, using `actions_in`, with no measurements.

  So K > 0 throws away the last K observations: no lifts and no node evidence from them. Larger K therefore gives more recurrence and less evidence at the same time, and the sweep cannot tell these apart.

  Also, K = 1 is K = 0 with ReLU applied before the transport, and with `h_{T−1}` metadata in place of `h_T`. The reason is that `lift(0) = conv_u(0) = 0` (no bias), so A(0 + M S h_{T−1}) = A(K0 tensor). That is why K0 and K1 differ by less than 2%. K = 1 is redundant as a sweep point, and K = 1 is also the default.

  The doc does disclose "K>0 deliberately excludes the last K observation lifts", which is honest. A variant that keeps all T observations and then runs K zero steps on `actions_out` is the natural alternative, and it is the "hybrid" tested under Q3.
- **Double activation for K > 0.** The anchor is already ReLU'd, and ReLU is applied again after transport. ReLU is idempotent, and a fractional spectral shift creates small negative ringing that the second ReLU clips. This is harmless and documented.

## Q3. Gate protocol and robustness of the NO-GO

The protocol is sound:

- Same cache and first-N sequences as the reviewer.
- fp32, no autocast, predictions clamped to [0, 1].
- Prefix MSE is the mean over the first H frames, sequences and pixels, which is the reviewer's definition.
- Prediction-time node updates are on for `flowm` and `zero`, as in B1. B1 measured that off vs on changes the result by less than 0.5%, so this does not bias against frozen.
- 16 vs 32 sequences does not matter at these margins. At MSE@20, frozen ≤ best baseline in 0/16 sequences, and the best frozen variant, prelift K4, in 1/16 (lock) and 0/16 (lk). The best frozen variant is 63% (lock) and 64% (lk) worse at MSE@20.
- Steps 10k/13k vs 9k/11k changes nothing qualitatively. Frozen reproduces exactly at 9k/11k.

The one built-in bias is unavoidable and should be stated. An evaluation-time gate asks a decoder trained on recurrent prediction states (`flowm` with tf = 0.125) to decode states it never saw. "Frozen is worse at evaluation time" therefore does not mean "frozen is worse when trained". The PR's training support would be the fair test, but the plan does not budget for it, and I do not recommend budgeting for it (see below).

**Ghosting is not the cause at short horizons (doc overclaim).** In the gate JSONs, frozen is already 1.5× (lock) and 1.9× (lk) worse than `zero` at τ = 1. At τ = 1 neighbouring copies are only about 1 px apart. I decoded `ReLU(M(S_τ h_T))`, that is, frozen transport followed by one mix and ReLU before decoding. It matches `zero` **exactly** at τ = 1 (0.013967 / 0.013331). So the τ = 1 gap is the missing learned mix step: decoder/state mismatch, not ghosting. Over longer horizons this "mix once" variant still loses: MSE@20/@150 is 0.0449/0.1170 for lock and 0.0480/0.1414 for lk.

**Ghosting is the cause at long horizons.** Evidence:

- The memory energy over the window is close to uniform across nodes: the effective number of nodes is 23.0/25 (lock) and 24.5/25 (lk), and the largest node share is 0.086 / 0.062. This confirms the lift duplicates content.
- Hard de-ghosting before transport recovers most of the MSE@150 gap, but not the MSE@20 gap (16 sequences, steps 10k/13k):

| Variant (evaluation time, 16 sequences) | lock MSE@20 / @50 / @150 | lk MSE@20 / @50 / @150 |
|---|---|---|
| min(flowm, zero) | 0.0268 / 0.0426 / 0.0834 | 0.0312 / 0.0542 / 0.0950 |
| frozen (PR) | 0.0604 / 0.0854 / 0.1241 | 0.0846 / 0.1204 / 0.1744 |
| per-pixel winner-take-all node, post-lift | 0.0511 / 0.0584 / **0.0800** | 0.0702 / 0.0819 / 0.1080 |
| per-pixel winner-take-all node, pre-lift + ReLU | 0.0714 / 0.0742 / 0.0911 | 0.0996 / 0.1016 / 0.1167 |
| oracle: true-velocity node inside digit boxes, WTA elsewhere | 0.0567 / 0.0632 / 0.0826 | 0.0735 / 0.0849 / 0.1099 |
| only locked nodes (lock flag > 0.5; mean 4.75 per sample) | 0.0751 / 0.0896 / 0.1114 | — |
| transport, then one M + ReLU | 0.0449 / 0.0712 / 0.1170 | 0.0480 / 0.0825 / 0.1414 |
| **hybrid:** zero-input for 20 steps (the training horizon), then frozen | 0.0268 / **0.0367** / **0.0660** | 0.0312 / **0.0487** / 0.0975 |

The hybrid's MSE@20 is identical to `zero` by construction.

**I re-ran the hybrid on the reviewer's checkpoints** (lock 9000 and lk 11000, 32 sequences):

| | lock MSE@20 / @50 / @150 | lk MSE@20 / @50 / @150 |
|---|---|---|
| hybrid | 0.0218 / 0.0344 / 0.0720 | 0.0284 / 0.0484 / 0.0946 |
| zero | 0.0218 / 0.0343 / 0.0689 | 0.0284 / 0.0475 / 0.0833 |
| flowm | — / — / 0.0749 | — / — / 0.0924 |

So the hybrid passes the gate in 1 of 4 checkpoint/rule combinations. It wins only where `zero`'s long-horizon error has degraded (lock 10k). It sits between `zero` and `flowm` otherwise.

**Conclusion.** The NO-GO holds for every pure-frozen and de-ghosted variant I could construct cheaply, including the oracle-node mask. The answer a maintainer would want is that de-ghosting fixes the tail but not τ ≤ 20, because the decoder needs the recurrence. Put this table, or a short version of it, in GATE_pred_input.md. Report the hybrid as "not robust: 1/4", so that nobody re-proposes it. The plan's earlier decision to drop the hybrid is confirmed by data, not just by assumption.

## Q4. node_concentration

- **Computation: correct as defined.**
  - The nearest node uses the carried world-frame ν against the true world velocity at the same predicted frame.
  - Masks use `cam[:, t]` and `cent[:, t]` from the prediction-aligned slice (diagnostics.py:42–48).
  - Energy is the squared feature channels only.
  - NaN marks invisible digits.
  - The test oracle and edge cases are good.
- **Weak as evidence.**
  1. It does not track MSE. `flowm` scores lower than `frozen` (lock 0.121 vs 0.126 at τ = 1; lk 0.081 vs 0.085), yet `flowm` has half the error.
  2. It is diluted by background pixels inside the box, by overlapping digits and by older trails.
  3. Its absolute level (0.08–0.20 against a uniform 0.04) and its flat τ profile in frozen modes do not show copies "spreading apart". If they did, the selected share should rise with τ, because wrong-velocity copies leave the box.

  The node energy share / effective node count above is the cleaner ghosting statistic. It shows about 23–24.5 of 25 nodes effective.
- **Recommendation.** Keep the diagnostic, which is cheap and correct. Add the effective node count at τ = 0, and soften the GATE doc: concentration is "consistent with but not diagnostic of" ghosting.

## Q5. zero vs flowm and H3

Paired per-sequence data from the gate JSONs (16 sequences, lock 10k / lk 13k), zero minus flowm:

| Horizon | lock: mean, t, zero wins | lk: mean, t, zero wins |
|---|---|---|
| @20 | −0.00259, t = −3.4, 14/16 | −0.00406, t = −6.1, 15/16 |
| @50 | t = −3.1, 12/16 | t = −4.4, 14/16 |
| @150 | +0.00145, t = +0.9, 6/16 | −0.0166, t = −2.4, 12/16 |

At 9k/11k, 32 sequences, zero wins at every horizon.

- Across 4 checkpoint/rule cells, `zero` beats `flowm` at @20 in 4/4 (−9% to −12%) and at @150 in only 3/4. The @150 numbers are also what lk's checkpoint spikes hit.
- H3 as written ("zero < flowm at MSE@150, lock model") picks the **least robust** endpoint, and the one cell where it currently fails is that very lock cell.
- **Recommendation: primary MSE@20.** It is consistent with H1 and is the trained horizon. MSE@50 and MSE@150 should be secondary, under Holm, or as a reported stability measure.
- Keep the rest of the design: lock model, checkpoint-averaged (16k–20k), held-out test seed, 5 seeds of PR1 checkpoints at evaluation time. Add the 3 trained `--dec_input zero` seeds.
- Pre-register one sentence: if trained-zero is not better than evaluation-time-zero, report evaluation-time zero only.
- The 3 trained-zero runs are reasonable, but they are the first cut if budget tightens. The evaluation-time switch is free.

## Q6. Tests, docs, style, cost

- **Default-identity tests compare against `old_forward`, not against the base commit's code** (tests/test_pred_input.py:41–59, 62, 309). `old_forward` reimplements the rollout loop on top of the *new* `step` and `_step`. It only monkeypatches `feature_update`, so any regression inside `_step`'s plumbing or the `return_prelift` branches would also appear in the reference and go unnoticed. My base-code check closes this gap for now. A stored golden tensor, or a test that imports a frozen copy of the base functions, would make it permanent.
- **Spec items covered:**
  - identity-tap at τ = 1, at 1e-5, with rounded actions and readout max (:95–112)
  - pre-lift K0 = post minus lift, on a constructed non-negative case (:115–139)
  - comoving on-lattice boost with interior/target masks, reusing the equiv helpers (:183–222)
  - suffix rule, including `_fk` (:325–338)
  - zero override equals the saved zero mode (:75–85)
  - concentration oracle (:254–289)
  - training gradients and checkpoint parity
  - 8.6 s total
  - CPU only, no `data/`
- **Test gap.** The boost test does not move scene content: obs is identical and only the camera actions are boosted. That is the correct comoving construction, but it should say so in a comment.
- **Docs overclaims and inaccuracies:**
  - PR3_pred_input.md "Pilot gate and limitations" and GATE "Runtime and limitations" state ghosting as the mechanism. Add the τ = 1 / mix-once finding.
  - The README says `kwm.train --dec_input` accepts `frozen_prelift` without the NO-GO next to it. It should at least say "experimental, failed the evaluation-time gate".
  - GATE's "30-minute budget was interpreted as wall time" is fine, but the 143 core-min should be the headline cost figure.
- **Style:**
  - The local import inside the prediction loop (model.py:356) should be a module-level import.
  - `_forward` has grown three interleaved mode flags (model.py:289–360). Pulling the frozen branch out into `_predict_frozen(anchor, actions_out, …)` would make the default path visibly untouched, and maintainers will ask for that.
  - `transport_frozen` and `with_features` are public methods without underscores; fine, but name them consistently.
  - `pred_input_gate.py:32` has `--n` default 32 while the committed results use 16. Make the default 16, or require `--n`.
  - `pred_input_gate.py:59` sets `node_update_pred=True` for frozen rows too. It is harmless, but the comment should say so.
- **Gate cost (143 core-min). Trim it:**
  1. Run the observation pass once per checkpoint and branch to all frozen variants. In my runs the observation pass took 21–28 s of the roughly 70 s per frozen row.
  2. Drop K = 1, which is redundant with K = 0 (see Q2), and K = 2.
  3. Reuse `flowm` and `zero` from the PR5a/PR1 evaluation instead of recomputing them. They are about 60% of the CPU time.

  This brings it to about 30–40 core-min, with no loss of information.

## BLOCKERS

None.

## MAJOR

1. **Training support for `frozen_prelift` should not ship** (train.py:164–167, model.py:290–342 training path, tests :225–251). It failed the gate, no budget plans to train it, and it creates a forward-compatibility hazard with pre-PR3 code (Q1). Either keep it behind an evaluation-only flag, or keep the CLI at `flowm|zero` and leave a one-line note in the doc that the analytic transport is differentiable.
2. **The ghosting narrative overclaims** (docs/PR3_pred_input.md "Pilot gate and limitations"; docs/GATE_pred_input.md last paragraph). At τ = 1 the whole gap is the missing mix step (mix-once equals zero exactly). Ghosting explains the tail: de-ghosting recovers MSE@150 to within 0–4% of the best baseline on lock, but not MSE@20. State both mechanisms.
3. **The gate doc lacks the variants a maintainer will ask about.** Add the de-ghosting / oracle-node / locked-nodes / hybrid rows above, with the hybrid's 1/4 result. This makes the NO-GO decisive rather than "we tried the pre-lift tensor".
4. **H3 primary horizon.** Change it from MSE@150 to MSE@20 (PLAN_v4_KWM.md PR3 section and amendment). Data in Q5.
5. **`--frozen_k` conflates recurrence and dropped evidence; K = 1 is effectively K = 0, and K = 1 is the default** (model.py:308–329; README; PR3 doc). Either make the default K = 0, the defined "frozen_prelift", or redefine K as "K zero steps after the last observation". The latter is the hybrid, which is now measured as not robust, so K = 0 as the default is simplest.

## MINOR

6. model.py:356: move the local import to module level.
7. Extract `_predict_frozen` from `_forward` (model.py:330–360) so the default path's diff is easier to review.
8. scripts/pred_input_gate.py:32: make the `--n` default match the committed run (16) or require it. :59: add a comment that the setting is ignored for frozen rows.
9. tests/test_pred_input.py:41: add a golden-tensor check against base-commit output, or a docstring stating the limitation. :183: add a comment that the comoving boost moves the camera, not the content.
10. node_concentration: add the effective-node-count statistic and soften the claims (Q4). The default evaluation now pays the concentration cost on every run (eval.py:400). Acceptable, but say so in the doc.
11. GATE doc: lead the runtime section with core-min. Trim the gate as described in Q6.
12. README lines 82–87: add "experimental; failed the pilot gate" next to the `frozen*` flags.

## RECOMMENDATION

**PR3 should ship:**

- `kwm.eval --dec_input {flowm,zero,frozen,frozen_prelift}` as an evaluation-only switch, with the suffix rule and JSON metadata.
- `frozen` as the documented negative control, and `frozen_prelift` with K = 0 as the default.
- The concentration diagnostic plus the effective node count.
- The gate doc, extended with the variant table and the two-mechanism explanation.
- **No** training support for frozen modes.
- The headline is the `zero` vs `flowm` evaluation-time switch, not freeze-and-render.

**H3 should be:** `zero` < `flowm` at **MSE@20** on K-real with the lock model, checkpoint-averaged (16k–20k), on the held-out test seed, over 5 seeds (evaluation time on the PR1 checkpoints), with a paired t-test. MSE@50 and @150 are secondary, with Holm, and MSE@150 stability is reported. The 3 trained `--dec_input zero` seeds are a secondary arm and the first cut if budget tightens. Freeze-and-render, the hybrid and de-ghosting are closed as negative results.

## Prioritized fix list

1. kwm/train.py:164–167: remove `frozen_prelift` from choices, or move it behind an experimental flag. Drop or relocate tests/test_pred_input.py:225–251.
2. docs/PR3_pred_input.md (Pilot gate and limitations) and docs/GATE_pred_input.md (Gate; Runtime and limitations): add the τ = 1 mix-once result and the de-ghosting/hybrid table. Restate ghosting as the long-horizon mechanism only.
3. Plan (PLAN_v4_KWM.md PR3 and the amendment): change the H3 primary to MSE@20.
4. kwm/model.py:63, kwm/train.py:166, kwm/eval.py:78: make the `frozen_k` default 0, or document K = 1 ≈ K = 0 and that K > 0 drops evidence.
5. kwm/model.py:330–360: extract `_predict_frozen`. Hoist the import at :356.
6. scripts/pred_input_gate.py:32 and :59: fix the `--n` default and add the comment. Share the observation pass and drop K ∈ {1, 2} to cut the core-min.
7. kwm/diagnostics.py: add the effective node count. Soften the concentration claims in the docs.
8. tests/test_pred_input.py:41: add a base golden reference. :183: add the comoving comment.
9. README.md:82–87: add the experimental/failed-gate note.
