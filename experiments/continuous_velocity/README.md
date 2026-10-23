# Continuous-velocity physics-attention pilot

For the complete experiment history, corrected follow-up metrics, current artifacts, and continuation instructions, start with the [research handoff](../../HANDOFF.md). This README describes the historical one-seed pilot.

This pilot tests learned correspondence attention → differentiable constant-velocity fit → trajectory-constrained memory attention → analytic continuous transport. The implicit appearance field is deferred.

Run scripts directly from the repository root, using the `flowm` environment. Importing through `python -m experiments...` invokes the repository's unrelated training-framework imports.

```bash
python experiments/continuous_velocity/test_model.py
python experiments/continuous_velocity/train.py --kind physical --steps 5000 --device cuda:1
python experiments/continuous_velocity/train.py --kind offset --steps 5000 --device cuda:1
python experiments/continuous_velocity/train.py --kind relaxed --steps 5000 --device cuda:1
python experiments/continuous_velocity/evaluate.py --models physical offset relaxed physical_untrained oracle_velocity --seeds 64 --device cuda:1
python experiments/continuous_velocity/evaluate.py --models flowm25 flowm81 --modes bilinear --seeds 64 --batch-size 4 --device cuda:1
python experiments/continuous_velocity/diagnostics.py --device cuda:1
python experiments/continuous_velocity/report.py
```

## Data and storage

Use the released validation shard and official MNIST IDX files in `data/fractional_velocity`. Training additionally requires `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` from the same official MNIST mirror used for test digits (`https://ossci-datasets.s3.amazonaws.com/mnist/`). Existing repository symlinks place data and all new checkpoints under `/media/diskstation/yixunhu/flowm`. Results remain in `outputs/continuous_velocity/pilot`.

- Training: MNIST training identities 0–54999; released camera episodes 64–223.
- Validation: MNIST training identities 55000–59999; camera episodes 224–255.
- Evaluation: official MNIST test identities; camera episodes 0–63.
- Training velocities: horizontal, uniform on [0,2], excluding ±0.03 around 0.37, 0.63, and 1.23; 15% of samples instead use integer speeds. Vertical velocity is zero in the data, but the model estimates both axes.
- Evaluation: original 13 speeds plus 0.37, 0.63, and 1.23. Note that 1.25 also lies inside an excluded interval.
- All training uses bilinear rendering. Nearest-pixel evaluation is explicitly a rendering-transfer control.
- Each paired evaluation seed fixes digit identity, integer start, and released camera trajectory across every speed and model.
- 50 context frames; training targets at horizons 1, 5, 10, 20, 30, 40. Evaluation predicts every horizon 1–40.

## Architecture

`model.py` has 1,540 learned parameters in its physical variant. A shared CNN produces descriptors, including a raw-intensity skip. Camera-compensated cross-correlation computes correspondence evidence under a shared object translation, using 12 frame pairs at each index lag 1, 2, and 4. Actual timestamps determine displacement fitting and spatial search size. Soft attention over *spatial displacement locations* yields continuous displacement estimates. There is no discrete velocity-channel bank, though spatial sampling and a bounded correspondence search remain.

Learned confidence weights and a differentiable robust weighted least-squares fit infer a single two-dimensional velocity. Temporal attention fuses aligned observations into a fixed 50×50 field that coincides with world coordinates at context time 49 and subsequently co-moves with the object. This retains position implicitly and avoids a separate position estimator. It differs from an object-centered 32×32 crop proposed in the initial sketch; the larger periodic field handles boundary crossings directly.

The canonical field is decoded once. Each future image samples it using absolute cumulative displacement, without recurrently resampling the previous prediction. Velocity, position, digit identity, and templates are never inputs to the learned model. True velocities are logged diagnostically but do not enter the training loss. The objective is foreground-weighted future-pixel squared error.

## Controls and interpretation

- `offset`: replaces the analytic time-times-velocity displacement with a learned MLP conditioned on correspondence statistics and elapsed time. It uses the same appearance encoder, memory attention, decoder, data sequence, optimizer, and update budget; its offset head adds parameters.
- `relaxed`: starts with the *identical physical function*, then adds independently learned per-time residual offsets. This stronger control separates good physical initialization from enforcing the law throughout training.
- `physical_untrained`: same physical architecture and initialization before any updates.
- `oracle_velocity`: separately labeled diagnostic using the trained physical model with true velocity substituted. It is not a deployable result.
- `flowm25`, `flowm81`: released frozen weights, reusing existing paired predictions where available. New cases use the unchanged released model or previously validated accumulated-offset densification. These models did not receive the pilot's continuous-velocity training; comparisons are not architecture-only causal evidence.

All learned models use one optimization seed, 5,000 updates, batch size 8, and Adam at 3e-4. Checkpoint selection uses held-out validation image loss only. Train and validation use identical data sequences across model variants. The strict and relaxed models' shared initial parameters are tested for equality. Training-source snapshots and SHA256 hashes are saved with the run records.

The shared evaluator uses the known digit template for localization, for *every* model, but never supplies it to the neural predictor. Missing and ghost detections are reported separately. Conditional trajectory errors exclude missing, duplicated, and ambiguous tracks, so always report coverage beside ADE. Pixel MSE includes all frames and pixels. Localizer calibration from the earlier experiment establishes a nonzero pixel-scale localization floor.

## Artifacts

- `outputs/continuous_velocity/pilot/report.html`: readable report and plots.
- `summary.json`: aggregate metrics and paired seed-bootstrap intervals.
- `all_metrics.csv`, `by_speed.csv`: sequence-level and speed-level measurements.
- `evaluation/<model>/sequences/*.npz`: actual predictions, GT, actions, visibility, and tracks.
- `<model>/training.csv`, `manifest.json`, `source/`: learning curves and provenance.
- `diagnostics.json`: temporal subsampling and latent time-composition checks.
- `downloaded_checkpoints/continuous_velocity/pilot/<model>/{best,last}.pt`: resumable checkpoints.

One training seed and a single rigid digit do not establish generalization to interacting objects, acceleration, or deformation. Measurements on a shared GPU are not a controlled speed benchmark. `--resume` resumes saved optimizer/data RNG state; evaluation reuses saved predictions. Do not treat a partially populated metrics CSV as a completed evaluation: completion is recorded in its manifest.
