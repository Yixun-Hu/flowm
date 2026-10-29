Item 1 of denzelzheng/kinetic-world-models#1: evaluation hygiene. Defaults are unchanged; omitting `--test_seed` (or passing 12345) reproduces the original cache names, output names and metrics.

## What it adds

- `kwm.eval --test_seed <S>`: scores a checkpoint on sequences generated from a seed other than the `VAL_SEED = 12345` set that `kwm.train` uses to select `best.pt`. Outputs for a non-default seed are named `eval_seed<S>_<ckpt>[_tag].json` so several checkpoints of one run can be kept side by side.
- Self-describing validation caches: new caches store seed, count, horizons, chunking, the environment config and its hash, the generator source hash and the MNIST test-bank hash. Loading rejects a cache whose generation parameters or config hash differ (`ValueError` naming the file and field); source changes only warn; legacy caches load with one warning.
- Provenance: source-file hashes, software versions and a git summary (scoped to `kwm/` and `scripts/`) are stored in checkpoints and copied into result JSON. Resuming a run appends to a `provenance_history` instead of overwriting, with a warning when the sources changed.
- Per-sequence errors (`per_seq_mse20`, `per_seq_mse150`) in result JSON, and a `sets` object that records which seed fed each reported number.
- `scripts/report.py`: paired multi-seed tables (paired t-test, exact sign-flip test, hierarchical bootstrap, Holm correction) that refuse to pair evaluations with different test seeds, sequence counts, environments or checkpoint names. Uses only numpy. Say if you would rather not have it upstream; it is self-contained.
- `docs/EVAL_HYGIENE.md` documents all of the above, including what a held-out seed does and does not hold out (trajectories and scenes, not a disjoint digit pool).

## Checks

- `pytest -q`: 131 passed (`tests/test_eval_hygiene.py` adds 40 tests: naming, cache verification and legacy fallback, provenance on save/resume, report-script statistics against closed forms, and refusal of mismatched pairs).
- Existing runs and caches work unchanged; `--test_seed` on an existing run only reads its checkpoint.

Code written with AI assistance (OpenAI Codex for implementation, Anthropic Claude for review) under my direction and review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
