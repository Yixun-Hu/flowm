"""One additional post-selection replicate for each mechanism control.

The tuning seed is disclosed separately. These fits give three fresh training
seeds (2027/2028/2029) for every core comparison, beyond tuning seed 2026.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.train import ROOT, run_training, atomic_json


def main():
    out = ROOT/'outputs/continuous_velocity/paper_v1'
    out.mkdir(parents=True, exist_ok=True)
    required = ['offset', 'mlp_velocity', 'mean_memory', 'raw_correlation']
    atomic_json(out/'additional_protocol.json', dict(
        seed=2029, kinds=required, steps=5000, learning_rate='selected by original validation search',
        purpose='three post-HPO training replicates for every core control',
        note='physical/relaxed already include 2029/2030 in the main campaign'))
    path = out/'additional_runs.json'
    added = json.loads(path.read_text()) if path.exists() else {}
    while len(added) < len(required):
        selected_path = out/'selected_runs.json'
        selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
        ready = [kind for kind in required if kind in selected and kind not in added]
        if not ready:
            time.sleep(10)
            continue
        kind = ready[0]
        lr = selected[kind]['lr']
        run_id = f'{kind}_bilinear_horizontal_s2029_lr{lr:g}'
        run_training(dict(kind=kind, seed=2029, lr=lr, run_id=run_id))
        added[kind] = dict(lr=lr, runs=[run_id])
        atomic_json(path, added)
    atomic_json(out/'additional_training_complete.json', dict(complete=True, runs=added))


if __name__ == '__main__':
    main()
