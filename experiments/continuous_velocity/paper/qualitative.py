"""Fixed-case qualitative evidence from completed schema-2 evaluations only.

No prediction is generated, no method is selected by its result, and no frame is
selected by visibility. Missing evaluations remain explicit missing panels.
Rerun after the campaign finishes to replace those panels with saved evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SPEEDS = (.37, .63, 1.23)
COLORS = dict(physical='#087f8c', relaxed='#7549a1', offset='#bb6044',
              raw_untrained='#60666a', est_q1_transport='#ba4b40',
              est_q05_transport='#cf981d')
LABELS = dict(physical='Physical fit', relaxed='Physical + correction',
              offset='Free offsets', raw_untrained='Raw geometric baseline',
              est_q1_transport='Integer velocity', est_q05_transport='Half-grid velocity')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Corpus:
    def __init__(self, root):
        self.root = root
        self.records = {}
        self.used = {}
        self.missing = {}
        self.rejected = []
        self.cache = {}
        for path in sorted((root/'evaluation').glob('*/*/manifest.json')):
            manifest = json.loads(path.read_text())
            if not manifest.get('complete'):
                continue
            if manifest.get('metric_schema_version') != 2:
                self.rejected.append(dict(path=str(path), reason='not explicit schema 2'))
                continue
            sources = manifest.get('source_sha256', {})
            changed = [name for name, digest in sources.items()
                       if not (ROOT/name).exists() or sha(ROOT/name) != digest]
            if not sources or changed:
                self.rejected.append(dict(path=str(path), reason='source mismatch or missing hashes',
                                          changed_sources=changed))
                continue
            family = manifest['job'].get('family', manifest['model']['kind'])
            seed = int(manifest['model']['train_seed'])
            key = (family, seed, manifest['suite'])
            if key in self.records:
                raise ValueError(f'Ambiguous completed evaluation for {key}')
            self.records[key] = (path, manifest)

    def example(self, family, train_seed, suite, mode, speed):
        if family == 'raw_untrained':
            train_seed = 2026  # Parameter-free; this identifies the saved job.
        case_id = f'seed000__{mode}__vx{speed:+.6g}_vy+0__mnist_regular_released'
        key = (family, train_seed, suite)
        request = '/'.join(map(str, (*key, case_id)))
        if request in self.cache:
            return self.cache[request]
        if key not in self.records:
            self.missing[request] = 'No completed current-source schema-2 evaluation'
            return None
        path, manifest = self.records[key]
        source = path.parent/'examples'/f'{case_id}.npz'
        if not source.exists():
            self.missing[request] = 'Completed evaluation has no requested saved example'
            return None
        with np.load(source) as saved:
            values = {key: saved[key] for key in saved.files}
        required = {'predictions', 'frames', 'positions', 'cameras', 'visibility',
                    'target_indices', 'context_indices', 'offsets', 'continuity_track',
                    'continuity_usable', 'frame_success'}
        if not required <= values.keys():
            self.missing[request] = f'Missing schema-2 arrays: {sorted(required-values.keys())}'
            return None
        self.used[request] = dict(
            family=family, train_seed=train_seed, suite=suite, case_id=case_id,
            example=str(source.relative_to(self.root)), example_sha256=sha(source),
            manifest=str(path.relative_to(self.root)), manifest_sha256=sha(path),
            evaluation_signature=manifest['signature'],
            checkpoint=manifest['model'].get('checkpoint'),
            checkpoint_sha256=manifest['model'].get('checkpoint_sha256'),
            selected_checkpoint_step=manifest['model'].get('train_steps'),
            requested_budget_checkpoint=manifest['job'].get('checkpoint'),
            evaluation_source_sha256=manifest['source_sha256'],
        )
        self.cache[request] = values
        return values


def ensure_paired(examples):
    available = [example for example in examples if example is not None]
    if not available:
        return None
    reference = available[0]
    for example in available[1:]:
        for key in ('frames', 'positions', 'cameras', 'target_indices', 'context_indices'):
            if not np.array_equal(reference[key], example[key]):
                raise ValueError(f'Qualitative panels are not paired: {key}')
    return reference


def save(fig, destination):
    for suffix in ('png', 'pdf'):
        fig.savefig(destination.with_suffix('.'+suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)


def frame_status(example, index, visibility):
    if visibility < .35:
        return 'GT partly/out of view'
    if not bool(example['present'][index]):
        return 'No detection'
    if bool(example['ghost'][index]):
        return 'Duplicate detected'
    return 'Localized within 2 px' if bool(example['frame_success'][index]) else 'Position error >2 px'


def image_sheet(corpus, mode, families, train_seed, suite, horizons, stem, title, note,
                speeds=SPEEDS):
    columns = [(speed, horizon) for speed in speeds for horizon in horizons]
    examples = {(family, speed): corpus.example(family, train_seed, suite, mode, speed)
                for family in families for speed in speeds}
    if not any(example is not None for example in examples.values()):
        return dict(stem=stem, generated=False, reason='No completed examples for this sheet')
    references = {speed: ensure_paired([examples[family, speed] for family in families])
                  for speed in speeds}
    fig, axes = plt.subplots(len(families)+1, len(columns),
                             figsize=(2.0*len(columns), 1.9*(len(families)+1)), squeeze=False)
    fig.subplots_adjust(left=.15, right=.99, bottom=.12, top=.9, wspace=.11, hspace=.35)
    displayed, unavailable = [], []
    for column, (speed, horizon) in enumerate(columns):
        reference = references[speed]
        index = horizon-1
        axes[0, column].set_title(f'$v_x={speed:g}$    $T={horizon}$', fontsize=11)
        for row in range(len(families)+1):
            ax = axes[row, column]
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                example = reference
                label = 'Ground truth'
            else:
                family = families[row-1]
                example = examples[family, speed]
                label = LABELS[family]
            if column == 0:
                ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=15, fontsize=10,
                              color='black' if row == 0 else COLORS[family])
            if example is None or index >= len(example['predictions']):
                ax.set_facecolor('#f3f3f3')
                ax.text(.5, .5, 'Evaluation\nnot complete /\nexample unavailable',
                        ha='center', va='center', transform=ax.transAxes, fontsize=8, color='#6d6262')
                unavailable.append(dict(family='ground_truth' if row == 0 else family,
                                        mode=mode, speed=speed, horizon=horizon))
                continue
            target_index = int(example['target_indices'][index])
            visibility = float(example['visibility'][target_index])
            truth = example['frames'][target_index]
            image = truth if row == 0 else example['predictions'][index]
            ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            if row == 0:
                caption = f'Visible energy {100*visibility:.0f}%'
            else:
                mse = float(np.mean((image-truth)**2))
                caption = f'MSE {mse:.2g}\n{frame_status(example, index, visibility)}'
            ax.set_xlabel(caption, fontsize=7.5, labelpad=3)
            displayed.append(dict(family='ground_truth' if row == 0 else family,
                                  mode=mode, speed=speed, horizon=horizon,
                                  absolute_time=float(example['times'][target_index]),
                                  camera=example['cameras'][target_index].tolist(),
                                  visible_energy=visibility))
    pending = f'  |  {len(unavailable)} panels unavailable' if unavailable else ''
    fig.suptitle(f'{title} — {mode} observations{pending}', fontsize=14)
    fig.text(.15, .047, 'Fixed test seed 0; every specified speed and horizon is retained, including low visibility.\n'
             + note + '\n32×32 camera crops; common grayscale [0,1]. Per-frame labels use the unchanged template detector; fragmented copies may escape detection.',
             fontsize=8, ha='left', va='bottom')
    save(fig, corpus.root/'figures'/stem)
    return dict(stem=stem, generated=True, panels=displayed, unavailable=unavailable)


def trajectory_sheet(corpus, mode, families, train_seed):
    examples = {(family, speed): corpus.example(family, train_seed, 'time', mode, speed)
                for family in families for speed in SPEEDS}
    stem = f'qualitative_trajectories_{mode}'
    if not any(example is not None for example in examples.values()):
        return dict(stem=stem, generated=False, reason='No completed time-suite examples')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.9, hspace=.3, wspace=.27)
    plotted, unavailable = [], []
    for column, speed in enumerate(SPEEDS):
        reference = ensure_paired([examples[family, speed] for family in families])
        if reference is None:
            for ax in axes[:, column]:
                ax.text(.5, .5, 'Completed example unavailable', ha='center', transform=ax.transAxes)
            continue
        last_context = int(reference['context_indices'][-1])
        ti = reference['target_indices']
        dt = reference['times'][ti]-reference['times'][last_context]
        truth = reference['positions'][ti, 0]-reference['positions'][last_context, 0]
        for ax in axes[:, column]:
            ax.plot(dt, truth, color='black', linestyle='--', linewidth=1.7, label='True displacement')
            ax.axvline(40, color='#9b9b9b', linestyle=':', linewidth=1)
            ax.grid(alpha=.18)
        for family in families:
            example = examples[family, speed]
            if example is None:
                unavailable.append(dict(family=family, mode=mode, speed=speed))
                continue
            track = example['continuity_track'][:, 0].copy()-reference['positions'][last_context, 0]
            valid = example['continuity_usable'].astype(bool)
            track[~valid] = np.nan
            axes[0, column].plot(dt, track, color=COLORS[family], linewidth=1.3,
                                 label=f'{LABELS[family]} ({int(valid.sum())}/160 frames)')
            axes[1, column].plot(dt, example['offsets'][:, 0], color=COLORS[family],
                                 linewidth=1.3, label=LABELS[family])
            plotted.append(dict(family=family, mode=mode, speed=speed,
                                continuity_scored_frames=int(valid.sum()), total_frames=len(valid)))
        axes[0, column].set_title(f'$v_x={speed:g}$', fontsize=12)
        axes[0, column].set_ylabel('Image-derived unwrapped Δx (px)')
        axes[1, column].set_ylabel('Internal transport Δx (px)')
        axes[1, column].set_xlabel('Frames after last context observation')
        for ax in axes[:, column]:
            ax.legend(fontsize=7, loc='upper left')
    pending = f'; {len(unavailable)} method/speed curves unavailable' if unavailable else ''
    fig.suptitle(f'Long-horizon evidence — {mode}; fixed test seed 0, train seed {train_seed}{pending}', fontsize=13)
    fig.text(.07, .045,
             'Top: positions from generated images, unwrapped using continuity; missing, duplicate and ambiguous detections create gaps.\n'
             'The context establishes one coordinate origin. GT does not reset winding at later frames. Lower coverage is not evidence of lower error.\n'
             'Bottom: the model’s internal displacement output; this diagnostic does not establish correct object appearance or localization.\n'
             'Dotted line marks the training horizon of 40. Curves are individual fixed examples, not aggregate results.',
             fontsize=8, va='bottom')
    save(fig, corpus.root/'figures'/stem)
    return dict(stem=stem, generated=True, curves=plotted, unavailable=unavailable)


def canonical_memory_checks(corpus):
    checks = []
    for mode in ('bilinear', 'nearest'):
        for speed in SPEEDS:
            continuous = corpus.example('physical', 2026, 'core', mode, speed)
            for family in ('est_q1_transport', 'est_q05_transport'):
                quantized = corpus.example(family, 2026, 'intervention', mode, speed)
                if continuous is None or quantized is None:
                    continue
                ensure_paired([continuous, quantized])
                error = float(np.max(np.abs(continuous['canonical_memory']-quantized['canonical_memory'])))
                checks.append(dict(mode=mode, speed=speed, family=family,
                                   canonical_memory_max_abs_difference=error))
                if error > 1e-5:
                    raise ValueError(f'Transport-only intervention changes canonical memory: {checks[-1]}')
    return checks


def run(root, train_seed=2027):
    root = Path(root).resolve()
    (root/'figures').mkdir(exist_ok=True)
    corpus = Corpus(root)
    figures = []
    families = ['physical', 'relaxed', 'offset', 'raw_untrained']
    for mode in ('bilinear', 'nearest'):
        seed_description = (f'independent post-HPO seed {train_seed}' if train_seed > 2026
                            else f'HPO reference seed {train_seed}')
        figures.append(image_sheet(
            corpus, mode, families, train_seed, 'core', (20, 40), f'qualitative_models_{mode}',
            'Fixed-case method comparison',
            f'Trained methods use {seed_description}; raw geometric baseline has no fitted parameters.'))
        for speed in SPEEDS:
            figures.append(image_sheet(
                corpus, mode, families, train_seed, 'core', tuple(range(5, 41, 5)),
                f'qualitative_uniform_{mode}_v{str(speed).replace(".", "p")}',
                f'Uniform temporal contact sheet, v={speed:g}',
                f'All multiples of 5 from T=5 to40 are shown; trained methods use {seed_description}.',
                speeds=(speed,)))
        # Source suite differs for continuous and quantized predictions. Build a
        # read-only view routing only the continuous reference to its core suite.
        original_example = corpus.example
        def transport_example(family, seed, suite, renderer, speed):
            return original_example(family, seed, 'core' if family == 'physical' else 'intervention', renderer, speed)
        corpus.example = transport_example
        try:
            figures.append(image_sheet(
                corpus, mode, ['physical', 'est_q1_transport', 'est_q05_transport'], 2026,
                'intervention', (20, 40), f'qualitative_transport_{mode}',
                'Fixed-checkpoint transport quantization',
                'All rows use checkpoint seed 2026 (HPO reference); estimated continuous memory is held fixed. Only future transport is quantized.'))
        finally:
            corpus.example = original_example
        figures.append(image_sheet(
            corpus, mode, families, train_seed, 'time', (40, 160), f'qualitative_long_{mode}',
            'Fixed-case extrapolation beyond training horizon',
            f'Training predicts at most 40 frames; trained methods use seed {train_seed}. Future targets through 160 are never used for fitting.'))
        figures.append(trajectory_sheet(corpus, mode, families, train_seed))
    memory_checks = canonical_memory_checks(corpus)
    provenance = dict(
        generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        selection=dict(test_seed=0, training_seed=train_seed, quantization_training_seed=2026,
                       speeds=SPEEDS, renderers=['bilinear', 'nearest'],
                       main_horizons=[20, 40], extrapolation_horizons=[40, 160],
                       supplementary_uniform_horizons=list(range(5, 41, 5)),
                       context='regular', frame_selection='fixed horizons, without visibility or error filtering'),
        script=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(__file__),
        metric_schema=2, completed_current_source_evaluations_only=True,
        figures=figures, examples=list(corpus.used.values()),
        missing_requests=corpus.missing, rejected_evaluations=corpus.rejected,
        canonical_memory_checks=memory_checks,
        caveats=[
            'Fixed examples illustrate behavior and do not establish aggregate superiority or training stability.',
            'Missing panels are pending/unavailable evaluations, not model failures.',
            'Nearest-rendered observations are a rendering transfer test for bilinear-trained methods.',
            'A black prediction may be correct when the ground-truth object is outside the camera view.',
            'The template detector can miss fragments, overlapping duplicates, or blurry appearances.',
            'Image-derived trajectories have explicit continuity gaps; internal transport curves are separate diagnostics.',
            'Raw baseline has no trained parameters; train seed 2026 only identifies its saved job.',
            'Transport quantization changes one inferred velocity, and does not emulate FloWM channel-wise readout.',
        ])
    (root/'figures/qualitative_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(generated_figures=[f['stem'] for f in figures if f['generated']],
                          pending_figures=[f['stem'] for f in figures if not f['generated']],
                          used_examples=len(corpus.used), missing_requests=len(corpus.missing)), indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/continuous_velocity/paper_v1')
    parser.add_argument('--train-seed', type=int, default=2027)
    args = parser.parse_args()
    run(args.root, args.train_seed)


if __name__ == '__main__':
    main()
