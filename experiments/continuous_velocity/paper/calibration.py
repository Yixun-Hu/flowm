"""Calibrate the fixed evaluation detector on known synthetic corruptions.

No thresholds are tuned here. Duplicates include close overlaps to disclose the
detector's resolution limit; fully/partially visible instances are separated.
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
import cv2
import numpy as np
from metrics import ROOT
from measurement import Detector, translate


def controls(template, seed):
    position = np.array([-4., 0.])
    image = translate(template, position)[:32, :32]
    energy = float((template**2).sum())
    def visibility(img):
        return float((img**2).sum()/max(energy, 1e-8))
    yield 'clean', image, True, False, visibility(image), 0.
    yield 'missing', np.zeros_like(image), False, False, 1., 0.
    for sigma in [1., 2., 3.]:
        yield f'blur_{sigma:g}', cv2.GaussianBlur(image, (0, 0), sigma), True, False, visibility(image), 0.
    for amplitude in [.2, .4, .7]:
        yield f'fade_{amplitude:g}', image*amplitude, True, False, visibility(image), 0.
    yy, xx = np.indices(image.shape)
    for kind, mask in [('fragment_half', xx < 16), ('fragment_stripes', xx % 4 < 2),
                       ('fragment_random_half', np.random.default_rng(seed).random(image.shape) < .5)]:
        yield kind, image*mask, True, False, visibility(image), 0.
    for distance in [4, 8, 12, 16]:
        second = translate(template, position+np.array([distance, 0.]))[:32, :32]
        yield f'duplicate_{distance}', np.clip(image+second, 0, 1), True, True, visibility(image), visibility(second)
    fragment = translate(template, position+np.array([12, 0]))[:32, :32]*(yy < 16)
    yield 'duplicate_fragment', np.clip(image+fragment, 0, 1), True, True, visibility(image), visibility(fragment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/continuous_velocity/paper/calibration')
    parser.add_argument('--digits', type=int, default=128)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT/'data/fractional_velocity/t10k-images-idx3-ubyte.gz', 'rb') as source:
        digits = np.frombuffer(source.read(), np.uint8, offset=16).reshape(-1, 28, 28)/255.
    rows, examples = [], {}
    began = time.monotonic()
    for index, digit in enumerate(digits[:args.digits]):
        template = np.zeros((50, 50), np.float32); template[:28, :28] = digit
        detector = Detector(template)
        for name, image, expected_present, expected_ghost, visibility, secondary_visibility in controls(template, index):
            measured = detector(image)
            eligible = visibility >= .35 and (not expected_ghost or secondary_visibility >= .35)
            rows.append(dict(digit_index=index, condition=name, expected_present=int(expected_present),
                             expected_ghost=int(expected_ghost), present=int(measured['present']),
                             ghost=int(measured['ghost']), eligible=int(eligible),
                             primary_visibility=visibility, secondary_visibility=secondary_visibility,
                             amplitude=measured['amplitude'], quality=measured['quality']))
            if index == 0:
                examples[name] = image
    with (args.out/'detections.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for condition in examples:
        group = [r for r in rows if r['condition'] == condition]
        eligible = [r for r in group if r['eligible']]
        def rate(key, data):
            return float(np.mean([r[key] for r in data])) if data else None
        summary[condition] = dict(n=len(group), n_eligible=len(eligible),
                                  present_rate=rate('present', group), ghost_rate=rate('ghost', group),
                                  eligible_present_rate=rate('present', eligible),
                                  eligible_ghost_rate=rate('ghost', eligible),
                                  expected_present=group[0]['expected_present'], expected_ghost=group[0]['expected_ghost'])
    result = dict(complete=True, digits=args.digits, elapsed_seconds=time.monotonic()-began,
                  detector='Fixed pilot least-squares oracle-template detector; no thresholds tuned',
                  thresholds=dict(visible_energy=.35, primary_amplitude=.25, residual_amplitude=.35,
                                  explained_energy=.30, gain_energy=.10, duplicate_exclusion_radius=8),
                  conditions=summary,
                  interpretation='A zero ghost rate means no ghosts detected by this detector, not absence of fragmented or overlapping copies. Calibration uses oracle template identity, as do all model evaluations.')
    (args.out/'summary.json').write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, (name, image) in zip(axes.flat, examples.items()):
        ax.imshow(image, cmap='gray', vmin=0, vmax=1); ax.set_title(name); ax.axis('off')
    for ax in list(axes.flat)[len(examples):]:
        ax.axis('off')
    fig.suptitle('Known synthetic corruption controls; MNIST test digit index 0')
    fig.tight_layout(); fig.savefig(args.out/'controls.png', dpi=160); fig.savefig(args.out/'controls.pdf'); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
