"""Plot training-only epoch losses from main.py's saved logs."""
import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ('loss_d', 'loss_g', 'loss_rec', 'loss_adv')
PATTERN = re.compile(r'epoch=(\d+)/(\d+)\s+' + r'\s+'.join(k + r'=(\S+)' for k in KEYS))


def plot(run_dir, output_dir=None):
    output_dir = output_dir or run_dir / 'loss_curves'
    stages = {}
    for stage in ('average', 'anomaly'):
        path = run_dir / 'logs' / (stage + '.log')
        rows = []
        for match in PATTERN.finditer(path.read_text()):
            values = tuple(float(v) for v in match.groups()[2:])
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'Nonfinite loss in {path}')
            rows.append((int(match[1]), *values))
        if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError(f'Missing or non-increasing epoch records in {path}')
        stages[stage] = rows
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'training_losses.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('stage', 'epoch', *KEYS))
        for stage, rows in stages.items():
            writer.writerows((stage, *row) for row in rows)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    names = ('Discriminator', 'Generator total', 'Reconstruction', 'Adversarial (generator)')
    for i, (stage, rows) in enumerate(stages.items()):
        for j, name in enumerate(names):
            ax = axes[i, j]
            ax.plot([r[0] for r in rows], [r[j+1] for r in rows], 'o-', color=('tab:blue', 'tab:orange')[i])
            ax.set(title=f'{stage.capitalize()}: {name}', xlabel='Epoch', ylabel=KEYS[j])
            ax.grid(alpha=.25)
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle('Training loss curves (no validation data)\nAverage reconstruction loss includes an epoch-dependent monthly weight', fontsize=13)
    for extension in ('png', 'pdf'):
        path = output_dir / ('training_losses.' + extension)
        fig.savefig(path, dpi=180)
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    plot(args.run_dir, args.output_dir)
