import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = Path(__file__).resolve().parent
BASENAME = 'memeqa-transitions'
NAVY = '#174B78'
OCHRE = '#A25A26'
INK = '#243749'


def draw(data_path, out):
    data = json.loads(data_path.read_text())
    rows = data['transitions']
    if (data['questions'], data['images']) != (816, 134) or [r['seed'] for r in rows] != [42, 43, 44]:
        raise ValueError('Unexpected primary cohort or seed order')
    for row in rows:
        counts = [row['corrected'], row['regressed']]
        if any(type(count) is not int or not 0 <= count <= 58 for count in counts):
            raise ValueError('Question count outside the plotted range')
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': BASENAME,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.36, 1.64))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.81)
        ax.set_xlim(-64, 64)
        ax.set_ylim(2.55, -0.55)
        ax.set_yticks(range(len(rows)), [str(r['seed']) for r in rows])
        ax.set_ylabel('Seed', labelpad=5)
        ax.set_xlabel('Questions', labelpad=3)
        ax.xaxis.set_major_locator(FixedLocator([-60, -30, 0, 30, 60]))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E0E7EF', linewidth=0.55)
        ax.axvline(0, color='#5E7387', linewidth=0.8, zorder=3)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#B8C5D0')
        ax.spines['bottom'].set_linewidth(0.65)
        ax.tick_params(axis='both', labelsize=7.8, length=0, pad=4)

        for y, row in enumerate(rows):
            regressed, corrected = row['regressed'], row['corrected']
            ax.barh(y, -regressed, height=0.48, color=OCHRE)
            ax.barh(y, corrected, height=0.48, color=NAVY)
            for x, value, anchor, color in [
                (-regressed - 2, regressed, 'right', OCHRE),
                (corrected + 2, corrected, 'left', NAVY),
            ]:
                ax.text(x, y, str(value), ha=anchor, va='center', color=color, fontweight='semibold')
        ax.text(0.25, 1.09, 'Regressed', ha='center', va='bottom', color=OCHRE,
                fontweight='semibold', transform=ax.transAxes)
        ax.text(0.75, 1.09, 'Corrected', ha='center', va='bottom', color=NAVY,
                fontweight='semibold', transform=ax.transAxes)
        for extension in ('pdf', 'png', 'svg'):
            fig.savefig(out / f'{BASENAME}.{extension}', dpi=300, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE / f'{BASENAME}.json')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    draw(args.data, args.out)
