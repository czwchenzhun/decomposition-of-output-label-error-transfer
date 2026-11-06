# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""45: month-coverage figure for the per-station de-meaned baseline period (review I8 / Fig. S3)

Data source: produced by 67_leakage_sensitivity.py
    02_Results_Data/truth_chain/multi/month_coverage.csv
Output (overwrites, 11 dpi):
    06_Reproduction_Outputs/28_monthcov/fig15_monthcov.png
    00_Manuscript/figures/fig15_monthcov.png
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Supplementary Fig. S3 (per-station month coverage, fig15_monthcov.png)
#   Data source: month_coverage.csv produced by 67_leakage_sensitivity.py. To regenerate: cd latex_paper && python _regen_figs.py
#

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---- EN-edition layout adaptation ---------------------------------------
# English panel titles are longer than the Chinese originals. Wrap long titles
# onto two lines at a word boundary (never inside maths) and cap the size so
# neighbouring panels do not collide.
import matplotlib.axes as _en_axes
_EN_ORIG_SET_TITLE = _en_axes.Axes.set_title


def _en_set_title(self, label, *args, **kwargs):
    if isinstance(label, str) and '\n' not in label and len(label) > 30:
        _words = label.split(' ')
        _best, _gap = None, 10 ** 9
        for _i in range(1, len(_words)):
            _left, _right = ' '.join(_words[:_i]), ' '.join(_words[_i:])
            if _left.count('$') % 2 or _right.count('$') % 2:
                continue
            _d = abs(len(_left) - len(_right))
            if _d < _gap:
                _gap, _best = _d, (_left, _right)
        if _best:
            label = _best[0] + '\n' + _best[1]
    _fd = kwargs.get('fontdict')
    _fs = kwargs.get('fontsize', (_fd or {}).get('fontsize', 12.5))
    try:
        _fs = float(_fs)
    except (TypeError, ValueError):
        _fs = 12.5
    kwargs['fontsize'] = min(_fs, 12.5)
    if isinstance(_fd, dict) and 'fontsize' in _fd:
        kwargs['fontdict'] = {_k: _v for _k, _v in _fd.items() if _k != 'fontsize'}
    return _EN_ORIG_SET_TITLE(self, label, *args, **kwargs)


_en_axes.Axes.set_title = _en_set_title
# -------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5

SRC = MULTI + '/month_coverage.csv'
OD = REPRO + '/process figure/28_monthcov'
LF = MANU_FIG
for d in (OD, LF):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito
C10, C30 = '#0072B2', '#D55E00'
GREY = '#999999'

REG_ORDER = ['R0_northchina', 'R1_yangtze', 'R3_northeast',
             'R2_loess', 'R4_longzhong', 'R5_sichuan']
REG_CN = {'R0_northchina': 'N. China', 'R1_yangtze': 'Yangtze',
          'R3_northeast': 'NE China', 'R2_loess': 'Loess',
          'R4_longzhong': 'Longzhong', 'R5_sichuan': 'Sichuan'}


def main():
    df = pd.read_csv(SRC)
    d10 = df[df['layer'] == '10cm'].set_index('region').loc[REG_ORDER].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    FS = 14.5 * 1.0          # consistent with the v19 scheme (base font size set to 14.5)
    LW = 1.5

    # ---------------- (a) distribution of months per station ----------------
    ax = axes[0]
    both = df.set_index('region').loc[REG_ORDER].reset_index()
    pos, labels = [], []
    for i, r in enumerate(REG_ORDER):
        for j, ly in enumerate(['10cm', '30cm']):
            row = both[(both['region'] == r) & (both['layer'] == ly)].iloc[0]
            x = i * 3 + j * 1.05
            # draw the five-number summary as "range bar + interquartile band + median line"
            ax.plot([x, x], [row['min_months'], row['max_months']],
                    color=(C10 if ly == '10cm' else C30), lw=LW * 1.6,
                    solid_capstyle='round', zorder=3)
            ax.fill_between([x - 0.22, x + 0.22], row['q1'], row['q3'],
                            color=(C10 if ly == '10cm' else C30), alpha=0.35,
                            zorder=2)
            ax.plot([x - 0.30, x + 0.30], [row['med_months']] * 2,
                    color=(C10 if ly == '10cm' else C30), lw=LW * 2.2, zorder=4)
        pos.append(i * 3 + 0.52)
        labels.append('%s\n$n$=%d' % (REG_CN[r], int(row['nsta'])))
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=FS * 0.68)
    ax.set_ylabel('months per station', fontsize=FS)
    ax.set_title('(a) Months per station', fontsize=FS * 1.05, pad=10)
    ax.grid(axis='y', ls=':', lw=0.8, color=GREY, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.9, 15.9)
    ax.set_ylim(14, 36)
    ax.tick_params(labelsize=FS * 0.85)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---------------- (b) monthly row counts (10 cm, normalised by region) ----------------
    ax = axes[1]
    cols = ['mon_%d' % m for m in range(1, 13)]
    M = d10[cols].to_numpy(dtype=float)
    Mn = M / M.max(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list(
        'okabe', ['#FFFFFF', '#56B4E9', '#0072B2'])
    im = ax.imshow(Mn, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=FS * 0.85)
    ax.set_yticks(range(len(REG_ORDER)))
    ax.set_yticklabels([REG_CN[r] for r in REG_ORDER], fontsize=FS * 0.85)
    ax.set_xlabel('month', fontsize=FS)
    ax.set_title('(b) Monthly pair counts (10 cm, scaled)',
                 fontsize=FS * 1.05, pad=10)
    for i in range(Mn.shape[0]):
        for j in range(Mn.shape[1]):
            ax.text(j, i, '%d' % int(M[i, j]), ha='center', va='center',
                    fontsize=FS * 0.62,
                    color='white' if Mn[i, j] > 0.55 else '#333333')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=FS * 0.78)
    cb.set_label('relative row count', fontsize=FS * 0.85)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(True)

    # ---------------- legend (outside, at the top) ----------------
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C10, lw=LW * 1.8, label='10 cm'),
               Line2D([0], [0], color=C30, lw=LW * 1.8, label='30 cm')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               frameon=False, fontsize=FS * 0.92, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.14, 1, 1.0))
    fig.text(0.5, 0.015,
             '(a) thick lines: min--max months per station; light bands: interquartile range; '
             'horizontal lines: median. (b) Grey cells: zero pair counts. '
             'Station counts in (a).',
             ha='center', fontsize=FS * 0.80, color='#333333')

    for d in (OD, LF):
        p = os.path.join(d, 'fig15_monthcov.png')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    main()
