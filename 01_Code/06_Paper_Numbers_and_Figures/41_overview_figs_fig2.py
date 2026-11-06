# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""41: I14 two overview figures
fig01_regions.png  study regions and station distribution (4 windows + 139 stations + 732 background stations)
fig02_design.png   experiment-design overview (form x strength plane, regime = shape, backbone = colour, parameter count annotated)
Output (overwrites):
    06_Reproduction_Outputs/27_overview/{fig01_regions,fig02_design}.png
    00_Manuscript/figures/{fig01_regions,fig02_design}.png
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Fig. 2 upper panel (experiment-design overview, fig02_design.png)
#   Note: this script also produces fig01_regions.png (six-region map); the main text has replaced that content with Table 1 (locations, extents and station counts of the six experimental windows), so this figure is no longer cited
#   To regenerate: cd latex_paper && python _regen_figs.py
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

from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5

STN = os.path.join(STATION, 'sm_stninfor.txt')
OD = REPRO + '/process figure/27_overview'
LF = MANU_FIG
for d in (OD, LF):
    os.makedirs(d, exist_ok=True)

# (key, name, lon0, lon1, lat0, lat1, climate, colour)
REGS = [
    ('R2', 'middle--lower Yangtze', 114, 117, 30, 33, 'humid',       '#0072B2'),
    ('R3', 'North China Plain',   114, 117, 36, 39, 'sub-humid',     '#009E73'),
    ('R4', 'Northeast China black soil',   125, 128, 44, 47, 'cold-humid / frozen soil',  '#D55E00'),
    ('R5', 'Loess Plateau',   108, 111, 35, 38, 'semi-arid',     '#CC79A7'),
    ('R7', 'Longzhong',      103, 106, 34, 37, 'semi-arid',      '#E69F00'),
    ('R8', 'Sichuan Basin',   105, 108, 29, 32, 'humid',       '#56B4E9'),
]
LAB_BELOW = {'R8'}          # the label is placed below the box when the space above is occupied by another window


def inwin(df, r):
    return ((df['lon'] >= r[2]) & (df['lon'] <= r[3])
            & (df['lat'] >= r[4]) & (df['lat'] <= r[5]))


def load_stations():
    df = pd.read_csv(STN, sep=r'\s+', header=None,
                     names=['id', 'lat', 'lon', 'elev'])
    return df


def fig_regions(df):
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    ax.scatter(df['lon'], df['lat'], s=7, c='#c9c9c9', zorder=1, lw=0,
               label='%d stations (outside window)'
               % (len(df) - int(sum(inwin(df, r).sum() for r in REGS))))
    for key, nm, lo0, lo1, la0, la1, clim, col in REGS:
        m = inwin(df, (key, nm, lo0, lo1, la0, la1, clim, col))
        n = int(m.sum())
        ax.scatter(df.loc[m, 'lon'], df.loc[m, 'lat'], s=16, c=col, zorder=3,
                   lw=0, edgecolors='none')
        ax.add_patch(Rectangle((lo0, la0), 3, 3, fill=False, ec=col, lw=2.7,
                               zorder=4))
        ax.add_patch(Rectangle((lo0 - 0.4, la0 - 0.4), 3.8, 3.8, fill=False,
                               ec=col, lw=1.4, ls='--', alpha=0.75, zorder=4))
        cx = (lo0 + lo1) / 2
        # the box for R8 sits directly below R7's expanded dashed box, so its label is placed below
        if key in LAB_BELOW:
            cy, va = la0 - 0.55, 'top'
        else:
            cy, va = la1 + 0.55, 'bottom'
        ax.annotate('%s %s\n%d stations, %s' % (key, nm, n, clim),
                    (cx, cy), ha='center', va=va, fontsize=13.4,
                    color=col, fontweight='bold', zorder=5)
        print('%s %s: %d stations' % (key, nm, n))
    ax.set_xlim(102.0, 129.4)
    ax.set_ylim(26.8, 48.6)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(38)))
    ax.set_xlabel('Longitude (°E)', fontsize=15.9)
    ax.set_ylabel('Latitude (°N)', fontsize=15.9)
    ax.grid(alpha=0.2, lw=0.8)
    n_in = int(sum(inwin(df, r).sum() for r in REGS))
    h = [plt.Line2D([], [], marker='o', ls='', ms=6, c='#c9c9c9',
                    label='%d stations (outside window)' % (len(df) - n_in))] + \
        [plt.Line2D([], [], marker='o', ls='', ms=6, c=r[7],
                    label='%s %s (%d stations)'
                    % (r[0], r[1], int(inwin(df, r).sum())))
         for r in REGS]
    # Legend placed at the top, outside the axes (the in-figure title duplicating the main-text caption was removed)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.legend(handles=h, loc='upper center', ncol=4, fontsize=12.6,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.985),
               columnspacing=1.6)
    for d in (OD, LF):
        fig.savefig(os.path.join(d, 'fig01_regions.png'), dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print('fig01 ok')


def fig_design():
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    BLUE, RED = '#0072B2', '#D55E00'
    C_SS, C_NS = 'o', 's'

    def put(x, y, ns, col, grey=False):
        c = '#8c8c8c' if grey else col
        ax.scatter([x], [y], s=170, marker=C_NS if ns else C_SS, c=c, zorder=3,
                   edgecolors='k', linewidths=0.7)

    def note(x, y, t, dy=9, fs=8.6, ha='center', color='#333333'):
        ax.annotate(t, (x, y), textcoords='offset points', xytext=(0, dy),
                    ha=ha, fontsize=fs, color=color)

    # ---- light horizontal bands for the three form levels ----
    for y0, c0 in [(0, '#eef3f8'), (1, '#eef7ef'), (2, '#fdf1ea')]:
        ax.axhspan(y0 - 0.42, y0 + 0.42, color=c0, zorder=0)

    # ---- FREE: form=0 (no state equation) ----
    put(-2.7, 0, False, BLUE)
    put(-2.30, 0, True, BLUE)
    note(-2.7, 0, 'LSTM', dy=-26)
    note(-2.30, 0, 'LSTMNS', dy=-26)
    note(-3.05, 0.13, 'Regime control: remove scheduled sampling only', dy=17, fs=8.2,
         ha='left', color='#555555')
    # ---- SOFT: form=1 (soft penalty), w continuous; the LPMIX series all use the LSTM backbone (blue) ----
    wv = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    xs = [np.log10(w) for w in wv]
    for x in xs:
        put(x, 1, False, BLUE)
    put(np.log10(3) + 0.75, 1.0, True, BLUE)
    note(np.log10(3) + 0.75, 1.0, 'LPMIXNS$\\times$3', dy=17)
    note(np.mean(xs), 1.0, 'LPMIX ($w\\in\\{0.03,...,30\\}$, + rerun seeds)',
         dy=-26, fs=8.6)
    # ---- HYBL: form=2, LSTM backbone ----
    put(2.15, 2.0, False, BLUE)
    put(2.55, 2.0, True, BLUE)
    note(2.15, 2.0, 'HYBLST', dy=-26)
    note(2.55, 2.0, 'HYBLSTNS', dy=17)
    # ---- HARD: form=2, GRU backbone, 3 gamma levels ----
    for i, g in enumerate([0, 5, 20]):
        put(3.05 + i * 0.36, 2.0, True, RED)
    note(3.41, 2.0, 'HYBNS', dy=-26, fs=9.8)
    note(3.41, 2.0, '$\\gamma=0$   $\\gamma=5$   $\\gamma=20$', dy=17, fs=8.6)
    # ---- DAMAGED: damaged control ----
    put(4.05, 0, False, BLUE, grey=True)
    note(4.05, 0, 'DAMAGED\nwd$\\geq$0.01', dy=11, color='#666666')

    # ---- axes ----
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['no state equation\n(free increment)', 'soft constraint\n(penalty term added to loss)',
                        'hard constraint\n(state equation holds exactly)'], fontsize=13.8)
    ax.set_ylim(-1.06, 2.75)
    ax.set_xlim(-3.3, 4.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(['0.03', '0.1', '0.3', '1', '3', '10', '30'], fontsize=12.3)
    ax.set_xlabel('Constraint strength: penalty weight $w$ (log)', fontsize=15.2)

    # ---- parameter count (bottom row, with a leading note) ----
    ax.text(-3.30, -0.56, 'params', ha='left', va='center', fontsize=8.2,
            color='#555555')
    for x, t in [(-2.45, '$5.3$/$7.9\\times10^{4}$'),
                 (np.mean(xs), '$2.1\\times10^{5}$'),
                 (2.35, '$2.1\\times10^{5}$'),
                 (3.41, '$9.3\\times10^{4}$')]:
        ax.annotate(t, (x, -0.56), ha='center', va='center', fontsize=11.3,
                    color='#555555')

    # ---- three single-variable control arrows ----
    arr = dict(arrowstyle='->', lw=2.4, color='#333333')
    ax.add_patch(FancyArrowPatch((np.log10(3) + 0.75, 1.0), (2.15, 1.95),
                                 connectionstyle='arc3,rad=-0.22',
                                 shrinkA=8, shrinkB=9, **arr))
    ax.annotate('FORM: change update rule only', (1.44, 1.55), fontsize=11.9, ha='center',
                color='#333333')
    ax.add_patch(FancyArrowPatch((2.62, 2.0), (2.98, 2.0), shrinkA=10, shrinkB=12,
                                 **arr))
    ax.annotate('BACKBONE: change backbone only', (2.80, 2.30), fontsize=11.9, ha='center',
                color='#333333')

    # ---- Legend placed at the top, outside the axes (shape x colour cross-decoding replaced by direct text) ----
    # fig.text(0.5, 0.975,
    #          'Shape: ○ = SS (scheduled sampling)   □ = NS (free-running rollout)    '
    #          'Colour: blue = LSTM backbone   orange-red = GRU backbone   grey = control group',
    #          ha='center', va='center', fontsize=10.4, color='#444444')
    ax.grid(alpha=0.18, lw=0.8, axis='x')
    fig.tight_layout()
    for d in (OD, LF):
        fig.savefig(os.path.join(d, 'fig02_design.png'), dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print('fig02 ok')


def main():
    df = load_stations()
    print('total stations:', len(df))
    fig_regions(df)
    fig_design()


if __name__ == '__main__':
    main()
