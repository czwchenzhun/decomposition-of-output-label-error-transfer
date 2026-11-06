# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""44: Model-architecture schematic (Fig. 2 lower panel)

Purpose: Table 1 lists the composition of the five families in words, which is hard to visualise; this figure draws four
      representative architectures (FREE / SOFT / HYBL / HARD) as four rows of one template, so that
      the three differences "where memory comes from", "what decides the state update" and "how the constraint enters"
      are readable at a glance. The damaged control has the same structure as FREE (only higher weight decay), so it is not shown as a separate row.

All structures follow the actual implementation in scripts/13_dl_hybrid.py:
  LSTM    : LSTMCell -> head -> free increment
  LSTMPHY : LSTMCell -> head -> free increment (state); flux head computes dtheta_wb for the penalty only
  LSTMHYB : LSTMCell -> head used as bounded residual U; flux head directly determines the state update
  PHY     : feed-forward flux head (memoryless) + ResFlux(GRUCell) residual -> hard water-balance update

Output: 06_Reproduction_Outputs/27_overview/fig02_arch.png
      00_Manuscript/figures/fig02_arch.png
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Fig. 2 lower panel (schematic of the four representative architectures, fig02_arch.png)
#   To regenerate: cd latex_paper && python _regen_figs.py
#

import os
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

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5

OD = REPRO + '/process figure/27_overview'
LF = MANU_FIG
for d in (OD, LF):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito (as used throughout)
C_FREE, C_SOFT, C_HYBL, C_HARD = '#0072B2', '#009E73', '#CC79A7', '#D55E00'
C_GREY = '#666666'

# ---- columns (data coordinates) ----
COL = {
    'in':  (0.15, 2.15),
    'bb':  (2.45, 4.45),
    'hd':  (4.75, 7.15),
    'up':  (7.45, 10.25),
    'ls':  (10.55, 13.25),
}
GAP = [('in', 'bb'), ('bb', 'hd'), ('hd', 'up'), ('up', 'ls')]

ROWS = [
    (5.05, C_FREE, '(a) FREE: unconstrained LSTM', [
        ('in', ['Input $x_t$, $\\theta_t$', 'forcing + static', 'attributes']),
        ('bb', ['LSTM memory', 'LSTMCell', '$h{=}128$']),
        ('hd', ['output head', 'gives directly', '$\\Delta\\theta$']),
        ('up', ['free increment', '$\\theta_{t+1}=\\theta_t$', '$+\\Delta\\theta$']),
        ('ls', ['MSE$(m,r)$', '(no physical term)']),
    ]),
    (3.80, C_SOFT, '(b) SOFT: soft penalty LPMIX', [
        ('in', ['input as in (a)']),
        ('bb', ['LSTM memory', 'LSTMCell', '$h{=}128$']),
        ('hd', ['flux head', 'ET/PET, $Q$, ', '$D_{12}$, $D_2$',
                '-> loss']),
        ('up', ['State still set by', 'free increment', '(loss only)']),
        ('ls', ['MSE$(m,r)$', '$+\\,w\\,r_{\\rm wb}^{2}$']),
    ]),
    (2.55, C_HYBL, '(c) HYBL: intermediate HYBLST', [
        ('in', ['input as in (a)']),
        ('bb', ['LSTM memory', 'LSTMCell', '$h{=}128$']),
        ('hd', ['flux head', 'ET/PET, $Q$, ', '$D_{12}$, $D_2$',
                '-> sets state']),
        ('up', ['hard balance', '$\\theta_{t+1}=\\theta_t$',
                '$+\\Delta\\theta_{\\rm wb}+U$']),
        ('ls', ['MSE$(m,r)$']),
    ]),
    (1.30, C_HARD, '(d) HARD: hard skeleton HYBNS', [
        ('in', ['input as in (a)']),
        ('bb', ['GRU residual', 'ResFlux', 'GRUCell $h{=}64$']),
        ('hd', ['FF flux head', '(MLP)', 'ET/PET, $Q$, $D$',
                '-> sets state']),
        ('up', ['hard balance', '+ runoff', '$+\\,U_{\\rm GRU}$']),
        ('ls', ['MSE$(m,r)$', '$+\\,\\gamma\\,L_{\\rm ET}$']),
    ]),
]

BH = 0.86          # box height
FS = 9.3          # in-box font size (about 6.5 pt after rendering)
FST = 10.8         # row-title font size


def box(ax, x0, x1, y, col, lines, lh=0.196):
    ax.add_patch(FancyBboxPatch(
        (x0, y - BH / 2), x1 - x0, BH,
        boxstyle='round,pad=0.02,rounding_size=0.09',
        linewidth=1.7, edgecolor=col, facecolor=to_rgba(col, 0.10),
        zorder=2))
    n = len(lines)
    y0 = y + (n - 1) * lh / 2.0
    for i, t in enumerate(lines):
        ax.text((x0 + x1) / 2, y0 - i * lh, t, ha='center', va='center',
                fontsize=FS, color='#111111', zorder=3, linespacing=1.25)


def arrow(ax, xa, xb, y, col, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (xa, y), (xb, y),
        arrowstyle='-|>', mutation_scale=13,
        linewidth=1.6, color=col, zorder=3,
        linestyle=(0, (4, 2.4)) if dashed else '-',
        shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(13.4, 5.70))
ax.set_xlim(0, 13.4)
ax.set_ylim(0.12, 5.72)
ax.axis('off')

for y, col, title, cells in ROWS:
    ax.text(0.15, y + 0.57, title, ha='left', va='center', fontsize=FST,
            color=col, fontweight='bold')
    for key, lines in cells:
        x0, x1 = COL[key]
        box(ax, x0, x1, y, col if key != 'in' else C_GREY, lines)
    for k0, k1 in GAP:
        arrow(ax, COL[k0][1] + 0.05, COL[k1][0] - 0.05, y, '#333333')

# (b) flux head -> loss: dashed (loss only, not part of the state update)
# route through the gap below that row (y=3.15) to avoid crossing the "state update" box and its text
YB = 3.80 - BH / 2          # lower edge of that row box
YU = 3.15                    # height of the polyline (between the lower edge of (b) and the upper edge of (c))
XS, XE = 6.60, 13.02
ax.plot([XS, XS, XE, XE], [YB, YU, YU, 3.46],
        color=C_SOFT, linewidth=1.7, linestyle=(0, (4.5, 2.6)), zorder=3)
ax.add_patch(FancyArrowPatch(
    (XE, 3.44), (XE, 3.62),
    arrowstyle='-|>', mutation_scale=13, linewidth=1.7, color=C_SOFT,
    zorder=3, shrinkA=0, shrinkB=0))

# (c) annotation of U (placed in the gap below that row)
ax.text(COL['up'][0], 2.55 - 0.62,
        '$U$: bounded residual from the LSTM head (zero-initialised)',
        ha='left', va='center', fontsize=9.4, color='#333333')

ax.text(0.15, 0.62,
        'Dashed = loss only; $r_{\\rm wb}$ = water-balance residual, '
        '$L_{\\rm ET}$ = ET auxiliary loss. Each row is one time step; $\\theta_{t+1}$ '
        'is fed back as the next initial state.',
        ha='left', va='center', fontsize=9.2, color='#333333')
ax.text(0.15, 0.28,
        'All four families share input, 30-day open-loop rollout and training budget; DAMAGED '
        'has the structure of (a) with weight decay $\\geq0.01$; not shown separately.',
        ha='left', va='center', fontsize=9.2, color='#333333')

fig.tight_layout()
for d in (OD, LF):
    fig.savefig(os.path.join(d, 'fig02_arch.png'), dpi=300,
                bbox_inches='tight')
plt.close(fig)
print('fig02_arch ok')
