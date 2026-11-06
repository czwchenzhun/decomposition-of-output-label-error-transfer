# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""406b: redraw the chain-decomposition figure using the new chain (left endpoint LPMIXNS)

Input: truth_chain/multi/chain_soft_ALL.csv (produced by 39)
Output (overwrites same-named files, same directory and resolution as 38):
    fig_c1_chain.png     new-chain FORM / BACKBONE / TOTAL, two rows of rho and E2/E1
    fig_c2_path.png      path of LPMIXNS -> HYBLSTNS -> HYBNS in the (x, E2/E1) plane
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Fig. 6 (form/backbone chain decomposition, fig07_chain.png); also produces fig08_path.png for the supplementary materials (not cited in the main text)
#   Data source: chain-decomposition outputs of 39_chain_soft.py and 60_form_ss_path.py. To regenerate: cd latex_paper && python _regen_figs.py
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


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5

OD = MULTI
FD = REPRO + '/process figure/26_confound'
LF = MANU_FIG
for d in (FD, LF):
    os.makedirs(d, exist_ok=True)

REGS = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast',
        'R4_longzhong', 'R5_sichuan']
REGCN = {'R0_northchina': 'N. China\n85 stations', 'R1_yangtze': 'Yangtze\n21 stations',
         'R2_loess': 'Loess\n19 stations', 'R3_northeast': 'NE China\n14 stations',
         'R4_longzhong': 'Longzhong\n32 stations', 'R5_sichuan': 'Sichuan\n23 stations'}
COL = {'FORM(new chain)': '#009E73', 'BACKBONE': '#D55E00', 'TOTAL(new chain)': '#0072B2'}
CN = {'FORM(new chain)': 'Form effect\n(same parameters)',
      'BACKBONE': 'Backbone effect\n(same hard update)',
      'TOTAL(new chain)': 'Total effect\n(= form + backbone)'}


def main():
    f = os.path.join(OD, 'chain_soft_ALL.csv')
    if not os.path.exists(f):
        print('!! missing %s, run 39 first' % f)
        return
    D = pd.read_csv(f, encoding='utf-8-sig')
    have = [r for r in REGS if (D.region == r).any()]
    print('regions:', have)

    # ---------------- fig c1 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 7.8))
    for ri, q in enumerate(['rho', 'ratio']):
        for li, lay in enumerate(['10cm', '30cm']):
            a = ax[ri][li]
            for i, reg in enumerate(have):
                for j, k in enumerate(['FORM(new chain)', 'BACKBONE', 'TOTAL(new chain)']):
                    g = D[(D.region == reg) & (D.layer == lay)
                          & (D.kind == k) & (D.label.str.contains('-'))]
                    if not len(g):
                        continue
                    r = g.iloc[0]
                    if not np.isfinite(r[q]):
                        continue
                    x0 = i + (j - 1) * 0.26
                    lo, hi = r[q] - r[q + '_lo'], r[q + '_hi'] - r[q]
                    a.errorbar([x0], [r[q]], yerr=[[lo], [hi]], fmt='o',
                               color=COL[k], ms=6, capsize=3, lw=2,
                               label=CN[k] if (i == 0 and li == 0 and ri == 0) else None)
            a.axhline(0, color='#333333', lw=1.5)
            a.set_xticks(range(len(have)))
            a.set_xticklabels([REGCN[r] for r in have], fontsize=10.4)
            a.grid(axis='y', alpha=0.25)
            a.set_title('(%s) %s' % ('abcd'[ri * 2 + li], lay), fontsize=15.2)
            if ri == 0:
                a.set_ylabel(r'$\Delta\rho$ (negative = better)', fontsize=14.5)
            else:
                a.set_ylabel(r'$\Delta\,(E_2/E_1)$ (negative = better)', fontsize=14.5)
    # Legend placed at the top, outside the axes (the in-figure suptitle duplicating the main-text caption was removed)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    fig.legend(loc='lower center', ncol=3, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_c1_chain.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('fig_c1 ok')

    # ---------------- fig c2 ----------------
    fig, ax = plt.subplots(2, 3, figsize=(14.4, 8.2), squeeze=False)
    xs = np.linspace(0.0, 0.85, 300)
    for p, reg in enumerate(have):
        sub = ax[p // 3][p % 3]
        drew = False
        pts = {}
        for lay in ['10cm', '30cm']:
            lv = D[(D.region == reg) & (D.layer == lay)
                   & (D.kind == 'level')].set_index('label')
            need = ['LPMIXNS', 'HYBLSTNS', 'HYBNS']
            if not all(k in lv.index for k in need):
                continue
            px = [float(lv.loc[k, 'x']) for k in need]
            py = [float(lv.loc[k, 'ratio']) for k in need]
            rho = float(lv.loc['HYBLSTNS', 'rho'])
            yy = np.sqrt((1 - rho ** 2) + (xs + rho) ** 2)
            sub.plot(xs, yy, '--', color='#bbbbbb', lw=1.5)
            sub.plot(px, py, '-o', ms=7, lw=2.7, alpha=0.9, label=lay,
                     color='#0072B2' if lay == '10cm' else '#E69F00')
            pts[lay] = (px, py)
            drew = True
        if drew and len(pts) == 2:
            # Label each configuration name only once: take the midpoint of the two same-named 10 cm/30 cm points,
            # place L on the left / M above / H on the right so the two layers do not overlap
            PLACE = [(-13, -12, 'right', 'center'),
                     (0, 15, 'center', 'bottom'),
                     (13, -12, 'left', 'center')]
            # the North China and Northeast China panels have special point positions and are offset separately
            OVE = {'R0_northchina': [(-16, -17, 'right', 'center'),
                                     (0, 15, 'center', 'bottom'),
                                     (17, -22, 'left', 'center')],
                   'R3_northeast': [(-13, 15, 'right', 'bottom'),
                                    (7, -19, 'left', 'top'),
                                    (15, 9, 'left', 'center')]}
            PLACE = OVE.get(reg, PLACE)
            for i, nm in enumerate(need):
                mx = pts['10cm'][0][i]
                my = pts['10cm'][1][i]
                dx, dy, ha_, va_ = PLACE[i]
                sub.annotate(nm, (mx, my), fontsize=11.1, ha=ha_, va=va_,
                             textcoords='offset points', xytext=(dx, dy),
                             color='#333333', zorder=5,
                             bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                       ec='none', alpha=0.85))
        if drew:
            lo, hi = sub.get_ylim()
            sub.set_ylim(lo - 0.055, hi + 0.02)
            sub.axhline(1.0, color='#999999', lw=1.4, ls=':')
            sub.set_xlabel('$x=E_3/E_1$ (deviation)', fontsize=13.8)
            sub.set_ylabel('$E_2/E_1$', fontsize=13.8)
            sub.set_title('%s' % REGCN[reg].replace('\n', ' '), fontsize=15.2)
            if reg == have[0]:
                hc, lc = sub.get_legend_handles_labels()
            sub.grid(alpha=0.22)
    for p in range(len(have), 6):
        ax[p // 3][p % 3].axis('off')
    # Legend placed at the top, outside the axes (the in-figure suptitle duplicating the main-text caption was removed)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    fig.legend(hc, lc, loc='lower center', ncol=2, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_c2_path.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('fig_c2 ok ->', FD)


if __name__ == '__main__':
    main()
