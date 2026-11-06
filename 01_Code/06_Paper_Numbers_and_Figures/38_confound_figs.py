# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""38: Confound-decomposition figures -- visualising the chain control enabled by the intermediate

Input: truth_chain/multi/confound_{region}.csv (produced by 37)
Output: 06_Reproduction_Outputs/26_confound/ and 00_Manuscript/figures/
    fig_c1_chain.png     chain decomposition of FORM / BACKBONE / TOTAL (three rows: rho, x, E2/E1)
    fig_c2_path.png      path of LSTMNS -> HYBLSTNS -> HYBNS in the (x, E2/E1) plane
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OD = MULTI
FD = REPRO + '/process figure/26_confound'
LF = MANU_FIG
for d in (FD, LF):
    os.makedirs(d, exist_ok=True)

REGS = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast']
REGCN = {'R0_northchina': 'North China\n85 stations', 'R1_yangtze': 'middle--lower Yangtze\n21 stations',
         'R2_loess': 'Loess Plateau\n19 stations', 'R3_northeast': 'Northeast China black soil\n14 stations'}
COL = {'FORM': '#2ca02c', 'BACKBONE': '#d62728', 'TOTAL': '#1f77b4'}
CN = {'FORM': 'Form effect\n(same LSTM backbone, soft to hard)',
      'BACKBONE': 'Backbone effect\n(same hard update, HYBNS design)',
      'TOTAL': 'Total effect\n(= form + backbone)'}


def load():
    out = {}
    for r in REGS:
        f = os.path.join(OD, 'confound_%s.csv' % r)
        if os.path.exists(f):
            out[r] = pd.read_csv(f, encoding='utf-8-sig')
    return out


def main():
    D = load()
    have = [r for r in REGS if r in D]
    print('regions:', have)
    if not have:
        return

    # ---------------- fig c1: chain decomposition (rho / E2/E1, two rows)----------------
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 7.6))
    for ri, q in enumerate(['rho', 'ratio']):
        for li, lay in enumerate(['10cm', '30cm']):
            a = ax[ri][li]
            n = 0
            for i, reg in enumerate(have):
                d = D[reg]
                for j, k in enumerate(['FORM', 'BACKBONE', 'TOTAL']):
                    g = d[(d.layer == lay) & (d.kind == k) & (d.label == q)]
                    if not len(g):
                        continue
                    r = g.iloc[0]
                    if not np.isfinite(r[q]):
                        continue
                    x0 = i + (j - 1) * 0.26
                    lo, hi = r[q] - r[q + '_lo'], r[q + '_hi'] - r[q]
                    a.errorbar([x0], [r[q]], yerr=[[lo], [hi]], fmt='o',
                               color=COL[k], ms=6, capsize=3, lw=1.3,
                               label=CN[k] if (i == 0 and li == 0 and ri == 0) else None)
                    n += 1
            a.axhline(0, color='#333333', lw=1.0)
            a.set_xticks(range(len(have)))
            a.set_xticklabels([REGCN[r] for r in have], fontsize=8.4)
            a.grid(axis='y', alpha=0.25)
            a.set_title('(%s) %s' % ('abcd'[ri * 2 + li], lay), fontsize=10.5)
            if ri == 0:
                a.set_ylabel(r'$\Delta\rho$ (negative = alignment improvement)', fontsize=10)
            else:
                a.set_ylabel(r'$\Delta\,(E_2/E_1)$ (negative = outcome improvement)', fontsize=10)
    fig.suptitle('Decomposing the total "hard skeleton vs unconstrained" gap into form and backbone effects (station-clustered bootstrap 95% CI)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.legend(loc='lower center', ncol=3, fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_c1_chain.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('fig_c1 npts=%d' % n)

    # ---------------- fig c2: path in the (x, E2/E1) plane ----------------
    # Background: draw the quadratic-form curve using the mean rho over all configurations for this region and layer
    nfig = sum(1 for reg in have
               for lay in ['10cm', '30cm']
               if len(D[reg][(D[reg].layer == lay) & (D[reg].kind == 'level')
                            & (D[reg].label == 'HYBLSTNS')]))
    print('path panels:', nfig)
    if nfig:
        fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
        xs = np.linspace(0.0, 0.85, 300)
        for p, reg in enumerate(have):
            sub = ax[p // 2][p % 2]
            drew = False
            for lay in ['10cm', '30cm']:
                d = D[reg]
                lv = d[(d.layer == lay) & (d.kind == 'level')].set_index('label')
                need = ['LSTMNS', 'HYBLSTNS', 'HYBNS']
                if not all(k in lv.index for k in need):
                    continue
                px = [lv.loc[k, 'x'] for k in need]
                py = [lv.loc[k, 'ratio'] for k in need]
                rho = float(lv.loc['HYBLSTNS', 'rho'])
                # quadratic-form background curve (using the intermediate rho)
                yy = np.sqrt((1 - rho ** 2) + (xs + rho) ** 2)
                sub.plot(xs, yy, '--', color='#bbbbbb', lw=1.0)
                sub.plot(px, py, '-o', ms=7, lw=1.8, alpha=0.9,
                         label=lay,
                         color='#1f77b4' if lay == '10cm' else '#ff7f0e')
                for xx_, yy_, nm in zip(px, py, need):
                    sub.annotate(nm, (xx_, yy_), fontsize=7.6,
                                 textcoords='offset points', xytext=(6, 4))
                drew = True
            if drew:
                sub.axhline(1.0, color='#999999', lw=0.9, ls=':')
                sub.set_xlabel('$x=E_3/E_1$ (relative deviation from label)', fontsize=9.5)
                sub.set_ylabel('$E_2/E_1$', fontsize=9.5)
                sub.set_title('%s' % REGCN[reg].replace('\n', ' '), fontsize=10.5)
                sub.legend(fontsize=8.6)
                sub.grid(alpha=0.22)
        for p in range(len(have), 4):
            ax[p // 2][p % 2].axis('off')
        fig.suptitle('Intermediate path: unconstrained -> intermediate (hard update + LSTM backbone) -> hard skeleton',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for d in (FD, LF):
            fig.savefig(os.path.join(d, 'fig_c2_path.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    print('Figures ->', FD)


if __name__ == '__main__':
    main()
