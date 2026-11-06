# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""Fidelity--bias-correction frontier: distribution of alpha / rho over constraint strength and seeds.

Reads err_transfer_lambda_grid.csv (produced by the 13 -> 16 -> 19 chain).
Usage:
  python 20_err_frontier.py [--grid err_transfer_lambda_grid.csv] [--x lam|umax]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = TRUTH
FDIR = REPRO + '/process_figures/23_err_transfer'

# Chinese font (silently falls back when glyphs are missing)
for f in ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

CLR = {'HYBNS': '#D62728', 'LSTM': '#1F77B4', 'LSTMNS': '#2CA02C'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='err_transfer_lambda_grid.csv')
    ap.add_argument('--x', default='lam', help='x-axis column name (lam / wbw / umax ...)')
    ap.add_argument('--out', default='err_frontier')
    args = ap.parse_args()

    fp = os.path.join(TDIR, args.grid)
    A = pd.read_csv(fp)
    if 'cfg' not in A.columns:
        A['cfg'] = A.get('model', 'LPMIX')
    for c in CLR:
        pass
    # Unknown cfg (e.g. LPMIX) is coloured automatically
    for c in sorted(set(A.cfg)):
        if c not in CLR:
            CLR[c] = '#7F3F98' if c == 'LPMIX' else '#666666'
    xv = args.x
    if xv not in A.columns:
        cand = [c for c in ('lam', 'w', 'umax') if c in A.columns]
        xv = cand[0] if cand else A.columns[0]
        print('!! column %s does not exist, using %s' % (args.x, xv))

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))
    # When the x-axis spans orders of magnitude, use equally spaced indices so the low-value region is not compressed
    _xs = np.sort(A[xv].unique()).astype(float)
    _logx = (len(_xs) > 3) and (_xs.min() >= 0) and (_xs[-1] / max(_xs[1], 1e-9) > 20)
    for j, lay in enumerate(['10cm', '30cm']):
        s = A[A.layer == lay].copy()
        if _logx:
            s['_x'] = [_xs.tolist().index(float(v)) for v in s[xv]]
        else:
            s['_x'] = s[xv].astype(float)
        _xt = {float(v): i for i, v in enumerate(_xs)} if _logx else {}
        # --- top row: alpha ---
        a0 = ax[0, j]
        for cfg in sorted(set(A.cfg)):
            v = s[s.cfg == cfg]
            if not len(v):
                continue
            g = v.groupby('_x').alpha.agg(['mean', 'std', 'min', 'max', 'count'])
            x = g.index.values.astype(float)
            a0.errorbar(x, 100 * g['mean'], yerr=100 * g['std'],
                        marker='o', capsize=4, color=CLR[cfg], label=cfg,
                        markersize=7, linewidth=1.6)
            a0.scatter(x, 100 * g['min'], marker='_', color=CLR[cfg], s=90, alpha=.75)
            a0.scatter(x, 100 * g['max'], marker='_', color=CLR[cfg], s=90, alpha=.75)
            for xi, gi in zip(x, g.itertuples()):
                if gi.count == 1:
                    a0.plot([xi], [100 * gi.mean], marker='s', color=CLR[cfg],
                            markersize=5, alpha=.6)
        a0.axhline(0, color='#888', linewidth=.8, linestyle='--')
        a0.set_title('%s  label-error elimination rate alpha' % lay, fontsize=11)
        a0.set_ylabel('alpha (%)')
        a0.set_xlabel('constraint strength  %s%s' % (xv, '(equally spaced ticks)' if _logx else ''))
        if _logx:
            a0.set_xticks(range(len(_xs)))
            a0.set_xticklabels([('%g' % v) for v in _xs])
        a0.grid(alpha=.25)
        a0.legend(fontsize=9)

        # --- bottom row: rho ---
        a1 = ax[1, j]
        for cfg in sorted(set(A.cfg)):
            v = s[s.cfg == cfg]
            if not len(v):
                continue
            g = v.groupby('_x').rho.agg(['mean', 'std', 'min', 'max'])
            x = g.index.values.astype(float)
            a1.errorbar(x, g['mean'], yerr=g['std'], marker='s', capsize=4,
                        color=CLR[cfg], label=cfg, markersize=6, linewidth=1.6)
            a1.scatter(x, g['min'], marker='_', color=CLR[cfg], s=90, alpha=.75)
            a1.scatter(x, g['max'], marker='_', color=CLR[cfg], s=90, alpha=.75)
        a1.axhline(0, color='#888', linewidth=.8, linestyle='--')
        a1.set_title('%s  correction efficiency rho (|rho|<=1, more negative means more bias correction)' % lay, fontsize=11)
        a1.set_ylabel('rho')
        a1.set_xlabel('constraint strength  %s%s' % (xv, '(equally spaced ticks)' if _logx else ''))
        if _logx:
            a1.set_xticks(range(len(_xs)))
            a1.set_xticklabels([('%g' % v) for v in _xs])
        a1.grid(alpha=.25)
        a1.legend(fontsize=9)

    fig.suptitle('Fidelity--bias-correction frontier (R0 North China 85 stations, monthly scale; error bars = seed standard deviation, short ticks = seed range)',
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(FDIR, exist_ok=True)
    fp2 = os.path.join(FDIR, '%s_%s.png' % (args.out, xv))
    fig.savefig(fp2, dpi=150)
    print('saved', fp2)

    # ---- Text summary ----
    P = sys.stdout.write
    P('\n=== seed-mean summary ===\n')
    for lay in ['10cm', '30cm']:
        s = A[A.layer == lay]
        for cfg in ['HYBNS', 'LSTM', 'LSTMNS']:
            v = s[s.cfg == cfg]
            if not len(v):
                continue
            P('%-5s %-7s n=%2d  alpha %5.2f%% (sd %.2f, %5.2f~%5.2f)  rho %.3f (%.3f~%.3f)\n'
              % (lay, cfg, len(v), 100 * v.alpha.mean(), 100 * v.alpha.std(),
                 100 * v.alpha.min(), 100 * v.alpha.max(),
                 v.rho.mean(), v.rho.min(), v.rho.max()))
    neg = int((A.rho < 0).sum())
    P('\nCov sign: %d/%d configurations rho<0\n' % (neg, len(A)))


if __name__ == '__main__':
    main()
