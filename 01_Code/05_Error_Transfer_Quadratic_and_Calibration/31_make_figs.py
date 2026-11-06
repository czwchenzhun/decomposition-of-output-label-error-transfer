# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""31 Draw three main figures from existing csv (no recomputation): (a) quadratic decomposition (b) optimal shrinkage (c) price-of-truth curve"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

B = TRUTH
O = REPRO + '/process_figures/23_err_transfer'
COL = {'FREE': '#1f77b4', 'SOFT': '#2ca02c', 'HARD': '#d62728', 'DAMAGED': '#7f7f7f'}
NAME = {'FREE': 'unconstrained LSTM', 'SOFT': 'soft constraint LPMIX',
        'HARD': 'hard skeleton HYBNS', 'DAMAGED': 'training-damaged wd'}
KS = [1, 2, 3, 5, 10, 20, 42]

Q = pd.read_csv(os.path.join(B, 'err_transfer_quadratic.csv'))
S = pd.read_csv(os.path.join(B, 'shrink_optimality.csv'))
P = pd.read_csv(os.path.join(B, 'shrink_price.csv'))
os.makedirs(O, exist_ok=True)


def fam(r):
    if r.run in ('gpu_degwd', 'gpu_wd0.01', 'gpu_wd0.03'):
        return 'DAMAGED'
    if r.cfg == 'HYBNS':
        return 'HARD'
    if r.cfg == 'LPMIX' or r.run.startswith('gpu_wbw'):
        return 'SOFT'
    return 'FREE'


for D in (Q, S, P):
    D['fam'] = D.apply(fam, axis=1)

# ---------------- (a) quadratic decomposition ----------------
fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
for j, lay in enumerate(['10cm', '30cm']):
    a = ax[j]
    s = Q[Q.layer == lay]
    xs = np.linspace(0, 0.65, 200)
    a.plot(xs, xs, '--', color='#999', lw=1.2, zorder=1)
    for k in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
        v = s[s.fam == k]
        if not len(v):
            continue
        a.scatter(-v.rho_w, v.x, s=48, alpha=.82, color=COL[k],
                  edgecolor='w', linewidth=.6, zorder=3, label=NAME[k])
    a.set_xlabel('alignment efficiency  -rho  (i.e. optimal deviation magnitude x*)')
    if j == 0:
        a.set_ylabel('actual deviation magnitude  x = E3/E1')
    a.set_title('%s   actual deviation vs optimal deviation' % lay, fontsize=11)
    a.grid(alpha=.25)
    a.legend(fontsize=8, loc='upper left')
ax[1].annotate('above the diagonal = excess deviation EXC > 0\nthe excess is squared into the penalty',
               xy=(0.04, 0.52), fontsize=9.5, color='#b00')
fig.suptitle('Beating the training label = ceiling(1-rho^2) + excess deviation^2  '
             '-- all four families lie above the diagonal (excess)', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(O, 'fig_a_quadratic.png'), dpi=170)
print('saved fig_a_quadratic.png')

# ---------------- (b) optimal shrinkage ----------------
fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2))
for j, lay in enumerate(['10cm', '30cm']):
    a = ax[j]
    s = S[S.layer == lay]
    for k in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
        v = s[s.fam == k]
        if not len(v):
            continue
        a.scatter(v.lam_th, v.lam_em, s=52, alpha=.85, color=COL[k],
                  edgecolor='w', linewidth=.6, label=NAME[k], zorder=3)
    lim = [0, 1.05]
    a.plot(lim, lim, '--', color='#999', lw=1.2, zorder=1)
    a.set_xlabel('theoretical optimal shrinkage factor  lam* = -rho/x')
    a.set_ylabel('empirical optimal shrinkage factor (grid scan)')
    a.set_title('%s   theory vs empirical (mean |error| 2e-4)' % lay, fontsize=11)
    a.grid(alpha=.25)
    a.legend(fontsize=8, loc='upper left')
fig.suptitle('P2 prediction: the empirical optimal lam equals the analytic solution -rho/x', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(O, 'fig_b_optimal_shrink.png'), dpi=170)
print('saved fig_b_optimal_shrink.png')

# ---------------- (c) price-of-truth curve ----------------
fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
for j, lay in enumerate(['10cm', '30cm']):
    a = ax[j]
    s = P[P.layer == lay]
    for k in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
        v = s[s.fam == k]
        if not len(v):
            continue
        ks = [x for x in KS if ('k%d' % x) in v.columns]
        capt = []
        for x in ks:
            den = (v['k%d_base' % x] - v.oracle).mean()
            capt.append(100 * (v['k%d_base' % x] - v['k%d' % x]).mean() / den
                        if abs(den) > 1e-9 else np.nan)
        a.plot(ks, capt, 'o-', color=COL[k], lw=1.9, ms=6, label=NAME[k])
    a.axhline(0, color='#333', lw=1)
    a.axhline(100, color='#2ca02c', lw=1, ls=':')
    a.set_xscale('log')
    a.set_xticks(KS)
    a.set_xticklabels([str(x) for x in KS])
    a.set_xlabel('number of stations k used to calibrate lam (01 random repeats, complement evaluation)')
    if j == 0:
        a.set_ylabel('gain capture fraction  %')
    a.set_title('%s' % lay, fontsize=11)
    a.grid(alpha=.25)
    a.legend(fontsize=8)
ax[0].annotate('Only when the model already aims well but deviates too much (hard skeleton)\nis calibration worth it; a model already near-optimal is hurt by small k',
               xy=(1.2, -700), fontsize=9, color='#444')
fig.suptitle('Price-of-truth curve: the truth needed to unlock bias correction is only a scalar', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(O, 'fig_c_price_of_truth.png'), dpi=170)
print('saved fig_c_price_of_truth.png')
