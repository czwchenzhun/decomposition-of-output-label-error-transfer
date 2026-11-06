# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
27 Exact quadratic-form decomposition (identity, no assumptions)

From the definitions E2^2 = E1^2 + E3^2 + 2 Cov,  rho = Cov/(E1 E3),  x = E3/E1  we obtain the identity:

      (E2/E1)^2  =  (1 - rho^2)  +  (x + rho)^2

where:
  FLOOR = sqrt(1 - rho^2)   : given alignment efficiency rho, the **insurmountable lower bound**
                              (independent of the deviation magnitude, depending only on "whether the deviation direction is well aimed")
  EXC   = x + rho           : **excess deviation**. The optimal deviation magnitude is exactly x* = -rho
                              (deviation = rho times the label error is optimal, any more is pure loss)

Corollary:
  * Whether the output can beat its own training label depends on only two numbers: how well it is aimed (rho), and whether there is excess deviation (x+rho)
  * Simply "deviating more" never improves the outcome: once x > -rho, each extra unit of deviation is a quadratic penalty
  * rho -> 0 when the lower bound -> 1: a model that cannot aim at all can never beat its label, however it is tuned
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

B = TRUTH
OUT = REPRO + '/process_figures/23_err_transfer'

# Note: the family is determined by **architecture configuration cfg**, not by the run name.
# Historical bug: this once read HARD = {'gpu', ...}, while under run='gpu' there are both
# cfg='HYBNS'(hard skeleton) and cfg='LSTM'(unconstrained), so unconstrained LSTM was labelled hard skeleton.
# This error was fixed by script 42 in the fam column of err_transfer_quadratic.csv.
DAMAGED = {'gpu_degwd', 'gpu_wd0.01', 'gpu_wd0.03'}
DEG = {'gpu_deg5', 'gpu_deg10', 'gpu_deg32'}

FAM_BY_CFG = {
    'LSTM': 'FREE unconstrained LSTM',
    'LSTMNS': 'FREE unconstrained LSTM',
    'LPMIX': 'SOFT soft constraint LPMIX',
    'LPMIXNS': 'SOFT soft constraint LPMIX',
    'HYBLST': 'HYBL intermediate HYBLST',
    'HYBLSTNS': 'HYBL intermediate HYBLST',
    'HYBNS': 'HARD hard skeleton HYBNS',
    'LGBM': 'MLBASE conventional ML baseline',
    'RIDGE': 'MLBASE conventional ML baseline',
}

FAMORDER = ['FREE unconstrained LSTM', 'SOFT soft constraint LPMIX', 'HYBL intermediate HYBLST',
            'HARD hard skeleton HYBNS', 'DAMAGED training-damaged wd']


def fam(r):
    # DAMAGED is a weight-decay control stacked on LSTM; it must be decided by run and take priority over cfg
    if r.run in DAMAGED:
        return 'DAMAGED training-damaged wd'
    if r.cfg not in FAM_BY_CFG:
        raise KeyError('unknown cfg=%s (run=%s): please add it to FAM_BY_CFG' % (r.cfg, r.run))
    return FAM_BY_CFG[r.cfg]


def main():
    R = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'))
    R['fam'] = R.apply(fam, axis=1)
    R['x'] = R.E3w / R.E1w
    R['floor'] = np.sqrt(np.clip(1 - R.rho_w ** 2, 0, None))
    R['exc'] = R.x + R.rho_w
    R['ratio'] = R.E2w / R.E1w
    # identity check
    R['chk'] = np.sqrt(R.floor ** 2 + R.exc ** 2)
    R['err'] = (R.chk - R.ratio).abs()

    P = []
    ap = P.append
    ap('=' * 100)
    ap('exact quadratic-form decomposition   (E2/E1)^2 = (1 - rho^2) + (x + rho)^2      x = E3/E1')
    ap('convention: within-station temporal anomaly (de-meaned by station), h=0, R0 North China 85 station-month pairs')
    ap('identity check: maximum reconstruction error %.2e  %s' % (R.err.max(),
                                       '(OK)' if R.err.max() < 1e-9 else '(!! abnormal)'))
    ap('=' * 100)

    for lay in ['10cm', '30cm']:
        s = R[R.layer == lay]
        ap('')
        ap('---- %s ----' % lay)
        ap('%-20s %3s | %7s %7s | %7s %7s | %8s | %8s %8s' %
           ('family', 'n', 'x=E3/E1', 'rho', 'FLOOR', 'EXC', 'E2/E1', 'optimal x*', 'over/under'))
        ap('-' * 100)
        for k in FAMORDER:
            v = s[s.fam == k]
            if not len(v):
                continue
            over = 'excess %.0f%%' % (100 * (v.x.mean() / (-v.rho_w.mean()) - 1)) \
                if v.rho_w.mean() < 0 else 'n/a'
            ap('%-20s %3d | %7.4f %+7.3f | %7.4f %+7.4f | %8.4f | %8.4f %8s' %
               (k, len(v), v.x.mean(), v.rho_w.mean(), v.floor.mean(),
                v.exc.mean(), v.ratio.mean(), -v.rho_w.mean(), over))
        ap('')
        ap('  Interpretation: FLOOR = the ceiling set by how well the family aims; EXC = how much it deviated beyond what it should.')
        ap('        Whether E2/E1 crosses 1.0 is decided jointly by these two terms; neither x nor rho alone is decisive.')

        # ---- unconfounded comparison of rho (rho is normalised by x and does not scale with the deviation magnitude) ----
        ap('')
        ap('  [rho between-group test] (rho is a normalised quantity, unconfounded by deviation magnitude)')
        g = {k: s[s.fam == k].rho_w.values for k in FAMORDER if len(s[s.fam == k])}
        for a, b in [('FREE unconstrained LSTM', 'SOFT soft constraint LPMIX'),
                     ('SOFT soft constraint LPMIX', 'HARD hard skeleton HYBNS'),
                     ('FREE unconstrained LSTM', 'HARD hard skeleton HYBNS'),
                     ('HARD hard skeleton HYBNS', 'DAMAGED training-damaged wd')]:
            if a not in g or b not in g:
                continue
            t, p = stats.ttest_ind(g[a], g[b], equal_var=False)
            u, pu = stats.mannwhitneyu(g[a], g[b])
            ap('    %-22s %+.3f±%.3f  vs  %-22s %+.3f±%.3f   Welch p=%.2g  MW p=%.2g' %
               (a, g[a].mean(), g[a].std(ddof=1), b, g[b].mean(), g[b].std(ddof=1), p, pu))

        # ---- EXC test ----
        ap('')
        ap('  [excess deviation EXC = x+rho between-group test]')
        ge = {k: s[s.fam == k].exc.values for k in FAMORDER if len(s[s.fam == k])}
        for a, b in [('FREE unconstrained LSTM', 'SOFT soft constraint LPMIX'),
                     ('SOFT soft constraint LPMIX', 'HARD hard skeleton HYBNS'),
                     ('HARD hard skeleton HYBNS', 'DAMAGED training-damaged wd')]:
            if a not in ge or b not in ge:
                continue
            t, p = stats.ttest_ind(ge[a], ge[b], equal_var=False)
            ap('    %-22s %+.4f±%.4f  vs  %-22s %+.4f±%.4f   Welch p=%.2g' %
               (a, ge[a].mean(), ge[a].std(ddof=1), b, ge[b].mean(),
                ge[b].std(ddof=1), p))

        # ---- Actual deviation magnitude vs optimal deviation magnitude for each family ----
        fr = s[s.fam == 'FREE unconstrained LSTM']
        ap('')
        ap('  [optimal-magnitude calibration] FREE family x=%.4f, -rho=%.4f, EXC=%+.4f (deviation from optimal %.0f%%)' %
           (fr.x.mean(), -fr.rho_w.mean(), fr.exc.mean(),
            100 * (fr.x.mean() / (-fr.rho_w.mean()) - 1)))
        for k in ['SOFT soft constraint LPMIX', 'HARD hard skeleton HYBNS', 'DAMAGED training-damaged wd']:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('    %-22s x=%.4f, -rho=%.4f, EXC=%+.4f (over-deviated %.0f%%)' %
               (k, v.x.mean(), -v.rho_w.mean(), v.exc.mean(),
                100 * (v.x.mean() / (-v.rho_w.mean()) - 1)))

    R.to_csv(os.path.join(B, 'err_transfer_quadratic.csv'),
             index=False, encoding='utf-8-sig')
    txt = '\n'.join(P)
    print(txt)
    with open(LOG + '/_27_quadratic.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    ap('saved err_transfer_quadratic.csv')

    # ---------------- Figure ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)
        COL = {'FREE unconstrained LSTM': '#0072B2', 'SOFT soft constraint LPMIX': '#009E73',
               'HYBL intermediate HYBLST': '#CC79A7', 'HARD hard skeleton HYBNS': '#D55E00',
               'DAMAGED training-damaged wd': '#000000'}
        for j, lay in enumerate(['10cm', '30cm']):
            a = ax[j]
            s = R[R.layer == lay]
            xs = np.linspace(0, 0.65, 300)
            a.plot(xs, xs, '--', color='#999', lw=1.2)
            a.text(0.52, 0.55, 'x = -rho\n(optimal deviation magnitude)', fontsize=9, color='#666')
            for k in FAMORDER:
                v = s[s.fam == k]
                if not len(v):
                    continue
                a.scatter(-v.rho_w, v.x, s=52, alpha=.8, color=COL[k], label=k,
                          edgecolor='w', linewidth=.6, zorder=3)
            a.axhspan(0, 1, color='none')
            a.set_xlabel('alignment efficiency  -rho  (optimal deviation magnitude x*)')
            a.set_title('%s  actual deviation magnitude x  vs  optimal deviation magnitude -rho' % lay, fontsize=11)
            a.grid(alpha=.25)
            if j == 0:
                a.set_ylabel('actual deviation magnitude  x = E3/E1')
            a.legend(fontsize=8, loc='upper left')
        # Diagonal annotation: above the diagonal = excess deviation
        ax[1].annotate('above the diagonal = excess deviation EXC>0\n(each extra unit of deviation is a quadratic penalty)',
                       xy=(0.30, 0.52), fontsize=9, color='#444')
        fig.suptitle('Whether the output can beat its own training label depends on two knobs: how well it is aimed (-rho),'
                     'and whether it has deviated too much (x)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        os.makedirs(OUT, exist_ok=True)
        fp = REPRO + '/process_figures/quadratic_decomposition.png'
        fig.savefig(fp, dpi=170)
        print('saved', fp)
    except Exception as e:
        print('plot fail:', e)


if __name__ == '__main__':
    main()
