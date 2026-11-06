# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
28 Optimal-shrinkage verification (zero training, pure post-processing)

Theoretical predictions:
  Let m_lam = r + lam*(m - r)  (scale the output's deviation from the label by lam)
  Then  Cov(lam) = lam*Cov,  E3(lam) = lam*E3  ->  rho is unchanged
  And  (E2/E1)^2 = (1-rho^2) + (lam*x + rho)^2      (using the identity in 27)
  => optimal lam* = -rho/x,  at which E2/E1 = sqrt(1-rho^2) = FLOOR (cannot be lowered further)

Hence this is a **falsifiable quantitative prediction**:
  (P1) rho is indeed invariant under lam scaling
  (P2) the empirical lam minimising E2/E1 equals the theoretical lam* = -rho/x
  (P3) E2/E1 at the optimum equals sqrt(1-rho^2)

If all three hold -> the identity is not merely a bookkeeping identity but an **actionable design rule**:
  To make an output better than its training label, first measure rho, then shrink the deviation to -rho times.
"""
import os
import numpy as np
import pandas as pd

B = TRUTH

HARD = {'gpu', 'gpu_lam0', 'gpu_lam20', 'gpu_lam5_s7', 'gpu_lam5_s21',
        'gpu_lam0_s7', 'gpu_lam0_s21', 'gpu_lam20_s7', 'gpu_lam20_s21'}
DAMAGED = {'gpu_degwd', 'gpu_wd0.01', 'gpu_wd0.03'}


def fam(r):
    if r.run in DAMAGED:
        return 'DAMAGED'
    if r.run in HARD or (r.cfg == 'HYBNS' and r.run.startswith('gpu_lam')):
        return 'HARD'
    if r.cfg == 'LPMIX' or r.run.startswith('gpu_wbw'):
        return 'SOFT'
    return 'FREE'


def budget(o, r, m):
    d1, d2, d3 = r - o, m - o, m - r
    E1 = np.sqrt(np.mean(d1 ** 2))
    E2 = np.sqrt(np.mean(d2 ** 2))
    E3 = np.sqrt(np.mean(d3 ** 2))
    C = np.mean(d1 * d3)
    return E1, E2, E3, (C / (E1 * E3) if E1 * E3 > 0 else 0.)


def within(df, c):
    return (df[c] - df.groupby('id')[c].transform('mean')).values


def main():
    import glob
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_gpu*.csv')))
    R = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'))
    R['fam'] = R.apply(fam, axis=1)

    P = []
    ap = P.append
    ap('=' * 104)
    ap('optimal-shrinkage verification: m_lam = r + lam*(m-r), theoretical lam* = -rho/x, theoretical lower bound sqrt(1-rho^2)')
    ap('=' * 104)

    rows = []
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        try:
            A = pd.read_csv(f)
        except Exception:
            continue
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        if not len(A):
            continue
        rr = R[(R.run == nm)]
        for cfg in sorted(set(A.cfg)):
            for lay in ['10cm', '30cm']:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 200:
                    continue
                ow, rw = within(v, 'obs'), within(v, 'smci')
                mw = within(v, 'model')
                E1, E2, E3, rho = budget(ow, rw, mw)
                x = E3 / E1
                lam_th = -rho / x if x > 0 else 1.0
                # empirical scan
                lams = np.linspace(0.0, 2.0, 2001)
                best = (None, 1e9)
                for L in lams:
                    mm = rw + L * (mw - rw)   # m_lam = r + lam*(m-r)
                    e2 = np.sqrt(np.mean((mm - ow) ** 2))
                    if e2 < best[1]:
                        best = (L, e2)
                _, E2opt, E3opt, rho_opt = budget(ow, rw, rw + best[0] * (mw - rw))
                _, _, _, rho_at = budget(ow, rw, rw + 0.5 * (mw - rw))
                rows.append(dict(run=nm, cfg=cfg, layer=lay,
                                 fam=fam(rr.iloc[0]) if len(rr) else '?',
                                 x=x, rho=rho, E2E1=E2 / E1,
                                 lam_th=lam_th, lam_em=best[0],
                                 E2E1_opt=best[1] / E1,
                                 floor=np.sqrt(max(0, 1 - rho ** 2)),
                                 rho_check=rho_at))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(B, 'shrink_optimality.csv'), index=False, encoding='utf-8-sig')

    for lay in ['10cm', '30cm']:
        s = D[D.layer == lay]
        ap('')
        ap('---- %s ----' % lay)
        ap('%-10s %-8s | %7s %7s | %8s | %8s %8s %8s | %8s %8s %8s' %
           ('family', 'n', 'x', 'rho', 'E2/E1(orig)', 'lam* theory', 'lam* empirical', 'error',
            'E2/E1(opt)', 'FLOOR', 'diff'))
        ap('-' * 104)
        for k in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('%-10s %-8d | %7.4f %+7.3f | %8.4f | %8.3f %8.3f %8.4f | %8.4f %8.4f %+8.4f' %
               (k, len(v), v.x.mean(), v.rho.mean(), v.E2E1.mean(),
                v.lam_th.mean(), v.lam_em.mean(),
                (v.lam_em - v.lam_th).abs().mean(),
                v.E2E1_opt.mean(), v.floor.mean(),
                (v.E2E1_opt - v.floor).abs().mean()))
        ap('')
        ap('  P1 rho invariance test: maximum deviation of rho across all configurations after lam scaling %.2e' %
           (D.rho_check - D.rho).abs().max())
        ap('  P2 theoretical vs empirical optimal lam: mean absolute error %.4f, maximum %.4f' %
           ((D.lam_em - D.lam_th).abs().mean(), (D.lam_em - D.lam_th).abs().max()))
        ap('  P3 at the optimum vs the theoretical lower bound: mean difference %.2e, maximum %.2e' %
           ((D.E2E1_opt - D.floor).abs().mean(), (D.E2E1_opt - D.floor).abs().max()))
        ap('')
        ap('  [how far each family is from beating its own label] (after shrinking to the optimum, E2/E1 < 1 is a genuine win)')
        for k in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('    %-8s n=%-3d  optimal E2/E1 = %.4f  (%.0f%% of points < 1)'
               '   original %.4f (%.0f%% < 1)  -> net improvement %.2f percentage points' %
               (k, len(v), v.E2E1_opt.mean(), 100 * (v.E2E1_opt < 1).mean(),
                v.E2E1.mean(), 100 * (v.E2E1 < 1).mean(),
                100 * (v.E2E1.mean() - v.E2E1_opt.mean())))

    txt = '\n'.join(P)
    print(txt)
    with open(LOG + '/_28_shrink.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
