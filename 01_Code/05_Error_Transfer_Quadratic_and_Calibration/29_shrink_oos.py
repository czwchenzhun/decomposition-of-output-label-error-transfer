# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
29 Out-of-sample transferability of optimal shrinkage (avoiding circular reasoning)

28 estimated lam* on the same test pairs -> "100% better than label" may just be overfitting.
This script performs a strict out-of-sample test:
  Scheme A  temporal hold-out: estimate lam* on 2011-01..2012-06, evaluate on 2012-07..2013-12
  Scheme B  spatial hold-out: estimate lam* on a random half of stations (42 stations), evaluate on the other half (43 stations)
  Control fac: lam=1 (no shrinkage, i.e. the original output) and lam = global theoretical value

If most of the gain survives out of sample -> this is a usable design rule;
If it works only in sample -> it is merely ex post accounting and cannot be written up as a method.
"""
import os
import glob
import numpy as np
import pandas as pd

B = TRUTH

# 2026-10-07 Fix: the old fam() judged family by run name, and the LSTM rows produced under run='gpu'
# were wrongly merged into HARD (second-round review R4). Switched to the (run, cfg) convention consistent with 33,
# and included the newly added intermediate family HYBL (HYBLST / HYBLSTNS).
FREE_RUNCFG = [
    ('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'), ('gpu_LSTMs21', 'LSTM'),
    ('gpu_lstmns', 'LSTMNS'),
    ('gpu_deg5', 'LSTM'), ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM'),
]
DAMAGED_RUNCFG = [('gpu_degwd', 'LSTM'), ('gpu_wd0.01', 'LSTM'),
                  ('gpu_wd0.03', 'LSTM')]


def fam(run, cfg):
    key = (run, cfg)
    if key in DAMAGED_RUNCFG:
        return 'DAMAGED'
    if key in FREE_RUNCFG:
        return 'FREE'
    if cfg in ('LPMIX', 'LPMIXNS'):
        return 'SOFT'
    if cfg == 'HYBNS':
        return 'HARD'
    if cfg in ('HYBLST', 'HYBLSTNS'):
        return 'HYBL'
    return 'OTHER'


FAMS = ['FREE', 'SOFT', 'HARD', 'HYBL', 'DAMAGED']


def within(ser, grp):
    return (ser - ser.groupby(grp).transform('mean')).values


def metrics(ow, rw, mw):
    d1, d2, d3 = rw - ow, mw - ow, mw - rw
    E1 = np.sqrt(np.mean(d1 ** 2)); E2 = np.sqrt(np.mean(d2 ** 2))
    E3 = np.sqrt(np.mean(d3 ** 2)); C = np.mean(d1 * d3)
    rho = C / (E1 * E3) if E1 * E3 > 0 else 0.
    return E1, E2, E3, rho


def main():
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_gpu*.csv')))
    rows = []
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        try:
            A = pd.read_csv(f)
        except Exception:
            continue
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        if 'year' not in A.columns or 'month' not in A.columns:
            continue
        A['_ym'] = A.year.astype(str) + '-' + A.month.astype(str).str.zfill(2)
        for cfg in sorted(set(A.cfg)):
            for lay in ['10cm', '30cm']:
                v = A[(A.cfg == cfg) & (A.layer == lay)].copy()
                if len(v) < 200:
                    continue
                ow = within(v.obs, v.id); rw = within(v.smci, v.id)
                mw = within(v['model'], v.id)
                E1, E2, E3, rho = metrics(ow, rw, mw)
                x = E3 / E1
                lam_th = -rho / x if x > 0 else 1.0
                re = dict(run=nm, cfg=cfg, layer=lay, fam=fam(nm, cfg),
                          x=x, rho=rho, lam_th=lam_th,
                          base=E2 / E1,
                          floor=np.sqrt(max(0., 1 - rho ** 2)))

                # ---- Scheme A: temporal hold-out (first half estimate, second half evaluate) ----
                v['_ord'] = v['_ym']
                med = sorted(v['_ym'].unique())[len(v['_ym'].unique()) // 2]
                tr = v['_ym'] < med; te = ~tr
                if tr.sum() > 100 and te.sum() > 100:
                    sub = v[tr]
                    o1 = within(sub.obs, sub.id); r1 = within(sub.smci, sub.id)
                    m1 = within(sub['model'], sub.id)
                    _, _, E3a, ra = metrics(o1, r1, m1)
                    lamA = -ra / (E3a / np.sqrt(np.mean((r1 - o1) ** 2)))
                    sub2 = v[te]
                    o2 = within(sub2.obs, sub2.id); r2 = within(sub2.smci, sub2.id)
                    m2 = within(sub2['model'], sub2.id)
                    e_base = np.sqrt(np.mean((m2 - o2) ** 2)) / np.sqrt(np.mean((r2 - o2) ** 2))
                    mm = r2 + lamA * (m2 - r2)
                    e_A = np.sqrt(np.mean((mm - o2) ** 2)) / np.sqrt(np.mean((r2 - o2) ** 2))
                    re.update(lam_A=lamA, base_A=e_base, shrink_A=e_A)

                # ---- Scheme B: spatial hold-out (random half of stations estimate, other half evaluate; 10 repeats averaged)----
                ids = sorted(v.id.unique())
                acc = []
                for rep in range(10):
                    rng = np.random.RandomState(1000 + rep)
                    pick = rng.permutation(ids)[:len(ids) // 2]
                    msk = v.id.isin(pick)
                    s1, s2 = v[msk], v[~msk]
                    o1 = within(s1.obs, s1.id); r1 = within(s1.smci, s1.id)
                    m1 = within(s1['model'], s1.id)
                    _, _, E3b, rb = metrics(o1, r1, m1)
                    lamB = -rb / (E3b / np.sqrt(np.mean((r1 - o1) ** 2)))
                    o2 = within(s2.obs, s2.id); r2 = within(s2.smci, s2.id)
                    m2 = within(s2['model'], s2.id)
                    den = np.sqrt(np.mean((r2 - o2) ** 2))
                    acc.append((np.sqrt(np.mean((m2 - o2) ** 2)) / den,
                                np.sqrt(np.mean((r2 + lamB * (m2 - r2) - o2) ** 2)) / den,
                                lamB))
                aa = np.array(acc)
                re.update(lam_B=aa[:, 2].mean(), base_B=aa[:, 0].mean(),
                          shrink_B=aa[:, 1].mean())
                rows.append(re)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(B, 'shrink_oos.csv'), index=False, encoding='utf-8-sig')

    P = []
    ap = P.append
    ap('=' * 108)
    ap('Out-of-sample test of optimal shrinkage (lambda estimated on a hold-out subset, evaluated outside it)')
    ap('=' * 108)
    for lay in ['10cm', '30cm']:
        s = D[D.layer == lay]
        ap('')
        ap('---- %s ----' % lay)
        ap('%-9s %3s | %7s | %16s | %22s | %22s' %
           ('family', 'n', 'lam*', 'in-sample optimum', 'A temporal', 'B spatial (10 folds)'))
        ap('%-9s %3s | %7s | %8s %7s | %10s %11s | %10s %11s' %
           ('', '', '', 'base', 'opt', 'base', 'shrink', 'base', 'shrink'))
        ap('-' * 108)
        for k in FAMS:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('%-9s %3d | %7.3f | %8.4f %7.4f | %10.4f %11.4f | %10.4f %11.4f' %
               (k, len(v), v.lam_th.mean(), v.base.mean(), v.floor.mean(),
                v.base_A.mean(), v.shrink_A.mean(), v.base_B.mean(), v.shrink_B.mean()))
        ap('')
        ap('  Out-of-sample gain (relative to each base):')
        for k in FAMS:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('    %-8s  A temporal %+.2f pp   B spatial %+.2f pp   '
               '(in-sample theoretical upper bound %.2f pp)' %
               (k, 100 * (v.base_A - v.shrink_A).mean(),
                100 * (v.base_B - v.shrink_B).mean(),
                100 * (v.base - v.floor).mean()))
        # Key: whether it crosses 1.0 after shrinkage
        ap('')
        ap('  Proportion with E2/E1 < 1 after shrinkage (better than its own training label):')
        for k in FAMS:
            v = s[s.fam == k]
            if not len(v):
                continue
            ap('    %-8s original %.0f%%  |  A temporal %.0f%%  |  B spatial %.0f%%' %
               (k, 100 * (v.base < 1).mean(), 100 * (v.shrink_A < 1).mean(),
                100 * (v.shrink_B < 1).mean()))
    txt = '\n'.join(P)
    print(txt)
    with open(LOG + '/_29_shrink_oos.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
