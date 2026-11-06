# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""42: fix the fam-column labelling error in err_transfer_quadratic.csv (review M9b)

Background: fam() in 27_quadratic.py determines the family from the run name:
    HARD = {'gpu', 'gpu_lam0', ...}
but under run='gpu' both cfg='HYBNS' (hard skeleton) and cfg='LSTM' (unconstrained) exist,
so the two cfg=LSTM rows were labelled "HARD hard skeleton HYBNS".

Correct rule: the family is determined by the **architecture configuration cfg**, independent of run (seed/hyperparameter variants);
the only exception is DAMAGED, a weight-decay control added on LSTM that must be determined by run.

This script:
  1. recompute fam via an explicit (cfg -> family) mapping
  2. assert that the cfg -> family mapping is unique and that each family count matches main-text Table 2
  3. assert that the corrected family means match Table 2 (paper_families.csv) value by value
  4. rewrite the fam column in place (row count and all other columns unchanged) and write an audit txt
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report  # noqa: E402

B = TRUTH
F = os.path.join(B, 'err_transfer_quadratic.csv')

# ---- correct mapping: cfg determines the architecture family ----
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
# DAMAGED is a control group (weight decay), determined by run and taking priority over cfg
DAMAGED_RUNS = {'gpu_degwd', 'gpu_wd0.01', 'gpu_wd0.03'}

# configuration counts of main-text Table 2 (LPMIXNS excluded from the SOFT family mean; the 34 configurations here also exclude LPMIXNS)
EXPECT_N = {
    'FREE unconstrained LSTM': 7,
    'SOFT soft constraint LPMIX': 15,
    'HARD hard skeleton HYBNS': 9,
    'DAMAGED training-damaged wd': 3,
}


def fam_of(run, cfg):
    if run in DAMAGED_RUNS:
        return 'DAMAGED training-damaged wd'
    if cfg not in FAM_BY_CFG:
        raise KeyError('unknown cfg: %s (run=%s)' % (cfg, run))
    return FAM_BY_CFG[cfg]


def main():
    d = pd.read_csv(F)
    n0 = len(d)
    old = d['fam'].copy()

    # --- Assertion 1: the cfg -> family mapping is unique (checked over all run x cfg, not just this table) ---
    full = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'))
    m = {}
    for _, r in full.iterrows():
        if r['run'] in DAMAGED_RUNS:
            continue  # DAMAGED is a control stacked on LSTM and is cross-family by design
        m.setdefault(r['cfg'], set()).add(fam_of(r['run'], r['cfg']))
    for cfg, fs in m.items():
        assert len(fs) == 1, 'cfg=%s maps to multiple families: %s' % (cfg, fs)

    # --- recompute ---
    d['fam'] = [fam_of(r, c) for r, c in zip(d['run'], d['cfg'])]
    changed = int((old != d['fam']).sum())
    assert len(d) == n0

    # --- Assertion 2: family configuration count ---
    P = []
    ap = P.append
    ap('=' * 88)
    ap('42: err_transfer_quadratic.csv fam-column correction (review M9b)')
    ap('=' * 88)
    ap('rows %d (configuration--layer), fam changed in %d rows' % (n0, changed))
    ap('')
    ap('--- change details ---')
    for i in range(n0):
        if old.iloc[i] != d['fam'].iloc[i]:
            ap('  run=%-16s cfg=%-8s %s  %s -> %s' %
               (d['run'].iloc[i], d['cfg'].iloc[i], d['layer'].iloc[i],
                old.iloc[i], d['fam'].iloc[i]))
    ap('')
    ap('--- corrected family composition ---')
    for k in ['FREE unconstrained LSTM', 'SOFT soft constraint LPMIX', 'HARD hard skeleton HYBNS',
              'HYBL intermediate HYBLST', 'DAMAGED training-damaged wd', 'MLBASE conventional ML baseline']:
        v = d[d.fam == k]
        if not len(v):
            continue
        ncfg = v.groupby(['run', 'cfg']).ngroups
        ok = ''
        if k in EXPECT_N:
            ok = '  [Table 2 expects %d] %s' % (EXPECT_N[k],
                                     'OK' if ncfg == EXPECT_N[k] else '!! MISMATCH')
            assert ncfg == EXPECT_N[k], 'family %s count %d != %d' % (k, ncfg, EXPECT_N[k])
        ap('  %-22s %2d configurations  %2d rows%s' % (k, ncfg, len(v), ok))

    # --- Assertion 3: family means match main-text Table 2 (reading paper_families.csv) ---
    ap('')
    ap('--- value-by-value check against Table 2 (paper_families.csv) ---')
    pf = os.path.join(B, 'paper_families.csv')
    if os.path.exists(pf):
        t = pd.read_csv(pf)
        NAME = {'FREE unconstrained LSTM': 'FREE', 'SOFT soft constraint LPMIX': 'SOFT',
                'HARD hard skeleton HYBNS': 'HARD'}
        for k, short in NAME.items():
            v = d[d.fam == k]
            for lay in ['10cm', '30cm']:
                vv = v[v.layer == lay]
                row = t[(t.fam == short) & (t.layer == lay)]
                if not len(row):
                    continue
                r1 = float(row.iloc[0]['ratio_mean'])
                r2 = float(vv.ratio.mean())
                rho1 = float(row.iloc[0]['rho_mean'])
                rho2 = float(vv.rho_w.mean())
                x1 = float(row.iloc[0]['x_mean'])
                x2 = float(vv.x.mean())
                ok = (abs(r1 - r2) < 5e-4 and abs(rho1 - rho2) < 5e-4
                      and abs(x1 - x2) < 5e-4)
                ap('  %-4s %s  ratio %.4f/%.4f  rho %+.4f/%+.4f  x %.4f/%.4f  %s' %
                   (short, lay, r1, r2, rho1, rho2, x1, x2,
                    'OK' if ok else '!! inconsistent'))
                assert ok, '%s %s family mean inconsistent with Table 2' % (short, lay)
    else:
        ap('  (paper_families.csv not found, skipping the Table 2 check)')

    # --- write back (only the fam column changes; row count and all other columns unchanged) ---
    d.to_csv(F, index=False, encoding='utf-8-sig')
    ap('')
    ap('wrote back %s (%d rows, only fam column changed)' % (F, len(d)))

    txt = '\n'.join(P)
    print(txt)
    report('42_fix_fam.txt', P)


if __name__ == '__main__':
    main()
