# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""69 Per-configuration station-clustered bootstrap intervals of ρ (second-round review R5)

Background
----
The first round conceded "family-level ρ≥0" to regime stratification; the second round pointed out: among the 9 **per-configuration** point estimates with ρ≥0 across the six regions,
5 are ≤ +0.0142 (noise level); writing them all as "correct predictions of the calibration condition"
would be overturned by recomputation at the point-estimate level. This script leaves no room for selective narrative -- it directly, for the ρ of each configuration,
performs a station-clustered bootstrap from the same source as the rest of the paper, reporting the 95% interval and "whether the interval lies entirely to the right of zero".

Method (same source as stage_bootstrap in 46)
--------------------------------------
Within-station de-meaning depends only on the station itself, so precompute per station A=Σe1², B=Σe3², C=Σe1e3 and the month count n,
sum after resampling stations: E1=√(A/n), E3=√(B/n), ρ=(C/n)/(E1E3). The resampling indices, within the same
(region x layer), are shared across configurations so that family-level differences and configuration-level ρ use the same randomness.

Output
----
  multi/rho_config_ci.csv  ρ, CI, whether it crosses zero, whether the CI is entirely positive, per (region x configuration x layer)
  scripts/69_rho_config_ci.txt
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '69_rho_config_ci.txt')

import importlib.util
spec = importlib.util.spec_from_file_location('m395', os.path.join(HERE, '46_region_analysis.py'))
m395 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m395)

B = m395.B
OD = m395.OD
NB = 4000
SEED = 20261029

REGS = [('R0_northchina', 'gpu*'), ('R1_yangtze', 'xr2_*'),
        ('R3_northeast', 'xr4_*'), ('R2_loess', 'xr5_*'),
        ('R4_longzhong', 'xr7_*'), ('R5_sichuan', 'xr8_*')]

CN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
      'R3_northeast': 'Northeast China black soil', 'R2_loess': 'Loess Plateau',
      'R4_longzhong': 'Longzhong', 'R5_sichuan': 'Sichuan Basin'}


def station_mats(P):
    """{(layer): {(run,cfg): dict(ids,n,A,B,C,fam,strat)}}"""
    raw = {}
    for (nm, cfg, lay), v in P.items():
        ow = m395.within(v.obs, v.id)
        rw = m395.within(v.smci, v.id)
        mw = m395.within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(v.id.values, return_inverse=True)
        k = len(u)
        raw.setdefault(lay, {})[(nm, cfg)] = dict(
            fam=m395.fam_of(nm, cfg), strat=m395.stratum_of(cfg), ids=u,
            n=np.bincount(inv, minlength=k).astype(float),
            A=np.bincount(inv, weights=e1 ** 2, minlength=k),
            B=np.bincount(inv, weights=e3 ** 2, minlength=k),
            C=np.bincount(inv, weights=e1 * e3, minlength=k))
    return raw


def main():
    o = []
    ap = o.append
    rows = []
    for reg, pat in REGS:
        P = m395.load_pairs(pat)
        if not P:
            ap('!! %s has no pairing file' % reg)
            continue
        raw = station_mats(P)
        for lay, cfgs in sorted(raw.items()):
            keys = sorted(cfgs.keys())
            ns = len(cfgs[keys[0]]['ids'])
            rng = np.random.default_rng(SEED + hash(lay) % 997)
            idx = rng.integers(0, ns, (NB, ns))
            for key in keys:
                d = cfgs[key]
                A, Bb, C, nn = d['A'], d['B'], d['C'], d['n']
                Aa = A[idx].sum(1)
                Bb2 = Bb[idx].sum(1)
                Cc = C[idx].sum(1)
                Nn = nn[idx].sum(1)
                E1 = np.sqrt(Aa / Nn)
                E3 = np.sqrt(Bb2 / Nn)
                rho = (Cc / Nn) / (E1 * E3)
                pt = (C.sum() / nn.sum()) / (np.sqrt(A.sum() / nn.sum())
                                             * np.sqrt(Bb.sum() / nn.sum()))
                rows.append(dict(region=reg, region_cn=CN[reg], run=key[0],
                                 cfg=key[1], layer=lay, fam=d['fam'],
                                 strat=d['strat'], nsta=ns,
                                 rho=float(pt),
                                 rho_lo=float(np.percentile(rho, 2.5)),
                                 rho_hi=float(np.percentile(rho, 97.5))))
    R = pd.DataFrame(rows)
    R['cross0'] = (R.rho_lo < 0) & (R.rho_hi > 0)
    R['all_pos'] = (~R.cross0) & (R.rho > 0)
    R['all_neg'] = (~R.cross0) & (R.rho < 0)
    R = R.sort_values(['region', 'layer', 'run', 'cfg'])
    R.to_csv(os.path.join(OD, 'rho_config_ci.csv'), index=False,
             encoding='utf-8-sig')

    ap('=' * 92)
    ap('69 per-configuration station-clustered bootstrap 95%% intervals of ρ (nboot=%d, seed=%d)' % (NB, SEED))
    ap('Convention: within-station de-meaning (W), Cov is the uncentred second moment; resampling unit = station')
    ap('=' * 92)
    pos = R[R.rho >= 0].sort_values('rho', ascending=False)
    ap('')
    ap('[A] All %d per-configuration entries with ρ ≥ 0 (second-round review R5)' % len(pos))
    for _, x in pos.iterrows():
        flag = 'CI entirely right of zero <- can assert ρ>0' if x.all_pos else 'CI contains zero (noise level)'
        ap('  %-8s %-14s %-12s %-5s ρ=%+.4f  95%%CI [%+.4f,%+.4f]  %d stations  %s'
           % (x.region_cn, x.run, x.cfg, x.layer, x.rho, x.rho_lo, x.rho_hi,
              x.nsta, flag))
    ap('')
    ap('  Of these, configurations whose CI lies entirely to the right of zero (ρ significantly positive): %d / %d'
       % (int(pos.all_pos.sum()), len(pos)))
    thr = pos[pos.rho <= 0.015]
    ap('  Of these, with point estimate ≤ +0.015: %d, whose CIs all contain zero: %s'
       % (len(thr), bool(thr.cross0.all())))

    ap('')
    ap('[B] Summary of configuration counts under the full convention')
    ap('  Total configuration--layer combinations: %d' % len(R))
    ap('  ρ point estimates ≥0: %d' % int((R.rho >= 0).sum()))
    ap('  ρ CIs entirely right of zero: %d' % int(R.all_pos.sum()))
    ap('  ρ CIs entirely left of zero (alignment significantly negative): %d' % int(R.all_neg.sum()))
    ap('  ρ CIs containing zero: %d' % int(R.cross0.sum()))
    ap('')
    ap('[C] Stratified by region (number with ρ CI entirely right of zero / number of combinations in that region)')
    g = R.groupby(['region_cn']).agg(n=('rho', 'size'), npos=('all_pos', 'sum'),
                                     nneg=('all_neg', 'sum'), ncr=('cross0', 'sum'))
    for k, x in g.iterrows():
        ap('  %-8s combos %2d  ρ>0 significant %d  ρ<0 significant %2d  contains zero %2d'
           % (k, int(x.n), int(x.npos), int(x.nneg), int(x.ncr)))

    txt = '\n'.join(o)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
