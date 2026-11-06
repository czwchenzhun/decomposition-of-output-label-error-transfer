# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""64: (x, rho) readout values for the North China primary region, 10cm, within-station convention

Purpose: provide recomputable numbers for two newly written passages, §3.3 "how to read Table 2" and §2.3 "the conditional-mean ideal",
      (all taken from region_fam_table.csv produced by 54, not computed by hand).

Output columns: fam n E1 E3 x rho ratio floor(=sqrt(1-rho^2)) x_plus_rho lambda*
       ratio_chk = sqrt((1-rho^2)+(x+rho)^2)  should match ratio value by value
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

P = os.path.join(MULTI, 'region_fam_table.csv')
df = pd.read_csv(P)
d = df[(df.region == 'R0_northchina') & (df.layer == '10cm')].copy()

L = ['64: North China primary region, 10cm, within-station convention (source region_fam_table.csv, produced by 54)', '']
L.append('%-8s %3s %10s %10s %8s %9s %9s %9s %9s %8s %9s' % (
    'fam', 'n', 'E1', 'E3', 'x', 'rho', 'E2/E1', 'floor', 'x+rho', 'lam*', 'chk'))
for _, r in d.iterrows():
    floor = float(np.sqrt(1.0 - r.rho ** 2))
    chk = float(np.sqrt((1.0 - r.rho ** 2) + (r.x + r.rho) ** 2))
    lam = -r.rho / r.x
    L.append('%-8s %3d %10.6f %10.6f %8.4f %9.4f %9.4f %9.4f %9.4f %8.3f %9.4f' % (
        r.fam, r.n, r.E1, r.E3, r.x, r.rho, r.ratio, floor, r.x + r.rho, lam, chk))
L.append('')
L.append('Consistency check: max |chk - E2/E1| = %.2e (should be < 1e-6)' %
         float(np.max(np.abs(
             np.sqrt((1 - d.rho ** 2) + (d.x + d.rho) ** 2) - d.ratio))))
L.append('')
L.append('Values used in the main text (rounded before writing):')
for _, r in d.iterrows():
    L.append('  %-8s x=%.3f rho=%.3f ratio=%.3f floor=%.3f x+rho=%.3f lam*=%.2f '
             'x/(-rho)=%.2f' % (r.fam, r.x, r.rho, r.ratio,
                                float(np.sqrt(1 - r.rho ** 2)), r.x + r.rho,
                                -r.rho / r.x, r.x / (-r.rho)))
L.append('')
L.append('E1(10cm)=%.6f  E1(30cm)=%.6f' % (
    float(d.E1.iloc[0]),
    float(df[(df.region == 'R0_northchina') &
             (df.layer == '30cm')].E1.iloc[0])))

common.report('64_quadratic_readout', L)
print('\n'.join(L))
