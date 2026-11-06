# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""68  Conventional supervised-learning baselines for the five replication regions (I7)

Review I7: the "trivial conditional-mean" argument for North China relies on the LightGBM / ridge-regression baselines, while the five replication regions
have no conventional machine-learning baseline, so one cannot answer whether "soft beating hard is also merely a manifestation of each one's default outcome".

This script reproduces, for R2/R4/R5/R7/R8, the same chain as North China 49 -> 16:
  49: construct window-level tabular samples on exactly the same npz as 13, train ridge regression and LightGBM;
  16: restore daily predictions to month--station pairs, following the same station-truth comparison chain.

To avoid contaminating existing products, both tag and out_suffix carry a region prefix; finally this script itself, following the
within-station (W) convention, aggregates and writes multi/mlbase_multiregion.csv (without overwriting withinstation_*.csv).
"""
import os
import re
import glob
import subprocess
import sys

import numpy as np
import pandas as pd

import common

B = TRUTH
OD = os.path.join(B, 'multi')
OUTB = BENCH
LAYERS = ['10cm', '30cm']

REGIONS = [
    ('R1_yangtze', 'xr2', 'R1_yangtze'),
    ('R3_northeast', 'xr4', 'R3_northeast'),
    ('R2_loess', 'xr5', 'R2_loess'),
    ('R4_longzhong', 'xr7', 'R4_longzhong'),
    ('R5_sichuan', 'xr8', 'R5_sichuan'),
]

_LOG = []


def P(s=''):
    print(s, flush=True)
    _LOG.append(str(s))


def run(cmd):
    # The subprocess scripts live in other directories of the package, resolved to absolute paths by script name (cwd is still 04_Runtime_Logs/)
    sc = cmd[0]
    if not os.path.isabs(sc):
        _here = os.path.dirname(os.path.abspath(__file__))
        for _base in (_here, os.path.join(CODE, '04_Truth_Verification_Chain')):
            _cand = os.path.join(_base, sc)
            if os.path.isfile(_cand):
                sc = _cand
                break
    P('$ ' + ' '.join(cmd))
    r = subprocess.run([sys.executable, sc] + list(cmd[1:]), cwd=LOG,
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    tail = (r.stdout or '')[-1200:]
    P(tail)
    if r.returncode != 0:
        P('!! FAILED rc=%d' % r.returncode)
        P((r.stderr or '')[-2000:])
    return r.returncode


def within(sr, grp):
    return (sr - sr.groupby(grp).transform('mean')).values


def budget(e1, e3):
    E1 = float(np.sqrt(np.mean(e1 ** 2)))
    E3 = float(np.sqrt(np.mean(e3 ** 2)))
    e2 = e1 + e3
    E2 = float(np.sqrt(np.mean(e2 ** 2)))
    C = float(np.mean(e1 * e3))
    rho = C / (E1 * E3) if E1 * E3 > 0 else 0.0
    return E1, E2, E3, rho


def fam_of(cfg):
    return {'RIDGE': 'RIDGE', 'LGBM': 'LGBM'}.get(cfg)


def main(do_train=True):
    for reg, pre, npre in REGIONS:
        npz = os.path.join(OUTB, 'dl_%s_nc256_20020101_20131231_sta.npz' % npre)
        if not os.path.exists(npz):
            P('!! missing npz: %s' % npz)
            continue
        tag = '_staG_mlbase_%s' % pre
        suf = 'mlbase_%s' % pre
        if do_train:
            rc = run(['49_ml_baseline.py', '--npz', npz, '--region', reg,
                      '--tag', tag, '--models', 'ridge,lgbm'])
            if rc != 0:
                continue
            rc = run(['16_model_vs_truth.py', '--npz', npz, '--tag', tag,
                      '--region', reg, '--cfgs', 'RIDGE,LGBM',
                      '--horizons', '0', '--out_suffix', suf])
            if rc != 0:
                continue

    # ---------------- aggregate under the within-station convention of 46 ----------------
    rows = []
    for reg, pre, npre in REGIONS:
        f = os.path.join(B, 'truth_chain_L2_monthly_mlbase_%s.csv' % pre)
        if not os.path.exists(f):
            P('!! missing %s' % f)
            continue
        A = pd.read_csv(f, encoding='utf-8-sig', dtype={'id': str})
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        for cfg in sorted(set(A.cfg)):
            fam = fam_of(cfg)
            if fam is None:
                continue
            for lay in LAYERS:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 60:
                    P('  skipped %s %s %s (n=%d)' % (reg, cfg, lay, len(v)))
                    continue
                ow = within(v.obs, v.id)
                rw = within(v.smci, v.id)
                mw = within(v['model'], v.id)
                e1, e3 = rw - ow, mw - rw
                E1, E2, E3, rho = budget(e1, e3)
                rows.append(dict(region=reg, cfg=cfg, fam=fam, layer=lay,
                                 n=len(v), nsta=v.id.nunique(),
                                 E1w=E1, E2w=E2, E3w=E3, rho_w=rho,
                                 x=E3 / E1, ratio=E2 / E1,
                                 beat=int(E2 / E1 < 1.0)))
    M = pd.DataFrame(rows)
    if len(M):
        M.to_csv(os.path.join(OD, 'mlbase_multiregion.csv'), index=False,
                 encoding='utf-8-sig')
        P('\n=== Conventional baselines in the replication regions (within-station W convention) ===')
        P(M.to_string())
        P('\nWROTE %s' % os.path.join(OD, 'mlbase_multiregion.csv'))
    common.report('68_mlbase_multiregion', _LOG)


if __name__ == '__main__':
    main(do_train=('--no_train' not in sys.argv))
