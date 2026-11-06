# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""51: M8 -- affine (with-intercept) calibration comparison and the test of $\\lambda^\\ast=1$

The first version of 50 read data by "file", but some files contain multiple cfg (e.g. gpu contains HYBNS and LSTM),
so HARD and FREE received the same set of numbers. This script reads data exactly by (file, cfg) and additionally includes
LPMIXNS and the conventional ML baselines.

Output: truth_chain/51_calib_affine.csv

Usage (geoenv): python 51_calib_affine.py
"""
import os
import numpy as np
import pandas as pd

import common

TC = TRUTH
RNG = np.random.default_rng(20261008)


def P(*a):
    print(*a, flush=True)


def run_one(v, nboot=2000):
    """v: monthly pairing table with obs/model/smci/id (single layer)"""
    e1 = (v['smci'] - v['obs']).values
    e3 = (v['model'] - v['smci']).values
    E1 = float(np.sqrt(np.mean(e1 ** 2)))
    lam_full = -np.cov(e1, e3)[0, 1] / np.var(e3)
    E2_full = float(np.sqrt(np.mean((e1 + lam_full * e3) ** 2))) / E1
    # Within-station (de-meaned by station)
    g = v.groupby('id')
    e1w = (g['smci'].transform(lambda s: s - s.mean())
           - g['obs'].transform(lambda s: s - s.mean())).values
    e3w = (g['model'].transform(lambda s: s - s.mean())
           - g['smci'].transform(lambda s: s - s.mean())).values
    E1w = float(np.sqrt(np.mean(e1w ** 2)))
    lam_w = -np.cov(e1w, e3w)[0, 1] / np.var(e3w)
    E2w = float(np.sqrt(np.mean((e1w + lam_w * e3w) ** 2))) / E1w
    # Affine (with intercept): full convention
    Xa = np.column_stack([np.ones_like(e3), e3])
    b, *_ = np.linalg.lstsq(Xa, -e1, rcond=None)
    E2_aff = float(np.sqrt(np.mean((e1 + b[0] + b[1] * e3) ** 2))) / E1
    # Affine: within-station convention
    Xw = np.column_stack([np.ones_like(e3w), e3w])
    bw, *_ = np.linalg.lstsq(Xw, -e1w, rcond=None)
    E2w_aff = float(np.sqrt(np.mean((e1w + bw[0] + bw[1] * e3w) ** 2))) / E1w
    # Station-clustered bootstrap of lambda* (within-station convention)
    ids = v['id'].values
    uid = np.unique(ids)
    idxby = {s: np.where(ids == s)[0] for s in uid}
    boots = np.empty(nboot)
    for k in range(nboot):
        pick = RNG.choice(uid, size=len(uid), replace=True)
        ii = np.concatenate([idxby[s] for s in pick])
        a1, a3 = e1w[ii], e3w[ii]
        boots[k] = -np.cov(a1, a3)[0, 1] / np.var(a3)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(v), n_sta=len(uid),
                lam_full=lam_full, lam_within=lam_w, lam_lo=lo, lam_hi=hi,
                reject_lam1=bool(not (lo <= 1.0 <= hi)),
                E2E1_noCal=1.0, E2E1_lam_within=E2w, E2E1_lam_full=E2_full,
                aff_a_full=b[0], aff_lam_full=b[1],
                aff_a_within=bw[0], aff_lam_within=bw[1],
                E2E1_aff_within=E2w_aff, E2E1_aff_full=E2_aff,
                gain_aff_pp=(E2w - E2w_aff) * 100)


# (label, file suffix, cfg)
CASES = [
    ('hard skeleton HYBNS', 'gpu', 'HYBNS'),
    ('hard skeleton $\\gamma=0$', 'gpu_lam0', 'HYBNS'),
    ('hard skeleton $\\gamma=20$', 'gpu_lam20', 'HYBNS'),
    ('unconstrained LSTM', 'gpu', 'LSTM'),
    ('unconstrained LSTMNS', 'gpu_lstmns', 'LSTMNS'),
    ('soft penalty LPMIX $w=1$', 'gpu_wbw1', 'LPMIX'),
    ('soft penalty LPMIXNS', 'gpu_lpmixns_s42', 'LPMIXNS'),
    ('intermediate HYBLSTNS', 'gpu_hbl_s42', 'HYBLSTNS'),
    ('damaged control', 'gpu_degwd', 'LSTM'),
    ('baseline LightGBM', 'mlbase', 'LGBM'),
    ('baseline ridge regression', 'mlbase', 'RIDGE'),
]


def main():
    rows = []
    for tag, suf, cfg in CASES:
        fp = os.path.join(TC, 'truth_chain_L2_monthly_%s.csv' % suf)
        if not os.path.exists(fp):
            P('  skip %s (missing %s)' % (tag, os.path.basename(fp)))
            continue
        d = pd.read_csv(fp, encoding='utf-8-sig')
        if 'cfg' in d.columns:
            d = d[d.cfg == cfg]
        for layer in ['10cm', '30cm']:
            v = d[d.layer == layer].dropna(subset=['obs', 'model', 'smci'])
            if len(v) < 200:
                continue
            r = run_one(v)
            r.update(dict(tag=tag, file=suf, cfg=cfg, layer=layer))
            rows.append(r)
            P('  %-22s %s  lam*=%.3f [%.3f,%.3f]  E2/E1: no calibration %.3f -> lambda* %.3f -> affine %.3f'
              % (tag, layer, r['lam_within'], r['lam_lo'], r['lam_hi'],
                 1.0, r['E2E1_lam_within'], r['E2E1_aff_within']))
    out = pd.DataFrame(rows)
    op = os.path.join(TC, '51_calib_affine.csv')
    out.to_csv(op, index=False, encoding='utf-8-sig')
    P('WROTE %s' % op)
    try:
        common.report('51_calib_affine', [])
    except Exception:
        pass


if __name__ == '__main__':
    main()
