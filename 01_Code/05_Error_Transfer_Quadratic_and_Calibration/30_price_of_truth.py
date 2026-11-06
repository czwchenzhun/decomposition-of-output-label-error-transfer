# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
30 The "price of truth" of calibration: how many stations are needed to estimate one lam?

C2 contains a lam that is **a single scalar**. This means the amount of truth information needed to unlock the model's latent bias-correction capability
is tiny. This script asks: estimating lam from k stations and applying it region-wide, how much gain is obtained?

Design (leakage-free):
  For each run / layer, draw k stations at random -> de-mean by station within the subset and compute lam_hat = -rho_hat/x_hat
  -> apply the same de-meaning transform on the **remaining complement** before evaluation (the complement does not enter the lam estimate)
  k = 1,2,3,5,10,20,42 (repeat 01 times per level)
  Controls: base = lam 1 (no calibration); oracle = full-sample optimal lam
"""
import os
import glob
import numpy as np
import pandas as pd

B = TRUTH
KS = [1, 2, 3, 5, 10, 20, 42]

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


def within(sr, grp):
    return (sr - sr.groupby(grp).transform('mean')).values


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
        if not len(A):
            continue
        for cfg in sorted(set(A.cfg)):
            for lay in ['10cm', '30cm']:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 200:
                    continue
                ow = within(v.obs, v.id); rw = within(v.smci, v.id)
                mw = within(v['model'], v.id)
                E1 = np.sqrt(np.mean((rw - ow) ** 2))
                base = np.sqrt(np.mean((mw - ow) ** 2)) / E1
                # oracle
                E3 = np.sqrt(np.mean((mw - rw) ** 2))
                rho = np.mean((rw - ow) * (mw - rw)) / (E1 * E3)
                lam_or = -rho / (E3 / E1)
                oracle = np.sqrt(np.mean((rw + lam_or * (mw - rw) - ow) ** 2)) / E1
                ids = v.id.values
                uid = np.unique(ids)
                idx = {s: np.where(ids == s)[0] for s in uid}
                re = dict(run=nm, cfg=cfg, layer=lay, fam=fam(nm, cfg),
                          lam_or=lam_or, base=base, oracle=oracle)
                for k in KS:
                    if k > len(uid) // 2:
                        continue
                    acc = []
                    for rep in range(200):
                        rng = np.random.RandomState(7 * rep + k)
                        pick = set(rng.choice(uid, size=k, replace=False))
                        ii = np.concatenate([idx[s] for s in pick])
                        # ---- de-mean by station on the complement (strictly leakage-free)----
                        comp = ~np.isin(ids, list(pick))
                        vc = v[comp]
                        oc = within(vc.obs, vc.id); rc = within(vc.smci, vc.id)
                        mc = within(vc['model'], vc.id)
                        E1c = np.sqrt(np.mean((rc - oc) ** 2))
                        # lam is estimated on only the drawn k stations (also de-meaned within the subset)
                        vh = v[np.isin(ids, list(pick))]
                        o1 = within(vh.obs, vh.id); r1 = within(vh.smci, vh.id)
                        m1 = within(vh['model'], vh.id)
                        E1k = np.sqrt(np.mean((r1 - o1) ** 2))
                        E3k = np.sqrt(np.mean((m1 - r1) ** 2))
                        rhok = np.mean((r1 - o1) * (m1 - r1)) / (E1k * E3k)
                        lh = float(np.clip(-rhok / (E3k / E1k), 0, 2))
                        mm = rc + lh * (mc - rc)
                        acc.append((np.sqrt(np.mean((mm - oc) ** 2)) / E1c,
                                    np.sqrt(np.mean((mc - oc) ** 2)) / E1c))
                    aa = np.array(acc)
                    re['k%d' % k] = aa[:, 0].mean()
                    re['k%d_sd' % k] = aa[:, 0].std()
                    re['k%d_base' % k] = aa[:, 1].mean()
                rows.append(re)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(B, 'shrink_price.csv'), index=False, encoding='utf-8-sig')

    P = []
    ap = P.append
    ap('=' * 104)
    ap('How many stations are needed to calibrate one scalar lam? (draw k stations at random to estimate lam, evaluate on all stations; 01 repeats per level)')
    ap('=' * 104)
    for lay in ['10cm', '30cm']:
        s = D[D.layer == lay]
        ap('')
        ap('---- %s ----' % lay)
        ks = [k for k in KS if ('k%d' % k) in D.columns]
        ap('%-9s %3s | %8s %8s | %s' % ('family', 'n', 'original (complement)', 'oracle',
                                      ' '.join('%8s' % ('k=%d' % k) for k in ks)))
        ap('-' * 104)
        for fo in FAMS:
            v = s[s.fam == fo]
            if not len(v):
                continue
            ap('%-9s %3d | %8.4f %8.4f | %s' %
               (fo, len(v), v['k1_base'].mean(), v.oracle.mean(),
                ' '.join('%8.4f' % v['k%d' % k].mean() for k in ks)))
        ap('')
        ap('  gain capture fraction  (base - k) / (base - oracle) -- what fraction of the oracle gain k stations can recover:')
        ap('  %-9s %s' % ('family', ' '.join('%8s' % ('k=%d' % k) for k in ks)))
        ap('  ' + '-' * 96)
        for fo in FAMS:
            v = s[s.fam == fo]
            if not len(v):
                continue
            cells = []
            for k in ks:
                den = (v['k%d_base' % k] - v.oracle).mean()
                cells.append('%7.0f%%' % (100 * (v['k%d_base' % k]
                                                - v['k%d' % k]).mean() / den)
                             if abs(den) > 1e-9 else '     n/a')
            ap('  %-9s %s' % (fo, ' '.join('%8s' % c for c in cells)))
    txt = '\n'.join(P)
    print(txt)
    with open(LOG + '/_30_price.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
