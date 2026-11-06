# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""67  Recomputation of label-leakage sensitivity + per-station month-coverage diagnostics (review C1 / I8)

C1: SMCI1.0 was trained on China Meteorological Administration station observations (dataset page: 1,648 stations; paper abstract: 1,789 stations),
the list of training stations is not public, so the "overlap fraction" cannot be verified directly. What can be verified is: if leakage exists, its fingerprint
should be **an anomalously small label error at that station**. This script uses the median of each station's raw label error RMSE(smci-obs)
to split the stations of each region x layer into halves, and on three subsets (all stations / lower |e1| half / upper |e1| half) separately
recomputes the three soft--hard contrasts and gives the station-clustered bootstrap 95% CI:
  A family-level mixed convention  HARD - SOFT(all configurations)    -- same convention as crossregion_tests.csv
  B within-regime        HARD_NS - SOFT_NS        -- the sole primary endpoint of the paper
  C cross-regime        HARD_NS - SOFT_SS

I8: report the distribution of the number of months used for within-station de-meaning per station, and the monthly sample sizes (data for Supplementary Fig. S3).

The aggregation convention is fully consistent with 46: de-mean by station first, then pool RMSE by row.
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: the 35/36 robustness numbers of §4.4 "sensitivity of the primary endpoint to label leakage", Supplementary Table S9, and the data source of Supplementary Fig. S3
#

import os
import re
import glob
import numpy as np
import pandas as pd

import common

B = TRUTH
OD = os.path.join(B, 'multi')
LAYERS = ['10cm', '30cm']

REGIONS = [('R0_northchina', 'gpu*', 'North China Plain'),
           ('R1_yangtze', 'xr2_*', 'middle--lower Yangtze'),
           ('R2_loess', 'xr5_*', 'Loess Plateau'),
           ('R3_northeast', 'xr4_*', 'Northeast China black soil'),
           ('R4_longzhong', 'xr7_*', 'Longzhong'),
           ('R5_sichuan', 'xr8_*', 'Sichuan Basin')]

NBOOT = 2000
SEED = 20261028

_LOG = []


def P(s=''):
    print(s, flush=True)
    _LOG.append(str(s))


def stratum_of(cfg):
    return 'NS' if cfg in ('LSTMNS', 'HYBNS', 'HYBLSTNS', 'LPMIXNS') else 'SS'


def fam_of(run, cfg):
    if cfg == 'HYBNS':
        return 'HARD'
    if cfg in ('HYBLST', 'HYBLSTNS'):
        return 'HYBL'
    if cfg in ('LPMIX', 'LPMIXNS'):
        return 'SOFT'
    if re.search(r'wd', run):
        return 'DAMAGED'
    if cfg in ('LSTM', 'LSTMNS'):
        return 'FREE'
    return None


def load_region(pattern):
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_%s.csv' % pattern)))
    out = {}
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        A = pd.read_csv(f, encoding='utf-8-sig', dtype={'id': str})
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        for cfg in sorted(set(A.cfg)):
            if fam_of(nm, cfg) is None:
                continue
            for lay in LAYERS:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 60:
                    continue
                out[(nm, cfg, lay)] = v[['id', 'month', 'obs', 'smci', 'model']].reset_index(drop=True)
    return out


def build_mats(Pp, lay):
    """Return (keys, stas, M1, M2, NM); missing stations recorded as NaN"""
    keys = [k for k in Pp if k[2] == lay]
    stas = sorted(set().union(*[set(Pp[k].id.unique()) for k in keys])) if keys else []
    si = {s: i for i, s in enumerate(stas)}
    nk, ns = len(keys), len(stas)
    M1 = np.full((nk, ns), np.nan)
    M2 = np.full((nk, ns), np.nan)
    NM = np.zeros((nk, ns))
    for j, k in enumerate(keys):
        v = Pp[k]
        g = v.groupby('id')
        ow = (v.obs - g.obs.transform('mean')).values
        rw = (v.smci - g.smci.transform('mean')).values
        mw = (v['model'] - g['model'].transform('mean')).values
        e1, e3 = rw - ow, mw - rw
        e2 = e1 + e3
        d = pd.DataFrame(dict(id=v.id.values, a=e1 ** 2, b=e2 ** 2)).groupby('id').mean()
        for s, row in d.iterrows():
            i = si[s]
            M1[j, i] = row.a
            M2[j, i] = row.b
            NM[j, i] = v.id.value_counts()[s]
    return keys, stas, M1, M2, NM


def ratios(M1, M2, NM, cols):
    """cols: array of column (configuration) indices -> E2/E1 of each configuration"""
    w = NM[cols]
    a = M1[cols]
    b = M2[cols]
    ok = ~np.isnan(a)
    ww = np.where(ok, w, 0.0)
    sw = ww.sum(axis=1)
    E1 = np.sqrt(np.nansum(np.where(ok, a, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
    E2 = np.sqrt(np.nansum(np.where(ok, b, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
    return E2 / E1


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for reg, pat, cn in REGIONS:
        P('=== %s (%s) ===' % (cn, reg))
        Pp = load_region(pat)
        for lay in LAYERS:
            keys, stas, M1, M2, NM = build_mats(Pp, lay)
            if not keys:
                continue
            # Raw label error per station ((obs, smci) is configuration-independent; take the first configuration)
            v0 = Pp[keys[0]]
            raw = np.sqrt(((v0.smci - v0.obs) ** 2).groupby(v0.id).mean())
            rraw = raw.reindex(stas).fillna(raw.median())
            med = float(np.median(rraw.values))
            idx_all = np.arange(len(stas))
            idx_low = np.where(rraw.values <= med)[0]
            idx_high = np.where(rraw.values > med)[0]
            P('  %s  stations %d  median raw |e1| RMSE %.4f  lower half %d / upper half %d'
              % (lay, len(stas), med, len(idx_low), len(idx_high)))

            fams = [fam_of(k[0], k[1]) for k in keys]
            strs = [stratum_of(k[1]) for k in keys]
            c_hard = np.where([f == 'HARD' for f in fams])[0]
            c_sall = np.where([f == 'SOFT' for f in fams])[0]
            c_sns = np.where([(f == 'SOFT') and (s == 'NS') for f, s in zip(fams, strs)])[0]
            c_sss = np.where([(f == 'SOFT') and (s == 'SS') for f, s in zip(fams, strs)])[0]

            subs = [('all stations', idx_all), ('lower |e1| half', idx_low), ('upper |e1| half', idx_high)]
            for tag, idx in subs:
                ra = ratios(M1, M2, NM, c_hard)[:, None][:, 0]
                # Point estimates of the three contrasts (take the column subset, then weight by the station subset)
                def sub_ratio(cols):
                    w = NM[np.ix_(cols, idx)]
                    a = M1[np.ix_(cols, idx)]
                    b = M2[np.ix_(cols, idx)]
                    ok = ~np.isnan(a)
                    ww = np.where(ok, w, 0.0)
                    sw = ww.sum(axis=1)
                    E1 = np.sqrt(np.nansum(np.where(ok, a, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
                    E2 = np.sqrt(np.nansum(np.where(ok, b, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
                    return E2 / E1
                rh = sub_ratio(c_hard)
                r_all = sub_ratio(c_sall)
                r_ns = sub_ratio(c_sns)
                r_ss = sub_ratio(c_sss)
                dA = float(rh.mean() - r_all.mean())
                dB = float(rh.mean() - r_ns.mean()) if len(r_ns) else np.nan
                dC = float(rh.mean() - r_ss.mean()) if len(r_ss) else np.nan

                # bootstrap: resample stations
                nb = len(idx)
                boots = {k: [] for k in ('A', 'B', 'C')}
                for _ in range(NBOOT):
                    take = idx[rng.integers(0, nb, nb)]
                    def br(cols):
                        w = NM[np.ix_(cols, take)]
                        a = M1[np.ix_(cols, take)]
                        b = M2[np.ix_(cols, take)]
                        ok = ~np.isnan(a)
                        ww = np.where(ok, w, 0.0)
                        sw = ww.sum(axis=1)
                        E1 = np.sqrt(np.nansum(np.where(ok, a, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
                        E2 = np.sqrt(np.nansum(np.where(ok, b, 0.0) * ww, axis=1) / np.maximum(sw, 1e-12))
                        return E2 / E1
                    hb = br(c_hard).mean()
                    boots['A'].append(hb - br(c_sall).mean())
                    if len(c_sns):
                        boots['B'].append(hb - br(c_sns).mean())
                    if len(c_sss):
                        boots['C'].append(hb - br(c_sss).mean())
                qs = {}
                for k in ('A', 'B', 'C'):
                    if boots[k]:
                        arr = np.array(boots[k])
                        qs[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                    else:
                        qs[k] = (np.nan, np.nan)
                rows.append(dict(region=reg, region_cn=cn, layer=lay, subset=tag,
                                 nsta=len(idx), nboot=NBOOT,
                                 dA=dA, dA_lo=qs['A'][0], dA_hi=qs['A'][1],
                                 dB=dB, dB_lo=qs['B'][0], dB_hi=qs['B'][1],
                                 dC=dC, dC_lo=qs['C'][0], dC_hi=qs['C'][1]))
                P('    %-12s A-family=%+.4f[%+.4f,%+.4f]  B-within-regime=%+.4f[%+.4f,%+.4f]  C-cross-regime=%+.4f[%+.4f,%+.4f]'
                  % (tag, dA, qs['A'][0], qs['A'][1], dB, qs['B'][0], qs['B'][1],
                     dC, qs['C'][0], qs['C'][1]))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OD, 'leakage_sensitivity.csv'), index=False, encoding='utf-8-sig')
    P('\n' + R.to_string())
    P('\nWROTE %s' % os.path.join(OD, 'leakage_sensitivity.csv'))

    # ---------------- I8: per-station month counts and monthly sample sizes ----------------
    mrows = []
    for reg, pat, cn in REGIONS:
        Pp = load_region(pat)
        for lay in LAYERS:
            ref = None
            for k in Pp:
                if k[2] == lay and fam_of(k[0], k[1]) == 'HARD':
                    ref = Pp[k]
                    break
            if ref is None:
                continue
            cnt = ref.groupby('id').size()
            ks = np.sort(cnt.values)
            mon = np.zeros(13, dtype=np.int64)
            for m, c in ref.month.value_counts().items():
                mon[int(m)] += int(c)
            d = dict(region=reg, region_cn=cn, layer=lay, nsta=len(ks),
                     min_months=int(ks.min()), q1=int(np.percentile(ks, 25)),
                     med_months=int(np.median(ks)), q3=int(np.percentile(ks, 75)),
                     max_months=int(ks.max()), frac_ge30=float(np.mean(ks >= 30)))
            d.update({'mon_%d' % i: int(mon[i]) for i in range(1, 13)})
            mrows.append(d)
    M = pd.DataFrame(mrows)
    M.to_csv(os.path.join(OD, 'month_coverage.csv'), index=False, encoding='utf-8-sig')
    P('\n' + M.to_string())
    P('\nWROTE %s' % os.path.join(OD, 'month_coverage.csv'))
    common.report('67_leakage_sensitivity', _LOG)


if __name__ == '__main__':
    main()
