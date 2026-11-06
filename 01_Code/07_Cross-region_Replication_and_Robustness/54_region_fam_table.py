# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""54: Per-region (region x family x layer) triplet table + full-convention evidence + absolute skill magnitude

Addresses the fourth-round review:
  M6  add a per-region (region x family x layer) table of (x, rho, E2/E1), replacing convention notes scattered across the text
  M7  state explicitly that "E2/E1 ~ 1.000 under the full convention" is quantitative evidence that the between-station static offset is not corrected
  I8  give absolute skill (absolute magnitudes of E1/E2, m3/m3) for comparison with the accuracy magnitude of published products
  R1  "family-level aggregation always excludes LPMIXNS" -- this script outputs both versions; the main text uses the one without it

Input: truth_chain/multi/withinstation_{region}.csv + raw monthly pairing files
Output: 02_Results_Data/truth_chain/multi/region_fam_table.csv
     02_Results_Data/truth_chain/multi/region_fullcaliber.csv
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: main-text Table 9 (deviation magnitude and alignment of SOFT and FREE under the cross-region aggregate) and Supplementary Table S5
#

import os
import re
import sys
import glob
import argparse
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    'm395', os.path.join(HERE, '46_region_analysis.py'))
M395 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M395)

B = M395.B
OD = M395.OD
LAYERS = M395.LAYERS
REGIONS = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast',
          'R4_longzhong', 'R5_sichuan']
REGCN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
         'R2_loess': 'Loess Plateau', 'R3_northeast': 'Northeast China black soil',
         'R4_longzhong': 'Longzhong loess', 'R5_sichuan': 'Sichuan Basin'}
PAT = {'R0_northchina': 'gpu*', 'R1_yangtze': 'xr2_*',
       'R2_loess': 'xr5_*', 'R3_northeast': 'xr4_*',
       'R4_longzhong': 'xr7_*', 'R5_sichuan': 'xr8_*'}
FAMS = ['FREE', 'SOFT', 'HYBL', 'HARD', 'DAMAGED']


def agg(g):
    """Equal-weight aggregation within a family (consistent with the main text)"""
    return dict(
        n=len(g), beat=int((g.ratio < 1).sum()),
        E1=float(g.E1w.mean()), E2=float(g.E2w.mean()), E3=float(g.E3w.mean()),
        x=float(g.x.mean()), rho=float(g.rho_w.mean()),
        ratio=float(g.ratio.mean()), floor=float(g.floorb.mean())
        if 'floorb' in g else float(np.sqrt(1 - g.rho_w.mean() ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude_lpmixns', type=int, default=1)
    a = ap.parse_args()

    o = ['=' * 100,
         '54 Per-region (region x family x layer) triplet table -- family-level aggregation %s LPMIXNS'
         % ('excludes' if a.exclude_lpmixns else 'includes'),
         '=' * 100]

    # ---------------- (1) within-station convention: aggregate from withinstation_{region}.csv ----------
    rows = []
    for r in REGIONS:
        f = os.path.join(OD, 'withinstation_%s.csv' % r)
        if not os.path.exists(f):
            continue
        W = pd.read_csv(f, encoding='utf-8-sig')
        if a.exclude_lpmixns:
            W = W[W.cfg != 'LPMIXNS']
        for lay in LAYERS:
            for fam in FAMS:
                g = W[(W.layer == lay) & (W.fam == fam)]
                if not len(g):
                    continue
                d = agg(g)
                d.update(region=r, region_cn=REGCN[r], layer=lay, fam=fam)
                rows.append(d)
    T = pd.DataFrame(rows)
    op = os.path.join(OD, 'region_fam_table.csv')
    T.to_csv(op, index=False, encoding='utf-8-sig')

    o.append('')
    o.append('[M6 per-region triplet table: within-station convention (W)]')
    o.append('%-12s %-5s %-7s %3s %5s %8s %8s %8s %8s' %
             ('region', 'layer', 'family', 'n', 'wins', 'x', 'rho', 'E2/E1', 'ceiling'))
    for _, r in T.iterrows():
        o.append('%-12s %-5s %-7s %3d %2d/%-2d %8.4f %+8.4f %8.4f %8.4f' %
                 (r.region_cn, r.layer, r.fam, r.n, r.beat, r.n,
                  r.x, r.rho, r.ratio, r.floor))

    # Cross-region aggregate: equal weight per configuration (concatenate per-configuration rows of the four regions, rather than averaging region means)
    ALL = []
    for r in REGIONS:
        f = os.path.join(OD, 'withinstation_%s.csv' % r)
        if not os.path.exists(f):
            continue
        W = pd.read_csv(f, encoding='utf-8-sig')
        if a.exclude_lpmixns:
            W = W[W.cfg != 'LPMIXNS']
        ALL.append(W)
    ALL = pd.concat(ALL, ignore_index=True)
    o.append('')
    o.append('[Cross-region aggregate (equal weight per configuration, n = sum of configuration counts over the four regions)]')
    for fam in ['FREE', 'SOFT', 'HYBL', 'HARD']:
        for lay in LAYERS:
            g = ALL[(ALL.fam == fam) & (ALL.layer == lay)]
            if not len(g):
                continue
            o.append('  %-6s %-5s n=%2d  wins %2d/%2d  x=%.3f  rho=%+.3f  E2/E1=%.3f'
                     % (fam, lay, len(g), int((g.ratio < 1).sum()), len(g),
                        g.x.mean(), g.rho_w.mean(), g.ratio.mean()))

    # Difference in x between SOFT and FREE (by region + four-region pool with equal weight per configuration)
    o.append('')
    o.append('[M6 key point: x of SOFT and FREE are close under the four-region pooled convention, but not within a single region]')
    for lay in LAYERS:
        for fam in ['SOFT', 'FREE']:
            g = T[(T.fam == fam) & (T.layer == lay)]
            s = '  '.join('%s %.3f' % (REGCN[r.region][:2], r.x) for _, r in g.iterrows())
            o.append('  %-5s %-5s per-region %s' % (lay, fam, s))
        gs = T[(T.fam == 'SOFT') & (T.layer == lay)]
        gf = T[(T.fam == 'FREE') & (T.layer == lay)]
        x3s = float(gs[gs.region == 'R0_northchina'].x.iloc[0])
        x3f = float(gf[gf.region == 'R0_northchina'].x.iloc[0])
        xas = float(ALL[(ALL.fam == 'SOFT') & (ALL.layer == lay)].x.mean())
        xaf = float(ALL[(ALL.fam == 'FREE') & (ALL.layer == lay)].x.mean())
        o.append('        -> North China single region SOFT/FREE x = %.3f / %.3f (differ by %.0f%%);'
                 'four regions, equal weight per configuration %.3f / %.3f (differ by %.0f%%)'
                 % (x3s, x3f, 100 * abs(x3s / x3f - 1),
                    xas, xaf, 100 * abs(xas / xaf - 1)))

    # ---------------- (2) full convention (P) -- M7 ----------------
    o.append('')
    o.append('[M7 E2/E1 under the full convention (P, no de-meaning): is the between-station static offset compressed]')
    prows = []
    for r in REGIONS:
        P = M395.load_pairs(PAT[r])
        for lay in LAYERS:
            for fam in FAMS:
                vals = []
                for (nm, cfg, ly), v in P.items():
                    if ly != lay or M395.fam_of(nm, cfg) != fam:
                        continue
                    if a.exclude_lpmixns and cfg == 'LPMIXNS':
                        continue
                    e1 = (v.smci - v.obs).values
                    e3 = (v['model'] - v.smci).values
                    e2 = e1 + e3
                    E1 = np.sqrt(np.mean(e1 ** 2))
                    E3 = np.sqrt(np.mean(e3 ** 2))
                    E2 = np.sqrt(np.mean(e2 ** 2))
                    rho = np.mean(e1 * e3) / (E1 * E3)
                    vals.append(dict(E1=E1, E2=E2, E3=E3, rho=rho,
                                     x=E3 / E1, ratio=E2 / E1))
                if not vals:
                    continue
                V = pd.DataFrame(vals)
                prows.append(dict(region=r, region_cn=REGCN[r], layer=lay, fam=fam,
                                  n=len(V), E1=V.E1.mean(), E2=V.E2.mean(),
                                  E3=V.E3.mean(), rho=V.rho.mean(),
                                  x=V.x.mean(), ratio=V.ratio.mean(),
                                  beat=int((V.ratio < 1).sum())))
    PR = pd.DataFrame(prows)
    op2 = os.path.join(OD, 'region_fullcaliber.csv')
    PR.to_csv(op2, index=False, encoding='utf-8-sig')
    o.append('%-12s %-5s %-7s %3s %9s %9s %8s %8s' %
             ('region', 'layer', 'family', 'n', 'E1(full)', 'E2(full)', 'rho', 'E2/E1'))
    for _, r in PR.iterrows():
        o.append('%-12s %-5s %-7s %3d %9.5f %9.5f %+8.4f %8.4f' %
                 (r.region_cn, r.layer, r.fam, r.n, r.E1, r.E2, r.rho, r.ratio))

    # ---------------- (2b) between-station static offset retention rate -- direct criterion of M7 ----------
    o.append('')
    o.append('[M7 direct criterion: between-station static offset retention rate = RMS_s(mean_t e2) / RMS_s(mean_t e1)]')
    o.append('  If the model can correct the between-station static offset, this ratio should be significantly below 1; if not corrected at all, it should be ~1.')
    o.append('%-12s %-5s %-7s %3s %10s %10s %8s' %
             ('region', 'layer', 'family', 'n', 'RMS(me1)', 'RMS(me2)', 'retention rate'))
    srows = []
    for r in REGIONS:
        P = M395.load_pairs(PAT[r])
        for lay in LAYERS:
            for fam in FAMS:
                rr = []
                for (nm, cfg, ly), v in P.items():
                    if ly != lay or M395.fam_of(nm, cfg) != fam:
                        continue
                    if a.exclude_lpmixns and cfg == 'LPMIXNS':
                        continue
                    e1 = (v.smci - v.obs).values
                    e2 = (v['model'] - v.obs).values
                    m1 = pd.Series(e1).groupby(v.id.values).transform('mean').values
                    m2 = pd.Series(e2).groupby(v.id.values).transform('mean').values
                    a1 = float(np.sqrt(np.mean(m1 ** 2)))
                    a2 = float(np.sqrt(np.mean(m2 ** 2)))
                    rr.append((a1, a2, a2 / a1 if a1 > 0 else np.nan))
                if not rr:
                    continue
                R = pd.DataFrame(rr, columns=['a1', 'a2', 'ret'])
                srows.append(dict(region=r, region_cn=REGCN[r], layer=lay,
                                  fam=fam, n=len(R), rms_me1=R.a1.mean(),
                                  rms_me2=R.a2.mean(), retention=R.ret.mean()))
                o.append('%-12s %-5s %-7s %3d %10.5f %10.5f %8.4f' %
                         (REGCN[r], lay, fam, len(R), R.a1.mean(), R.a2.mean(),
                          R.ret.mean()))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(OD, 'region_static_retention.csv'),
             index=False, encoding='utf-8-sig')
    if len(S):
        o.append('  -> Retention rate over all %d (region x family x layer) cells: mean %.4f, range [%.4f, %.4f]'
                 % (len(S), S.retention.mean(), S.retention.min(), S.retention.max()))

    # ---------------- (3) absolute skill (I8) ----------------
    o.append('')
    o.append('[I8 absolute magnitude: within-station E1 and E2 of each family (m3/m3), for comparison with ubRMSE in the literature]')
    o.append('%-12s %-5s %10s %10s %10s %10s %10s' %
             ('region', 'layer', 'E1(within-station)', 'E2 FREE', 'E2 SOFT', 'E2 HYBL', 'E2 HARD'))
    for r in REGIONS:
        for lay in LAYERS:
            g = T[(T.region == r) & (T.layer == lay)]
            if not len(g):
                continue
            def gv(f):
                s = g[g.fam == f]
                return float(s.E2.iloc[0]) if len(s) else float('nan')
            o.append('%-12s %-5s %10.5f %10.5f %10.5f %10.5f %10.5f' %
                     (REGCN[r], lay, float(g.E1.iloc[0]),
                      gv('FREE'), gv('SOFT'), gv('HYBL'), gv('HARD')))
    o.append('')
    o.append('  Reference magnitude: the ubRMSE of satellite soil-moisture products relative to station observations is usually reported as '
             '0.04--0.06 m3/m3 (the conventional evaluation convention of ESA CCI / SMAP).')

    # ---------------- (4) cross-layer consistency (I2) ----------------
    o.append('')
    o.append('[I2 cross-layer consistency: whether the E2/E1 differences SOFT--HARD and SOFT--FREE have the same sign in both layers]')
    for fam2 in ['HARD', 'FREE']:
        same = 0
        tot = 0
        for r in REGIONS:
            for lay in LAYERS:
                pass
        for r in REGIONS:
            s = T[(T.region == r) & (T.fam == 'SOFT')]
            h = T[(T.region == r) & (T.fam == fam2)]
            if not len(s) or not len(h):
                continue
            d10 = float(h[h.layer == '10cm'].ratio.iloc[0]) - float(s[s.layer == '10cm'].ratio.iloc[0])
            d30 = float(h[h.layer == '30cm'].ratio.iloc[0]) - float(s[s.layer == '30cm'].ratio.iloc[0])
            tot += 1
            same += int(d10 > 0) == int(d30 > 0)
            o.append('  %-12s  SOFT-%s  E2/E1 diff: 10cm %+.4f  30cm %+.4f  same sign %s'
                     % (REGCN[r], fam2, d10, d30, 'Yes' if (d10 > 0) == (d30 > 0) else 'No'))
        o.append('  -> SOFT vs %s: %d/%d regions have the same sign in both layers' % (fam2, same, tot))

    txt = '\n'.join(o)
    with open(os.path.join(HERE, '_407.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print()
    print('written ->', op)
    print('written ->', op2)


if __name__ == '__main__':
    main()
