# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""52 Soft penalty vs hard skeleton under the same training regime (NS) (fourth-round review M2)

Background
----
The headline conclusion "the soft-penalty family outperforms the hard-skeleton family" was, in the original manuscript, a **cross-training-regime** comparison of SOFT(SS) vs HARD(NS)
; moreover, §5.5 measured that scheduled sampling alone worsens rho by +0.093, on the same order as the effect to be demonstrated.
M2 requires a within-regime control: LPMIXNS = component-wise identical to LPMIX with only scheduled sampling removed,
so SOFT-NS and HARD-NS form a within-regime comparison.

This script performs a station-clustered bootstrap on the "family x regime" grouping (same algorithm and same seed as 46),
and reports three sets of quantities:
  (a) E2/E1 and rho means + CI for each group (fam_strat);
  (b) within-regime gap  gap_same  = HARD_NS - SOFT_NS
      cross-regime gap  gap_cross = HARD_NS - SOFT_SS
      retention rate = gap_same / gap_cross
  (c) the regime effect itself: the SS minus NS difference within one family (SOFT_SS - SOFT_NS, etc.)

Usage
----
  python 52_regime_contrast.py

Output
----
  02_Results_Data/truth_chain/52_regime_contrast.csv
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: main-text Table 7 (two conventions for the soft-penalty--hard-skeleton contrast, with differences and bootstrap intervals) -- the sole primary endpoint of the paper
#

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import importlib.util

# Reuse the family/regime assignment and data loading of 46 to prevent convention drift
_spec = importlib.util.spec_from_file_location(
    'm395', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '46_region_analysis.py'))
m395 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m395)

B = m395.B
OD = m395.OD
LAYERS = m395.LAYERS

REGIONS = [
    ('R0_northchina', 'gpu*'),
    ('R1_yangtze', 'xr2_*'),
    ('R3_northeast', 'xr4_*'),
    ('R2_loess', 'xr5_*'),
    ('R4_longzhong', 'xr7_*'),
    ('R5_sichuan', 'xr8_*'),
]

NB = 2000
SEED = 20261006


# ----------------------------------------------------------- group means (point estimates)
def point_estimates(region):
    f = os.path.join(OD, 'withinstation_%s.csv' % region)
    if not os.path.exists(f):
        return None
    W = pd.read_csv(f, encoding='utf-8-sig')
    W['grp'] = W.fam + '_' + W.strat
    g = W.groupby(['layer', 'grp']).agg(
        ncfg=('run', 'size'), ratio=('ratio', 'mean'),
        ratio_sd=('ratio', 'std'), rho=('rho_w', 'mean'),
        x=('x', 'mean'), beat=('beat', 'sum')).reset_index()
    g['region'] = region
    return g


# ----------------------------------------------------------- cluster bootstrap
def boot_groups(P, region, nboot=NB, seed=SEED):
    """Same algorithm as 46.stage_bootstrap, but the grouping key is fam_strat"""
    raw = {}
    for (nm, cfg, lay), v in P.items():
        ids = v.id.values
        ow = m395.within(v.obs, v.id)
        rw = m395.within(v.smci, v.id)
        mw = m395.within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(ids, return_inverse=True)
        k = len(u)
        grp = '%s_%s' % (m395.fam_of(nm, cfg), m395.stratum_of(cfg))
        raw.setdefault(lay, {})[(nm, cfg)] = dict(
            grp=grp, ids=u,
            n=np.bincount(inv, minlength=k).astype(float),
            A=np.bincount(inv, weights=e1 ** 2, minlength=k),
            B=np.bincount(inv, weights=e3 ** 2, minlength=k),
            C=np.bincount(inv, weights=e1 * e3, minlength=k))

    rng = np.random.default_rng(seed)
    out = []
    for lay, cfgs in raw.items():
        keys = list(cfgs.keys())
        nst = [len(cfgs[k]['ids']) for k in keys]
        if max(nst) != min(nst):
            common = set(cfgs[keys[0]]['ids'])
            for k in keys[1:]:
                common &= set(cfgs[k]['ids'])
            common = np.sort(np.array(sorted(common)))
        else:
            common = cfgs[keys[0]]['ids']
        prep = {}
        for k in keys:
            pos = {s: i for i, s in enumerate(cfgs[k]['ids'])}
            ii = np.array([pos[s] for s in common])
            prep[k] = dict(grp=cfgs[k]['grp'], n=cfgs[k]['n'][ii],
                           A=cfgs[k]['A'][ii], B=cfgs[k]['B'][ii],
                           C=cfgs[k]['C'][ii])
        ns = len(common)
        grps = sorted(set(v['grp'] for v in prep.values()))
        acc = {g: [] for g in grps}
        acc_r = {g: [] for g in grps}
        for _ in range(nboot):
            ii = rng.integers(0, ns, ns)
            for g in grps:
                rs, rr = [], []
                for k in [k for k in keys if prep[k]['grp'] == g]:
                    d = prep[k]
                    A = d['A'][ii].sum()
                    Bb = d['B'][ii].sum()
                    C = d['C'][ii].sum()
                    nn = d['n'][ii].sum()
                    E1 = np.sqrt(A / nn)
                    E3 = np.sqrt(Bb / nn)
                    rho = (C / nn) / (E1 * E3) if E1 * E3 > 0 else 0.0
                    E2 = np.sqrt((A + Bb + 2 * C) / nn)
                    rs.append(rho)
                    rr.append(E2 / E1)
                acc[g].append(np.mean(rs))
                acc_r[g].append(np.mean(rr))
        # Group-mean CI
        for g in grps:
            a, ar = np.array(acc[g]), np.array(acc_r[g])
            nf = len([k for k in keys if prep[k]['grp'] == g])
            out.append(dict(region=region, layer=lay, kind='grp', name=g,
                            ncfg=nf, nsta=ns,
                            rho=a.mean(), rho_lo=np.percentile(a, 2.5),
                            rho_hi=np.percentile(a, 97.5),
                            ratio=ar.mean(), ratio_lo=np.percentile(ar, 2.5),
                            ratio_hi=np.percentile(ar, 97.5)))
        # Key contrasts
        pairs = [('HARD_NS', 'SOFT_NS'), ('HARD_NS', 'SOFT_SS'),
                 ('SOFT_SS', 'SOFT_NS'), ('HYBL_SS', 'HYBL_NS'),
                 ('FREE_SS', 'FREE_NS'), ('HARD_NS', 'FREE_NS')]
        for g1, g2 in pairs:
            if g1 not in acc or g2 not in acc:
                continue
            d = np.array(acc[g1]) - np.array(acc[g2])
            dr = np.array(acc_r[g1]) - np.array(acc_r[g2])
            out.append(dict(region=region, layer=lay, kind='diff',
                            name='%s - %s' % (g1, g2), ncfg=0, nsta=ns,
                            rho=d.mean(), rho_lo=np.percentile(d, 2.5),
                            rho_hi=np.percentile(d, 97.5),
                            ratio=dr.mean(), ratio_lo=np.percentile(dr, 2.5),
                            ratio_hi=np.percentile(dr, 97.5)))
    return pd.DataFrame(out)


def main():
    o = []
    ap = o.append
    pts, bts = [], []
    for region, pat in REGIONS:
        W = point_estimates(region)
        if W is None:
            ap('!! %s has no withinstation table, skipped' % region)
            continue
        pts.append(W)
        P = m395.load_pairs(pat)
        if not P:
            ap('!! %s has no paired data, bootstrap skipped' % region)
            continue
        bts.append(boot_groups(P, region))
        ap('   %s done (%d groups)' % (region, len(P)))

    PT = pd.concat(pts, ignore_index=True)
    PT.to_csv(os.path.join(B, '52_regime_groups.csv'),
              index=False, encoding='utf-8-sig')
    BT = pd.concat(bts, ignore_index=True)
    BT.to_csv(os.path.join(B, '52_regime_contrast.csv'),
              index=False, encoding='utf-8-sig')

    ap('')
    ap('=' * 100)
    ap('[Group point estimates] (E2/E1 and rho, within-station convention)')
    for lay in LAYERS:
        s = PT[PT.layer == lay].sort_values(['region', 'grp'])
        for _, r in s.iterrows():
            ap('  %-16s %-10s n=%2d  E2/E1=%.4f±%.4f  rho=%+.4f  x=%.3f  better than label %d/%d'
               % (r.region, r.grp, r.ncfg, r.ratio,
                  r.ratio_sd if not np.isnan(r.ratio_sd) else 0.0,
                  r.rho, r.x, r.beat, r.ncfg))

    ap('')
    ap('[Within-regime vs cross-regime: soft-penalty--hard-skeleton gap]')
    for region, _ in REGIONS:
        for lay in LAYERS:
            d = BT[(BT.region == region) & (BT.layer == lay) & (BT.kind == 'diff')]
            same = d[d.name == 'HARD_NS - SOFT_NS']
            cross = d[d.name == 'HARD_NS - SOFT_SS']
            if not len(same) or not len(cross):
                continue
            gs, gc = float(same.ratio.iloc[0]), float(cross.ratio.iloc[0])
            ap('  %-16s %s  cross-regime %.4f [%.4f,%.4f]  ->  within-regime %.4f [%.4f,%.4f]   retained %.0f%%'
               % (region, lay, gc, float(cross.ratio_lo.iloc[0]),
                  float(cross.ratio_hi.iloc[0]), gs,
                  float(same.ratio_lo.iloc[0]), float(same.ratio_hi.iloc[0]),
                  100.0 * gs / gc if gc != 0 else float('nan')))

    ap('')
    ap('[The regime effect itself: SS - NS within one family]')
    for region, _ in REGIONS:
        for lay in LAYERS:
            d = BT[(BT.region == region) & (BT.layer == lay) & (BT.kind == 'diff')]
            for nm in ['SOFT_SS - SOFT_NS', 'HYBL_SS - HYBL_NS', 'FREE_SS - FREE_NS']:
                s = d[d.name == nm]
                if not len(s):
                    continue
                ap('  %-16s %s %-18s  Δrho=%+.4f [%.4f,%.4f]  Δratio=%+.4f [%.4f,%.4f]'
                   % (region, lay, nm, float(s.rho.iloc[0]), float(s.rho_lo.iloc[0]),
                      float(s.rho_hi.iloc[0]), float(s.ratio.iloc[0]),
                      float(s.ratio_lo.iloc[0]), float(s.ratio_hi.iloc[0])))

    ap('')
    ap('Output: %s' % os.path.join(B, '52_regime_contrast.csv'))
    common.report('52_regime_contrast.txt', o)


if __name__ == '__main__':
    main()
