# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""53 M4: cross-region family means recomputed on the ``standard capacity'' subset (excluding deliberately degraded FREE configurations)

Problem (fourth-round review M4)
--------------------
The 7 configurations of FREE in the primary region include 3 **explicitly flagged as degraded** capacity/epoch variants
(5 epochs, 10 epochs, hidden size 32; run names gpu_deg5 / gpu_deg10 / gpu_deg32),
yet they are weighted equally with full-capacity configurations in the cross-region family means. The manuscript applied a remedy in one place for North China
(restricting to the 4 non-degraded standard configurations, p=0.381), but the cross-region aggregate did not receive the same treatment,
and the cross-region aggregate is precisely the source of the headline conclusion.

Note: run names containing ``degwd'' are the **damaged control** (weight decay disrupts training, DAMAGED family),
not degraded capacity variants, so they must be retained.

Output
----
  02_Results_Data/truth_chain/53_std_capacity.csv
"""
import os
import numpy as np
import pandas as pd
import common

OD = MULTI
B = TRUTH
REGIONS = ['R0_northchina', 'R1_yangtze', 'R3_northeast', 'R2_loess']
DEGRADED = ['gpu_deg5', 'gpu_deg10', 'gpu_deg32']   # only these 3 are degraded capacity variants


def load():
    rows = []
    for reg in REGIONS:
        f = os.path.join(OD, 'withinstation_%s.csv' % reg)
        if not os.path.exists(f):
            continue
        W = pd.read_csv(f, encoding='utf-8-sig')
        W['region'] = reg
        rows.append(W)
    return pd.concat(rows, ignore_index=True)


def main():
    W = load()
    W['degraded'] = W.run.isin(DEGRADED)
    assert W[W.degraded].fam.unique().tolist() == ['FREE'], W[W.degraded].fam.unique()

    o = []
    ap = o.append
    out = []

    for subset, keep in [('all', None), ('std', False)]:
        D = W if keep is None else W[W.degraded == keep]
        # Per-region family means
        g = D.groupby(['region', 'layer', 'fam']).agg(
            n=('run', 'size'), ratio=('ratio', 'mean'),
            rho=('rho_w', 'mean'), x=('x', 'mean'),
            beat=('beat', 'sum')).reset_index()
        g['subset'] = subset
        out.append(g)
        # Cross-region aggregate (regions equally weighted, then averaged at configuration level)
        p = D.groupby(['layer', 'fam']).agg(
            n=('run', 'size'), ratio=('ratio', 'mean'),
            rho=('rho_w', 'mean'), x=('x', 'mean'),
            beat=('beat', 'sum')).reset_index()
        p['subset'] = subset
        p['region'] = 'CROSS'
        out.append(p)

        ap('=' * 92)
        ap('[Cross-region aggregate - %s]' % subset)
        for _, r in p.iterrows():
            ap('  %-5s %-8s n=%2d  E2/E1=%.4f  rho=%+.4f  x=%.3f  better than label %d/%d'
               % (r.layer, r.fam, r.n, r.ratio, r.rho, r.x, r.beat, r.n))

    # Whether the SOFT vs FREE contrast changes after removing degraded configurations across the 8 (region x layer) combinations
    ap('')
    ap('[8 (region x layer) combinations: SOFT - FREE]')
    for lay in ['10cm', '30cm']:
        for reg in REGIONS:
            line = '  %-16s %s' % (reg, lay)
            for subset, keep in [('all', None), ('std', False)]:
                D = W if keep is None else W[W.degraded == keep]
                s = D[(D.region == reg) & (D.layer == lay)]
                fr = s[s.fam == 'FREE']
                so = s[s.fam == 'SOFT']
                if not len(fr) or not len(so):
                    continue
                line += '   %s: dRho=%+.4f dRatio=%+.4f (n_FREE=%d)' % (
                    subset, so.rho_w.mean() - fr.rho_w.mean(),
                    so.ratio.mean() - fr.ratio.mean(), len(fr))
            ap(line)

    R = pd.concat(out, ignore_index=True)
    R.to_csv(os.path.join(B, '53_std_capacity.csv'), index=False,
             encoding='utf-8-sig')
    ap('')
    ap('Output: %s' % os.path.join(B, '53_std_capacity.csv'))
    common.report('53_std_capacity.txt', o)


if __name__ == '__main__':
    main()
