# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""66: Cross-region statistical summary (for main-text numbers, all computed by script, no manual filling)

Input: multi/crossregion_tests.csv (produced by 47, six-region station-clustered bootstrap contrasts)
      multi/withinstation_*.csv (produced by 46, per-configuration within-station convention)

Output (all cross-region counts needed by the main text):
  1. delta_E2/E1 and delta_rho of HARD-SOFT: numbers of positive/negative, numbers with CI excluding zero, stratified by region
  2. FREE-HARD / FREE-SOFT likewise
  3. List of configurations with rho>=0 by region (to show lambda* is truncated by the lower bound rather than a failure of the calibration condition)
  4. Two-sided sign-test p (a region counts as same-sign only if both layers have the same sign)
"""
import os
import sys
import math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

ODT = MULTI

ORDER = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast',
         'R4_longzhong', 'R5_sichuan']
REGCN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
         'R2_loess': 'Loess Plateau', 'R3_northeast': 'Northeast China black soil',
         'R4_longzhong': 'Longzhong loess', 'R5_sichuan': 'Sichuan Basin'}
NSTA = {'R0_northchina': 85, 'R1_yangtze': 21, 'R2_loess': 19,
        'R3_northeast': 14, 'R4_longzhong': 32, 'R5_sichuan': 24}


def binom_two_sided(k, n):
    """Two-sided exact sign-test p (<=1)"""
    def tail(kk):
        return sum(math.comb(n, i) for i in range(kk, n + 1)) / (2.0 ** n)
    return min(1.0, 2 * min(tail(k), tail(n - k)))


def main():
    o = ['=' * 100, '66 Cross-region statistical summary (six regions)', '=' * 100]
    T = pd.read_csv(os.path.join(ODT, 'crossregion_tests.csv'),
                    encoding='utf-8-sig')
    T = T[T.region.isin(ORDER)]
    T['drsig'] = T.dratio_lo * T.dratio_hi > 0      # CI of delta_E2/E1 excludes zero
    o.append('%d contrast records; %d regions involved'
             % (len(T), T.region.nunique()))

    # ---------------- 1. signs + CI of the three contrasts ----------------
    for contrast in ['HARD-SOFT', 'FREE-HARD', 'FREE-SOFT']:
        sub = T[T.contrast == contrast]
        o.append('')
        o.append('[%s] %d (region x layer) records' % (contrast, len(sub)))
        for lay in ['10cm', '30cm']:
            g = sub[sub.layer == lay]
            if not len(g):
                continue
            pos = int((g.dratio > 0).sum())
            sig = int(g.drsig.sum())          # CI of delta_E2/E1 excludes zero
            posr = int((g.drho > 0).sum())
            sigr = int(g.sig.sum())           # CI of delta_rho excludes zero
            o.append('  %-5s delta_E2/E1 > 0  %2d/%2d  its CI excludes zero %2d/%2d   |   '
                     'delta_rho > 0  %2d/%2d  its CI excludes zero %2d/%2d'
                     % (lay, pos, len(g), sig, len(g), posr, len(g), sigr, len(g)))
            o.append('         ' + '  '.join(
                '%s %+.4f[%+.4f,%+.4f]%s'
                % (REGCN[r.region][:2], r.dratio, r.dratio_lo, r.dratio_hi,
                   '' if r.dratio_lo * r.dratio_hi > 0 else '*')
                for _, r in g.iterrows()))
        # Counted by region: both layers have the same sign
        by = {}
        for reg in ORDER:
            g = sub[sub.region == reg]
            if len(g) < 2:
                continue
            both_pos = bool((g.dratio > 0).all())
            both_sig = bool((g.dratio_lo * g.dratio_hi > 0).all())
            by[reg] = (int(both_pos), int(both_sig))
        kp = sum(v[0] for v in by.values())
        ks = sum(v[1] for v in by.values())
        n = len(by)
        o.append('  By region (both layers satisfied): delta_E2/E1 same sign %d/%d  p=%.4f;'
                 'both-layer CIs exclude zero %d/%d  p=%.4f'
                 % (kp, n, binom_two_sided(kp, n), ks, n, binom_two_sided(ks, n)))
        o.append('      ' + '  '.join(
            '%s %d/%s' % (REGCN[r], v[1], '2') for r, v in by.items()))

    # ---------------- 2. configurations with rho >= 0 (lambda* truncated by the lower bound) ----------------
    o.append('')
    o.append('[Configurations with rho >= 0 (here lambda* = -rho/x <= 0, truncated by the feasible-region lower bound)]')
    tot = []
    for reg in ORDER:
        f = os.path.join(ODT, 'withinstation_%s.csv' % reg)
        if not os.path.exists(f):
            continue
        W = pd.read_csv(f, encoding='utf-8-sig')
        bad = W[W.rho_w >= 0]
        tot.append(len(bad))
        if len(bad):
            o.append('  %-6s %d: %s'
                     % (REGCN[reg], len(bad),
                        '  '.join('%s/%s/%s rho=%+.4f' % (r.run, r.cfg, r.layer,
                                                          r.rho_w)
                                  for _, r in bad.iterrows())))
        else:
            o.append('  %-6s 0' % REGCN[reg])
    o.append('  Total %d' % sum(tot))

    common.report('66_crossregion_stats', o)
    print('WROTE scripts/66_crossregion_stats.txt')
    print('\n'.join(o[2:14]))


if __name__ == '__main__':
    main()
