# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""65: Unified computation of cross-region sign consistency (retention) -- including new regions R7/R8

The "retention" of the three on-paper criteria O1/O2/O3 must be recomputed by the same pipeline; hand-counting is forbidden.

  O1  HARD E2/E1 > 1                       (hard skeleton worse than label)
  O2  rho_FREE - rho_HARD < 0                 (hard skeleton aligns worse than unconstrained)
  O3  rho of SOFT is the most negative of the three families (FREE/SOFT/HARD)
  O3b E2/E1 of FREE and SOFT both < 1          (the absolute outcome "both better than label")
  O4  E2/E1 of SOFT < E2/E1 of HARD           (between-family ranking)

The baseline sign is taken from the corresponding layer of North China R3; independent units are counted by region (not region x layer).
Output: 02_Results_Data/truth_chain/multi/crossregion_sign.csv + log
"""
import os
import sys
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
SORT_KEY = {'R4_longzhong': 'Longzhong semi-arid', 'R5_sichuan': 'Sichuan Basin humid'}
LAYERS = ['10cm', '30cm']
FAMS3 = ['FREE', 'SOFT', 'HARD']
EXCLUDE_LPMIXNS = True


def load(reg):
    f = os.path.join(ODT, 'withinstation_%s.csv' % reg)
    if not os.path.exists(f):
        return None
    W = pd.read_csv(f, encoding='utf-8-sig')
    if EXCLUDE_LPMIXNS:
        W = W[W.cfg != 'LPMIXNS']
    return W


def fam_row(W, lay, fam):
    g = W[(W.layer == lay) & (W.fam == fam)]
    if not len(g):
        return None
    return dict(n=len(g), beat=int((g.ratio < 1).sum()),
                ratio=float(g.ratio.mean()), rho=float(g.rho_w.mean()),
                x=float(g.x.mean()), floor=float(g.floor.mean()))


def main():
    data = {}
    for r in ORDER:
        W = load(r)
        if W is None:
            continue
        data[r] = W

    have = [r for r in ORDER if r in data]
    o = ['=' * 96, '65 Cross-region sign-consistency (retention) computation', '=' * 96,
         'Participating regions: ' + ', '.join('%s(%s)' % (r, REGCN[r]) for r in have),
         'Convention: within-station W; family means; %s LPMIXNS'
         % ('excludes' if EXCLUDE_LPMIXNS else 'includes'), '']

    # ---------- triplets per region ----------
    o.append('[Family-level aggregation per region]')
    o.append('%-14s %-5s %-5s %3s %5s %9s %9s %9s'
             % ('region', 'layer', 'family', 'n', 'wins', 'x', 'rho', 'E2/E1'))
    T = {}
    for r in have:
        for lay in LAYERS:
            for fam in ['FREE', 'SOFT', 'HYBL', 'HARD', 'DAMAGED']:
                d = fam_row(data[r], lay, fam)
                if d is None:
                    continue
                T[(r, lay, fam)] = d
                o.append('%-14s %-5s %-5s %3d %2d/%-2d %9.4f %+9.4f %9.4f'
                         % (REGCN[r], lay, fam, d['n'], d['beat'], d['n'],
                            d['x'], d['rho'], d['ratio']))
        o.append('')

    # ---------- sign matrix of the four criteria ----------
    def crit(r, lay, key):
        if (r, lay, 'FREE') not in T or (r, lay, 'HARD') not in T:
            return None
        if (r, lay, 'SOFT') not in T:
            return None
        rf, rh, rs = T[(r, lay, 'FREE')], T[(r, lay, 'HARD')], T[(r, lay, 'SOFT')]
        if key == 'O1':
            return 1 if rh['ratio'] > 1 else -1
        if key == 'O2':
            return 1 if rf['rho'] - rh['rho'] < 0 else -1
        if key == 'O3':
            rho = {f: T[(r, lay, f)]['rho'] for f in FAMS3}
            return 1 if min(rho, key=rho.get) == 'SOFT' else -1
        if key == 'O3b':
            return 1 if (rf['ratio'] < 1 and rs['ratio'] < 1) else -1
        if key == 'O4':
            return 1 if rs['ratio'] < rh['ratio'] else -1
        return None

    base = 'R0_northchina'
    o.append('[Sign matrix (1 = same-sign direction as North China; differences in parentheses are observed values)]')
    keys = ['O1', 'O2', 'O3', 'O3b', 'O4']
    rows = []
    for lay in LAYERS:
        brow = {k: crit(base, lay, k) for k in keys}
        o.append('  %s  North China baseline sign: %s' % (lay, ' '.join(
            '%s=%+d' % (k, brow[k]) for k in keys if brow[k])))
        for r in have:
            if r == base:
                continue
            cells = []
            row = dict(region=r, region_cn=REGCN[r], layer=lay)
            for k in keys:
                s = crit(r, lay, k)
                row[k] = s
                row[k + '_keep'] = int((s is not None) and (s == brow[k]))
                cells.append('%s%s' % (k, '+' if s == brow[k] else '-'))
            rows.append(row)
            o.append('    %-10s %s' % (REGCN[r], '  '.join(cells)))
        o.append('')

    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(ODT, 'crossregion_sign.csv'),
             index=False, encoding='utf-8-sig')

    # ---------- retention summary ----------
    o.append('[Retention: joint count over region x layer; independent units counted by region]')
    for k in keys:
        sub = S[S[k + '_keep'].notna()]
        keep = int(sub[k + '_keep'].sum())
        tot = len(sub)
        # Counted by region: a region counts as retained only if both layers are retained
        by_region = {}
        for r in set(sub.region):
            b = sub[sub.region == r][k + '_keep']
            by_region[r] = int(b.sum())
        nreg = len(by_region)
        nreg_keep = sum(1 for v in by_region.values() if v == 2)
        import math
        p = 2 * 0.5 ** nreg
        o.append('  %-4s region x layer %2d/%2d   by region (both layers retained) %d/%d   '
                 'sign test p=%.4f' % (k, keep, tot, nreg_keep, nreg, p))
        o.append('        per-region detail: %s' % '  '.join(
            '%s %d/2' % (REGCN[r], v) for r, v in by_region.items()))

    # ---------- auxiliary: E2/E1 of HARD 10cm by region ----------
    o.append('')
    o.append('[E2/E1 of HARD by region (direct reading of O1)]')
    for lay in LAYERS:
        s = '  '.join('%s %.4f' % (REGCN[r], T[(r, lay, 'HARD')]['ratio'])
                      for r in have if (r, lay, 'HARD') in T)
        o.append('  %-5s %s' % (lay, s))
    o.append('')
    o.append('[rho_FREE - rho_HARD by region (direct reading of O2)]')
    for lay in LAYERS:
        s = '  '.join('%s %+.4f' % (REGCN[r], T[(r, lay, 'FREE')]['rho']
                                    - T[(r, lay, 'HARD')]['rho'])
                      for r in have if (r, lay, 'FREE') in T
                      and (r, lay, 'HARD') in T)
        o.append('  %-5s %s' % (lay, s))

    common.report('65_crossregion_sign', o)
    print('\n'.join(o[:6]))
    print('...(full table in scripts/65_crossregion_sign)')


if __name__ == '__main__':
    main()
