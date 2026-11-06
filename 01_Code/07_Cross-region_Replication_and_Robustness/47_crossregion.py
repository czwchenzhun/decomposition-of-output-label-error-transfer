# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""47 Cross-region consistency check -- does the main conclusion replicate across 4 climatic regions

Read the withinstation/quadratic/price/bootstrap tables produced by 46 in each region, and answer:
  Q1 Is the family ranking consistent across regions? (FREE/SOFT better than label, HARD worse than label)
  Q2 Is "hard skeleton destroys alignment" consistent across regions? (rho_FREE - rho_HARD < 0, CI excludes zero)
  Q3 Do the three predictions of the calibration condition hold across regions?
  Q4 Does the practical criterion of the price of truth hold across regions? (the larger the mismatch, the larger the calibration gain)

Output
----
  02_Results_Data/truth_chain/multi/crossregion_families.csv
  02_Results_Data/truth_chain/multi/crossregion_tests.csv
  scripts/_47_crossregion.txt
"""
import os
import glob
import numpy as np
import pandas as pd

OD = MULTI
TXT = LOG + '/_47_crossregion.txt'
LAYERS = ['10cm', '30cm']
FAMS = ['FREE', 'SOFT', 'HYBL', 'HARD', 'DAMAGED']

REGCN = {'R0_northchina': 'R0 North China Plain (85 stations, sub-humid)',
         'R1_yangtze': 'R1 middle--lower Yangtze (21 stations, humid)',
         'R2_loess': 'R2 Loess Plateau (19 stations, semi-arid)',
         'R3_northeast': 'R3 Northeast China black soil (14 stations, cold-humid)',
         'R4_longzhong': 'R4 Longzhong loess (32 stations, semi-arid)',
         'R5_sichuan': 'R5 Sichuan Basin (24 stations, humid)'}
ORDER = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast',
         'R4_longzhong', 'R5_sichuan']


def main():
    o = []

    def ap(s=''):
        o.append(s)
        print(s)

    ws, bs, qd, pr = {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(OD, 'withinstation_*.csv'))):
        reg = os.path.basename(f).replace('withinstation_', '').replace('.csv', '')
        ws[reg] = pd.read_csv(f, encoding='utf-8-sig')
    for f in sorted(glob.glob(os.path.join(OD, 'bootstrap_*.csv'))):
        reg = os.path.basename(f).replace('bootstrap_', '').replace('.csv', '')
        bs[reg] = pd.read_csv(f, encoding='utf-8-sig')
    for f in sorted(glob.glob(os.path.join(OD, 'quadratic_*.csv'))):
        reg = os.path.basename(f).replace('quadratic_', '').replace('.csv', '')
        qd[reg] = pd.read_csv(f, encoding='utf-8-sig')
    for f in sorted(glob.glob(os.path.join(OD, 'price_*.csv'))):
        reg = os.path.basename(f).replace('price_', '').replace('.csv', '')
        pr[reg] = pd.read_csv(f, encoding='utf-8-sig')

    regs = [r for r in ORDER if r in ws]
    ap('=' * 100)
    ap('47 Cross-region consistency check    region: %s' % ', '.join(regs))
    ap('=' * 100)
    ap('')

    # ---------------- Q1 family outcomes ----------------
    ap('[Q1] Family outcomes (within-station convention E2/E1, <1 means better than its own training label)')
    rows = []
    for lay in LAYERS:
        ap('')
        ap('  ---- %s ----' % lay)
        ap('  %-18s %8s %8s %8s %8s %8s' % ('region', 'FREE', 'SOFT', 'HYBL', 'HARD', 'DAMAGED'))
        for reg in regs:
            W = ws[reg]
            s = W[W.layer == lay]
            cells = []
            for f in FAMS:
                g = s[s.fam == f]
                if not len(g):
                    cells.append('    --  ')
                    continue
                cells.append('%5.4f(%d/%d)' % (g.ratio.mean(), int(g.beat.sum()), len(g)))
                rows.append(dict(region=reg, layer=lay, fam=f, n=len(g),
                                 ratio=g.ratio.mean(), sd=g.ratio.std(),
                                 rho=g.rho_w.mean(), floor=g.floor.mean(),
                                 beat=int(g.beat.sum())))
            ap('  %-18s %8s %8s %8s %8s %8s'
               % (REGCN.get(reg, reg), *cells))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OD, 'crossregion_families.csv'),
             index=False, encoding='utf-8-sig')
    ap('')
    ap('  Criterion: FREE/SOFT all <1 and HARD all >1 -> %s'
       % ('holds' if ((F[(F.fam.isin(['FREE', 'SOFT']))].ratio < 1).all()
                     and (F[F.fam == 'HARD'].ratio > 1).all())
          else 'does not hold'))
    ap('')

    # ---------------- Q2 hard skeleton destroys alignment ----------------
    ap('[Q2] Does the hard skeleton destroy alignment (rho_FREE - rho_HARD, station-clustered bootstrap 95% CI)')
    trows = []
    for lay in LAYERS:
        ap('')
        ap('  ---- %s ----' % lay)
        for reg in regs:
            B = bs[reg]
            if reg not in bs:
                continue
            s = B[(B.layer == lay)]
            # The contrast names in 46 follow fam lexicographic order: HARD < HYBL, hence HARD-HYBL
            for key in ['FREE-HARD', 'FREE-SOFT', 'HARD-SOFT',
                        'FREE-HYBL', 'HARD-HYBL']:
                g = s[s.fam == key]
                if not len(g):
                    continue
                r = g.iloc[0]
                sig = 'significant' if (r.rho_lo * r.rho_hi > 0) else 'not significant'
                ap('  %-18s %-10s  Δrho %+7.4f [%+.4f,%+.4f]  %s   ΔE2/E1 %+7.4f [%+.4f,%+.4f]'
                   % (REGCN.get(reg, reg), key, r.rho, r.rho_lo, r.rho_hi, sig,
                      r.ratio, r.ratio_lo, r.ratio_hi))
                trows.append(dict(region=reg, layer=lay, contrast=key,
                                  drho=r.rho, drho_lo=r.rho_lo, drho_hi=r.rho_hi,
                                  dratio=r.ratio, dratio_lo=r.ratio_lo,
                                  dratio_hi=r.ratio_hi,
                                  sig=bool(r.rho_lo * r.rho_hi > 0)))  # 46 output column names are rho/*
    T = pd.DataFrame(trows)
    T.to_csv(os.path.join(OD, 'crossregion_tests.csv'),
             index=False, encoding='utf-8-sig')
    g = T[T.contrast == 'FREE-HARD']
    nsig = int((g.drho < 0).sum())
    ap('')
    ap('  Criterion: delta_rho(FREE-HARD) < 0 holds for %d/%d region x layer combinations; CI excludes zero for %d/%d'
       % (nsig, len(g), int(g.sig.sum()), len(g)))
    ap('')

    # ---------------- Q3 calibration condition ----------------
    ap('[Q3] Maximum deviation of the three predictions of the calibration condition (all should be at machine precision)')
    for lay in LAYERS:
        for reg in regs:
            D = qd.get(reg)
            if D is None:
                continue
            s = D[D.layer == lay]
            if not len(s):
                continue
            ap('  %-6s %-18s P1 %.2e   P2 %.2e   P3 %.2e   (n=%d)'
               % (lay, REGCN.get(reg, reg),
                  np.abs(s.rho_check - s.rho).max(),
                  np.abs(s.lam_th - s.lam_em).max(),
                  np.abs(s.E2E1_opt - s.floor).max(), len(s)))
    ap('')

    # ---------------- Q4 price of truth ----------------
    ap('[Q4] Price of truth: E2/E1 after calibration with 3 stations (capture fraction in parentheses)')
    for lay in LAYERS:
        ap('')
        ap('  ---- %s ----' % lay)
        for reg in regs:
            P = pr.get(reg)
            if P is None:
                continue
            s = P[P.layer == lay]
            cells = []
            for f in ['FREE', 'SOFT', 'HYBL', 'HARD']:
                gg = s[s.fam == f]
                if not len(gg) or 'k3' not in gg.columns:
                    cells.append('   --   ')
                    continue
                cap = (gg.base - gg.k3) / (gg.base - gg.oracle)
                cells.append('%.4f(%+.0f%%)' % (gg.k3.mean(), 100 * cap.mean()))
            ap('  %-18s FREE %-16s SOFT %-16s HYBL %-16s HARD %s'
               % (REGCN.get(reg, reg), *cells))
    ap('')
    ap('  Criterion: calibration gain should correlate positively with the degree of mismatch -- the capture fraction of HARD should be positive and largest in every region.')
    ap('')

    # ---------------- Q5 consistency of the rho family ranking ----------------
    ap('[Q5] Whose alignment (rho, the more negative the better) is best? Is it consistent across regions')
    rank = {}
    for lay in LAYERS:
        for reg in regs:
            B = bs[reg]
            s = B[(B.layer == lay) & (B.fam.isin(['FREE', 'SOFT', 'HYBL', 'HARD']))]
            if len(s) < 3:
                continue
            v = s.set_index('fam').rho
            best = v.idxmin()
            rank[(reg, lay)] = (best, v.min())
    allA = [k[0] for k in rank.values()]
    ns = len(rank)
    cnt = {f: allA.count(f) for f in ['FREE', 'SOFT', 'HYBL', 'HARD']}
    for lay in LAYERS:
        seg = ['%s:%s' % (REGCN[k[0]].split()[0], rank[k][0])
               for k in rank if k[1] == lay]
        ap('  %s  %s' % (lay, '   '.join(seg)))
    ap('  Best-rho family counts: SOFT %d / HYBL %d / HARD %d / FREE %d  (%d region x layer combos in total)'
       % (cnt['SOFT'], cnt['HYBL'], cnt['HARD'], cnt['FREE'], ns))
    g5 = T[(T.contrast == 'FREE-SOFT') | (T.contrast == 'HARD-SOFT')]
    n5 = int(((g5.drho > 0) & g5.sig).sum())
    ap('   rho_%s - rho_SOFT > 0 combinations: %d/%d; of which CI excludes zero: %d/%d'
       % ('FREE/HARD', int((g5.drho > 0).sum()), len(g5), n5, len(g5)))
    ap('  Note: the 8 region x layer combinations are not independent samples; the sign test counted over 4 independent regions '
       'p = 2x0.5^4 = 0.125, so this is descriptive evidence rather than a significance conclusion.')
    ap('')

    # ---------------- Synthesis ----------------
    ap('=' * 100)
    ok1 = bool((F[(F.fam.isin(['FREE', 'SOFT']))].ratio < 1).all()
               and (F[F.fam == 'HARD'].ratio > 1).all())
    gg = T[T.contrast == 'FREE-HARD']
    ok2 = int((gg.drho < 0).sum()) >= len(gg) - 1
    ap('Synthesis: Q1 family ranking consistent across regions = %s ; Q2 hard skeleton destroys alignment consistent across regions = %s (%d/%d)'
       % ('Yes' if ok1 else 'No', 'Yes' if ok2 else 'No',
          int((gg.drho < 0).sum()), len(gg)))
    ap('=' * 100)

    with open(TXT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(o) + '\n')
    print('\n-> %s' % TXT)


if __name__ == '__main__':
    main()
