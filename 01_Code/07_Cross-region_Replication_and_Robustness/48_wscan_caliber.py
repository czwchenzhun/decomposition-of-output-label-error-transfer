# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
48_wscan_caliber.py -- responding to review comments M1 (first round) and R1/r1/r2/r3 (second round)

Purpose: recompute, for the SOFT family (LPMIX, soft water-balance penalty w scan),
      the regression of rho / x / E2E1 / alpha on log10(w) under **three error conventions**
      one by one, and output a CSV for citation in the main text.

Three conventions:
  W  within-station anomaly convention (primary convention): computed after de-meaning by station; rho_w / E1w / E2w / E3w from script 25.
     This is the convention used throughout the rest of the paper (64--74% of the label error is an uncorrectable between-station static offset).
  P  full convention: no de-meaning; rho_f / E1f / E2f / E3f from script 25.
     Like W, this is an **exact convention**, but with a different estimand: its denominator E1 contains 64--74% uncorrectable
     between-station static variance, so P and W are not equivalent and cannot be substituted for one another.
  G  globally de-meaned convention (approximate, diagnostic convention): debias=True in script 19,
     the three series each have their **overall mean** subtracted (not de-meaning by station), and the quadratic-form identity no longer holds exactly here
     -- the source comments of 19 already state that this is a "supplementary convention".
     Data path: rows with debias=True in `err_transfer_region_gpu_wbw*.csv`,
     1 row per layer per run file; the 15 wbw* files with w>0 give n=15 per layer.

2026-10-07 revision (second-round review R1):
  The original version computed rho and alpha only for P / G, so 8 of the 24 cells in main-text Table 5
  (4 cells each for x and E2/E1 under the P and G conventions) had no computational basis in the archive.
  This version: script 25 now supplies the E3f / E2f columns, so under P, x = E3f/E1f and E2/E1 = E2f/E1f can be computed directly;
  under G, x = E3/E1 and E2/E1 = E2/E1 are computed directly from the output of 19. All 24 cells of Table 5 now have a source.

2026-10-07 revision (second-round review r2):
  The configuration-level regression treats w levels and seed re-runs as independent samples, overstating the nominal degrees of freedom.
  This version reports three inferences in parallel:
    (a) configuration-level OLS (original convention, retained for comparison with the earlier version);
    (b) cluster-robust SE clustered by w (number of clusters = number of w levels);
    (c) regression on w-level means (each w level collapsed to 1 point, n = number of levels, most conservative).
  (c) is the primary inference convention in the main text, (a) serves only as a comparison.

2026-10-07 revision (second-round review r1 / r3):
  r3  w=0 cannot take log10, so it is excluded from the regression (Table 1 still lists this level).
  r1  The three replication regions have only three levels, w in {1,3,10}, too few to support a regression;
      this version instead reports a sign-consistency test of delta_x = x(w=10) - x(w=1) and outputs a separate CSV.
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: main-text Table 6 (convention sensitivity of the soft-constraint strength scan) and the strength-scan part of Fig. 5
#   Output: wscan_caliber_sensitivity.csv
#

import os
import sys
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report  # noqa: E402

B = TRUTH
MULTI = os.path.join(B, 'multi')
OUT = B

REPRO = [('R1_yangtze', 'middle--lower Yangtze'), ('R3_northeast', 'Northeast China black soil'),
         ('R2_loess', 'Loess Plateau')]


def w_of(run):
    m = re.match(r'gpu_wbw([0-9.]+)(?:s(\d+))?$', run)
    if not m:
        return np.nan, None
    return float(m.group(1)), (m.group(2) or '42')


def lin(x, y):
    """Configuration-level OLS"""
    if len(x) < 3 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, len(x)
    sl, ic, r, p, _ = stats.linregress(x, y)
    return sl, p, r ** 2, len(x)


def ols_cluster(x, y, g):
    """Cluster-robust OLS clustered by g: returns (slope, p_cluster, n, nclus)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = np.asarray(g)
    n = len(x)
    if n < 3 or np.std(y) == 0:
        return np.nan, np.nan, n, 0
    X = np.column_stack([np.ones(n), x])
    XtXinv = np.linalg.pinv(X.T @ X)
    b = XtXinv @ (X.T @ y)
    u = y - X @ b
    ug = np.unique(g)
    G = len(ug)
    meat = np.zeros((2, 2))
    for c in ug:
        msk = g == c
        s = X[msk].T @ u[msk]
        meat += np.outer(s, s)
    k = 2
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtXinv @ meat @ XtXinv * corr
    se = np.sqrt(np.diag(V))[1]
    if se <= 0:
        return b[1], np.nan, n, G
    t = b[1] / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=max(G - 1, 1)))
    return b[1], p, n, G


def wlevel_fit(d, wc, yc):
    """Regress after collapsing each w level to 1 point (most conservative convention)"""
    g = d.groupby(wc)[yc].mean().sort_index()
    xx = np.log10(g.index.values.astype(float))
    return lin(xx, g.values)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--tag', default='wscan_caliber')
    args = ap_.parse_args()

    _lines = []

    def ap(s=''):
        _lines.append(s)

    Wtab = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'),
                       encoding='utf-8-sig')
    Wtab['w'], Wtab['seed'] = zip(*Wtab.run.map(w_of))
    S = Wtab.dropna(subset=['w']).copy()
    S0 = S[S.w <= 0]              # r3: w=0 cannot take log10, excluded
    S = S[S.w > 0].copy()
    S['x'] = S.E3w / S.E1w
    S['xf'] = S.E3f / S.E1f
    S['ratio'] = S.E2w / S.E1w
    S['ratiof'] = S.E2f / S.E1f

    # Per-run output of 19: the same file contains debias=False(full convention) and debias=True(globally de-meaned)
    grow, gfiles = [], []
    for f in sorted(glob.glob(os.path.join(B, 'err_transfer_region_gpu_wbw*.csv'))):
        run = os.path.basename(f).replace('err_transfer_region_', '').replace('.csv', '')
        w, sd = w_of(run)
        if np.isnan(w) or w <= 0:
            continue
        d = pd.read_csv(f, encoding='utf-8-sig')
        for _, r in d.iterrows():
            if not bool(r.debias):
                continue
            grow.append(dict(run=run, w=w, layer=r.layer, rho=r.rho,
                             E1=r.E1, E3=r.E3, E2=r.E2, alpha=r.alpha))
        gfiles.append(run)
    G = pd.DataFrame(grow)
    G['x'] = G.E3 / G.E1
    G['ratio'] = G.E2 / G.E1

    rows = []
    for lay in ['10cm', '30cm']:
        d = S[S.layer == lay].copy()
        specs = [('W_within-station(primary convention)', 'rho_w', 'x', 'ratio', 'a_w'),
                 ('P_full convention', 'rho_f', 'xf', 'ratiof', 'a_f')]
        for cname, crho, cx, crat, calp in specs:
            for metric, col in [('rho', crho), ('x', cx),
                                ('E2/E1', crat), ('alpha', calp)]:
                sl, p, r2, n = lin(np.log10(d.w), d[col])
                slc, pc, nc, ng = ols_cluster(np.log10(d.w.values), d[col].values,
                                              d.w.values)
                slw, pw, r2w, nw = wlevel_fit(d, 'w', col)
                rows.append(dict(caliber=cname, layer=lay, metric=metric,
                                 slope=sl, p=p, R2=r2, n=n,
                                 slope_cl=slc, p_cl=pc, nclus=ng,
                                 slope_wlvl=slw, p_wlvl=pw, n_wlvl=nw))
        # 19 globally de-meaned (approximate diagnostic convention)
        sub = G[G.layer == lay].copy()
        for metric, col in [('rho', 'rho'), ('x', 'x'),
                            ('E2/E1', 'ratio'), ('alpha', 'alpha')]:
            sl, p, r2, n = lin(np.log10(sub.w), sub[col])
            slc, pc, nc, ng = ols_cluster(np.log10(sub.w.values), sub[col].values,
                                          sub.w.values)
            slw, pw, r2w, nw = wlevel_fit(sub, 'w', col)
            rows.append(dict(caliber='G_globally de-meaned(approx. diagnostic)', layer=lay, metric=metric,
                             slope=sl, p=p, R2=r2, n=n,
                             slope_cl=slc, p_cl=pc, nclus=ng,
                             slope_wlvl=slw, p_wlvl=pw, n_wlvl=nw))

    R = pd.DataFrame(rows)
    op = os.path.join(OUT, args.tag + '_sensitivity.csv')
    R.to_csv(op, index=False, encoding='utf-8-sig')

    # ---------------- Strength-direction test in the replication regions (r1) ----------------
    rep = []
    for reg, cn in REPRO:
        w = pd.read_csv(os.path.join(MULTI, 'withinstation_%s.csv' % reg),
                        encoding='utf-8-sig')
        for lay in ['10cm', '30cm']:
            d = w[(w.layer == lay) & (w.cfg == 'LPMIX')].copy()
            d['w'] = d.run.str.replace(r'^xr\d_w', '', regex=True).astype(float)
            d = d.sort_values('w')
            if not len(d):
                continue
            x1 = d[d.w == 1].x.values
            x10 = d[d.w == 10].x.values
            r1_ = d[d.w == 1].rho_w.values
            r10 = d[d.w == 10].rho_w.values
            rep.append(dict(region=reg, region_cn=cn, layer=lay,
                            x_w1=float(np.mean(x1)), x_w10=float(np.mean(x10)),
                            dx=float(np.mean(x10) - np.mean(x1)),
                            rho_w1=float(np.mean(r1_)), rho_w10=float(np.mean(r10)),
                            drho=float(np.mean(r10) - np.mean(r1_)),
                            n_levels=int(d.w.nunique())))
    RP = pd.DataFrame(rep)
    RP.to_csv(os.path.join(OUT, args.tag + '_repro.csv'), index=False,
              encoding='utf-8-sig')

    # ---------------- Report ----------------
    ap('[Convention sensitivity of the soft-constraint strength scan] Output: %s (all 24 cells have a source)' % op)
    ap('  Soft-constraint configurations n=%d (%d w levels with w>0); plus %d configurations with w=0'
       'excluded because log10 is undefined (r3).'
       % (len(S[S.layer == '10cm']), S[S.layer == '10cm'].w.nunique(), len(S0)))
    ap('')
    ap('%-22s %-5s %-6s %9s %9s %9s %9s'
       % ('convention', 'layer', 'quantity', 'slope', 'p(config)', 'p(cluster)', 'p(w level)'))
    for _, r in R.iterrows():
        ap('%-22s %-5s %-6s %+9.4f %9.3g %9.3g %9.3g'
           % (r.caliber, r.layer, r.metric, r.slope, r.p, r.p_cl, r.p_wlvl))

    ap('')
    ap('  Note 1: the unit of alpha is the "alpha increment per decade", likewise for E2/E1; multiply by 100 for percentage points per decade.')
    ap('  Note 2: the three conventions are not equivalent in status -- W and P are two **exact conventions** (same samples,')
    ap('       with e2=e1+e3 pointwise), but with different estimands: the denominator of P contains 64--74% uncorrectable')
    ap('       between-station static variance; G is an **approximate diagnostic convention** (globally de-meaned; the source comments of 19 state')
    ap('       that the identity does not hold here). Hence results from the three conventions should not be counted with equal weight.')
    ap('  Note 3: the data path for G = rows with debias=True in err_transfer_region_gpu_wbw*.csv,')
    ap('       1 row per layer in each run file with w>0, %d files in total -> n=%d per layer.'
       % (len(gfiles), len(G[G.layer == '10cm'])))
    ap('       File list: ' + ', '.join(gfiles))
    ap('  Note 4: inference convention (r2) -- the configuration-level regression treats w levels and seed re-runs as independent samples,')
    ap('       overstating the nominal degrees of freedom; p(cluster) clusters by w level (%d clusters);'
       % R.nclus.max())
    ap('       p(w level) regresses after collapsing each w level to 1 point (n=%d, most conservative).'
       % int(R.n_wlvl.max()))
    ap('       The main text uses the w-level convention as primary; configuration-level results serve only as comparison.')

    ap('')
    ap('[Sign-count recheck (response to R2)]')
    for metric in ['rho', 'E2/E1']:
        s = R[R.metric == metric]
        pos = int((s.slope > 0).sum())
        ap('  %-6s %d slopes in total: %d positive, %d negative; of which significant (p<0.05) positive %d, negative %d'
           % (metric, len(s), pos, len(s) - pos,
              int(((s.slope > 0) & (s.p < 0.05)).sum()),
              int(((s.slope < 0) & (s.p < 0.05)).sum())))
        for _, r in s.iterrows():
            ap('      %-22s %-5s %+8.4f  p=%.3g%s'
               % (r.caliber, r.layer, r.slope, r.p,
                  '*' if r.p < 0.05 else ''))

    # Per-level detail
    for lay in ['10cm', '30cm']:
        d = S[S.layer == lay].copy()
        g = d.groupby('w').agg(ncfg=('x', 'size'), rho_w=('rho_w', 'mean'),
                               x=('x', 'mean'), ratio=('ratio', 'mean'),
                               xf=('xf', 'mean'), ratiof=('ratiof', 'mean'))
        ap('')
        ap('  %s by w level (primary convention W; x / E2E1 under the full convention P in parentheses)' % lay)
        for w, r in g.iterrows():
            ap('    w=%-6g n=%d  rho=%.4f  x=%.4f  E2/E1=%.4f   '
               '(P: x=%.4f  E2/E1=%.4f)'
               % (w, int(r.ncfg), r.rho_w, r.x, r.ratio, r.xf, r.ratiof))

    ap('')
    ap('[Strength-direction test in the replication regions (r1)] Output: %s'
       % os.path.join(OUT, args.tag + '_repro.csv'))
    ap('  The replication regions have only three levels, w in {1,3,10}, too few to support a regression; instead a sign-consistency test of delta_x is used.')
    ap('  %-14s %-8s %5s %9s %9s %9s' % ('region', 'layer', 'n levels', 'x(w=1)', 'x(w=10)', 'delta_x'))
    for _, r in RP.iterrows():
        ap('  %-14s %-8s %5d %9.4f %9.4f %+9.4f'
           % (r.region_cn, r.layer, r.n_levels, r.x_w1, r.x_w10, r.dx))
    if len(RP):
        ap('  Fraction of positive delta_x: %d/%d' % (int((RP.dx > 0).sum()), len(RP)))

    ap('=== done ===')
    report(os.path.basename(__file__).replace('.py', ''), _lines)
    print('\n'.join(_lines))


if __name__ == '__main__':
    main()
