# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""61: Cost--benefit analysis of station coverage (answering "only 139 of 732 stations were used")

Output:
  1) Station counts and shares of the 4 existing 3x3 windows
  2) Nationwide 3x3 sliding windows (step 0.5 degrees), top 20 by station count
  3) Cumulative coverage curve of greedy non-overlapping window selection (K = 4..20 windows)
  4) Station counts, grid-cell counts and volume estimates after enlarging the four existing regions to 4x4 / 5x5 / 6x6 degrees
  5) Number of windows needed to cover 50%/70% of stations under "adaptive partitioning by station density"
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report, OUT  # noqa

STN = os.path.join(STATION, 'sm_stninfor.txt')
OD = DATA
os.makedirs(OD, exist_ok=True)

df = pd.read_csv(STN, sep=r'\s+', header=None, names=['id', 'lat', 'lon', 'elev'])
N = len(df)
LOG = []
LOG.append('=' * 74)
LOG.append('61 Station coverage analysis (dataset %d stations)' % N)
LOG.append('=' * 74)

# ---------- 1. The four existing regions ----------
REGS = [
    ('R3', 'North China Plain', 114, 117, 36, 39),
    ('R2', 'middle--lower Yangtze', 114, 117, 30, 33),
    ('R5', 'Loess Plateau', 108, 111, 35, 38),
    ('R4', 'Northeast China black soil', 125, 128, 44, 47),
]
LOG.append('')
LOG.append('[I] Station counts of the four existing regions (3x3 degrees)')
tot = 0
used = set()
for k, nm, lo0, lo1, la0, la1 in REGS:
    m = ((df.lon >= lo0) & (df.lon <= lo1) & (df.lat >= la0) & (df.lat <= la1))
    n = int(m.sum())
    tot += n
    used |= set(df.loc[m, 'id'])
    LOG.append('  %s %-8s lon %3d-%3d lat %2d-%2d  -> %3d stations' % (k, nm, lo0, lo1, la0, la1, n))
LOG.append('  Four regions total %d stations, out of %d stations total (%.1f%%)' % (tot, N, 100.0 * tot / N))
LOG.append('  Unused outside windows: %d stations (%.1f%%)' % (N - tot, 100.0 * (N - tot) / N))

# ---------- 2. Nationwide sliding-window ranking ----------
STEP = 0.5
HALF = 1.5
lats = np.arange(22.0, 52.01, STEP)
lons = np.arange(75.0, 135.01, STEP)
recs = []
for la in lats:
    for lo in lons:
        m = ((df.lon >= lo - HALF) & (df.lon <= lo + HALF)
             & (df.lat >= la - HALF) & (df.lat <= la + HALF))
        recs.append((la, lo, int(m.sum())))
R = pd.DataFrame(recs, columns=['lat', 'lon', 'n'])
R.to_csv(os.path.join(OD, 'station_coverage_windows.csv'), index=False,
         encoding='utf-8-sig')
top = R.sort_values('n', ascending=False).head(20)
LOG.append('')
LOG.append('[II] Nationwide 3x3-degree sliding windows (step 0.5 degrees, %d candidates in total), top 20 by station count' % len(R))
LOG.append('  centre lat   centre lon   station count')
for _, r in top.iterrows():
    LOG.append('   %5.1f     %5.1f     %3d' % (r.lat, r.lon, r.n))

# ---------- 3. Greedy non-overlapping cumulative coverage ----------
LOG.append('')
LOG.append('[III] Cumulative coverage curve of greedy window selection (window centres >=3 degrees apart count as non-overlapping)')
Rs = R.sort_values('n', ascending=False).reset_index(drop=True)
sel = []
cov = set()
curve = []
for _, r in Rs.iterrows():
    if any(abs(r.lat - s[0]) < 3.0 and abs(r.lon - s[1]) < 3.0 for s in sel):
        continue
    sel.append((r.lat, r.lon, r.n))
    m = ((df.lon >= r.lon - HALF) & (df.lon <= r.lon + HALF)
         & (df.lat >= r.lat - HALF) & (df.lat <= r.lat + HALF))
    cov |= set(df.loc[m, 'id'])
    curve.append((len(sel), r.lat, r.lon, r.n, len(cov), 100.0 * len(cov) / N))
LOG.append('  K  new window (centre)      stations in window  cumulative coverage  share of %d stations' % N)
for k, la, lo, n, c, p in curve:
    if k <= 20 or k in (25, 30):
        LOG.append('  %2d  (%5.1f,%5.1f)   %5d   %5d    %5.1f%%'
                   % (k, la, lo, n, c, p))
for target in (50, 70, 80):
    hit = [c for c in curve if c[4] >= target / 100.0 * N]
    if hit:
        LOG.append('  Covering %d%% of stations (>=%d stations) needs at least %d 3x3 windows'
                   % (target, int(target / 100.0 * N), hit[0][0]))
    else:
        LOG.append('  Covering %d%% of stations: not reachable within 30 windows (30 windows accumulate %d stations)'
                   % (target, curve[-1][4] if curve else 0))
pd.DataFrame(curve, columns=['K', 'lat', 'lon', 'n_win', 'covered', 'pct']).to_csv(
    os.path.join(OD, 'station_coverage_curve.csv'), index=False, encoding='utf-8-sig')

# ---------- 4. Enlarging the four existing regions ----------
LOG.append('')
LOG.append('[IV] Station counts and volume of enlarged windows for the four existing regions (grid-cell count after 0.4-degree outward extension, 0.1-degree grid)')
LOG.append('  region        3x3  4x4  5x5  6x6   (grid cells 3x3=37x37=1369)')
grand = {3: 0, 4: 0, 5: 0, 6: 0}
for k, nm, lo0, lo1, la0, la1 in REGS:
    row = []
    for w in (3, 4, 5, 6):
        m = ((df.lon >= lo0 - (w - 3) / 2.0) & (df.lon <= lo1 + (w - 3) / 2.0)
             & (df.lat >= la0 - (w - 3) / 2.0) & (df.lat <= la1 + (w - 3) / 2.0))
        row.append(int(m.sum()))
        grand[w] += int(m.sum())
    LOG.append('  %-12s %4d %4d %4d %4d' % (nm, row[0], row[1], row[2], row[3]))
LOG.append('  Four-region total  ' + ''.join('%5d' % grand[w] for w in (3, 4, 5, 6)))
LOG.append('  share of %d stations ' % N
           + ''.join('%4.0f%%' % (100.0 * grand[w] / N) for w in (3, 4, 5, 6)))
LOG.append('  Volume estimate (8341 steps x 78-dim float32, per region):')
for w in (3, 4, 5, 6):
    cell = int((w / 0.1 + 1 + 8) ** 2)   # 0.4-degree outward extension -> +8 cells per side
    gb = cell * 8341 * 78 * 4 / 1024.0 ** 3
    LOG.append('    %dx%d degrees: %5d grid cells  %.2f GB/region (two variants full+nolag about %.1f GB)'
               % (w, w, cell, gb, 2 * gb))

# ---------- 5. Station-count quantiles: how many windows have >=10 stations ----------
LOG.append('')
LOG.append('[V] Station-count quantiles of candidate windows (%d sliding windows)' % len(R))
for thr in (5, 10, 15, 20, 30, 50):
    LOG.append('  Windows with >= %2d stations: %d' % (thr, int((R.n >= thr).sum())))
LOG.append('  Station-count distribution: median %.0f, 90th percentile %.0f, max %d'
           % (R.n.median(), R.n.quantile(0.9), R.n.max()))

report('61_station_coverage', LOG)
