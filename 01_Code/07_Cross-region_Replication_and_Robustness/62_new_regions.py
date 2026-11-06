# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""62: Select candidate windows for "adding 2 new regions"

Constraints (consistent with the four existing regions):
  - 3x3 degrees, integer longitude/latitude grid (lon0, lat0 are integers)
  - No station overlap with the four existing regions (including the 0.4-degree outward extension)
  - As many stations as possible (>=30), and climatically distinct from the four existing regions
Output top 20 candidates: station count, extent, distance to the nearest existing region
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report  # noqa

STN = os.path.join(STATION, 'sm_stninfor.txt')
OD = DATA

df = pd.read_csv(STN, sep=r'\s+', header=None, names=['id', 'lat', 'lon', 'elev'])

EXIST = [
    ('R2', 'middle--lower Yangtze', 114, 117, 30, 33),
    ('R3', 'North China Plain', 114, 117, 36, 39),
    ('R4', 'Northeast China black soil', 125, 128, 44, 47),
    ('R5', 'Loess Plateau', 108, 111, 35, 38),
]
PAD = 0.4
used = set()
for k, nm, lo0, lo1, la0, la1 in EXIST:
    m = ((df.lon >= lo0 - PAD) & (df.lon <= lo1 + PAD)
         & (df.lat >= la0 - PAD) & (df.lat <= la1 + PAD))
    used |= set(df.loc[m, 'id'])
LOG = []
LOG.append('=' * 76)
LOG.append('62 Candidate windows for new regions (3x3-degree integer grid, non-overlapping with the four existing regions + outward extension)')
LOG.append('=' * 76)
LOG.append('Stations occupied by the four existing regions (incl. outward extension): %d stations' % len(used))

rows = []
for la0 in range(22, 50):
    for lo0 in range(75, 133):
        la1, lo1 = la0 + 3, lo0 + 3
        m = ((df.lon >= lo0) & (df.lon <= lo1)
             & (df.lat >= la0) & (df.lat <= la1))
        ids = set(df.loc[m, 'id'])
        if len(ids) < 20:
            continue
        ov = len(ids & used)
        # Distance to the centre of the nearest existing region (degrees)
        dmin = min(abs((la0 + la1) / 2 - (a4 + a5) / 2)
                   + abs((lo0 + lo1) / 2 - (b4 + b5) / 2)
                   for _, _, b4, b5, a4, a5 in EXIST)
        rows.append(dict(lat0=la0, lon0=lo0, n=len(ids), overlap=ov,
                         dmin=round(dmin, 1)))
R = pd.DataFrame(rows).sort_values('n', ascending=False)
R.to_csv(os.path.join(OD, 'new_region_candidates.csv'), index=False,
         encoding='utf-8-sig')

LOG.append('')
LOG.append('%d candidate windows (station count >=20); top 20 sorted by station count' % len(R))
LOG.append('  window extent            station count  overlap with existing regions  distance to nearest existing region (degrees)')
for _, r in R.head(20).iterrows():
    LOG.append('  lat %2d-%2d lon %3d-%3d  %4d      %3d          %5.1f'
               % (r.lat0, r.lat0 + 3, r.lon0, r.lon0 + 3, r.n, r.overlap, r.dmin))

LOG.append('')
LOG.append('[Top 10 with full non-overlap (overlap=0)]')
R0 = R[R.overlap == 0].head(10)
for _, r in R0.iterrows():
    LOG.append('  lat %2d-%2d lon %3d-%3d  %4d stations   distance to nearest region %5.1f degrees'
               % (r.lat0, r.lat0 + 3, r.lon0, r.lon0 + 3, r.n, r.dmin))

# Pairwise combinations: the two new regions also do not overlap each other, and each has as many stations as possible
LOG.append('')
LOG.append('[Pairwise-combination top 10] (the two new regions do not overlap each other, nor the four existing regions)')
best = []
sub = R[R.overlap == 0].reset_index(drop=True)
for i in range(len(sub)):
    for j in range(i + 1, len(sub)):
        a, b = sub.iloc[i], sub.iloc[j]
        if abs(a.lat0 - b.lat0) < 3 and abs(a.lon0 - b.lon0) < 3:
            continue
        ma = ((df.lon >= a.lon0) & (df.lon <= a.lon0 + 3)
              & (df.lat >= a.lat0) & (df.lat <= a.lat0 + 3))
        mb = ((df.lon >= b.lon0) & (df.lon <= b.lon0 + 3)
              & (df.lat >= b.lat0) & (df.lat <= b.lat0 + 3))
        ov2 = len(set(df.loc[ma, 'id']) & set(df.loc[mb, 'id']))
        if ov2:
            continue
        best.append((a.n + b.n, a.lat0, a.lon0, a.n, b.lat0, b.lon0, b.n))
best.sort(reverse=True)
for t in best[:10]:
    LOG.append('  Total %3d stations = A(lat %2d-%2d lon %3d-%3d) %2d stations '
               '+ B(lat %2d-%2d lon %3d-%3d) %2d stations'
               % (t[0], t[1], t[1] + 3, t[2], t[2] + 3, t[3],
                  t[4], t[4] + 3, t[5], t[5] + 3, t[6]))

LOG.append('')
LOG.append('Reference: the four existing regions have 139 stations; adding %d stations in total -> %d stations (%.1f%% of 732)'
           % (best[0][0], 139 + best[0][0], 100.0 * (139 + best[0][0]) / 732))
report('62_new_regions', LOG)
