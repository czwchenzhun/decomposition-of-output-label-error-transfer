# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""63: Generate station-matched grid-cell tables cells_{REG}.csv (id,iy,ix) for the new regions

The grid is fully consistent with 02-06: 0.1-degree cell centres after a 0.4-degree outward extension of the 3x3-degree central window.
First reverse-engineer and verify the algorithm against the official R2_loess file (should reproduce value by value), then write the new regions.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report  # noqa

STN = os.path.join(STATION, 'sm_stninfor.txt')
CELLDIR = TRUTH + '/station_cells'
BUF = 0.4
# (lon0, lat0, lon1, lat1)
REGIONS = {
    'R2_loess':      (108.0, 35.0, 111.0, 38.0),   # for verification
    'R4_longzhong':  (103.0, 34.0, 106.0, 37.0),
    'R5_sichuan':    (105.0, 29.0, 108.0, 32.0),
}
df = pd.read_csv(STN, sep=r'\s+', header=None, names=['id', 'lat', 'lon', 'elev'])
LOG = []
LOG.append('=' * 72)
LOG.append('63 Station-matched grid-cell tables (verification + new-region generation)')
LOG.append('=' * 72)


def axis(a, b, buf=BUF):
    lo, hi = a - buf, b + buf
    c0 = np.ceil((lo + 0.05) * 10) / 10.0
    c1 = np.floor((hi - 0.05) * 10) / 10.0
    n = int(round((c1 - c0) / 0.1)) + 1
    return np.round(c0 + 0.1 * np.arange(n), 4)


def cells_for(reg):
    lo0, la0, lo1, la1 = REGIONS[reg]
    lats, lons = axis(la0, la1), axis(lo0, lo1)
    m = ((df.lon >= lo0) & (df.lon <= lo1) & (df.lat >= la0) & (df.lat <= la1))
    sub = df[m].copy()
    sub['iy'] = [int(np.argmin(np.abs(lats - x))) for x in sub.lat]
    sub['ix'] = [int(np.argmin(np.abs(lons - x))) for x in sub.lon]
    return lats, lons, sub[['id', 'iy', 'ix']]


# ---------- Verification: R5 should reproduce value by value ----------
lats5, lons5, c5 = cells_for('R2_loess')
old = pd.read_csv(os.path.join(CELLDIR, 'cells_R2_loess.csv'), encoding='utf-8-sig')
mg = old.merge(c5, on='id', how='outer', suffixes=('_old', '_new'))
same = (mg.iy_old == mg.iy_new).all() and (mg.ix_old == mg.ix_new).all()
LOG.append('')
LOG.append('[Verify R2_loess] official %d rows, recomputed %d rows' % (len(old), len(c5)))
LOG.append('  grid lat %s ... %s (%d)' % (lats5[0], lats5[-1], len(lats5)))
LOG.append('  grid lon %s ... %s (%d)' % (lons5[0], lons5[-1], len(lons5)))
LOG.append('  id sets identical: %s' % (set(old.id) == set(c5.id)))
LOG.append('  indices identical value by value: %s' % same)
if not same:
    LOG.append('  !! mismatch details (first 10)')
    for _, r in mg[mg.iy_old != mg.iy_new].head(10).iterrows():
        LOG.append('     %s old=(%s,%s) new=(%s,%s)' % (r.id, r.iy_old, r.ix_old,
                                                        r.iy_new, r.ix_new))
assert same, 'R5 verification failed: algorithm inconsistent with the official convention'

# ---------- Generate new regions ----------
for reg in ('R4_longzhong', 'R5_sichuan'):
    la, lo, c = cells_for(reg)
    out = os.path.join(CELLDIR, 'cells_%s.csv' % reg)
    c.to_csv(out, index=False, encoding='utf-8-sig')
    LOG.append('')
    LOG.append('[%s] %d stations -> %s' % (reg, len(c), out))
    LOG.append('  grid lat %.1f..%.1f (%d)  lon %.1f..%.1f (%d)'
               % (la[0], la[-1], len(la), lo[0], lo[-1], len(lo)))
    LOG.append('  index range iy %d-%d  ix %d-%d' % (c.iy.min(), c.iy.max(),
                                                  c.ix.min(), c.ix.max()))

report('63_station_cells', LOG)
