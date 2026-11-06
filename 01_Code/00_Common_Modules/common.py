# -*- coding: utf-8 -*-
"""Project-wide configuration: paths, study-region bounds, environment (all paths point inside the package)"""
import os
from paths import (PKG, DATA, TRUTH, MULTI, RAW, WORK, REPRO, LOG, BENCH,
                   OUT_COMPAT, STATION, CODE)

ROOT = PKG                 # package root directory, see paths.py
SCRIPTS = LOG              # report() output directory
OUT = OUT_COMPAT           # backward-compatible with the old upstream-chain usage; analysis scripts always use TRUTH/MULTI
SCRATCH = os.path.join(WORK, '_scratch')

CLIP_BOUNDS = (113.913607, 24.486252, 116.638102, 27.143578)
BUF = 0.4                      # window expansion buffer (degrees), used by 02 series and 63 use
BBOX = (CLIP_BOUNDS[0] - BUF, CLIP_BOUNDS[1] - BUF,
        CLIP_BOUNDS[2] + BUF, CLIP_BOUNDS[3] + BUF)
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = BBOX
WKT = 'EPSG:4326'


def clip_box(ds, lon_name='lon', lat_name='lat'):
    """Crop to the expanded study-region bounds xr.Dataset/DataArray, automatically handling ascending/descending latitude"""
    lo, la = ds[lon_name].values, ds[lat_name].values
    lo_sl = slice(LON_MIN, LON_MAX) if lo[-1] > lo[0] else slice(LON_MAX, LON_MIN)
    la_sl = slice(LAT_MIN, LAT_MAX) if la[-1] > la[0] else slice(LAT_MAX, LAT_MIN)
    return ds.sel({lon_name: lo_sl, lat_name: la_sl})

for _d in (WORK, SCRATCH):
    os.makedirs(_d, exist_ok=True)


def report(name, lines):
    """Write the runtime log to 04_Runtime_Logs/, without modifying the archived results data."""
    import io
    p = os.path.join(LOG, name)
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(x) for x in lines))
    print('WROTE', p)
