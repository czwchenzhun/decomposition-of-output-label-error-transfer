# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""395a Generate a station-matched grid-cell list for the extended regions (for 11_dl_prep --station_cells for use)

From 15, filter stations_in_windows.csv by region, output only iy,ix in a csv, 
ensure all grid cells containing stations in that region are forcibly included in 11 grid-cell subset (otherwise 21/19/14  stations
most would fall outside the randomly drawn 256 grid cells,L2 chain would idle outright).
"""
import os
import pandas as pd

B = TRUTH
OUT = os.path.join(B, 'station_cells')
os.makedirs(OUT, exist_ok=True)

sta = pd.read_csv(os.path.join(B, 'stations_in_windows.csv'),
                  encoding='utf-8-sig', dtype={'id': str})
for reg, g in sta.groupby('region'):
    if reg == 'OUT':
        continue
    g = g.dropna(subset=['iy', 'ix'])
    sub = g[['id', 'iy', 'ix']].copy()
    sub['iy'] = sub['iy'].astype(int)
    sub['ix'] = sub['ix'].astype(int)
    sub = sub.drop_duplicates(subset=['iy', 'ix'])
    p = os.path.join(OUT, 'cells_%s.csv' % reg)
    sub.to_csv(p, index=False, encoding='utf-8-sig')
    print('%-16s stations %3d  deduplicated grid cells %3d -> %s'
          % (reg, len(g), len(sub), os.path.basename(p)))
