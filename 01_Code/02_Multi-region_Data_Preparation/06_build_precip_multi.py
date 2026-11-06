# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""06: multi-region CHM_PRE V2 daily precipitation forcing (0.1deg -> target 0.1deg grid, cell centres aligned with .X0)

same convention as the single-region 11_build_precip.py (same data source, same linear interpolation), 
This keeps the dynamic precipitation features comparable across the three regions.CMFD of prec It is used only for cross-checking, not in the main chain.

Output: results_data/04_ready/multi/{region}/chm_precip_daily.nc

Usage:
  python 06_build_precip_multi.py --region R0_northchina --year 2020
  python 06_build_precip_multi.py
"""
import os, time, argparse, glob
import numpy as np, pandas as pd, xarray as xr

RAW8 = RAW + '/data8/daily'
OUT = REPRO + '/_Upstream_Intermediate_Products/out'
BUF = 0.4
START_Y, END_Y = 2000, 2022

REGIONS = {
    'R1_yangtze':    (114.0, 30.0, 117.0, 33.0),
    'R0_northchina': (114.0, 36.0, 117.0, 39.0),
    'R3_northeast':  (125.0, 44.0, 128.0, 47.0),
    'R2_loess':      (108.0, 35.0, 111.0, 38.0),
    'R4_longzhong':  (103.0, 34.0, 106.0, 37.0),
    'R5_sichuan':    (105.0, 29.0, 108.0, 32.0),
}

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def grid_for(lat0, lat1, lon0, lon1, buf=BUF):
    def axis(a, b):
        lo, hi = a - buf, b + buf
        c0 = np.ceil((lo + 0.05) * 10) / 10.0
        c1 = np.floor((hi - 0.05) * 10) / 10.0
        n = int(round((c1 - c0) / 0.1)) + 1
        return np.round(c0 + 0.1 * np.arange(n), 4)
    return axis(lat0, lat1), axis(lon0, lon1)


def build(region, years):
    lon0, lat0, lon1, lat1 = REGIONS[region]
    LATg, LONg = grid_for(lat0, lat1, lon0, lon1)
    outdir = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi', region)
    os.makedirs(outdir, exist_ok=True)
    P('=== %s ===  Grid %dx%d  Years %d-%d' % (region, len(LATg), len(LONg), years[0], years[-1]))

    files = sorted(glob.glob(os.path.join(RAW8, 'CHM_PRE_V2_daily_*.nc')))
    files = [f for f in files if int(os.path.basename(f).split('_')[-1][:4]) in set(years)]
    P('  files to process %d entries' % len(files))
    if not files:
        P('  !! no matching year files'); return None

    das = []
    t0 = time.time()
    for f in files:
        yr = int(os.path.basename(f).split('_')[-1][:4])
        try:
            with xr.open_dataset(f) as ds:
                v = 'prec' if 'prec' in ds.data_vars else list(ds.data_vars)[0]
                da = ds[v]
                sub = da.sel(lat=slice(LATg[0] - 0.2, LATg[-1] + 0.2),
                             lon=slice(LONg[0] - 0.2, LONg[-1] + 0.2)).load()
        except Exception as e:
            P('  !! %d failed to open, skipped: %s' % (yr, str(e)[:100]))
            continue
        sub = sub.interp(lat=LATg, lon=LONg, method='linear').astype('float32').sortby('lat')
        t = pd.to_datetime(np.asarray(sub['time'].values)).normalize()
        sub = sub.assign_coords(time=t)
        das.append(sub)
        a = sub.values
        P('  %d  shape=%s  annual total=%.0f mm  max daily=%.1f mm  nan%%=%.2f  (%.0fs)' % (
            yr, a.shape, float(np.nansum(a)), float(np.nanmax(a)),
            100 * float(np.isnan(a).mean()), time.time() - t0))

    full = xr.concat(das, dim='time').sortby('time').rename('precip')
    allnan = np.isnan(full.values).all(axis=0)
    P('  grid cells missing for the whole year = %d' % int(allnan.sum()))
    if allnan.any():
        ys, xs = np.where(allnan)
        P('    location examples: ' + ', '.join('(%.1f,%.1f)' % (full.lat.values[y], full.lon.values[x])
                                      for y, x in list(zip(ys, xs))[:12]))

    # Cross-check: compare with CMFD prec annual-total comparison (same region)
    cmfd_fp = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi', region, 'cmfd_meteo_daily.nc')
    if os.path.exists(cmfd_fp):
        try:
            with xr.open_dataset(cmfd_fp) as dsc:
                pc = dsc['precip']
            common = sorted(set(int(y) for y in pc.groupby('time.year').groups)
                            & set(int(y) for y in full.groupby('time.year').groups))
            if common:
                cm = np.array([float(np.nanmean(pc.groupby('time.year')[y].sum(skipna=True).values))
                               for y in common])
                ch = np.array([float(np.nanmean(full.groupby('time.year')[y].sum(skipna=True).values))
                               for y in common])
                P('  [Cross-source check] %d-%d  CMFD %.1f mm/yr  vs  CHM_PRE %.1f mm/yr  relative difference %.1f%%' % (
                    common[0], common[-1], float(cm.mean()), float(ch.mean()),
                    100 * float((ch - cm).mean()) / max(float(cm.mean()), 1e-9)))
                for y, a_, b_ in zip(common, cm, ch):
                    P('      %d  CMFD %7.1f  CHM_PRE %7.1f  %+6.1f%%' % (
                        y, a_, b_, 100 * (b_ - a_) / max(a_, 1e-9)))
        except Exception as e:
            P('  [Cross-source check] skipped: %r' % (e,))
    else:
        P('  [Cross-source check] missing cmfd_meteo_daily.nc, skipped')

    full.attrs.update(dict(units='mm/day', long_name='daily precipitation',
                           source='CHM_PRE V2.0 (TPDC), 0.1deg daily, regridded to target grid'))
    ds = full.to_dataset()
    ds.attrs.update(dict(title='CHM_PRE V2 daily precipitation, %s + %.1fdeg buffer' % (region, BUF),
                         grid='0.1deg %dx%d' % (len(LATg), len(LONg)),
                         bbox='lon %.4f-%.4f, lat %.4f-%.4f' % (LONg[0], LONg[-1], LATg[0], LATg[-1])))
    fp = os.path.join(outdir, 'chm_precip_daily.nc')
    ds.to_netcdf(fp, encoding={'precip': dict(zlib=True, complevel=4, dtype='float32')})
    P('==> Wrote %s  %.1f MB  %s ~ %s total %d days' % (
        fp, os.path.getsize(fp) / 1e6, str(full['time'].values[0])[:10],
        str(full['time'].values[-1])[:10], full.sizes['time']))
    return ds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    ap.add_argument('--year', type=int, default=None)
    a = ap.parse_args()
    if a.year:
        years = [a.year]
    else:
        years = list(range(START_Y, END_Y + 1))
    regs = [a.region] if a.region else list(REGIONS)
    for r in regs:
        build(r, years)
    rep = LOG + '/_06_precip_report.txt'
    with open(rep, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    print('\n[written] %s' % rep)
