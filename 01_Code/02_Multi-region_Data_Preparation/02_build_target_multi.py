# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""02: multi-region SMCI target field (daily 10cm/30cm, 2000-2022)

and 10_build_target.py same algorithm as (0.001 scale factor,-900 missing,2021/22 time-axis reconstruction),
but with the region parameterised, output to results_data/04_ready/multi/{region}/

Usage:
  python 02_build_target_multi.py --region R0_northchina --year 2020   # single-region single-year trial run
  python 02_build_target_multi.py                                       # full run R3+R6
"""
import os, sys, zipfile, time, argparse
import numpy as np, xarray as xr, netCDF4

RAW = RAW
OUT = REPRO + '/_Upstream_Intermediate_Products/out'
BUF = 0.4

# central window 3°×3°: (lon0, lat0, lon1, lat1)
REGIONS = {
    'R1_yangtze':    (114.0, 30.0, 117.0, 33.0),
    'R0_northchina': (114.0, 36.0, 117.0, 39.0),
    'R3_northeast':  (125.0, 44.0, 128.0, 47.0),
    'R2_loess':      (108.0, 35.0, 111.0, 38.0),
    'R4_longzhong':  (103.0, 34.0, 106.0, 37.0),
    'R5_sichuan':    (105.0, 29.0, 108.0, 32.0),
}
DEPTHS = ('10cm', '30cm')
YEARS = list(range(2000, 2023))

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def grid_for(lon0, lat0, lon1, lat1, buf=BUF):
    def axis(a, b):
        lo, hi = a - buf, b + buf
        c0 = np.ceil((lo + 0.05) * 10) / 10.0
        c1 = np.floor((hi - 0.05) * 10) / 10.0
        n = int(round((c1 - c0) / 0.1)) + 1
        return np.round(c0 + 0.1 * np.arange(n), 4)
    return axis(lat0, lat1), axis(lon0, lon1)


def build(region, years, depths=DEPTHS):
    lon0, lat0, lon1, lat1 = REGIONS[region]
    LATg, LONg = grid_for(lon0, lat0, lon1, lat1)
    outdir = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi', region)
    os.makedirs(outdir, exist_ok=True)
    P('=== %s ===' % region)
    P('  Grid LAT %.1f..%.1f (%d)  LON %.1f..%.1f (%d)  -> %d grid cells' % (
        LATg[0], LATg[-1], len(LATg), LONg[0], LONg[-1], len(LONg), len(LATg) * len(LONg)))

    single = (len(years) == 1)
    for dep in depths:
        t0 = time.time(); das = []
        for yr in years:
            z = os.path.join(RAW, 'data5', 'SMCI_9km', dep, 'SMCI_9km_%d_%s.zip' % (yr, dep))
            if not os.path.exists(z):
                P('  !! missing %s' % z); continue
            with zipfile.ZipFile(z) as zf:
                nm = [n for n in zf.namelist() if n.endswith('.nc')][0]
                buf = zf.open(nm).read()
            ncd = netCDF4.Dataset('mem', memory=buf)
            ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ncd))
            v = list(ds.data_vars)[0]
            la, lo = ds['lat'].values, ds['lon'].values
            la_sl = slice(LATg[0] - 0.05, LATg[-1] + 0.05) if la[-1] > la[0] \
                else slice(LATg[-1] + 0.05, LATg[0] - 0.05)
            lo_sl = slice(LONg[0] - 0.05, LONg[-1] + 0.05)
            sub = ds[v].sel(lat=la_sl, lon=lo_sl)
            sub = sub.sel(lat=LATg, lon=LONg, method='nearest').load()
            a = sub.values.astype('float32')
            # ds.close() also closes the underlying netCDF4 Dataset, ncd.close() must not be called again ncd.close() (would raise Not a valid ID)
            ds.close(); ncd = None
            a = np.where(a <= -900, np.nan, a) / 1000.0     # 0.001 m3/m3 -> m3/m3
            # Source defect: 2021/2022 files start time at day1, shifted by one day overall -> rebuilt from 1/1 of each year
            n = a.shape[0]
            times = np.arange('%d-01-01' % yr, '%d-01-01' % (yr + 1),
                              dtype='datetime64[D]').astype('datetime64[ns]')[:n]
            das.append(xr.DataArray(a, dims=('time', 'lat', 'lon'),
                                    coords=dict(time=times, lat=LATg, lon=LONg)))
            P('  %s %d ok shape=%s mean=%.4f nan%%=%.2f' % (
                dep, yr, a.shape, np.nanmean(a), 100 * np.isnan(a).mean()))
            del buf

        full = xr.concat(das, dim='time').sortby('time')
        da = full.rename('SM_%s' % dep)
        da.attrs.update(units='m3/m3',
                        long_name='volumetric soil water content at %s' % dep,
                        source='SMCI1.0 (TPDC) 0.1deg daily')
        dso = da.to_dataset()
        tag = '' if not single else '_test%d' % years[0]
        dso.attrs.update(title='SMCI soil moisture %s (%s)' % (dep, region),
                         grid='0.1deg %dx%d' % (len(LATg), len(LONg)),
                         center_window='lon %.1f-%.1f lat %.1f-%.1f' % (lon0, lon1, lat0, lat1))
        fp = os.path.join(outdir, 'SMCI_%s_daily_2000_2022%s.nc' % (dep, tag))
        dso.to_netcdf(fp, encoding={'SM_%s' % dep: dict(zlib=True, complevel=4, dtype='float32')})
        P('==> %s %s Wrote %s (%.1f MB) %d days  elapsed %.0f s' % (
            region, dep, fp, os.path.getsize(fp) / 1e6, da.sizes['time'], time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    ap.add_argument('--year', type=int, default=None)
    a = ap.parse_args()
    regions = [a.region] if a.region else list(REGIONS)
    years = [a.year] if a.year else YEARS
    for r in regions:
        build(r, years)
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_02_target_report.txt')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print('WROTE', rp)
