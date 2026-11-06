# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""05: multi-region CMFD 2.0 meteorological forcing (8 variables -> 11 derived variables, daily 2000-2022)

Algorithm follows 60_build_cmfd_meteo.py (cell centres .X5 -> .X0 bilinear interpolation, unit conversion,
Tetens VPD, Priestley-Taylor PET), with the region parameterised.
Output: results_data/04_ready/multi/{region}/cmfd_meteo_daily.nc

Usage:
  python 05_build_meteo_multi.py --region R0_northchina --year 2020
  python 05_build_meteo_multi.py
"""
import os, time, argparse
import numpy as np, pandas as pd, xarray as xr

FORCING = RAW + '/data7/Data_forcing_01dy_010deg'
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
VARS = ['temp', 'pres', 'shum', 'rhum', 'wind', 'srad', 'lrad', 'prec']
UNITS = {
    't2m': ('2m air temperature', 'degC'),
    'spressure': ('surface air pressure', 'hPa'),
    'shum': ('near-surface specific humidity', 'kg/kg'),
    'rhum': ('near-surface relative humidity', '%'),
    'wspeed': ('10m wind speed', 'm/s'),
    'swrad': ('downward shortwave radiation', 'MJ/m2/day'),
    'lwrad': ('downward longwave radiation', 'MJ/m2/day'),
    'precip': ('precipitation', 'mm/day'),
    'vpd': ('vapor pressure deficit', 'kPa'),
    'netrad': ('net radiation (albedo 0.18, Ts~Ta)', 'MJ/m2/day'),
    'pet': ('potential ET (Priestley-Taylor, alpha=1.26)', 'mm/day'),
}

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


def fname(v, y):
    return os.path.join(FORCING, '%s_CMFD_V0200_B-01_01dy_010deg_%04d01-%04d12.nc' % (v, y, y))


def load_year(v, y, LATg, LONg):
    with xr.open_dataset(fname(v, y)) as ds:
        vv = list(ds.data_vars)[0]
        sub = ds[vv].sel(lat=slice(LATg[0] - 0.2, LATg[-1] + 0.2),
                         lon=slice(LONg[0] - 0.2, LONg[-1] + 0.2)).load()
    da = sub.interp(lat=LATg, lon=LONg, method='linear')
    t = pd.to_datetime(np.asarray(da['time'].values)).normalize()
    return da.assign_coords(time=t)


def build(region, years):
    lon0, lat0, lon1, lat1 = REGIONS[region]
    LATg, LONg = grid_for(lon0, lat0, lon1, lat1)
    outdir = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi', region)
    os.makedirs(outdir, exist_ok=True)
    P('=== %s ===  Grid %dx%d  Years %d-%d' % (region, len(LATg), len(LONg), years[0], years[-1]))
    target_time = pd.date_range('%d-01-01' % years[0], '%d-12-31' % years[-1], freq='D')
    t_start = time.time()

    raw = {}
    for v in VARS:
        t0 = time.time(); parts = []
        for y in years:
            parts.append(load_year(v, y, LATg, LONg))
        da = xr.concat(parts, dim='time').sortby('time').reindex(time=target_time).rename(v)
        raw[v] = da
        P('  %-5s shape=%s  %.0f s' % (v, dict(da.sizes), time.time() - t0))

    tk = raw['temp']
    t2m = tk - 273.15
    sp_hpa = raw['pres'] / 100.0
    swrad = raw['srad'] * 0.0864
    lwrad = raw['lrad'] * 0.0864
    precip = raw['prec'] * 86400.0
    es = 0.6108 * np.exp(17.27 * t2m / (t2m + 237.3))
    ea = es * raw['rhum'] / 100.0
    vpd = (es - ea).clip(min=0)
    rlu = 4.903e-9 * (tk ** 4)
    netrad = 0.82 * swrad + lwrad - rlu
    delta = 4098.0 * es / ((t2m + 237.3) ** 2)
    gamma = 0.000665 * (sp_hpa / 10.0)
    pet = (1.26 * (delta / (delta + gamma)) * netrad / 2.45).clip(min=0)

    out = xr.Dataset({
        't2m': t2m.astype('float32'), 'spressure': sp_hpa.astype('float32'),
        'shum': raw['shum'].astype('float32'), 'rhum': raw['rhum'].astype('float32'),
        'wspeed': raw['wind'].astype('float32'), 'swrad': swrad.astype('float32'),
        'lwrad': lwrad.astype('float32'), 'precip': precip.astype('float32'),
        'vpd': vpd.astype('float32'), 'netrad': netrad.astype('float32'),
        'pet': pet.astype('float32'),
    })
    for k, (ln, u) in UNITS.items():
        out[k].attrs.update(long_name=ln, units=u)
    out.attrs.update(title='CMFD 2.0 daily meteorology (%s)' % region,
                     source='CMFD V0200 Data_forcing_01dy_010deg (TPDC), data7',
                     note='source grid centers at .X5 deg; bilinearly interpolated to .X0 modeling grid')
    tag = '' if len(years) > 1 else '_test%d' % years[0]
    fp = os.path.join(outdir, 'cmfd_meteo_daily%s.nc' % tag)
    enc = {k: dict(zlib=True, complevel=4, dtype='float32') for k in out.data_vars}
    out.to_netcdf(fp, engine='netcdf4', encoding=enc)
    P('==> %s Wrote %s (%.1f MB) elapsed %.0f s' % (region, fp, os.path.getsize(fp) / 1e6, time.time() - t_start))

    P('  -- QC (climatological monthly means)')
    for k in ['t2m', 'precip', 'vpd', 'pet']:
        clim = out[k].groupby('time.month').mean()
        vals = [float(clim.sel(month=m).mean()) for m in range(1, 13)]
        P('     %-7s' % k + ''.join('%8.2f' % x for x in vals))
    for k in out.data_vars:
        a = out[k].values
        P('     %-10s NaN=%d  min=%9.4f p50=%9.4f max=%9.4f' % (
            k, int(np.isnan(a).sum()), np.nanmin(a), np.nanpercentile(a, 50), np.nanmax(a)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    ap.add_argument('--year', type=int, default=None)
    a = ap.parse_args()
    regions = [a.region] if a.region else list(REGIONS)
    years = [a.year] if a.year else list(range(START_Y, END_Y + 1))
    for r in regions:
        build(r, years)
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_05_meteo_report.txt')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print('WROTE', rp)
