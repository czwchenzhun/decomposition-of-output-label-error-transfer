# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""03: multi-region CSDLv2 soil properties (data6, 1km -> 0.1deg within-cell mean)

Algorithm follows the validated 12_build_soil_props.py (nearest-bin averaging, scale factor 100), with the region parameterised.
Output: results_data/01_static/multi/{region}/soil_props_0p1deg.nc

Usage:
  python 03_build_soil_multi.py --region R0_northchina
  python 03_build_soil_multi.py
"""
import os, glob, zipfile, re, time, argparse
import numpy as np, xarray as xr, netCDF4

RAW = RAW
OUT = REPRO + '/_Upstream_Intermediate_Products/out'
BUF = 0.4

REGIONS = {
    'R1_yangtze':    (114.0, 30.0, 117.0, 33.0),
    'R0_northchina': (114.0, 36.0, 117.0, 39.0),
    'R3_northeast':  (125.0, 44.0, 128.0, 47.0),
    'R2_loess':      (108.0, 35.0, 111.0, 38.0),
    'R4_longzhong':  (103.0, 34.0, 106.0, 37.0),
    'R5_sichuan':    (105.0, 29.0, 108.0, 32.0),
}
PROPS = ['sand', 'silt', 'clay', 'BD', 'OC', 'pH', 'CEC', 'gravel',
         'Wet_R', 'Wet_G', 'Wet_B', 'Dry_R', 'Dry_G', 'Dry_B']
UNITS = {'sand': '%', 'silt': '%', 'clay': '%', 'BD': 'g/cm3', 'OC': 'g/100g',
         'pH': '-', 'CEC': 'me/100g', 'gravel': '% by volume',
         'Wet_R': '0-255', 'Wet_G': '0-255', 'Wet_B': '0-255',
         'Dry_R': '0-255', 'Dry_G': '0-255', 'Dry_B': '0-255'}

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


def build(region):
    lon0, lat0, lon1, lat1 = REGIONS[region]
    LATg, LONg = grid_for(lon0, lat0, lon1, lat1)
    outdir = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/01_static/multi', region)
    os.makedirs(outdir, exist_ok=True)
    P('=== %s ===  Grid %dx%d' % (region, len(LATg), len(LONg)))
    t0 = time.time()
    out = {}
    for prop in PROPS:
        pdir = os.path.join(RAW, 'data6', prop)
        if not os.path.isdir(pdir):
            P('  !! missing directory %s' % pdir); continue
        zs = sorted(glob.glob(os.path.join(pdir, '*.zip')))
        for z in zs:
            m = re.search(r'_(\d+-\d+cm)_1km', os.path.basename(z))
            dep = m.group(1) if m else 'unknown'
            with zipfile.ZipFile(z) as zf:
                nm = [n for n in zf.namelist() if n.endswith('.nc')][0]
                buf = zf.open(nm).read()
            ncd = netCDF4.Dataset('mem', memory=buf)
            vn = [v for v in ncd.variables if v not in ('latitude', 'longitude')][0]
            lat = np.asarray(ncd.variables['latitude'][:])
            lon = np.asarray(ncd.variables['longitude'][:])
            ilat = np.where((lat >= LATg[0] - 0.02) & (lat <= LATg[-1] + 0.02))[0]
            ilon = np.where((lon >= LONg[0] - 0.02) & (lon <= LONg[-1] + 0.02))[0]
            if len(ilat) == 0 or len(ilon) == 0:
                P('  !! %s %s crop is empty' % (prop, dep)); ncd.close(); continue
            i0, i1 = int(ilat.min()), int(ilat.max())
            j0, j1 = int(ilon.min()), int(ilon.max())
            arr = np.asarray(ncd.variables[vn][i0:i1 + 1, j0:j1 + 1])
            sub_lat = lat[i0:i1 + 1]; sub_lon = lon[j0:j1 + 1]
            ncd.close()
            if sub_lat[-1] < sub_lat[0]:
                arr = arr[::-1, :]; sub_lat = sub_lat[::-1]
            arr = arr.astype('float32')
            arr[arr <= -30000] = np.nan
            iy = np.clip(np.round((sub_lat - LATg[0]) / 0.1).astype(int), 0, len(LATg) - 1)
            ix = np.clip(np.round((sub_lon - LONg[0]) / 0.1).astype(int), 0, len(LONg) - 1)
            flat = arr.ravel(); valid = np.isfinite(flat)
            YY, XX = np.meshgrid(iy, ix, indexing='ij')
            key = (YY.ravel() * len(LONg) + XX.ravel())[valid]
            cnt = np.bincount(key, minlength=len(LATg) * len(LONg)).astype('float64')
            ssum = np.bincount(key, weights=flat[valid].astype('float64'),
                               minlength=len(LATg) * len(LONg))
            with np.errstate(invalid='ignore', divide='ignore'):
                grid = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan).reshape(len(LATg), len(LONg))
            grid = grid / 100.0
            name = '%s_%s' % (prop, dep.replace('-', '_'))
            out[name] = (('lat', 'lon'), grid.astype('float32'))
            P('  %-22s source pixels=%7d  within-cell mean %.3f  empty cells=%d' % (
                name, int(valid.sum()), np.nanmean(grid), int(np.isnan(grid).sum())))
            del buf, arr

    ds = xr.Dataset(out, coords=dict(lat=LATg, lon=LONg))
    for k in ds.data_vars:
        ds[k].attrs['units'] = UNITS.get(k.split('_')[0], '?')
    ds.attrs.update(title='CSDLv2 soil properties 0.1deg (%s)' % region,
                    source='data6 CSDLv2 1km, TPDC',
                    aggregation='area mean of 1km pixels within each 0.1deg cell',
                    scale='raw integer / 100')
    fp = os.path.join(outdir, 'soil_props_0p1deg.nc')
    ds.to_netcdf(fp)
    P('==> %s Wrote %s (%.2f MB, %d variables) elapsed %.0f s' % (
        region, fp, os.path.getsize(fp) / 1e6, len(ds.data_vars), time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    a = ap.parse_args()
    regions = [a.region] if a.region else list(REGIONS)
    for r in regions:
        build(r)
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_03_soil_report.txt')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print('WROTE', rp)
