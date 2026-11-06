# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""04: multi-region terrain covariates (Copernicus GLO-30 -> 0.001deg -> slope/aspect/TPI -> 0.1deg)

by region bbox select tiles from the unified tile directory (no longer glob all), algorithm follows 13_build_terrain.py. 
Output: results_data/01_static/multi/{region}/terrain_0p1deg.nc

Usage:python 04_build_terrain_multi.py [--region R0_northchina]
"""
import os, time, argparse
import numpy as np, xarray as xr, rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

OUT = REPRO + '/_Upstream_Intermediate_Products/out'
DEM_DIR = RAW + '/dem/DEMtile data'
BUF = 0.4
RES = 0.001

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


def grid_for(lon0, lat0, lon1, lat1, buf=BUF):
    def axis(a, b):
        lo, hi = a - buf, b + buf
        c0 = np.ceil((lo + 0.05) * 10) / 10.0
        c1 = np.floor((hi - 0.05) * 10) / 10.0
        n = int(round((c1 - c0) / 0.1)) + 1
        return np.round(c0 + 0.1 * np.arange(n), 4)
    return axis(lat0, lat1), axis(lon0, lon1)


def pick_tiles(LATg, LONg):
    """Select the 1x1 deg tiles required by the grid 1°×1° tile selection"""
    need = set()
    for la in LATg:
        for lo in LONg:
            need.add((int(np.floor(la)), int(np.floor(lo))))
    files, miss = [], []
    for nla, nlo in sorted(need):
        fn = 'Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM.tif' % (nla, nlo)
        p = os.path.join(DEM_DIR, fn)
        (files if os.path.exists(p) else miss).append(p if os.path.exists(p) else fn)
    return files, miss


def build(region):
    lon0, lat0, lon1, lat1 = REGIONS[region]
    LATg, LONg = grid_for(lon0, lat0, lon1, lat1)
    outdir = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/01_static/multi', region)
    os.makedirs(outdir, exist_ok=True)
    P('=== %s ===  Grid %dx%d' % (region, len(LATg), len(LONg)))

    tiles, miss = pick_tiles(LATg, LONg)
    P('  tiles needed %d  missing %d %s' % (len(tiles) + len(miss), len(miss), miss[:5]))
    if miss:
        P('  !! missing tiles, abort this region'); return

    WEST, EAST = LONg[0] - 0.05, LONg[-1] + 0.05
    SOUTH, NORTH = LATg[0] - 0.05, LATg[-1] + 0.05
    W = int(round((EAST - WEST) / RES))
    H = int(round((NORTH - SOUTH) / RES))
    P('  merge bounds W/E/S/N = %.3f/%.3f/%.3f/%.3f  intermediate grid %dx%d' % (WEST, EAST, SOUTH, NORTH, W, H))

    t0 = time.time()
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(srcs, bounds=(WEST, SOUTH, EAST, NORTH),
                              res=(RES, RES), resampling=Resampling.average, nodata=-32768)
    for s in srcs: s.close()
    dem = mosaic[0].astype('float32')
    dem[dem <= -32000] = np.nan
    P('  Mosaicked %s  %.0f s  valid %.1f%%  min=%.0f p50=%.0f max=%.0f' % (
        dem.shape, time.time() - t0, 100 * np.isfinite(dem).mean(),
        np.nanmin(dem), np.nanpercentile(dem, 50), np.nanmax(dem)))
    H, W = dem.shape

    # slope / aspect (central differences)
    lat2d = np.arange(H)[:, None] * (-RES) + NORTH - RES / 2
    dy_m = RES * 110574.0
    dx_m = RES * 111320.0 * np.cos(np.deg2rad(lat2d))
    dzdy, dzdx = np.gradient(dem, axis=(0, 1))
    dzdx = dzdx / dx_m
    dzdy = dzdy / dy_m
    slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    aspect = np.arctan2(dzdy, -dzdx)
    northness = np.cos(aspect)
    eastness = np.sin(aspect)

    def agg(a, how='mean'):
        n_y, n_x = a.shape[0] // 100, a.shape[1] // 100
        b = a[:n_y * 100, :n_x * 100].reshape(n_y, 100, n_x, 100)
        with np.errstate(invalid='ignore'):
            return np.nanmean(b, axis=(1, 3)) if how == 'mean' else np.nanstd(b, axis=(1, 3))

    dem01 = agg(dem)
    tpi01 = dem01 - uniform_filter(dem01, size=5, mode='nearest')

    out = xr.Dataset(
        data_vars=dict(
            dem_mean=(('lat', 'lon'), dem01.astype('float32')),
            dem_std=(('lat', 'lon'), agg(dem, 'std').astype('float32')),
            slope=(('lat', 'lon'), agg(slope).astype('float32')),
            northness=(('lat', 'lon'), agg(northness).astype('float32')),
            eastness=(('lat', 'lon'), agg(eastness).astype('float32')),
            tpi=(('lat', 'lon'), tpi01.astype('float32')),
        ), coords=dict(lat=LATg, lon=LONg))
    for k, u in dict(dem_mean='m', dem_std='m', slope='degree',
                     northness='-1..1', eastness='-1..1', tpi='m').items():
        out[k].attrs['units'] = u
    out.attrs.update(title='Terrain covariates (%s)' % region,
                     source='Copernicus GLO-30 DEM (AWS Open Data), 30m -> 0.001deg -> 0.1deg',
                     note='slope/aspect from Horn gradient at 0.001deg; TPI on 0.1deg grid, 5x5 window')
    fp = os.path.join(outdir, 'terrain_0p1deg.nc')
    out.to_netcdf(fp)
    for k in out.data_vars:
        a = out[k].values
        P('  %-10s min=%.3f p50=%.3f max=%.3f nan=%d' % (
            k, np.nanmin(a), np.nanpercentile(a, 50), np.nanmax(a), int(np.isnan(a).sum())))
    P('==> %s Wrote %s (%.2f MB) elapsed %.0f s' % (region, fp, os.path.getsize(fp) / 1e6, time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    a = ap.parse_args()
    regions = [a.region] if a.region else list(REGIONS)
    for r in regions:
        build(r)
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_04_terrain_report.txt')
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    print('WROTE', rp)
