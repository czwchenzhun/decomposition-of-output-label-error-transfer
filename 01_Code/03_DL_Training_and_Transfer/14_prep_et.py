# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""14: PSP evapotranspiration dataset (data10): extract regional daily ET components, aligned to the mult grid

Data source: National Tibetan Plateau Data Center PSP model (phenology-snow Penman), HCD01 forcing
  Raw dataset/data10/daily/{ET,PET,Tr,Eb,Ei,fsc}.zip  yearly nc, 0.1°, units mm/day
  Grid:lat 54.95..15.05 (descending),lon 70.05..139.95 (ascending), cell centres .X5
  Our grid: cell centres .X0 (e.g. 35.70), exactly at the centre of PSP the centre of four PSP grid cells -> 2x2 arithmetic mean of 2x2 cells is exactly bilinear interpolation

Usage (geoenv): 
  python 14_prep_et.py --npz out/_bench/dl_R0_northchina_nc64_20160101_20191231_lag.npz

Output:out/_bench/et_{region}_{y0}_{y1}.npz
  Fields ET/PET/Tr/Eb/Ei/ETsoil(=ET-Ei, the part taken from soil)  (T,H,W) float32, units mm/day
"""
import os
import sys
import zipfile
import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, report

LOG = []


def P(s):
    print(s)
    LOG.append(str(s))


SRC = RAW + '/data10/daily'
BENCH = BENCH
VARS = ['ET', 'PET', 'Tr', 'Eb', 'Ei', 'fsc']


def ps_open(var, year):
    """From zip read a given year nc directly, return Dataset (not written to disk)"""
    zp = os.path.join(SRC, '%s.zip' % var)
    nm = '%s_%d_daily.nc' % (var, year)
    with zipfile.ZipFile(zp) as z:
        if nm not in z.namelist():
            return None
        return nc.Dataset('mem', memory=z.read(nm))


def nearest_idx(ax, target):
    """find the index closest to target in the coordinate axis target index closest to target (avoiding floating-point error)"""
    return int(np.argmin(np.abs(np.asarray(ax) - target)))


def block_mean(ds, var, year, lat_ax, lon_ax, gla, glo):
    """Split PSP data for one year 2x2 averaged 2x2 onto our grid, returning (nday,H,W) float32"""
    # PSP grid cells one ring outside the target grid PSP grid cells
    i_hi = [nearest_idx(lat_ax, la + 0.05) for la in gla]
    i_lo = [nearest_idx(lat_ax, la - 0.05) for la in gla]
    j_lo = [nearest_idx(lon_ax, lo - 0.05) for lo in glo]
    j_hi = [nearest_idx(lon_ax, lo + 0.05) for lo in glo]
    i0, i1 = min(min(i_hi), min(i_lo)), max(max(i_hi), max(i_lo)) + 1
    j0, j1 = min(min(j_lo), min(j_hi)), max(max(j_lo), max(j_hi)) + 1
    raw = ds[var][:, i0:i1, j0:j1]                  # (nday, h, w)
    a = np.asarray(raw, dtype='float64')
    if np.ma.isMaskedArray(raw):
        a = np.asarray(raw.filled(np.nan), dtype='float64')
    a[a < -1e-3] = np.nan                            # -9999*scale -> missing
    sl = np.asarray(lat_ax[i0:i1])
    sl2 = np.asarray(lon_ax[j0:j1])
    ia_h = np.array([nearest_idx(sl, la + 0.05) for la in gla])
    ia_l = np.array([nearest_idx(sl, la - 0.05) for la in gla])
    ja_l = np.array([nearest_idx(sl2, lo - 0.05) for lo in glo])
    ja_h = np.array([nearest_idx(sl2, lo + 0.05) for lo in glo])
    if np.array_equal(ia_h, ia_l) or np.array_equal(ja_l, ja_h):
        raise SystemExit('!! grid does not form a 2x2, check region coordinates (%s %d)' % (var, year))
    q = (a[:, ia_h, :][:, :, ja_l] + a[:, ia_h, :][:, :, ja_h] +
         a[:, ia_l, :][:, :, ja_l] + a[:, ia_l, :][:, :, ja_h]) / 4.0
    return q.astype('float32'), float(np.isnan(q).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='11_dl_prep produced by npz, used to obtain the time axis and grid')
    ap.add_argument('--region', default=None)
    ap.add_argument('--vars', default='ET,PET,Tr,Eb,Ei')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    npz = a.npz if os.path.isabs(a.npz) else os.path.join(ROOT, a.npz)
    d = np.load(npz, allow_pickle=True)
    tm = d['time'].astype('datetime64[D]')
    region = a.region or str(d['region'])
    P('Source npz: %s' % os.path.basename(npz))
    P('Region %s   time %s ~ %s   total %d days' % (region, tm[0], tm[-1], len(tm)))

    # Target grid: read the region cmfd product coordinates directly to ensure consistency with the training data
    gnc = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi', region, 'cmfd_meteo_daily.nc')
    g = nc.Dataset(gnc)
    gla = np.round(np.asarray(g['lat'][:]), 4)
    glo = np.round(np.asarray(g['lon'][:]), 4)
    g.close()
    P('target grid lat %.2f..%.2f (%d)  lon %.2f..%.2f (%d)' %
      (gla.min(), gla.max(), len(gla), glo.min(), glo.max(), len(glo)))

    yrs = sorted(set(tm.astype('datetime64[Y]').astype(int) + 1970))
    P('years to extract: %s' % yrs)

    wanted = [v for v in a.vars.split(',') if v]
    out = {}
    nan_report = {}
    for var in wanted:
        if not os.path.exists(os.path.join(SRC, '%s.zip' % var)):
            P('!! skipped %s (no zip)' % var)
            continue
        blocks, nanf = [], []
        for y in yrs:
            ds = ps_open(var, y)
            if ds is None:
                P('!! %s missing %d year, using NaN filled with' % (var, y))
                nd = int((tm.astype('datetime64[Y]').astype(int) + 1970 == y).sum())
                blocks.append(np.full((nd, len(gla), len(glo)), np.nan, 'float32'))
                continue
            lat_ax = np.asarray(ds['lat'][:])
            lon_ax = np.asarray(ds['lon'][:])
            q, nf = block_mean(ds, var, y, lat_ax, lon_ax, gla, glo)
            ds.close()
            blocks.append(q)
            nanf.append(nf)
        arr = np.concatenate(blocks, axis=0)
        if arr.shape[0] != len(tm):
            # PSP extracted by whole years, while npz time axis may already have been warmup trimmed (e.g. 2000-03-01 onwards),
            # When lengths differ, take the **date** intersection instead of raising an error.
            full = np.arange(np.datetime64('%d-01-01' % yrs[0]),
                             np.datetime64('%d-01-01' % (yrs[-1] + 1)))
            if arr.shape[0] != len(full):
                raise SystemExit('!! %s time length %d != %d (and the whole-year axis %d days either)'
                                 % (var, arr.shape[0], len(tm), len(full)))
            idx = np.clip(np.searchsorted(full, tm), 0, len(full) - 1)
            if not np.array_equal(full[idx], tm):
                raise SystemExit('!! %s date alignment failed' % var)
            arr = arr[idx]
            if var == wanted[0]:
                P('  [date alignment] whole-year axis %d days -> take by date the npz of %d days (%s ~ %s)'
                  % (len(full), len(tm), tm[0], tm[-1]))
        out[var] = arr
        nan_report[var] = float(np.isnan(arr).mean())
        P('  %-4s extraction complete  p50 %.4f  annual total %7.1f mm/yr  missing %.4f' %
          (var, np.nanpercentile(arr, 50), np.nanmean(arr) * 365.25, nan_report[var]))

    if 'ET' in out and 'Ei' in out:
        out['ETsoil'] = (out['ET'] - out['Ei']).astype('float32')
        P('  ETsoil = ET - Ei (interception removed, the part truly taken from soil) annual total %.1f mm/yr'
          % (np.nanmean(out['ETsoil']) * 365.25))
        if 'PET' in out:
            P('  ET/PET = %.3f    ETsoil/PET = %.3f' %
              (np.nanmean(out['ET']) / np.nanmean(out['PET']),
               np.nanmean(out['ETsoil']) / np.nanmean(out['PET'])))
    if 'Tr' in out and 'ET' in out:
        P('  Tr/ET %.3f   Eb/ET %.3f' % (np.nanmean(out['Tr']) / np.nanmean(out['ET']),
                                         np.nanmean(out['Eb']) / np.nanmean(out['ET'])))

    out['time'] = tm
    out['region'] = np.array(region)
    op = os.path.join(BENCH, 'et_%s_%d_%d%s.npz' % (region, yrs[0], yrs[-1], a.tag))
    np.savez_compressed(op, **out)
    P('WROTE %s' % op)
    report('310_prep_et_%s' % region, LOG)


if __name__ == '__main__':
    main()
