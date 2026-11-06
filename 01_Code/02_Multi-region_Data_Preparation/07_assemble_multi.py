# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""07: multi-region data assembly  --  same convention as the single-region 30_assemble.py, producing a modelling-ready cube

Input (results_data/04_ready/multi/{region}/): 
    SMCI_10cm_daily_2000_2022.nc / SMCI_30cm_daily_2000_2022.nc   target fields (02)
    soil_props_0p1deg.nc                                          soil properties (03)
    terrain_0p1deg.nc                                            terrain (04)
    cmfd_meteo_daily.nc                                          CMFD 11 variables (05)
    chm_precip_daily.nc                                          CHM_PRE daily precipitation (06)

Output (same directory):
    cube_target_SM.nc        SM_10cm, SM_30cm
    cube_precip_dynamic.nc   precip_daily + sum01/03/07/15/30/60d + API_k85/95
    static_covariates.nc     terrain + soil
    and consistency checks (grid/time axis/missing values) and QC figure

Note: the single region uses ganzhou_frac mask-selected 68  grid cells; the multi-region window is a regular rectangle,
no administrative mask; all n×n all grid cells are modelled (invalid cells are reported explicitly).
"""
import os, sys, glob, time, argparse
import numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
except Exception:
    pass

OUT = REPRO + '/_Upstream_Intermediate_Products/out'
MULTI = REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi'
STATIC_MULTI = REPRO + '/_Upstream_Intermediate_Products/out/01_static/multi'
T0, T1 = '2000-01-01', '2022-12-31'

regions = sorted([d for d in os.listdir(MULTI) if os.path.isdir(os.path.join(MULTI, d))])

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def find_input(region, fname):
    """03/04 output to 01_static/multi, the rest in 04_ready/multi  --  compatible with lookups in both places"""
    for base in (MULTI, STATIC_MULTI):
        p = os.path.join(base, region, fname)
        if os.path.exists(p):
            return p
    return None


def build_precip_cube(pr_da):
    """CHM_PRE daily precipitation -> multi-window accumulation + API, same convention as the single-region cube_precip_dynamic.nc"""
    p = np.asarray(pr_da.values, dtype='float32')
    c = np.nancumsum(p, axis=0)
    out = {}
    for w in (1, 3, 7, 15, 30, 60):
        s = np.empty_like(p); s[:] = np.nan
        s[w - 1:] = c[w - 1:] - np.concatenate([np.zeros((1,) + p.shape[1:], p.dtype), c[:-w]], axis=0)
        out['precip_sum%02dd' % w] = (('time', 'lat', 'lon'), s.astype('float32'))
    for k in (0.85, 0.95):
        api = np.empty_like(p); acc = np.zeros(p.shape[1:], 'float32')
        for i in range(p.shape[0]):
            acc = k * acc + p[i]
            api[i] = acc
        out['API_k%02d' % int(k * 100)] = (('time', 'lat', 'lon'), api.astype('float32'))
    dyn = xr.Dataset(out, coords=dict(time=pr_da.time.values, lat=pr_da.lat.values, lon=pr_da.lon.values))
    dyn['precip_daily'] = (('time', 'lat', 'lon'), p)
    for v in dyn.data_vars:
        dyn[v].attrs['units'] = 'mm' if ('sum' in v or 'API' in v) else 'mm/day'
        dyn[v].attrs['source'] = 'CHM_PRE V2 (TPDC)'
    return dyn


def assemble(region):
    d = os.path.join(MULTI, region)
    P('')
    P('================ %s ================' % region)
    t0 = time.time()

    # ---------- Target ----------
    f10 = os.path.join(d, 'SMCI_10cm_daily_2000_2022.nc')
    f30 = os.path.join(d, 'SMCI_30cm_daily_2000_2022.nc')
    with xr.open_dataset(f10) as ds:
        t10 = ds['SM_10cm']
        LATg, LONg = t10.lat.values, t10.lon.values
        nT = t10.sizes['time']
        t10 = t10.load()
    with xr.open_dataset(f30) as ds:
        t30 = ds['SM_30cm']
        if t30.sizes['time'] != nT:
            P('  !! 30cm time-axis lengths differ, align to 10cm aligned')
        t30 = t30.load().assign_coords(time=t10.time.values)
    target = xr.Dataset({'SM_10cm': t10, 'SM_30cm': t30})
    fp = os.path.join(d, 'cube_target_SM.nc')
    target.to_netcdf(fp, encoding={k: dict(zlib=True, complevel=4, dtype='float32') for k in target.data_vars})
    P('[Target] %s  Grid %dx%d  time %d days  %.1f MB' % (
        os.path.basename(fp), len(LATg), len(LONg), nT, os.path.getsize(fp) / 1e6))
    P('        SM_10cm nan%%=%.3f  SM_30cm nan%%=%.3f' % (
        100 * float(np.isnan(t10.values).mean()), 100 * float(np.isnan(t30.values).mean())))

    # ---------- precipitation ----------
    with xr.open_dataset(os.path.join(d, 'chm_precip_daily.nc')) as ds:
        pr = ds['precip'].sel(time=slice(T0, T1)).load()
    assert pr.sizes['time'] == nT, (pr.sizes['time'], nT)
    dyn = build_precip_cube(pr)
    fp = os.path.join(d, 'cube_precip_dynamic.nc')
    dyn.to_netcdf(fp, encoding={v: dict(zlib=True, complevel=4, dtype='float32') for v in dyn.data_vars})
    P('[precipitation] %d  variables  %.1f MB  nan%%=%.3f' % (
        len(dyn.data_vars), os.path.getsize(fp) / 1e6, 100 * float(np.isnan(pr.values).mean())))

    # ---------- static ----------
    fter = find_input(region, 'terrain_0p1deg.nc')
    fsoi = find_input(region, 'soil_props_0p1deg.nc')
    if fter is None or fsoi is None:
        raise FileNotFoundError('missing terrain/soil inputs: %s / %s' % (fter, fsoi))
    with xr.open_dataset(fter) as ds:
        ter = ds.load()
    with xr.open_dataset(fsoi) as ds:
        soi = ds.load()
    # Unify to the target grid (if sub-metre differences exist)
    soi = soi.interp(lat=LATg, lon=LONg, method='nearest') if not np.array_equal(soi.lat.values, LATg) else soi
    ter = ter.interp(lat=LATg, lon=LONg, method='nearest') if not np.array_equal(ter.lat.values, LATg) else ter
    static = xr.merge([ter, soi])
    fp = os.path.join(d, 'static_covariates.nc')
    static.to_netcdf(fp)
    P('[static] terrain %d + soil %d = %d  variables' % (len(ter.data_vars), len(soi.data_vars), len(static.data_vars)))
    bad = [v for v in static.data_vars if not np.isfinite(static[v].values).all()]
    if bad:
        for v in bad:
            a = static[v].values
            P('       !! %s contains invalid values nan%%=%.2f' % (v, 100 * float(np.isnan(a).mean())))
    else:
        P('       All static variables finite: OK')

    # ---------- CMFD consistency ----------
    fcm = os.path.join(d, 'cmfd_meteo_daily.nc')
    if os.path.exists(fcm):
        with xr.open_dataset(fcm) as cm:
            need = ['t2m', 'rhum', 'wspeed', 'spressure', 'swrad', 'lwrad', 'netrad', 'vpd', 'pet', 'precip']
            miss = [v for v in need if v not in cm.data_vars]
            nTcm = cm.sizes['time']
            P('[CMFD] variables %d  variables, missing among those required for modelling: %s ; time axis %d days' % (len(cm.data_vars), miss or 'none', nTcm))
            if not np.array_equal(cm.lat.values, LATg) or not np.array_equal(cm.lon.values, LONg):
                P('       !! CMFD grid does not match the target')
            nan_pct = {v: 100 * float(np.isnan(cm[v].values).mean()) for v in need if v in cm.data_vars}
            P('       key variables nan%%: ' + '  '.join('%s=%.3f' % (k, v) for k, v in nan_pct.items()))

    # ---------- QC figure ----------
    fig_dir = os.path.join(OUT, '05_qc_figures', 'multi_' + region)
    os.makedirs(fig_dir, exist_ok=True)
    sm_mean = t10.mean('time').values
    pp = pr.mean('time').values * 365.25

    def qmap(a, title, fn, cmap='viridis', unit=''):
        plt.figure(figsize=(5.0, 4.2))
        im = plt.pcolormesh(LONg, LATg, np.where(np.isfinite(a), a, np.nan), cmap=cmap, shading='auto')
        plt.colorbar(im, label=unit)
        plt.xlabel('Lon'); plt.ylabel('Lat'); plt.title('%s  %s' % (region, title))
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, fn), dpi=130); plt.close()

    qmap(sm_mean, 'Mean SM 0-10cm', '01_mean_SM10cm.png', 'YlGnBu', 'm3/m3')
    if 'dem_mean' in static.data_vars:
        qmap(static['dem_mean'].values, 'Elevation', '02_elevation.png', 'terrain', 'm')
    if 'clay_0_5cm' in static.data_vars:
        qmap(static['clay_0_5cm'].values, 'Clay 0-5cm', '03_clay.png', 'YlOrRd', '%')
    qmap(pp, 'Annual precipitation', '04_annual_precip.png', 'Blues', 'mm/yr')

    tt = pd.to_datetime(np.asarray(t10.time.values))
    sm_m = [float(np.nanmean(t10.values[tt.month == m])) for m in range(1, 13)]
    pr_m = [float(np.nansum(pr.values[tt.month == m])) / ((tt.year.max() - tt.year.min() + 1)) for m in range(1, 13)]
    tmean = None
    if os.path.exists(fcm):
        with xr.open_dataset(fcm) as cm:
            tmean = np.asarray(cm['t2m'].values, dtype='float32')
    fig, ax = plt.subplots(1, 3 if tmean is not None else 2, figsize=(12.5, 3.6))
    ax[0].plot(range(1, 13), sm_m, 'o-', color='#185FA5'); ax[0].set_title('SM 0-10cm monthly')
    ax[0].set_xlabel('Month'); ax[0].set_ylabel('m3/m3'); ax[0].grid(alpha=.3)
    ax[1].bar(range(1, 13), pr_m, color='#0F6E56'); ax[1].set_title('Monthly precipitation')
    ax[1].set_xlabel('Month'); ax[1].set_ylabel('mm/month'); ax[1].grid(alpha=.3)
    if tmean is not None:
        tm = [float(np.nanmean(tmean[tt.month == m])) for m in range(1, 13)]
        ax[2].plot(range(1, 13), tm, 'o-', color='#993C1D'); ax[2].set_title('Monthly T2m')
        ax[2].set_xlabel('Month'); ax[2].set_ylabel('degC'); ax[2].grid(alpha=.3)
    plt.suptitle(region)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '05_seasonal_cycle.png'), dpi=130); plt.close()

    P('[QC figure] %s' % fig_dir)
    P('elapsed %.1f s' % (time.time() - t0))
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=None)
    a = ap.parse_args()
    regs = [a.region] if a.region else regions
    for r in regs:
        try:
            assemble(r)
        except Exception as e:
            import traceback
            P('!! %s failed: %r' % (r, e))
            LOG.append(traceback.format_exc())
    rep = LOG + '/_07_assemble_report.txt'
    open(rep, 'w', encoding='utf-8').write('\n'.join(LOG))
    print('\n[written] %s' % rep)
