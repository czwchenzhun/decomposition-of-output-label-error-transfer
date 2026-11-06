# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""09: multi-region modelling sample construction  --  same convention as the single-region 80_build_samples.py

78 -dimensional features = static28(terrain6+soil22) + longitude/latitude2 + CMFD 26 + CHM_PRE 8 + time encoding2 + lagSM 12
Output (results_data/04_ready/multi/{region}/samples/):
    X.dat      float32 memmap (T=8341, N=1369, F=78)
    y.npy      float32 (2, T, N)  [SM_10cm, SM_30cm]
    time.npy / cells.npz / meta.json

Unlike the single region: no administrative mask,37x37=1369 all grid cells are modelled.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import xarray as xr

OUT = REPRO + '/_Upstream_Intermediate_Products/out'
MULTI = REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi'
T0, T1 = '2000-01-01', '2022-12-31'

WARMUP = 60
LAGS = [1, 2, 3, 5, 7, 15]

STATIC_VARS = ['dem_mean', 'dem_std', 'slope', 'northness', 'eastness', 'tpi']
SOIL_SETS = [
    ('0_5cm', ['sand', 'silt', 'clay', 'BD', 'OC', 'pH', 'CEC', 'gravel', 'Wet_R']),
    ('15_30cm', ['sand', 'silt', 'clay', 'BD', 'OC', 'pH', 'CEC', 'gravel', 'Wet_R']),
    ('30_60cm', ['sand', 'silt', 'clay', 'BD']),
]
CMFD_VARS = ['t2m', 'rhum', 'wspeed', 'spressure', 'swrad', 'lwrad', 'netrad', 'vpd', 'pet', 'precip']
CHM_VARS = ['precip_daily', 'precip_sum03d', 'precip_sum07d', 'precip_sum15d',
            'precip_sum30d', 'precip_sum60d', 'API_k85', 'API_k95']

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def roll(a, w, how):
    df = pd.DataFrame(np.asarray(a, dtype='float64'))
    out = getattr(df.rolling(w, min_periods=1), how)()
    return out.to_numpy(dtype='float32')


def build(region):
    d = os.path.join(MULTI, region)
    sd = os.path.join(d, 'samples')
    os.makedirs(sd, exist_ok=True)
    t_start = time.time()
    P('')
    P('================ %s ================' % region)

    # 1. Target
    with xr.open_dataset(os.path.join(d, 'cube_target_SM.nc')) as tg:
        time_all = tg['time'].values
        LATg, LONg = tg.lat.values, tg.lon.values
        T_all = tg.sizes['time']
        ny, nx = tg.sizes['lat'], tg.sizes['lon']
        N = ny * nx
        sm10 = np.asarray(tg['SM_10cm'].values, dtype='float32').reshape(T_all, N)
        sm30 = np.asarray(tg['SM_30cm'].values, dtype='float32').reshape(T_all, N)
    P('[Target] %d x %d = %d grid cells  %d days' % (ny, nx, N, T_all))

    # 2. preallocate the feature matrix
    names = []
    feats_spec = []          # (name, (T,N) array or 'repeat2d')

    with xr.open_dataset(os.path.join(d, 'static_covariates.nc')) as st:
        for v in STATIC_VARS:
            a = np.asarray(st[v].values, dtype='float32').reshape(N)
            feats_spec.append((v, np.repeat(a[None, :], T_all, axis=0)))
            names.append(v)
            P('  static %-12s [%.3f, %.3f]' % (v, float(a.min()), float(a.max())))
        for layer, props in SOIL_SETS:
            for p in props:
                v = '%s_%s' % (p, layer)
                a = np.asarray(st[v].values, dtype='float32').reshape(N)
                feats_spec.append((v, np.repeat(a[None, :], T_all, axis=0)))
                names.append(v)
        lat_arr = np.repeat(LATg.reshape(-1, 1), nx, axis=1).reshape(N)
        lon_arr = np.repeat(LONg.reshape(1, -1), ny, axis=0).reshape(N)
        feats_spec.append(('lat', np.repeat(lat_arr[None, :], T_all, axis=0)))
        names.append('lat')
        feats_spec.append(('lon', np.repeat(lon_arr[None, :], T_all, axis=0)))
        names.append('lon')
    n_static = len(names)
    P('[static] %d  features' % n_static)

    # CMFD
    with xr.open_dataset(os.path.join(d, 'cmfd_meteo_daily.nc')) as cm:
        assert cm.sizes['time'] == T_all
        cm_cache = {v: np.asarray(cm[v].values, dtype='float32').reshape(T_all, N) for v in CMFD_VARS}
    for v in CMFD_VARS:
        feats_spec.append(('cmfd_' + v, cm_cache[v])); names.append('cmfd_' + v)
    for w in (7, 30):
        for v in ['t2m', 'rhum', 'vpd', 'pet', 'wspeed', 'swrad']:
            feats_spec.append(('cmfd_%s_mean%dd' % (v, w), roll(cm_cache[v], w, 'mean')))
            names.append('cmfd_%s_mean%dd' % (v, w))
    for w in (7, 30):
        for v in ['precip', 'pet']:
            feats_spec.append(('cmfd_%s_sum%dd' % (v, w), roll(cm_cache[v], w, 'sum')))
            names.append('cmfd_%s_sum%dd' % (v, w))
    P('[CMFD] %d  features' % (len(names) - n_static))

    # CHM_PRE
    with xr.open_dataset(os.path.join(d, 'cube_precip_dynamic.nc')) as pv:
        assert pv.sizes['time'] == T_all
        for v in CHM_VARS:
            feats_spec.append(('chm_' + v, np.asarray(pv[v].values, dtype='float32').reshape(T_all, N)))
            names.append('chm_' + v)
    P('[CHM_PRE] %d  features' % len(CHM_VARS))

    # time encoding
    doy = pd.DatetimeIndex(time_all).dayofyear.values.astype('float64')
    feats_spec.append(('doy_sin', np.repeat(np.sin(2 * np.pi * doy / 365.25)[:, None], N, axis=1).astype('float32')))
    names.append('doy_sin')
    feats_spec.append(('doy_cos', np.repeat(np.cos(2 * np.pi * doy / 365.25)[:, None], N, axis=1).astype('float32')))
    names.append('doy_cos')

    # lag SM
    for dep, src in [('10cm', sm10), ('30cm', sm30)]:
        for k in LAGS:
            a = np.full_like(src, np.nan)
            a[k:] = src[:-k]
            feats_spec.append(('SM%s_lag%02d' % (dep, k), a))
            names.append('SM%s_lag%02d' % (dep, k))
    F = len(names)
    assert F == 78, 'Number of features should be 78, actual %d' % F
    P('[features] Total %d entries' % F)

    # 3. assembly (preallocated, filled feature by feature, avoiding stack double memory)
    X = np.empty((T_all, N, F), dtype='float32')
    t0 = time.time()
    for j, (_, a) in enumerate(feats_spec):
        X[:, :, j] = a
    P('[assembly] X %s  %.2f GB  elapsed %.1f s' % (X.shape, X.nbytes / 1e9, time.time() - t0))
    del feats_spec, cm_cache

    X = X[WARMUP:]
    y = np.stack([sm10[WARMUP:], sm30[WARMUP:]], axis=0).astype('float32')
    tt = time_all[WARMUP:]
    T = X.shape[0]
    P('[warm-up removed] T=%d (%s ~ %s)  y=%s' % (T, str(tt[0])[:10], str(tt[-1])[:10], y.shape))

    bad = ~np.isfinite(X).all(axis=(0, 1))
    if bad.any():
        P('  !! features containing invalid values: %s' % list(np.array(names)[bad]))
    else:
        P('  All feature values finite: OK')
    ybad = ~np.isfinite(y)
    P('  target missing: %d (%.3f%%)' % (int(ybad.sum()), 100 * float(ybad.mean())))

    # 4. write to disk
    xp = os.path.join(sd, 'X.dat')
    X.tofile(xp)
    np.save(os.path.join(sd, 'y.npy'), y)
    np.save(os.path.join(sd, 'time.npy'), tt)
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    np.savez(os.path.join(sd, 'cells.npz'), iy=iy.reshape(-1), ix=ix.reshape(-1),
             lat=np.repeat(LATg.reshape(-1, 1), nx, axis=1).reshape(-1),
             lon=np.repeat(LONg.reshape(1, -1), ny, axis=0).reshape(-1))
    meta = dict(region=region, T=int(T), N=int(N), F=int(F), warmup=WARMUP, lags=LAGS,
                features=names, n_static=n_static,
                target=['SM_10cm', 'SM_30cm'],
                time_start=str(tt[0])[:10], time_end=str(tt[-1])[:10],
                note='X dimension order (time, cell, feature) float32; the multi-region regular window has no administrative mask, all grid cells modelled')
    with open(os.path.join(sd, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    P('[write to disk] X.dat %.2f GB  y.npy  y=%s' % (os.path.getsize(xp) / 1e9, y.shape))
    P('[Done] Total elapsed %.1f s' % (time.time() - t_start))
    return T, N, F


if __name__ == '__main__':
    import sys
    sel = None
    force = False
    for arg in sys.argv[1:]:
        if arg.startswith('--region='):
            sel = arg.split('=', 1)[1].split(',')
        elif arg == '--force':
            force = True
    allr = sorted(d for d in os.listdir(MULTI) if os.path.isdir(os.path.join(MULTI, d)))
    regions = [r for r in allr if r in sel] if sel else allr
    for r in regions:
        if not force and os.path.exists(os.path.join(MULTI, r, 'samples', 'meta.json')):
            P('[skip] %s samples already exist (--force can rebuild)' % r); continue
        try:
            build(r)
        except Exception as e:
            import traceback
            P('!! %s failed: %r' % (r, e))
            LOG.append(traceback.format_exc())
    open(LOG + '/_09_samples_report.txt', 'w', encoding='utf-8').write('\n'.join(LOG))
    print('\n[written] _09_samples_report.txt')
