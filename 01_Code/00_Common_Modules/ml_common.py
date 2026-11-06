# -*- coding: utf-8 -*-
from paths import DATA, REPRO
"""Shared modelling: sample loading, metric computation, train/test index construction"""
import os, json
import numpy as np

SD = REPRO + '/_Upstream_Intermediate_Products/out/06_samples'
MD = REPRO + '/_Upstream_Intermediate_Products/out/07_models'
FD = REPRO + '/_Upstream_Intermediate_Products/out/08_figures'
for _d in (MD, FD):
    os.makedirs(_d, exist_ok=True)

DEPTHS = ['10cm', '30cm']
SCHEMES = ['temporal', 'spatial', 'random']


def load_samples():
    """Return X(memmap float32, T,N,F), y(2,T,N), meta, cells(dict), time(datetime64)"""
    with open(os.path.join(SD, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    T, N, F = meta['T'], meta['N'], meta['F']
    X = np.memmap(os.path.join(SD, 'X.dat'), dtype='float32', mode='r').reshape(T, N, F)
    y = np.load(os.path.join(SD, 'y.npy'))                 # (2,T,N) float32
    tt = np.load(os.path.join(SD, 'time.npy'))
    c = np.load(os.path.join(SD, 'cells.npz'))
    cells = dict(iy=c['iy'], ix=c['ix'], lat=c['lat'], lon=c['lon'], frac=c['frac'])
    return X, y, meta, cells, tt


def lag_feature_names(names):
    return [i for i, n in enumerate(names) if n.startswith('SM') and 'lag' in n]


def metrics(obs, pred):
    o = np.asarray(obs, dtype='float64').ravel()
    p = np.asarray(pred, dtype='float64').ravel()
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if o.size == 0:
        return dict(n=0)
    err = p - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ub = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))
    r = float(np.corrcoef(o, p)[0, 1])
    return dict(n=int(o.size), R=r, R2=r * r, RMSE=rmse, ubRMSE=ub,
                MAE=float(np.mean(np.abs(err))), Bias=bias,
                obs_mean=float(o.mean()), pred_mean=float(p.mean()),
                obs_sd=float(o.std()), pred_sd=float(p.std()))


def per_cell_metrics(obs2d, pred2d):
    """(T,N) -> per-cell R / RMSE, return dict of arrays"""
    o = np.asarray(obs2d, 'float64'); p = np.asarray(pred2d, 'float64')
    ok = np.isfinite(o) & np.isfinite(p)
    po = np.where(ok, p, np.nan)
    om = o.mean(0); pm = np.nanmean(po, 0)
    dov = o - om; dpv = po - pm
    num = np.nansum(dov * np.where(ok, dpv, np.nan), 0)
    den = np.sqrt(np.nansum(dov ** 2, 0) * np.nansum(np.where(ok, dpv, np.nan) ** 2, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    rmse = np.sqrt(np.nanmean(np.where(ok, (po - o) ** 2, np.nan), 0))
    return dict(R=r, RMSE=rmse)


def fmt_metrics(d):
    return ('R=%.4f  RMSE=%.4f  ubRMSE=%.4f  MAE=%.4f  Bias=%+.4f'
            % (d['R'], d['RMSE'], d['ubRMSE'], d['MAE'], d['Bias']))


def make_split(scheme, tt, ncell, seed=42):
    """Return dict(train=..., test=...), each being (time_idx, cell_idx)"""
    T = len(tt)
    yrs = np.asarray(tt, dtype='datetime64[Y]').astype(int) + 1970
    rng = np.random.default_rng(seed)
    if scheme == 'temporal':
        tr_t = np.where(yrs <= 2017)[0]
        te_t = np.where(yrs >= 2018)[0]
        ci = np.arange(ncell)
        return dict(train=(tr_t, ci), test=(te_t, ci), desc='temporal: 2000-2017train / 2018-2022test')
    if scheme == 'spatial':
        perm = rng.permutation(ncell)
        k = int(ncell * 0.25)
        te_c = np.sort(perm[:k]); tr_c = np.sort(perm[k:])
        ti = np.arange(T)
        return dict(train=(ti, tr_c), test=(ti, te_c),
                    desc='spatial: randomly hold out 25%% grid cells (%d/%d) as test' % (k, ncell))
    if scheme == 'random':
        ti = np.arange(T); ci = np.arange(ncell)
        return dict(train=(ti, ci), test=(ti, ci), random_row=True, seed=seed,
                    desc='random: row-wise random 80/20 split')
    raise ValueError(scheme)


def take(X, t_idx, c_idx, f_idx=None):
    """X(T,N,F) -> (len(t_idx)*len(c_idx), len(f_idx)) float32"""
    sub = X[np.asarray(t_idx)[:, None], np.asarray(c_idx)[None, :]]   # (t,n,F)
    if f_idx is not None:
        sub = sub[:, :, np.asarray(f_idx)]
    return np.ascontiguousarray(sub.reshape(-1, sub.shape[-1]))
