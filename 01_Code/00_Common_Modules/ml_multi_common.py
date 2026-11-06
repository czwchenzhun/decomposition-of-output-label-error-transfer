# -*- coding: utf-8 -*-
from paths import DATA, REPRO
"""Shared multi-region modelling (01/240/241 series) --  same as the single-region ml_common.py metric conventions identical, only paths parameterised

Do not modify ml_common.py (protecting reproducibility of the archived single-region results).
Metric functions are imported directly from ml_common imported from ml_common, ensuring R/RMSE/ubRMSE/bias definitions are exactly identical.
"""
import os, json
import numpy as np

sys_root = REPRO + '/_Upstream_Intermediate_Products/out'

def sample_dir(region):
    return os.path.join(sys_root, '04_ready', 'multi', region, 'samples')

def model_dir(region):
    d = os.path.join(sys_root, '07_models', 'multi', region)
    os.makedirs(d, exist_ok=True)
    return d

def fig_dir(region):
    d = os.path.join(sys_root, '08_figures', 'multi', region)
    os.makedirs(d, exist_ok=True)
    return d

def regions():
    m = os.path.join(sys_root, '04_ready', 'multi')
    return sorted(d for d in os.listdir(m) if os.path.isdir(os.path.join(m, d))
                  and os.path.exists(os.path.join(m, d, 'samples', 'meta.json')))

def load_samples_multi(region):
    """Return X(memmap float32, T,N,F), y(2,T,N), meta, cells, time"""
    sd = sample_dir(region)
    with open(os.path.join(sd, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    T, N, F = meta['T'], meta['N'], meta['F']
    X = np.memmap(os.path.join(sd, 'X.dat'), dtype='float32', mode='r').reshape(T, N, F)
    y = np.load(os.path.join(sd, 'y.npy'))
    tt = np.load(os.path.join(sd, 'time.npy'))
    c = np.load(os.path.join(sd, 'cells.npz'))
    cells = dict(iy=c['iy'], ix=c['ix'], lat=c['lat'], lon=c['lon'])
    return X, y, meta, cells, tt

def years_of(tt):
    return np.asarray(tt, dtype='datetime64[Y]').astype(int) + 1970

def months_of(tt):
    return np.asarray(tt, dtype='datetime64[M]').astype(int) % 12 + 1

def take(X, t_idx, c_idx, f_idx=None):
    sub = X[np.asarray(t_idx)[:, None], np.asarray(c_idx)[None, :]]
    if f_idx is not None:
        sub = sub[:, :, np.asarray(f_idx)]
    return np.ascontiguousarray(sub.reshape(-1, sub.shape[-1]))

def make_split(scheme, tt, ncell, seed=42, tr_end=2017, te_start=2018):
    """temporal: <=tr_end train / >=te_start test;spatial: hold out 25% grid cells;random: row-level 80/20"""
    T = len(tt)
    yrs = years_of(tt)
    rng = np.random.default_rng(seed)
    if scheme == 'temporal':
        tr_t = np.where(yrs <= tr_end)[0]
        te_t = np.where(yrs >= te_start)[0]
        ci = np.arange(ncell)
        return dict(train=(tr_t, ci), test=(te_t, ci),
                    desc='temporal extrapolation: <=%d train / >=%d test' % (tr_end, te_start))
    if scheme == 'spatial':
        perm = rng.permutation(ncell)
        k = int(ncell * 0.25)
        te_c = np.sort(perm[:k]); tr_c = np.sort(perm[k:])
        ti = np.arange(T)
        return dict(train=(ti, tr_c), test=(ti, te_c),
                    desc='spatial block hold-out: random 25%% grid cells (%d/%d) as test' % (k, ncell))
    if scheme == 'random':
        ti = np.arange(T); ci = np.arange(ncell)
        return dict(train=(ti, ci), test=(ti, ci), random_row=True, seed=seed,
                    desc='random row split 80/20')
    raise ValueError(scheme)

def lag_feature_names(names):
    return [i for i, n in enumerate(names) if n.startswith('SM') and 'lag' in n]
