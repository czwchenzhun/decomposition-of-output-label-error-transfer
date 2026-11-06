# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""49: Conventional supervised-learning baselines (M3, fourth-round review) -- ridge regression / LightGBM

Review point
--------
From the variance-budget decomposition of the three error layers (M1), any conditional-mean model that targets MSE and approximates E[r|X]
**automatically** obtains Cov(e1,e3)<0; therefore "model output better than its training label" is the theoretical default outcome of this framework.
To judge whether the deep model additionally extracts "alignment information", a **non-deep, non-physical** conventional
supervised-learning baseline must be trained on exactly the same inputs, labels, time splits and rollout protocol.

This script does two things:
  1. Using exactly the same npz as 13 (same grid cells, same features, same time axis), construct
     **window-level tabular samples**: each row = (grid cell, day t within the window), features = same-day forcing and static covariates
     + the label value s0 at the window start (aligned with the initial state of the deep model) + relative position t/(K-1) within the window;
     target = the SMCI value of that day.
  2. Train ridge regression (linear) and LightGBM (trees), reconstruct daily predictions of shape (B,N,K,2) over the test-period windows,
     save in exactly the same pred_*.npz format as 13, so that 16 follows **the same** station-truth comparison chain.

The only remaining difference between the baselines and the deep model is then the "function class":
  - ridge regression: linear, memoryless, stateless;
  - LightGBM: non-linear, memoryless, stateless;
  - LSTM/LPMIX/HYBNS: with memory/state, recursively rollable.

Usage (geoenv, CPU):
  python 49_ml_baseline.py --npz <Training_Inputs npz> \
      --region R0_northchina --tag _staG_mlbase --models ridge,lgbm
  # Afterwards:
  python 16_model_vs_truth.py --npz <the same npz> --tag _staG_mlbase \
      --region R0_northchina --cfgs RIDGE,LGBM --out_suffix mlbase
  python 46_region_analysis.py --region R0_northchina --pattern "*" --nboot 2000
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: main-text Table 4 (conventional supervised-learning baselines) and the baseline comparison in §4.1
#

import os, sys, argparse, time
import numpy as np

import common

ROOT = common.ROOT                      # package root, resolved relatively by paths.py
OUTB = BENCH


def P(*a):
    print(*a, flush=True)


def sl_of(time, s0, s1):
    """Date string -> index slice (same convention as sl_of in 13)"""
    t0 = np.datetime64(s0, 'D')
    t1 = np.datetime64(s1, 'D')
    i0 = int(np.searchsorted(time, t0, 'left'))
    i1 = int(np.searchsorted(time, t1, 'right'))
    return slice(i0, i1)


def make_rows(X, Y, starts, K):
    """Window-level tabular samples.

    X (T,N,F), Y (T,N,2) -> features (nsw*K*N, F+3), target (nsw*K*N, 2)
    The 3 additional columns: s0_10cm, s0_30cm (label values at the window start, aligned with the initial state of the deep model),
                 step = t/(K-1) (relative position within the window)
    """
    nsw = len(starts)
    N = X.shape[1]
    F = X.shape[2]
    s0 = Y[starts]                       # (nsw, N, 2)
    xs = np.empty((K, nsw, N, F + 3), 'float32')
    ys = np.empty((K, nsw, N, 2), 'float32')
    for t in range(K):
        ti = starts + t + 1
        xs[t, :, :, :F] = X[ti]                       # (nsw,N,F)
        xs[t, :, :, F:F + 2] = s0                     # label values at the window start (two layers)
        xs[t, :, :, F + 2] = t / max(1, K - 1)        # relative position
        ys[t] = Y[ti]
    x = xs.reshape(K * nsw * N, F + 3)
    y = ys.reshape(K * nsw * N, 2)
    ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    return x[ok], y[ok], ok.reshape(K, nsw, N)


def fit_predict(xtr, ytr, xva, yva, xte, model, layer, seed=42):
    """Fit per layer; return test-period predictions (ntest_rows,)"""
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(xtr)
        best, ba = None, None
        for a in (0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=a).fit(sc.transform(xtr), ytr)
            r = float(np.sqrt(np.mean((m.predict(sc.transform(xva)) - yva) ** 2)))
            if ba is None or r < ba:
                ba, best = r, (a, m)
        a, m = best
        P('    [ridge] layer=%s  alpha=%.3g  val RMSE=%.6f' % (layer, a, ba))
        return m.predict(sc.transform(xte))
    if model == 'lgbm':
        import lightgbm as lgb
        ds_tr = lgb.Dataset(xtr, label=ytr)
        ds_va = lgb.Dataset(xva, label=yva, reference=ds_tr)
        p = dict(objective='regression', learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, seed=seed, verbose=-1, num_threads=8)
        m = lgb.train(p, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        P('    [lgbm] layer=%s  best_iter=%d' % (layer, m.best_iteration))
        return m.predict(xte, num_iteration=m.best_iteration)
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz',
                    default=os.path.join(BENCH, 'dl_R0_northchina_nc256_20020101_20131231_sta.npz'))
    ap.add_argument('--region', default='R0_northchina')
    ap.add_argument('--tag', default='_staG_mlbase')
    ap.add_argument('--models', default='ridge,lgbm')
    ap.add_argument('--ttrain', default='2002-01-01,2009-12-31')
    ap.add_argument('--tval', default='2010-01-01,2010-12-31')
    ap.add_argument('--ttest', default='2011-01-01,2013-12-31')
    ap.add_argument('--K', type=int, default=30)
    ap.add_argument('--stride', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t00 = time.time()
    d = np.load(args.npz, allow_pickle=True)
    X, Y, time_ = d['X'], d['Y'], d['time']
    feats = list(d['feats'])
    N = X.shape[1]
    T = X.shape[0]
    P('Data %s: X%s Y%s  feats=%d  N=%d  T=%d' % (args.region, X.shape, Y.shape,
                                                  len(feats), N, T))
    P('Features: %s' % ', '.join(feats))

    sl_tr = sl_of(time_, *args.ttrain.split(','))
    sl_va = sl_of(time_, *args.tval.split(','))
    sl_te = sl_of(time_, *args.ttest.split(','))
    K, st = args.K, args.stride

    def starts_of(sl):
        return np.arange(sl.start, sl.stop - K, st, dtype=np.int64)

    st_tr, st_va, st_te = starts_of(sl_tr), starts_of(sl_va), starts_of(sl_te)
    P('Windows: train %d  val %d  test %d  (K=%d stride=%d)'
      % (len(st_tr), len(st_va), len(st_te), K, st))

    xtr, ytr, _ = make_rows(X, Y, st_tr, K)
    xva, yva, _ = make_rows(X, Y, st_va, K)
    xte, yte, okte = make_rows(X, Y, st_te, K)
    P('Samples: train %d  val %d  test %d  dim %d' % (len(xtr), len(xva), len(xte), xtr.shape[1]))

    for model in args.models.split(','):
        model = model.strip()
        if not model:
            continue
        P('---- Fitting %s ----' % model)
        pred = np.empty((K, len(st_te), N, 2), 'float32')
        for li, lay in enumerate(['10cm', '30cm']):
            p = fit_predict(xtr, ytr[:, li], xva, yva[:, li], xte, model, lay, args.seed)
            full = np.full((K * len(st_te) * N,), np.nan, 'float32')
            # make_rows returns the filtered subset; back-fill by okte
            full[okte.reshape(-1)] = p
            pred[:, :, :, li] = full.reshape(K, len(st_te), N)
        # -> (nsw, N, K, 2), then flatten to (nsw*N, K, 2) as saved by 13
        pred = np.transpose(pred, (1, 2, 0, 3)).astype('float32')   # (nsw, N, K, 2)
        cfg = 'RIDGE' if model == 'ridge' else 'LGBM'
        pred_out = pred.reshape(len(st_te) * N, K, 2)
        op = os.path.join(OUTB, 'pred_%s%s.npz' % (cfg, args.tag))
        np.savez_compressed(op, pred=pred_out, obs=Y, s0=Y[st_te],
                            starts=st_te, N=np.int64(N), K=np.int64(K),
                            cfg=cfg, feats=np.array(feats))
        P('  WROTE %s  pred%s' % (op, pred_out.shape))
        # Fidelity to the label (E3); the outcome against truth, E2, is given by 16 on the same chain
        idx = st_te[:, None] + np.arange(K)[None, :] + 1      # (nsw, K)
        obs = np.transpose(Y[idx], (0, 2, 1, 3))              # (nsw, N, K, 2)
        for li, lay in enumerate(['10cm', '30cm']):
            e3 = float(np.sqrt(np.nanmean((pred[..., li] - obs[..., li]) ** 2)))
            P('  %s %s label RMSE(E3) = %.6f' % (cfg, lay, e3))

    P('Elapsed %.1f min' % ((time.time() - t00) / 60.0))
    P('Next step: python 16_model_vs_truth.py --npz %s --tag %s --region %s '
      '--cfgs RIDGE,LGBM --out_suffix mlbase' % (args.npz, args.tag, args.region))
    try:
        common.report('401_ml_baseline_%s' % args.region, [])
    except Exception:
        pass


if __name__ == '__main__':
    main()
