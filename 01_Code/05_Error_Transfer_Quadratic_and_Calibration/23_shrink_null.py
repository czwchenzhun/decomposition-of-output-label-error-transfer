# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
23 Key falsification test: is "bias correction" a mathematical necessity of MSE training?

Theoretical derivation (strict):
  Let θ=grid truth, η=station representativeness error (independent of everything at grid scale), b=reference-field error,
     t = θ + η        station observation
     r = θ + b        reference field (label)
     m = θ + c        model output
  Then  Cov[(r-t),(m-r)] = Cov[b,c] - Var[b]
      α = -Cov/E1² = (Var[b] - Cov[b,c]) / (Var[b] + Var[η])
  Decompose c orthogonally against b as c = β·b + δ:
      α = (1-β)·Var[b] / E1²

  If the model is MSE-optimal, c = E[b|X] + e_θ, hence
      β = Var[E[b|X]] / Var[b] = R² of predicting the reference-field error from meteorological forcing
      α = Var[u_b] / E1²       (u_b = the part of the reference-field error not predictable from the forcing)

  ⇒ α>0 is a **mathematical necessity** for any MSE-trained model (unpredictable error is regressed away),
    not the model "knowing" where the truth is. Any operation that increases shrinkage (regularisation) raises α.

  Therefore the test: construct a series of **zero-training, pure-shrinkage** predictors and see whether their α is
  already ≥ the deep-learning models. If so, "bias correction" is not model ability but a shrinkage artefact.
"""
import os
import numpy as np
import pandas as pd

B = TRUTH


def budget(t, r, m):
    """Three-layer error budget: returns E1,E2,E3,Cov,rho,alpha"""
    d1 = r - t
    d2 = m - t
    d3 = m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    rho = Cov / (E1 * E3) if E1 * E3 > 0 else 0.0
    alpha = -Cov / (E1 ** 2)
    return E1, E2, E3, Cov, rho, alpha


def main():
    P = 'truth_chain_L2_monthly_gpu.csv'
    A = pd.read_csv(os.path.join(B, P))
    if 'horizon' in A.columns:
        A = A[A.horizon == 0]

    for lay in ['10cm', '30cm']:
        D = A[A.layer == lay].copy()
        print('=' * 108)
        print('[%s]  n=%d pairs   -- pure-shrinkage zero-training predictors vs trained models' % (lay, len(D)))
        print('-' * 108)
        print('%-36s %8s %8s %8s %8s %8s %7s' %
              ('predictor', 'E1', 'E2', 'E3', 'rho', 'alpha', 'E2/E1'))
        print('-' * 108)

        t = D.obs.values
        r = D.smci.values

        rows = []
        for cfg in sorted(set(D.cfg)):
            v = D[D.cfg == cfg]
            rows.append([cfg + '  ← trained model'] +
                        list(budget(v.obs.values, v.smci.values, v.model.values)))

        clim = D.groupby('id').smci.transform('mean').values
        seas = D.groupby(['id', 'month']).smci.transform('mean').values
        rows.append(['CLIM station temporal mean <- zero training'] + list(budget(t, r, clim)))
        rows.append(['SEAS station monthly climatology <- zero training'] + list(budget(t, r, seas)))
        for g in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
            rows.append(['shrink g=%.2f (clim+g·(smci−clim))' % g] +
                        list(budget(t, r, clim + g * (r - clim))))

        for nm, E1, E2, E3, Cov, rho, al in rows:
            print('%-36s %8.5f %8.5f %8.5f %+8.3f %6.2f%% %7.4f' %
                  (nm, E1, E2, E3, rho, 100 * al, E2 / E1))
        print()


if __name__ == '__main__':
    main()