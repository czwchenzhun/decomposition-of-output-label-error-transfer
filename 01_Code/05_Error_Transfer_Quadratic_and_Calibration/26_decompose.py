# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
26 Interpretable decomposition: necessary and sufficient condition for an output to beat its training label

Theory (within-station anomaly convention, η already de-meaned by station):
  t = θ + η   station observation      r = θ + b   reference field (label)     m = θ + c   model
  Decompose c into μ + δ, where μ = the part of label error predictable from forcing X (learned and reproduced by the model),
  δ = the deviation additionally injected by the model; σ = the model's own irreducible prediction error.
  Identity (strict): E2² - E1² = Var[σ] + Var[δ] - Var[u_b]
  where u_b = the part of label error not predictable from the forcing (= the part regressed away by MSE).

  ⇒ output better than label  ⟺  Var[σ] + Var[δ]  <  Var[u_b]
    "the error you cannot shed + the deviation you inject"  is less than  "the part of label error you cannot learn"

  Observable proxies (under the MSE-optimal approximation Cov ≈ -Var[u_b]):
    U   = -Cov               (variance of the unlearnable label error)
    ID  = E2² - E1² - Cov    (own irreducible + injected deviation)
"""
import os
import numpy as np
import pandas as pd

B = TRUTH


def parts(o, r, m):
    d1, d2, d3 = r - o, m - o, m - r
    E1 = float(np.mean(d1 ** 2))
    E2 = float(np.mean(d2 ** 2))
    Cov = float(np.mean(d1 * d3))
    return E1, E2, Cov


def main():
    A = pd.read_csv(os.path.join(B, 'truth_chain_L2_monthly_gpu.csv'))
    A = A[A.horizon == 0]
    files = sorted(set(os.listdir(B)))
    mfiles = [f for f in files if f.startswith('truth_chain_L2_monthly_gpu')]

    rows = []
    for f in mfiles:
        nm = f.replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        try:
            D = pd.read_csv(os.path.join(B, f))
        except Exception:
            continue
        if 'horizon' in D.columns:
            D = D[D.horizon == 0]
        for cfg in sorted(set(D.cfg)):
            for lay in ['10cm', '30cm']:
                v = D[(D.cfg == cfg) & (D.layer == lay)]
                if len(v) < 200:
                    continue
                vv = v.copy()
                for c in ['obs', 'smci', 'model']:
                    vv[c] = vv[c] - vv.groupby('id')[c].transform('mean')
                E1, E2, Cov = parts(vv.obs.values, vv.smci.values, vv.model.values)
                U = -Cov
                ID = E2 - E1 - Cov
                rows.append(dict(run=nm, cfg=cfg, layer=lay,
                                 E1=np.sqrt(E1), U=U, ID=ID,
                                 ratio=np.sqrt(E2 / E1),
                                 verdict=('better than label' if ID < U else 'worse than label')))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(B, 'err_transfer_decomp_components.csv'),
             index=False, encoding='utf-8-sig')

    for lay in ['10cm', '30cm']:
        s = R[R.layer == lay].copy()
        s['grp'] = np.where(s.cfg == 'HYBNS', 'hard skeleton HYBNS',
                            np.where(s.cfg == 'LPMIX', 'soft constraint LPMIX', 'free LSTM'))
        print('=' * 96)
        print('[%s] within-station anomaly convention   unit = 1e-4 (m³/m³)²   E1²=%.4f' % (lay, s.E1.iloc[0] ** 2 * 1e4))
        print('-' * 96)
        print('%-16s %4s | %10s %10s | %8s | %s' %
              ('group', 'n', 'U=unlearnable label', 'ID=own+injected', 'E2/E1', 'verdict'))
        print('-' * 96)
        for g in ['free LSTM', 'soft constraint LPMIX', 'hard skeleton HYBNS']:
            v = s[s.grp == g]
            if not len(v):
                continue
            U, ID = v.U.mean() * 1e4, v.ID.mean() * 1e4
            sdU = v.U.std(ddof=1) * 1e4 if len(v) > 1 else 0
            sdI = v.ID.std(ddof=1) * 1e4 if len(v) > 1 else 0
            print('%-16s %4d | %6.3f±%.3f %6.3f±%.3f | %8.4f | %s' %
                  (g, len(v), U, sdU, ID, sdI, v.ratio.mean(),
                   'better than label' if ID < U else 'worse than label'))
        print()


if __name__ == '__main__':
    main()
