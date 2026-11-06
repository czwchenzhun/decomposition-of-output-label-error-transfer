# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
24 E1 structural diagnostic: what exactly is this "label error"?

Split d1 = smci - obs into
  between-group (station-mean offset, reflecting point-to-grid representativeness differences / instrument calibration)
  within-group (within-station temporal anomaly, reflecting dynamic mismatch)
Then recompute α under the "within-group anomaly only" convention to see whether bias correction happens in time or in space.
"""
import os
import numpy as np
import pandas as pd

B = TRUTH


def budget(t, r, m):
    d1, d2, d3 = r - t, m - t, m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    return E1, E2, E3, Cov, (Cov / (E1 * E3) if E1 * E3 > 0 else 0.), -Cov / (E1 ** 2)


def main():
    A = pd.read_csv(os.path.join(B, 'truth_chain_L2_monthly_gpu.csv'))
    if 'horizon' in A.columns:
        A = A[A.horizon == 0]

    for lay in ['10cm', '30cm']:
        D = A[A.layer == lay].copy()
        print('=' * 100)
        print('[%s] %d stations, pairs: %d' % (lay, D.id.nunique(), len(D)))
        print('-' * 100)

        # ---- variance decomposition of d1 ----
        g = D.groupby('id').smci.transform('mean') - D.groupby('id').obs.transform('mean')
        d1 = D.smci - D.obs
        tot = float(np.mean(d1 ** 2))
        btw = float(np.mean(g ** 2))
        wit = float(np.mean((d1 - g) ** 2))
        print('MSE(d1) = %.6f   between-station offset %.6f (%.1f%%)   within-station temporal anomaly %.6f (%.1f%%)'
              % (tot, btw, 100 * btw / tot, wit, 100 * wit / tot))
        print('  → E1=%.5f; after removing the between-station static offset, the residual RMSE=%.5f'
              % (np.sqrt(tot), np.sqrt(wit)))
        print()

        # ---- α under the two conventions ----
        print('%-12s | %-30s | %-30s' % ('cfg', 'full convention E1 / rho / alpha', 'within-station anomaly E1 / rho / alpha'))
        print('-' * 100)
        for cfg in sorted(set(D.cfg)):
            v = D[D.cfg == cfg]
            o, r, m = v.obs.values, v.smci.values, v.model.values
            e1, e2, e3, cv, rh, al = budget(o, r, m)
            # within-station anomaly convention: de-mean each series by station
            vv = v.copy()
            for c in ['obs', 'smci', 'model']:
                vv[c] = vv[c] - vv.groupby('id')[c].transform('mean')
            e1b, e2b, e3b, cvb, rhb, alb = budget(vv.obs.values, vv.smci.values,
                                                  vv.model.values)
            print('%-12s | %.5f %+7.3f %6.2f%%        | %.5f %+7.3f %6.2f%%'
                  % (cfg, e1, rh, 100 * al, e1b, rhb, 100 * alb))
        print()


if __name__ == '__main__':
    main()
