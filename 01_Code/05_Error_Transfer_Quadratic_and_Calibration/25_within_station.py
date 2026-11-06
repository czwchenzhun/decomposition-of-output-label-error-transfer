# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
25 Full recomputation for all configurations: within-station temporal anomaly convention (removing point-to-grid static representativeness offset)

E1 comprises 64-74% between-station static offset (cannot be corrected by a grid-scale model); the remaining 26-36% is dynamic mismatch.
The primary convention should be "within-station temporal anomalies after de-meaning by station", with the full convention reported alongside to avoid convention-shopping.
"""
import os
import glob
import numpy as np
import pandas as pd

B = TRUTH


def budget(o, r, m):
    d1, d2, d3 = r - o, m - o, m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    return dict(E1=E1, E2=E2, E3=E3, Cov=Cov,
                rho=(Cov / (E1 * E3) if E1 * E3 > 0 else 0.),
                alpha=-Cov / (E1 ** 2))


def main():
    # Primary-region GPU configurations + conventional supervised-learning baselines (produced by 49, M3); replication-region xr* handled separately by 46
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_gpu*.csv')))
    _ml = os.path.join(B, 'truth_chain_L2_monthly_mlbase.csv')
    if os.path.exists(_ml):
        files.append(_ml)
    rows = []
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        try:
            A = pd.read_csv(f)
        except Exception as e:
            print('skip', nm, e)
            continue
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        if not len(A):
            continue
        for cfg in sorted(set(A.cfg)):
            for lay in ['10cm', '30cm']:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 200:
                    continue
                full = budget(v.obs.values, v.smci.values, v.model.values)
                vv = v.copy()
                for c in ['obs', 'smci', 'model']:
                    vv[c] = vv[c] - vv.groupby('id')[c].transform('mean')
                wit = budget(vv.obs.values, vv.smci.values, vv.model.values)
                # 2026-10-07 Added: the full convention also outputs E3f / E2f,
                # so that script 48 can compute x=E3f/E1f and E2/E1=E2f/E1f
                #  (second-round review R1: 8 cells in Table 5 had no computational basis in the archive).
                # Only new columns are added; existing column values and conventions are completely unchanged.
                rows.append(dict(run=nm, cfg=cfg, layer=lay, n=len(v),
                                 E1f=full['E1'], E3f=full['E3'], E2f=full['E2'],
                                 rho_f=full['rho'], a_f=full['alpha'],
                                 E1w=wit['E1'], E3w=wit['E3'], E2w=wit['E2'],
                                 rho_w=wit['rho'], a_w=wit['alpha']))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(B, 'err_transfer_withinstation.csv'),
             index=False, encoding='utf-8-sig')

    for lay in ['10cm', '30cm']:
        s = R[R.layer == lay].sort_values('a_w', ascending=False)
        print('=' * 112)
        print('[%s] ranked by α under the within-station anomaly convention' % lay)
        print('-' * 112)
        print('%-26s %-7s %5s | %7s %7s %7s | %7s %7s %7s' %
              ('run', 'cfg', 'n', 'αfull', 'ρfull', 'E3full', 'αwithin', 'ρwithin', 'E2/E1within'))
        print('-' * 112)
        for _, r in s.iterrows():
            print('%-26s %-7s %5d | %6.2f%% %+7.3f         | %6.2f%% %+7.3f %7.4f' %
                  (r.run, r.cfg, r.n, 100 * r.a_f, r.rho_f,
                   100 * r.a_w, r.rho_w, r.E2w / r.E1w))
        print()


if __name__ == '__main__':
    main()
