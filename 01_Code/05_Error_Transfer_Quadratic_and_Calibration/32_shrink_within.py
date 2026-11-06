# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
32 Within-station-convention version of the zero-training shrinkage family (supplement to 23)

23 is the full convention. Under the within-station convention:
  CLIM (station mean) has a within-station component = 0, i.e. "no within-station anomaly is predicted";
  SEAS (monthly climatology) retains monthly anomalies;
  shrink g: m_within = g · r_within.
If trained models cannot beat these zero-training baselines under the within-station convention, "beating the label" is empty talk.

Output: truth_chain/paper_null_within.csv + scripts/_32_shrink_within.txt
"""
import os
import io
import numpy as np
import pandas as pd

B = TRUTH
TXT = LOG + '/_32_shrink_within.txt'


def budget(o, r, m):
    d1, d2, d3 = r - o, m - o, m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    rho = Cov / (E1 * E3) if E1 * E3 > 0 else 0.0
    return E1, E2, E3, rho, E2 / E1


def main():
    A = pd.read_csv(os.path.join(B, 'truth_chain_L2_monthly_gpu.csv'), encoding='utf-8-sig')
    if 'horizon' in A.columns:
        A = A[A.horizon == 0]
    out = []
    rows = []

    def ap(s=''):
        out.append(s)
        print(s)

    for lay in ['10cm', '30cm']:
        D = A[A.layer == lay].copy()
        ap('=' * 100)
        ap('[%s] n=%d' % (lay, len(D)))
        ap('%-30s %9s %9s %9s %8s %8s    (full convention | within-station convention)' %
           ('predictor', 'E1f', 'E2f', 'E3f', 'rho_f', 'E2/E1f'))
        for name, m in [
            ('CLIM station mean (zero training)', D.groupby('id').smci.transform('mean').values),
            ('SEAS station-month climatology (zero training)',
             D.groupby(['id', 'month']).smci.transform('mean').values),
        ] + [('shrink g=%.2f' % g,
              (D.groupby('id').smci.transform('mean') +
               g * (D.smci - D.groupby('id').smci.transform('mean'))).values)
             for g in (0.25, 0.5, 0.75, 0.9)]:
            E1f, E2f, E3f, rf, rr = budget(D.obs.values, D.smci.values, m)
            # within-station convention
            V = D.copy()
            for c, mv in [('obs', D.obs.values), ('smci', D.smci.values), ('m', m)]:
                V[c] = mv - pd.Series(mv).groupby(V.id.values).transform('mean').values
            E1w, E2w, E3w, rw, rww = budget(V.obs.values, V.smci.values, V.m.values)
            ap('%-30s %9.5f %9.5f %9.5f %+8.3f %8.4f   | within-station: E1=%.5f E3=%.5f '
               'rho=%+.3f E2/E1=%.4f' % (name, E1f, E2f, E3f, rf, rr,
                                         E1w, E3w, rw, rww))
            rows.append(dict(layer=lay, pred=name, E1f=E1f, E2f=E2f, E3f=E3f,
                             rho_f=rf, ratio_f=rr, E1w=E1w, E2w=E2w, E3w=E3w,
                             rho_w=rw, ratio_w=rww))
        # trained-model comparison
        for cfg in sorted(set(D.cfg)):
            v = D[D.cfg == cfg]
            E1f, E2f, E3f, rf, rr = budget(v.obs.values, v.smci.values, v.model.values)
            V = v.copy()
            for c in ['obs', 'smci', 'model']:
                V[c] = V[c] - V.groupby('id')[c].transform('mean')
            E1w, E2w, E3w, rw, rww = budget(V.obs.values, V.smci.values, V.model.values)
            ap('%-30s %9.5f %9.5f %9.5f %+8.3f %8.4f   | within-station: E1=%.5f E3=%.5f '
               'rho=%+.3f E2/E1=%.4f' % (cfg + ' (trained)', E1f, E2f, E3f, rf, rr,
                                         E1w, E3w, rw, rww))
            rows.append(dict(layer=lay, pred=cfg + '_trained', E1f=E1f, E2f=E2f,
                             E3f=E3f, rho_f=rf, ratio_f=rr,
                             E1w=E1w, E2w=E2w, E3w=E3w, rho_w=rw, ratio_w=rww))
        ap('')

    pd.DataFrame(rows).to_csv(os.path.join(B, 'paper_null_within.csv'),
                              index=False, encoding='utf-8-sig')
    with io.open(TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


if __name__ == '__main__':
    main()
