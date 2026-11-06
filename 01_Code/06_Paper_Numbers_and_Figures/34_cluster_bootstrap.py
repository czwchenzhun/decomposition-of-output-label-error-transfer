# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
34 station-level clustered bootstrap -- 95% CI for family-level rho / E2/E1

Motivation: the sd in 33 is the spread "across model configurations", not sampling uncertainty.
      the true sampling unit is the station (paired (station, month) values are not independent within a station),
      so here the station is the bootstrap cluster (resampling 85 stations with replacement),
      one bootstrap resample index is used for all configurations of that layer, so that
      the CI of "between-family differences" (e.g. rho_FREE - rho_HARD) can be given directly.

Output: 
  02_Results_Data/truth_chain/paper_bootstrap.csv
  scripts/_34_cluster_bootstrap.txt
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: station-clustered bootstrap 95% intervals in main-text Tables 5, 7 and 8
#

import os
import io
import glob
import numpy as np
import pandas as pd

B = TRUTH
TXT = LOG + '/_34_cluster_bootstrap.txt'
NBOOT = 2000
SEED = 20261005

FREE_RUNCFG = [
    ('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'), ('gpu_LSTMs21', 'LSTM'),
    ('gpu_lstmns', 'LSTMNS'),
    ('gpu_deg5', 'LSTM'), ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM'),
]
DAMAGED = [('gpu_degwd', 'LSTM'), ('gpu_wd0.01', 'LSTM'), ('gpu_wd0.03', 'LSTM')]
FAMS = ['FREE', 'SOFT', 'HARD', 'DAMAGED']


def fam_of(run, cfg):
    if (run, cfg) in DAMAGED:
        return 'DAMAGED'
    if (run, cfg) in FREE_RUNCFG:
        return 'FREE'
    if cfg == 'LPMIX':
        return 'SOFT'
    if cfg == 'HYBNS':
        return 'HARD'
    return None


def load():
    """returns {(layer): dict(keys, stations, n, A, B, C)}"""
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_gpu*.csv')))
    raw = {}
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        A = pd.read_csv(f, encoding='utf-8-sig')
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        for cfg in sorted(set(A.cfg)):
            fam = fam_of(nm, cfg)
            if fam is None:
                continue
            for lay in ['10cm', '30cm']:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 200:
                    continue
                ids = v.id.values
                d1 = v.smci.values - v.obs.values
                d3 = v.model.values - v.smci.values
                u, inv = np.unique(ids, return_inverse=True)
                k = len(u)
                cnt = np.bincount(inv, minlength=k).astype(float)
                s1 = np.bincount(inv, weights=d1, minlength=k)
                s3 = np.bincount(inv, weights=d3, minlength=k)
                d1c = d1 - (s1 / cnt)[inv]
                d3c = d3 - (s3 / cnt)[inv]
                raw.setdefault(lay, {})[(nm, cfg)] = dict(
                    fam=fam, ids=u,
                    n=cnt,
                    A=np.bincount(inv, weights=d1c ** 2, minlength=k),
                    Bb=np.bincount(inv, weights=d3c ** 2, minlength=k),
                    C=np.bincount(inv, weights=d1c * d3c, minlength=k))
    return raw


def main():
    raw = load()
    out = []

    def ap(s=''):
        out.append(s)
        print(s)

    ap('=' * 96)
    ap('34 station-level clustered bootstrap (%d resamples, mvn.seed=%d)' % (NBOOT, SEED))
    ap('=' * 96)

    rows = []
    for lay in ['10cm', '30cm']:
        D = raw[lay]
        keys = sorted(D.keys())
        stations = sorted(set().union(*[set(D[k]['ids']) for k in keys]))
        sidx = {s: i for i, s in enumerate(stations)}
        S = len(stations)
        ap('')
        ap('[%s] stations %d, configurations %d' % (lay, S, len(keys)))
        ap('  median month-station pair count per file = %d'
           % int(np.median([len(D[k]['n']) for k in keys])))

        N = np.zeros((len(keys), S))
        # Note: n / A / B / C must all be aligned by station
        mats = {}
        for name in ['n', 'A', 'Bb', 'C']:
            M = np.zeros((len(keys), S))
            for i, k in enumerate(keys):
                pos = [sidx[s] for s in D[k]['ids']]
                M[i, pos] = D[k][name]
            mats[name] = M
        N = mats['n']

        fam_arr = np.array([D[k]['fam'] for k in keys])
        rng = np.random.default_rng(SEED)
        ns = N.sum(axis=1)
        boot_idx = rng.integers(0, S, size=(NBOOT, S))

        res = {k: [] for k in ['rho', 'x', 'ratio', 'exc', 'floor']}
        fam_res = {f: {k: np.empty(NBOOT) for k in ['rho', 'x', 'ratio']} for f in FAMS}
        diff_rho = np.empty(NBOOT)
        diff_ratio = np.empty(NBOOT)

        # prefetch columns
        nSumm = {}
        for name in ['A', 'Bb', 'C']:
            nSumm[name] = mats[name]

        for b in range(NBOOT):
            ix = boot_idx[b]
            cnt = np.bincount(ix, minlength=S).astype(float)
            # weighted sum per configuration (missing stations with n=0 are excluded automatically)
            A1 = nSumm['A'] @ cnt
            B1 = nSumm['Bb'] @ cnt
            C1 = nSumm['C'] @ cnt
            NN = N @ cnt
            E1 = np.sqrt(A1 / NN)
            E3 = np.sqrt(B1 / NN)
            Cov = C1 / NN
            rho = Cov / (E1 * E3)
            E2 = np.sqrt(E1 ** 2 + E3 ** 2 + 2 * Cov)
            x = E3 / E1
            ratio = E2 / E1
            for f in FAMS:
                m = fam_arr == f
                if m.sum():
                    fam_res[f]['rho'][b] = rho[m].mean()
                    fam_res[f]['x'][b] = x[m].mean()
                    fam_res[f]['ratio'][b] = ratio[m].mean()
            mF = fam_arr == 'FREE'
            mH = fam_arr == 'HARD'
            diff_rho[b] = rho[mF].mean() - rho[mH].mean()
            diff_ratio[b] = ratio[mF].mean() - ratio[mH].mean()

        ap('')
        ap('  family            rho [95% CI]              x [95% CI]           E2/E1 [95% CI]       P(E2/E1<1)')
        for f in FAMS:
            m = fam_arr == f
            if not m.sum():
                continue
            for name in ['rho', 'x', 'ratio']:
                v = fam_res[f][name]
                if name == 'rho':
                    lo, hi = np.percentile(v, [2.5, 97.5])
                    s_rho = '%.4f [%.4f, %.4f]' % (v.mean(), lo, hi)
                elif name == 'x':
                    lo, hi = np.percentile(v, [2.5, 97.5])
                    s_x = '%.4f [%.4f, %.4f]' % (v.mean(), lo, hi)
                else:
                    lo, hi = np.percentile(v, [2.5, 97.5])
                    s_r = '%.4f [%.4f, %.4f]' % (v.mean(), lo, hi)
                    pbeat = float((v < 1).mean())
            ap('  %-8s n=%-3d %s  %s  %s  %.3f'
               % (f, int(m.sum()), s_rho, s_x, s_r, pbeat))
            rows.append(dict(layer=lay, fam=f, n=int(m.sum()),
                             rho=fam_res[f]['rho'].mean(),
                             rho_lo=np.percentile(fam_res[f]['rho'], 2.5),
                             rho_hi=np.percentile(fam_res[f]['rho'], 97.5),
                             x=fam_res[f]['x'].mean(),
                             x_lo=np.percentile(fam_res[f]['x'], 2.5),
                             x_hi=np.percentile(fam_res[f]['x'], 97.5),
                             ratio=fam_res[f]['ratio'].mean(),
                             ratio_lo=np.percentile(fam_res[f]['ratio'], 2.5),
                             ratio_hi=np.percentile(fam_res[f]['ratio'], 97.5),
                             p_beat=pbeat))
        lo, hi = np.percentile(diff_rho, [2.5, 97.5])
        ap('')
        ap('  difference rho(FREE) - rho(HARD) = %+.4f [%+.4f, %+.4f]   P(<0) = %.3f'
           % (diff_rho.mean(), lo, hi, float((diff_rho < 0).mean())))
        lo, hi = np.percentile(diff_ratio, [2.5, 97.5])
        ap('  difference E2/E1(FREE) - E2/E1(HARD) = %+.4f [%+.4f, %+.4f]   P(<0) = %.3f'
           % (diff_ratio.mean(), lo, hi, float((diff_ratio < 0).mean())))
        rows.append(dict(layer=lay, fam='DIFF_FREE_MINUS_HARD', n=-1,
                         rho=diff_rho.mean(), rho_lo=np.percentile(diff_rho, 2.5),
                         rho_hi=np.percentile(diff_rho, 97.5),
                         x=np.nan, x_lo=np.nan, x_hi=np.nan,
                         ratio=diff_ratio.mean(),
                         ratio_lo=np.percentile(diff_ratio, 2.5),
                         ratio_hi=np.percentile(diff_ratio, 97.5),
                         p_beat=float((diff_ratio < 0).mean())))

    pd.DataFrame(rows).to_csv(os.path.join(B, 'paper_bootstrap.csv'),
                              index=False, encoding='utf-8-sig')
    with io.open(TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print('\nwrote', TXT)


if __name__ == '__main__':
    main()
