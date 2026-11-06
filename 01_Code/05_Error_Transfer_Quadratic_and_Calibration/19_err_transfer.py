# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""19_err_transfer.py -- error transfer decomposition (Gate experiment E1)

Question: for a model m trained with the land-surface product r as label, how is its error against ground truth t composed?

Identity (strict, holds point by point, because m-t = (r-t) + (m-r))
------------------------------------------------------
    MSE(m,t) = MSE(r,t) + MSE(m,r) + 2·Cov[(r-t), (m-r)]
       E2²   =    E1²   +    E3²   +      2·Cov

  E1 = Error of label (SMCI) against truth      -- exogenous, cannot be removed by improving the model
  E2 = Error of model output against truth        -- the quantity actually of interest
  E3 = Error of model reproducing the label          -- what all existing literature reports
  Cov= whether the direction of the model's deviation from the label is opposite to the label's error direction (i.e. "bias correction")

Two interpretable metrics are derived
------------------------------------------------------
  ρ = Cov / (E1·E3) ∈ [-1,1]          correction efficiency (directional consistency)
  α = -Cov / E1²                      label-error elimination rate
      Derivation: if m = r - α(r-t) + ε, then m-t = (1-α)(r-t) + ε
      α=0 pure reproduction; α=1 completely removes the label error

Null hypothesis H0: the model is just a noisy copy of the label (ε ⊥ (r-t))→ Cov=0 → E2² = E1² + E3²
Alternative H1: Cov < 0, the model is quietly correcting bias

Input: truth_chain_L2_monthly.csv produced by 16 (obs/model/smci triples for the same station and month)
      -- the three have been monthly-averaged over the same set of days, with strictly identical conventions, so no model needs retraining
"""
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import OUT, report

OUTDIR = TRUTH
FIGDIR = REPRO + '/process_figures/23_err_transfer'
os.makedirs(FIGDIR, exist_ok=True)

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() >= 3 else np.nan


def decomp(o, r, m, debias=False):
    """o=truth r=label m=model; if debias=True each series is de-meaned first (approximate convention)"""
    v = np.isfinite(o) & np.isfinite(r) & np.isfinite(m)
    o, r, m = o[v], r[v], m[v]
    n = len(o)
    if n < 10:
        return None
    if debias:
        # de-mean each separately so bias does not enter; the identity is then no longer strict, used as a supplementary convention
        o = o - o.mean()
        r = r - r.mean()
        m = m - m.mean()
    e1 = rmse(r, o)
    e2 = rmse(m, o)
    e3 = rmse(m, r)
    cov = float(np.mean((r - o) * (m - r)))
    rho = cov / (e1 * e3) if (e1 > 0 and e3 > 0) else np.nan
    alpha = -cov / (e1 ** 2) if e1 > 0 else np.nan
    orth = float(np.sqrt(e1 ** 2 + e3 ** 2))
    return dict(n=n, E1=e1, E2=e2, E3=e3, Cov=cov, rho=rho, alpha=alpha,
                E2_orth=orth, gain=orth - e2,
                share_E1=100 * e1 ** 2 / (e1 ** 2 + e3 ** 2))


def boot_rho(df, nboot=1000, seed=42):
    """Station-clustered bootstrap (months within a station are not independent), returning the 95% CI of rho / alpha"""
    rng = np.random.default_rng(seed)
    ids = df['id'].unique()
    out = {'rho': [], 'alpha': []}
    for _ in range(nboot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        parts = [df[df['id'] == i] for i in pick]
        o = np.concatenate([p['obs'].values for p in parts])
        r = np.concatenate([p['smci'].values for p in parts])
        m = np.concatenate([p['model'].values for p in parts])
        d = decomp(o, r, m)
        if d:
            out['rho'].append(d['rho'])
            out['alpha'].append(d['alpha'])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in out.items() if len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default=os.path.join(OUTDIR, 'truth_chain_L2_monthly.csv'))
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--min_n', type=int, default=8, help='Minimum number of months per station')
    ap.add_argument('--out_suffix', default='', help='output filename suffix (to distinguish CPU/GPU versions)')
    args = ap.parse_args()
    SUF = ('_' + args.out_suffix) if args.out_suffix else ''

    P('=== 19 error transfer decomposition (Gate experiment E1)===')
    df = pd.read_csv(args.pair, dtype={'id': str})
    P('paired samples %d rows, %d stations, model %s, layer %s'
      % (len(df), df['id'].nunique(),
         ','.join(sorted(df.cfg.unique())), ','.join(sorted(df.layer.unique()))))

    rows, st_rows = [], []
    for (cfg, lay), g in df.groupby(['cfg', 'layer']):
        o = g['obs'].values.astype(float)
        r = g['smci'].values.astype(float)
        m = g['model'].values.astype(float)

        d = decomp(o, r, m)
        if not d:
            continue
        ci = boot_rho(g, nboot=args.nboot)
        d.update(cfg=cfg, layer=lay, scope='pooled', debias=False,
                 rho_lo=ci.get('rho', (np.nan, np.nan))[0],
                 rho_hi=ci.get('rho', (np.nan, np.nan))[1],
                 alpha_lo=ci.get('alpha', (np.nan, np.nan))[0],
                 alpha_hi=ci.get('alpha', (np.nan, np.nan))[1])
        rows.append(d)

        du = decomp(o, r, m, debias=True)
        if du:
            du.update(cfg=cfg, layer=lay, scope='pooled', debias=True)
            rows.append(du)

        # per station
        for sid, gs in g.groupby('id'):
            ds = decomp(gs['obs'].values.astype(float),
                        gs['smci'].values.astype(float),
                        gs['model'].values.astype(float))
            if ds and ds['n'] >= args.min_n:
                # the label bias at that station (systematic deviation of SMCI relative to station observations), used to test
                # "whether the model is correcting the systematic bias of the label" -- if larger |bias_L1| gives larger α,
                # then the model captures not random noise but the structural error of the label
                ds['bias_L1'] = float(np.nanmean(gs['smci'].values.astype(float)
                                                 - gs['obs'].values.astype(float)))
                ds['sd_L1'] = float(np.nanstd(gs['smci'].values.astype(float)
                                              - gs['obs'].values.astype(float)))
                # mean deviation magnitude of the model relative to the label (its sign should be opposite to the label bias)
                ds['d_model'] = float(np.nanmean(gs['model'].values.astype(float)
                                                 - gs['smci'].values.astype(float)))
                ds.update(cfg=cfg, layer=lay, id=sid)
                st_rows.append(ds)

    R = pd.DataFrame(rows)
    ST = pd.DataFrame(st_rows)
    R.to_csv(os.path.join(OUTDIR, 'err_transfer_region%s.csv' % SUF),
             index=False, encoding='utf-8-sig')
    ST.to_csv(os.path.join(OUTDIR, 'err_transfer_station%s.csv' % SUF),
              index=False, encoding='utf-8-sig')

    # ---- Main table
    P('')
    P('--- Error budget (pooled, RMSE convention, identity holds strictly)---')
    P('%-7s %-5s %6s %8s %8s %8s %11s %8s %8s %7s  %s'
      % ('cfg', 'layer', 'n', 'E1 label', 'E2 output', 'E3 reproduction',
         'Cov', 'rho', 'alpha', 'E1 share%', 'rho 95%CI'))
    for _, r in R[(R.scope == 'pooled') & (~R.debias)].iterrows():
        P('%-7s %-5s %6d %8.5f %8.5f %8.5f %11.3e %+8.3f %+8.3f %7.1f  [%+.3f, %+.3f]'
          % (r.cfg, r.layer, r.n, r.E1, r.E2, r.E3, r.Cov, r.rho, r.alpha,
             r.share_E1, r.rho_lo, r.rho_hi))

    P('')
    P('--- De-biased convention ubRMSE (each series de-meaned separately, identity approximate)---')
    P('%-7s %-5s %8s %8s %8s %11s %8s %8s'
      % ('cfg', 'layer', 'E1', 'E2', 'E3', 'Cov', 'rho', 'alpha'))
    for _, r in R[(R.scope == 'pooled') & (R.debias)].iterrows():
        P('%-7s %-5s %8.5f %8.5f %8.5f %11.3e %+8.3f %+8.3f'
          % (r.cfg, r.layer, r.E1, r.E2, r.E3, r.Cov, r.rho, r.alpha))

    # ---- Significance test
    P('')
    P('--- Criterion: is Cov significantly < 0 (H0: the model is just a noisy copy of the label)---')
    try:
        from scipy import stats
        for (cfg, lay), g in ST.groupby(['cfg', 'layer']):
            v = g['rho'].dropna().values
            a = g['alpha'].dropna().values
            if len(v) < 10:
                continue
            w = stats.wilcoxon(v, alternative='less')
            wa = stats.wilcoxon(a, alternative='greater')
            P('  %-7s %-5s nsta=%3d  median rho %+.3f (negative %d/%d, p=%.4g) | '
              'alpha median %+.3f (positive %d/%d, p=%.4g)'
              % (cfg, lay, len(v), np.median(v), int((v < 0).sum()), len(v),
                 w.pvalue, np.median(a), int((a > 0).sum()), len(a), wa.pvalue))
    except Exception as e:
        P('  test failed: %r' % e)

    # ---- Supplementary test: is the model correcting "the systematic bias of the label" or random noise?
    P('')
    P('--- Supplementary: station-level regression of "label bias -> model deviation direction" ---')
    P('  If the model is just a noisy copy of the label, d_model is unrelated to bias_L1 (slope ≈ 0);')
    P('  If the model is correcting bias, d_model ≈ -alpha * bias_L1 (slope significantly negative).')
    try:
        from scipy import stats
        P('  %-7s %-5s %10s %10s %10s' % ('cfg', 'layer', 'slope', 'p', 'Spearman'))
        for (cfg, lay), g in ST.groupby(['cfg', 'layer']):
            b = g['bias_L1'].values
            d = g['d_model'].values
            m = np.isfinite(b) & np.isfinite(d)
            if m.sum() < 10:
                continue
            lr = stats.linregress(b[m], d[m])
            sp = stats.spearmanr(b[m], d[m])
            P('  %-7s %-5s %+10.4f %10.3g %10.3f (p=%.3g)'
              % (cfg, lay, lr.slope, lr.pvalue, sp.statistic, sp.pvalue))
    except Exception as e:
        P('  regression failed: %r' % e)

    # ---- Gate verdict
    P('')
    P('--- Gate verdict ---')
    ok = 0
    tot = 0
    for _, r in R[(R.scope == 'pooled') & (~R.debias)].iterrows():
        tot += 1
        good = (r.rho < 0) and (r.rho_hi < 0)
        ok += int(good)
        P('  %-7s %-5s rho=%+.3f CI upper bound=%+.3f  -> %s'
          % (r.cfg, r.layer, r.rho, r.rho_hi,
             'PASS (Cov significantly negative)' if good else 'FAIL'))
    P('  Total PASS %d/%d' % (ok, tot))
    P('  Gate pass condition: ≥1 configuration with rho<0 whose bootstrap 95%% CI upper bound < 0')

    # ---- Figure
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        Rm = R[(R.scope == 'pooled') & (~R.debias)]
        labels = ['%s\n%s' % (a, b) for a, b in zip(Rm.cfg, Rm.layer)]
        x = np.arange(len(Rm))
        w = 0.26
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        ax[0].bar(x - w, Rm.E1, w, color='#7F77DD', label='E1 label vs truth')
        ax[0].bar(x, Rm.E2, w, color='#378ADD', label='E2 output vs truth')
        ax[0].bar(x + w, Rm.E3, w, color='#EF9F27', label='E3 output vs label')
        ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
        ax[0].set_ylabel('RMSE (m3/m3)')
        ax[0].set_title('Magnitude of the three error layers (same month, same station pairing)')
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')

        ax[1].bar(x, Rm.alpha, color='#1D9E75')
        ax[1].errorbar(x, Rm.alpha,
                       yerr=[Rm.alpha - Rm.alpha_lo, Rm.alpha_hi - Rm.alpha],
                       fmt='none', ecolor='#444441', capsize=3, lw=1)
        ax[1].axhline(0, color='k', lw=.8)
        ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
        ax[1].set_title('Label-error elimination rate α (>0 = the model is correcting bias)')
        ax[1].grid(alpha=.3, axis='y')

        for (cfg, lay), g in ST.groupby(['cfg', 'layer']):
            ax[2].hist(g['alpha'].dropna(), bins=20, alpha=.55,
                       label='%s %s' % (cfg, lay))
        ax[2].axvline(0, color='k', lw=.8, ls='--')
        ax[2].set_xlabel('per-station α')
        ax[2].set_ylabel('station count')
        ax[2].set_title('Distribution of per-station α')
        ax[2].legend(fontsize=7)

        plt.tight_layout()
        fp = os.path.join(FIGDIR, 'err_transfer_R3%s.png' % SUF)
        plt.savefig(fp, dpi=160)
        plt.close()
        P('WROTE %s' % fp)
    except Exception as e:
        P('figure failed: %r' % e)

    report('19_err_transfer', LOG)


if __name__ == '__main__':
    main()
