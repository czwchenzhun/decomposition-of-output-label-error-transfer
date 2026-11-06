# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""16_model_vs_truth.py -- the "model vs truth" verification chain L2

L1 (15) only answered "how credible is the reference field". L2 answers the question that is really being asked:
For the extrapolated soil moisture, **relative to ground observations**, how large is the error?

Approach (three-way comparison under the same convention, to avoid talking past each other)
------------------------------------
  Truth   = station oven-drying monthly values (China 732 stations, 85 stations in the R0 North China window)
  Reference field = SMCI (same grid cell, same period, same monthly averaging)
  Model   = free-running rollout prediction from 13 (K=30 days, no SMCI is seen during the prediction period)

All three are compared after the same step of "grid-cell daily values -> monthly average -> pairing with station monthly values", therefore:
  * model vs truth   -> true error of the extrapolated output
  * reference field vs truth -> error of the reference field itself (L1 made concrete for this period)
  * model vs reference field -> the existing L3 convention, used to verify that "the model has not drifted"

Key: station truth **never participated in training** (the model has only seen meteorological forcing + SMCI),
so this is a strictly out-of-sample test, not goodness of fit.

Sequence-reconstruction convention (consistent with SeqSet in 13)
------------------------------------
  pred flattened index i = k(start index) * N + c(grid-subset position)
  Step t (0-based) corresponds to time starts[k] + t + 1
  In the test period stride=K and sequences do not overlap; here we still accumulate and average in an overlap-additive way, compatible with any stride

Output (02_Results_Data/truth_chain/)
----------------------------------------
  truth_chain_L2_station.csv   metrics per station × convention (model/SMCI) × layer × lead time
  truth_chain_L2_region.csv    Regional summary (median + IQR + station count)
  truth_chain_L2_monthly.csv   Monthly paired samples
  L2_fig_*.png                 Figures
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ROOT, OUT

OUTDIR = TRUTH
FIGDIR = REPRO + '/process_figures/17_truth_chain'
os.makedirs(FIGDIR, exist_ok=True)

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def metrics(o, s):
    o = np.asarray(o, float)
    s = np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    n = len(o)
    if n < 3:
        return dict(n=n, R=np.nan, bias=np.nan, rmse=np.nan, ubRMSE=np.nan,
                    obs_std=np.nan, sim_std=np.nan)
    b = float(np.mean(s - o))
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    ub = float(np.sqrt(np.mean(((s - b) - o) ** 2)))
    sd_o, sd_s = float(np.std(o)), float(np.std(s))
    R = float(np.corrcoef(o, s)[0, 1]) if (sd_o > 0 and sd_s > 0) else np.nan
    return dict(n=n, R=R, bias=b, rmse=rmse, ubRMSE=ub,
                obs_std=sd_o, sim_std=sd_s)


def daily_from_seq(pred, starts, Nsub, Tfull):
    """pred (B,K,2) -> (Tfull,Nsub,2) daily values; overlapping sequences are averaged"""
    ns = len(starts)
    pr = pred.reshape(ns, Nsub, pred.shape[1], 2)
    acc = np.zeros((Tfull, Nsub, 2), 'float64')
    cnt = np.zeros((Tfull, Nsub), 'float64')
    for t in range(pred.shape[1]):
        ti = starts + t + 1
        m = ti < Tfull
        if not m.any():
            continue
        np.add.at(acc, ti[m], pr[m][:, :, t, :])
        np.add.at(cnt, ti[m], 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = acc / np.maximum(cnt, 1)[:, :, None]
    out[cnt < 1] = np.nan
    return out, cnt


def month_index(time):
    tv = np.asarray(time, dtype='datetime64[D]')
    y = tv.astype('datetime64[Y]').astype(int) + 1970
    m = (tv.astype('datetime64[M]').astype(int) % 12) + 1
    return y, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz',
                    default=os.path.join(BENCH, 'dl_R0_northchina_nc256_20020101_20131231_sta.npz'),
                    help='11 produced dl_*.npz (included in the package, can be omitted)')
    ap.add_argument('--tag', default='_sta', help='the --tag of 13, used to locate pred_*.npz')
    ap.add_argument('--region', default='R0_northchina')
    ap.add_argument('--cfgs', default='HYBNS,LSTM')
    ap.add_argument('--horizons', default='1,7,15,30')
    ap.add_argument('--min_months', type=int, default=6)
    ap.add_argument('--station_cells', default='')
    ap.add_argument('--out_suffix', default='',
                    help='output filename suffix, to distinguish CPU/GPU and other weight versions')
    args = ap.parse_args()
    SUF = ('_' + args.out_suffix) if args.out_suffix else ''

    HZ = [int(x) for x in args.horizons.split(',')]
    P('=== 16 model vs truth (L2) ===')
    P('Data %s' % os.path.basename(args.npz))

    d = np.load(args.npz, allow_pickle=True)
    Y = d['Y']                       # (T,Nsub,2) SMCI reference field
    time = d['time']
    cells = d['cells']               # flattened index of the subset grid cells in the full-region grid
    T, Nsub, _ = Y.shape
    P('T=%d  grid-cell subset n=%d  time %s .. %s' % (T, Nsub, time[0], time[-1]))

    # Subset position <-> (iy,ix)
    cfull = np.load(os.path.join(MULTI, args.region,
                                 'samples', 'cells.npz'))
    key2flat = {(int(cfull['iy'][n]), int(cfull['ix'][n])): n
                for n in range(len(cfull['iy']))}
    pos2flat = {p: int(cells[p]) for p in range(Nsub)}

    # ---- Station table
    sta = pd.read_csv(os.path.join(OUTDIR, 'stations_in_windows.csv'),
                      dtype={'id': str})
    sta = sta[sta.region == args.region].copy()
    obs = pd.read_csv(os.path.join(OUTDIR, 'truth_chain_L1_monthly.csv'),
                      dtype={'id': str})
    obs = obs[obs.region == args.region]
    P('stations inside windows: %d; station monthly-value pairs: %d rows' % (len(sta), len(obs)))

    sta['flat'] = [key2flat.get((int(r.iy), int(r.ix)), -1) for _, r in sta.iterrows()]
    flat2pos = {v: k for k, v in pos2flat.items()}
    sta['pos'] = [flat2pos.get(int(f), -1) for f in sta['flat']]
    n_in = int((sta.pos >= 0).sum())
    P('of which %d stations fall in the model grid-cell subset (the other %d are not among the randomly drawn 256 cells)'
      % (n_in, len(sta) - n_in))
    if n_in < 5:
        P('!! too few stations hit; please re-draw grid cells with --station_cells')
        from common import report
        report('16_model_vs_truth', LOG)
        return

    # ---- Load daily predictions of each model
    preds = {}
    for cfg in args.cfgs.split(','):
        fp = os.path.join(BENCH, 'pred_%s%s.npz' % (cfg, args.tag))
        if not os.path.exists(fp):
            P('!! missing %s (add --savepred when running 13)' % fp)
            continue
        z = np.load(fp)
        starts = z['starts']
        K = int(z['K'])
        daily, cnt = daily_from_seq(z['pred'], starts, int(z['N']), T)
        preds[cfg] = (daily, cnt)
        P('loaded %s: %d sequences, K=%d, reconstructed daily values %.1f%% valid'
          % (cfg, len(starts), K, 100.0 * np.mean(cnt > 0)))
    if not preds:
        from common import report
        report('16_model_vs_truth', LOG)
        return

    # Daily values for each lead time: keep only the step corresponding to that horizon (step h-1)
    # Note: h=1 is "1-day lead", h=30 is "30-day lead", used to see how error grows with lead time
    #
    # Important: in the test period stride=K and windows **do not overlap**, so each calendar day falls on exactly one lead time.
    # If cross-tabulated by month × lead time, each month has only 1 day of samples, which is extremely noisy.
    # Therefore we use two conventions:
    #   h=0  → the complete daily series mixing all lead times (one value per day), used for **monthly-scale** comparison with station monthly values (primary convention)
    #   h>0  → **daily-scale** metrics grouped by lead time, pooled over the whole region / multiple stations, used to plot the error-decay curve
    preds_h = {}
    for cfg, (daily, cnt) in preds.items():
        preds_h[(cfg, 0)] = daily
        for h in HZ:
            fp = os.path.join(BENCH, 'pred_%s%s.npz' % (cfg, args.tag))
            z = np.load(fp)
            pr = z['pred'].reshape(len(z['starts']), int(z['N']), int(z['K']), 2)
            j = min(h, int(z['K'])) - 1
            ti = z['starts'] + j + 1
            m = ti < T
            acc = np.full((T, Nsub, 2), np.nan)
            acc[ti[m]] = pr[m][:, :, j, :]
            preds_h[(cfg, h)] = acc

    # ---- Per-station pairing (monthly scale uses only the h=0 all-lead mixed series)
    yy, mm = month_index(time)
    rows = []
    pair_rows = []
    for _, r in sta.iterrows():
        if r['pos'] < 0:
            continue
        p = int(r['pos'])
        so = obs[obs.id == r['id']]
        if len(so) == 0:
            continue
        df = pd.DataFrame(dict(year=yy, month=mm,
                               smci10=Y[:, p, 0], smci30=Y[:, p, 1]))
        for (cfg, h), acc in preds_h.items():
            if h != 0:
                continue
            df['p10'] = acc[:, p, 0]
            df['p30'] = acc[:, p, 1]
            # The reference field must be monthly-averaged over **the same set of days**, otherwise the model covers only the test period,
            # while the reference field uses the full period, making the two incomparable (this is the step most likely to produce a wrong conclusion)
            v = np.isfinite(acc[:, p, 0]) & np.isfinite(acc[:, p, 1])
            df['smci10'] = np.where(v, Y[:, p, 0], np.nan)
            df['smci30'] = np.where(v, Y[:, p, 1], np.nan)
            g = df.groupby(['year', 'month'], as_index=False).mean(numeric_only=True)
            for dep in ('10', '30'):
                # 15's pairing table is long-form: 10cm/30cm are distinguished by the layer column, with columns named obs / smci
                so_l = so[so.layer == dep + 'cm']
                if len(so_l) == 0:
                    continue
                j = so_l.merge(g, on=['year', 'month'], how='inner')
                j = j[np.isfinite(j['obs'])]
                o = j['obs'].values
                for src in ('p', 'smci'):
                    s = j[src + dep].values
                    mm_ = np.isfinite(s)
                    if mm_.sum() < 3:
                        continue
                    mt = metrics(o[mm_], s[mm_])
                    rows.append(dict(id=r['id'], region=args.region, layer=dep + 'cm',
                                     source=('MODEL:' + cfg) if src == 'p' else 'SMCI',
                                     horizon=h, gdist=float(r['gdist']), **mt))
                    if src == 'p':
                        pair_rows.append(pd.DataFrame(dict(
                            id=r['id'], year=j['year'].values[mm_],
                            month=j['month'].values[mm_], layer=dep + 'cm',
                            cfg=cfg, horizon=h,
                            obs=o[mm_], model=s[mm_],
                            smci=j['smci' + dep].values[mm_])))

    ST = pd.DataFrame(rows)
    PAIR = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    ST.to_csv(os.path.join(OUTDIR, 'truth_chain_L2_station%s.csv' % SUF),
              index=False, encoding='utf-8-sig')
    if len(PAIR):
        PAIR.to_csv(os.path.join(OUTDIR, 'truth_chain_L2_monthly%s.csv' % SUF),
                    index=False, encoding='utf-8-sig')
    P('')
    P('per-station metrics: %d rows; monthly pairs: %d rows' % (len(ST), len(PAIR)))
    P('Note: stations are monthly values, so lead-time-wise (h>0) truth-comparison samples are unavoidably too sparse (test windows do not overlap,'
      'about 1 day per month per lead time), so here we report only monthly-scale metrics for the all-lead mixed (h=0) case;'
      'For the lead-time decay curve see the L3 table in 13.')

    # ---- Regional summary
    P('')
    P('--- L2 summary (median of per-station metrics; nsta=number of valid stations)---')
    P('%-12s %-6s %6s %5s %6s %8s %8s %8s' % (
        'source', 'layer', 'horizon', 'nsta', 'R', 'bias', 'RMSE', 'ubRMSE'))
    summ = []
    for (src, lay, h), g in ST.groupby(['source', 'layer', 'horizon']):
        g = g[g.n >= args.min_months]
        if len(g) < 5:
            continue
        row = dict(region=args.region, source=src, layer=lay, horizon=h,
                   nsta=len(g),
                   R_med=float(g.R.median()),
                   R_iqr=float(g.R.quantile(.75) - g.R.quantile(.25)),
                   bias_med=float(g.bias.median()),
                   rmse_med=float(g.rmse.median()),
                   ub_med=float(g.ubRMSE.median()),
                   ub_iqr=float(g.ubRMSE.quantile(.75) - g.ubRMSE.quantile(.25)),
                   nobs=int(g.n.sum()))
        summ.append(row)
        P('%-12s %-6s %6d %5d %6.3f %+8.4f %8.4f %8.4f' % (
            src, lay, h, row['nsta'], row['R_med'], row['bias_med'],
            row['rmse_med'], row['ub_med']))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUTDIR, 'truth_chain_L2_region%s.csv' % SUF),
             index=False, encoding='utf-8-sig')

    # ---- Key criterion: model error vs reference-field error
    P('')
    P('--- Criterion: is the error of the extrapolated output relative to truth significantly larger than that of the reference field itself?---')
    for lay in ('10cm', '30cm'):
        for h in sorted(ST.horizon.unique()):
            base = S[(S.source == 'SMCI') & (S.layer == lay) & (S.horizon == h)]
            if len(base) == 0:
                continue
            b0 = float(base.ub_med.iloc[0])
            bR = float(base.R_med.iloc[0])
            for src in S.source.unique():
                if src == 'SMCI':
                    continue
                g = S[(S.source == src) & (S.layer == lay) & (S.horizon == h)]
                if len(g) == 0:
                    continue
                du = float(g.ub_med.iloc[0]) - b0
                dR = float(g.R_med.iloc[0]) - bR
                P('  %-6s h=%-3d %-10s  ubRMSE %+0.4f (%+.1f%%)   R %+0.3f  vs SMCI(ub %.4f / R %.3f)'
                  % (lay, h, src, du, 100 * du / b0, dR, b0, bR))

    # ---- Figure
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        # A) Distribution of per-station ubRMSE (monthly scale, all-lead mixed)
        lay = '10cm'
        h0 = 0
        g = ST[(ST.layer == lay) & (ST.horizon == h0) & (ST.n >= args.min_months)]
        srcs = sorted(g.source.unique())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].boxplot([g[g.source == s].ubRMSE.dropna().values for s in srcs],
                      tick_labels=[s.replace('MODEL:', '') for s in srcs],
                      showmeans=True)
        ax[0].set_title('%s per-station ubRMSE vs station observations (%s)' % (lay, args.region))
        ax[0].set_ylabel('ubRMSE (m3/m3)')
        ax[0].grid(alpha=.3, axis='y')
        # B) Monthly-scale scatter: model / reference field each vs observations
        pl = PAIR[(PAIR.layer == lay)]
        cfg0 = args.cfgs.split(',')[0]
        pl = pl[pl.cfg == cfg0] if 'cfg' in pl.columns else pl
        if len(pl):
            ax[1].scatter(pl['obs'], pl['smci'], s=6, alpha=.35,
                          color='steelblue', label='SMCI reference field')
            ax[1].scatter(pl['obs'], pl['model'], s=6, alpha=.35,
                          color='crimson', label='model %s' % cfg0)
            lim = [float(np.nanmin(pl['obs'])), float(np.nanmax(pl['obs']))]
            ax[1].plot(lim, lim, 'k--', lw=.8)
            ax[1].set_xlabel('station observations θ (m3/m3)')
            ax[1].set_ylabel('grid-cell monthly mean θ (m3/m3)')
            ax[1].set_title('%s monthly-scale comparison (n=%d)' % (lay, len(pl)))
            ax[1].legend(fontsize=8)
            ax[1].grid(alpha=.3)
        plt.tight_layout()
        fp = os.path.join(FIGDIR, 'L2_station_%s_%s%s.png' % (args.region, lay, SUF))
        plt.savefig(fp, dpi=160)
        plt.close()
        P('WROTE %s' % fp)
    except Exception as e:
        P('figure failed: %r' % e)

    from common import report
    report('16_model_vs_truth', LOG)


if __name__ == '__main__':
    main()
