# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""57: M4 parameter-count check + M5 persistence (PERS) brought into the truth chain + m13 traceability of the PERS h1 RMSE

Background (found in the fourth-round self-review):
  M4  The parameter counts in Table tab:cap (LSTM 5.27e4 / HYBNS 9.25e4 / HYBLSTNS 2.08e5)
      are not found in the archived JSON; moreover, the same architecture under different regimes (LSTM and LSTMNS,
      HYBLST and HYBLSTNS) must have identical parameter counts, so the numbers in the main text are self-contradictory.
  M5  Persistence is the most natural zero-parameter predictor (set here as re-initialisation with the product every 30 days,
      30-day open-loop rollout), but it never entered the truth chain, so (rho, x, E2/E1) was unknown.
  m13  The PERS h=1 RMSE=0.0094 in Table 3 does not match 0.00861 in the archived main JSON; traceability is needed.

PERS is an analytic baseline (in 13, `pers_rmse(S0te, Yte)`, S0=Y[s] is the label value at the window start),
it requires no training: the predicted series = the starting SMCI value held constant for K days. It can thus be constructed directly from the npz,
then the error triple is computed with exactly the same pairing and convention as 16/25.
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: §4.1 persistence baseline PERS (analytic, zero-parameter, zero-training) and main-text Tables 3/4
#

import os
import glob
import json
import io
import numpy as np
import pandas as pd

from common import OUT, report

B = TRUTH
BENCH = BENCH
# MULTI is provided by paths.py: the runtime copy of 02_Results_Data/truth_chain/multi (including samples/cells.npz)

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


REG = [
    ('R0_northchina', 'dl_R0_northchina_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG.npz'),
    ('R1_yangtze', 'dl_R1_yangtze_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG_R1_yangtze_a.npz'),
    ('R3_northeast', 'dl_R3_northeast_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG_R3_northeast_a.npz'),
    ('R2_loess', 'dl_R2_loess_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG_R2_loess_a.npz'),
    ('R4_longzhong', 'dl_R4_longzhong_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG_R4_longzhong_a.npz'),
    ('R5_sichuan', 'dl_R5_sichuan_nc256_20020101_20131231_sta.npz',
     'pred_HYBNS_staG_R5_sichuan_a.npz'),
]
CN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
      'R3_northeast': 'Northeast China black soil', 'R2_loess': 'Loess Plateau',
      'R4_longzhong': 'Longzhong loess', 'R5_sichuan': 'Sichuan Basin'}


def budget(o, r, m):
    """Fully consistent with 25_within_station.budget: Cov is the uncentred second moment"""
    d1, d2, d3 = r - o, m - o, m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    return dict(E1=E1, E2=E2, E3=E3, Cov=Cov,
                rho=(Cov / (E1 * E3) if E1 * E3 > 0 else 0.),
                alpha=-Cov / (E1 ** 2))


def month_index(time):
    tv = np.asarray(time, dtype='datetime64[D]')
    y = tv.astype('datetime64[Y]').astype(int) + 1970
    m = (tv.astype('datetime64[M]').astype(int) % 12) + 1
    return y, m


def daily_from_seq(pred, starts, Nsub, Tfull):
    """Fully consistent with 16: pred (B*N,K,2) -> (Tfull,Nsub,2), overlapping windows averaged"""
    ns = len(starts)
    pr = pred.reshape(ns, Nsub, pred.shape[1], 2)
    acc = np.zeros((Tfull, Nsub, 2), 'float64')
    cnt = np.zeros((Tfull, Nsub), 'float64')
    for t in range(pred.shape[1]):
        ti = starts + t + 1
        msk = ti < Tfull
        if not msk.any():
            continue
        np.add.at(acc, ti[msk], pr[msk][:, :, t, :])
        np.add.at(cnt, ti[msk], 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = acc / np.maximum(cnt, 1)[:, :, None]
    out[cnt < 1] = np.nan
    return out, cnt


# =====================================================================
# [A] parameter-count check
# =====================================================================
def part_capacity():
    P('=' * 78)
    P('[A] Parameter-count check: the _staG series actually used in the paper (including per-region weights)')
    P('=' * 78)
    want = ['LSTM', 'LSTMNS', 'LPMIX', 'LPMIXNS', 'HYBNS', 'HYBLST', 'HYBLSTNS']
    agg = {}
    for f in glob.glob(os.path.join(BENCH, 'dl_hybrid_result*.json')):
        try:
            j = json.load(io.open(f, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(j, list):
            continue
        tag = os.path.basename(f)[len('dl_hybrid_result'):-5]
        for r in j:
            if isinstance(r, dict) and r.get('nparam'):
                agg.setdefault(tag, {})[r['cfg']] = int(r['nparam'])
    keep = [t for t in sorted(agg) if t.lstrip('_').startswith('staG')]
    P('%-30s %s' % ('training label', '  '.join('%-9s' % c for c in want)))
    P('-' * 78)
    for t in keep:
        d = agg[t]
        P('%-30s %s' % (t, '  '.join('%-9s' % (d.get(c, '-')) for c in want)))
    # Paper convention: main weights of the four-region _staG
    main = agg.get('_staG', {})
    vals = {}
    for c in want:
        vs = set()
        for t in keep:
            if c in agg[t]:
                vs.add(agg[t][c])
        vals[c] = sorted(vs)
    P('')
    P('Set of nparam values of each configuration across all _staG* runs (should be unique, otherwise the variants differ):')
    for c in want:
        flag = '' if len(vals[c]) <= 1 else '  <-- not unique!'
        P('  %-10s %s%s' % (c, vals[c], flag))
    P('')
    P('The main-text convention (Table tab:cap) should read:')
    tab = {}
    for c in ['LSTM', 'LSTMNS', 'LPMIX', 'LPMIXNS', 'HYBNS', 'HYBLST', 'HYBLSTNS']:
        v = vals[c][0] if len(vals[c]) == 1 else (main.get(c) or vals[c][0])
        tab[c] = v
        P('  %-10s %d  (%.3f x 10^5)' % (c, v, v / 1e5))
    P('')
    P('---- Capacity multiples (corrected) ----')
    P('  SOFT(LPMIX) / FREE(LSTM)      = %.3f times   diff %.3e' %
      (tab['LPMIX'] / tab['LSTM'], tab['LPMIX'] - tab['LSTM']))
    P('  SOFT(LPMIX) / HARD(HYBNS)     = %.3f times' %
      (tab['LPMIX'] / tab['HYBNS']))
    P('  HARD(HYBNS) / FREE(LSTM)      = %.3f times' %
      (tab['HYBNS'] / tab['LSTM']))
    P('  HYBLSTNS(=HYBLST)             = %d' % tab['HYBLST'])
    P('')
    P('---- Comparison with the original main-text values (the main text uses the erroneous values flagged by M4) ----')
    old = {'LSTM': 52700, 'LSTMNS': 79600, 'LPMIX': 212000, 'LPMIXNS': 212000,
           'HYBNS': 92500, 'HYBLST': 212000, 'HYBLSTNS': 208000}
    for c in ['LSTM', 'LSTMNS', 'LPMIX', 'HYBNS', 'HYBLSTNS']:
        P('  %-10s main text %-8d  archive %-8d  %s' %
          (c, old[c], tab[c], 'OK' if old[c] == tab[c] else '*** mismatch ***'))
    P('  Main-text multiples: FREE-SOFT 4.0x / SOFT-HARD 2.3x / extra parameters 1.6e5 / HARD:FREE 1.75x')
    P('  Actual multiples: FREE-SOFT %.2fx / SOFT-HARD %.2fx / extra parameters %.2e / HARD:FREE %.2fx' %
      (tab['LPMIX'] / tab['LSTM'], tab['LPMIX'] / tab['HYBNS'],
       tab['LPMIX'] - tab['LSTM'], tab['HYBNS'] / tab['LSTM']))
    return tab


# =====================================================================
# [B] PERS brought into the truth chain
# =====================================================================
def part_pers():
    P('')
    P('=' * 78)
    P('[B] Persistence (PERS) brought into the truth chain: analytic construction + exactly the same monthly pairing as 16')
    P('=' * 78)
    sta_all = pd.read_csv(os.path.join(B, 'stations_in_windows.csv'),
                          dtype={'id': str})
    obs_all = pd.read_csv(os.path.join(B, 'truth_chain_L1_monthly.csv'),
                          dtype={'id': str})
    pairs, rows = [], []
    for region, fnpz, fprd in REG:
        fp = os.path.join(BENCH, fnpz)
        if not os.path.exists(fp):
            P('!! missing %s' % fp)
            continue
        d = np.load(fp, allow_pickle=True)
        Y = d['Y']
        time = d['time']
        cells = d['cells']
        T, Nsub, _ = Y.shape
        z = np.load(os.path.join(BENCH, fprd))
        starts = z['starts']
        K = int(z['K'])
        N = int(z['N'])
        if N != Nsub:
            P('!! %s: pred N=%d != npz Nsub=%d' % (region, N, Nsub))
        # ---- PERS series: starting label value held constant for K days
        s0 = Y[starts]                                   # (nw, N, 2)
        pr = np.repeat(s0[:, :, None, :], K, axis=2)     # (nw, N, K, 2)
        pers_flat = pr.reshape(-1, K, 2)
        daily, cnt = daily_from_seq(pers_flat, starts, Nsub, T)
        # ---- Same grid-cell mapping as 16
        cfull = np.load(os.path.join(MULTI, region, 'samples', 'cells.npz'))
        key2flat = {(int(cfull['iy'][n]), int(cfull['ix'][n])): n
                    for n in range(len(cfull['iy']))}
        pos2flat = {p: int(cells[p]) for p in range(Nsub)}
        flat2pos = {v: k for k, v in pos2flat.items()}
        sta = sta_all[sta_all.region == region].copy()
        sta['flat'] = [key2flat.get((int(r.iy), int(r.ix)), -1)
                       for _, r in sta.iterrows()]
        sta['pos'] = [flat2pos.get(int(f), -1) for f in sta['flat']]
        obs = obs_all[obs_all.region == region]
        yy, mm = month_index(time)
        nrow = 0
        for _, r in sta.iterrows():
            p = int(r['pos'])
            if p < 0:
                continue
            so = obs[obs.id == r['id']]
            if len(so) == 0:
                continue
            df = pd.DataFrame(dict(year=yy, month=mm,
                                   smci10=Y[:, p, 0], smci30=Y[:, p, 1]))
            df['p10'] = daily[:, p, 0]
            df['p30'] = daily[:, p, 1]
            v = np.isfinite(daily[:, p, 0]) & np.isfinite(daily[:, p, 1])
            df['smci10'] = np.where(v, Y[:, p, 0], np.nan)
            df['smci30'] = np.where(v, Y[:, p, 1], np.nan)
            g = df.groupby(['year', 'month'], as_index=False).mean(numeric_only=True)
            for dep in ('10', '30'):
                so_l = so[so.layer == dep + 'cm']
                if len(so_l) == 0:
                    continue
                j = so_l.merge(g, on=['year', 'month'], how='inner')
                j = j[np.isfinite(j['obs'])]
                o = j['obs'].values
                s = j['p' + dep].values
                sm = j['smci' + dep].values
                k = np.isfinite(s) & np.isfinite(sm)
                if k.sum() < 3:
                    continue
                pairs.append(pd.DataFrame(dict(
                    region=region, id=r['id'], year=j['year'].values[k],
                    month=j['month'].values[k], layer=dep + 'cm',
                    cfg='PERS', horizon=0,
                    obs=o[k], model=s[k], smci=sm[k])))
                nrow += int(k.sum())
        P('%-16s %s  stations in window %d, matched grid cells %d, PERS pairs %d rows'
          % (region, CN[region], len(sta), int((sta.pos >= 0).sum()), nrow))
        # ---- m13: traceability of the PERS h=1 RMSE (same convention as pers_rmse in 13)
        yi = Y[starts + 1]                    # (nw, N, 2) first-step target
        with np.errstate(invalid='ignore'):
            e1 = np.nanmean((s0 - yi) ** 2, axis=(0, 1))
        rows.append(dict(region=region, region_cn=CN[region],
                         p_h1_10cm=float(np.sqrt(e1[0])),
                         p_h1_30cm=float(np.sqrt(e1[1]))))
    PAIR = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    if len(PAIR):
        PAIR.to_csv(os.path.join(B, 'truth_chain_L2_monthly_pers.csv'),
                    index=False, encoding='utf-8-sig')
    P('')
    P('PERS pairs total %d rows, written to truth_chain_L2_monthly_pers.csv' % len(PAIR))

    # ---- Error triples under three conventions
    out = []
    for (region, lay), g in PAIR.groupby(['region', 'layer']):
        o = g['obs'].values.astype(float)
        r = g['smci'].values.astype(float)
        m = g['model'].values.astype(float)
        full = budget(o, r, m)                                  # P full convention
        vv = g.copy()
        for c in ['obs', 'smci', 'model']:
            vv[c] = vv[c] - vv.groupby('id')[c].transform('mean')
        wit = budget(vv.obs.values, vv.smci.values, vv.model.values)   # W within-station
        # G globally de-meaned (approximate diagnostic)
        glo = budget(o - o.mean(), r - r.mean(), m - m.mean())
        def pack(tag, b):
            return dict(region=region, region_cn=CN[region], layer=lay,
                        caliber=tag, n=len(g), nsta=g['id'].nunique(),
                        E1=b['E1'], E2=b['E2'], E3=b['E3'], rho=b['rho'],
                        x=b['E3'] / b['E1'], ratio=b['E2'] / b['E1'],
                        floor=float(np.sqrt(max(0., 1 - b['rho'] ** 2))),
                        alpha=b['alpha'],
                        lam=(-b['rho'] / (b['E3'] / b['E1'])
                             if b['E3'] > 0 else np.nan))
        out += [pack('P', full), pack('W', wit), pack('G', glo)]
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(B, 'pers_truthchain.csv'), index=False,
             encoding='utf-8-sig')
    P('')
    P('%-14s %-6s %-3s %5s %5s | %7s %7s %7s | %7s %7s %7s %7s'
      % ('region', 'layer', 'convention', 'n', 'station', 'E1', 'E3', 'E2', 'rho', 'x',
         'E2/E1', 'floor'))
    P('-' * 100)
    for _, r in R.iterrows():
        P('%-14s %-6s %-3s %5d %5d | %7.4f %7.4f %7.4f | %+7.3f %7.3f %7.4f %7.4f'
          % (r.region_cn, r.layer, r.caliber, r.n, r.nsta,
             r.E1, r.E3, r.E2, r.rho, r.x, r.ratio, r.floor))
    P('')
    P('Note: x=E3/E1; E2/E1<1 means better than label; floor=sqrt(1-rho^2) is the lower bound of E2/E1.')
    P('    W convention = pooled after de-meaning obs/smci/model simultaneously per station (primary convention of the main text).')
    P('')
    P('---- [C] traceability of the PERS h=1 RMSE (m13) ----')
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(B, 'pers_h1_rmse.csv'), index=False,
             encoding='utf-8-sig')
    for _, r in H.iterrows():
        P('  %-14s h1 RMSE: 10cm=%.5f  30cm=%.5f'
          % (r.region_cn, r.p_h1_10cm, r.p_h1_30cm))
    P('  Main-text Table tab:cap reports PERS h=1=0.0094; the archived main JSON gives 0.00861.')
    P('  Note: the h=1 RMSE is a pooled value "across all grid cells + all windows",'
      'a different convention from the per-station monthly values above; the two are not directly comparable;'
      'the main text should label the h=1 column explicitly as "grid-cell scale, daily scale, all-window pooled".')
    return PAIR, R


def main():
    P('=== 57: M4 parameter-count check + M5 PERS brought into the truth chain ===')
    part_capacity()
    part_pers()
    report('57_pers_capacity', LOG)


if __name__ == '__main__':
    main()
