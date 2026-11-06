# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""58: Per-configuration fingerprints of the damaged control (DAMAGED) vs analytic persistence (PERS)

Origin: after script 57 brought PERS into the truth chain, it was found that under the within-station convention for North China 10cm
PERS has E3=0.024584 / rho=-0.041691 / E2E1=1.140489,
which matches the main-text DAMAGED family means E3=0.024580 / rho=-0.041706 / E2E1=1.140439
to within 1e-5. The two sources are fully independent (one analytically constructed, one a real training product),
so it must be determined whether DAMAGED contains one configuration that "degenerates to persistence",
or all three configurations degenerate, or it is pure coincidence.

This script reports, per configuration:
  (a) daily scale: RMSE between prediction and PERS, std of the prediction along the 30-day rollout axis
  (b) monthly-scale within-station convention: E1 / E3 / rho / x / E2E1
"""
import os
import numpy as np
import pandas as pd

from common import OUT, report

B = TRUTH
BENCH = BENCH
# MULTI is provided by paths.py: the runtime copy of 02_Results_Data/truth_chain/multi (including samples/cells.npz)
REGION = 'R0_northchina'
NPZ = 'dl_R0_northchina_nc256_20020101_20131231_sta.npz'

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def budget(o, r, m):
    d1, d2, d3 = r - o, m - o, m - r
    E1 = float(np.sqrt(np.mean(d1 ** 2)))
    E2 = float(np.sqrt(np.mean(d2 ** 2)))
    E3 = float(np.sqrt(np.mean(d3 ** 2)))
    Cov = float(np.mean(d1 * d3))
    return dict(E1=E1, E2=E2, E3=E3, Cov=Cov,
                rho=(Cov / (E1 * E3) if E1 * E3 > 0 else 0.))


def month_index(time):
    tv = np.asarray(time, dtype='datetime64[D]')
    y = tv.astype('datetime64[Y]').astype(int) + 1970
    m = (tv.astype('datetime64[M]').astype(int) % 12) + 1
    return y, m


def daily_from_seq(pred, starts, Nsub, Tfull):
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


CFGS = [('PERS', None),
        ('wd0.01', 'pred_LSTM_wd0.01.npz'),
        ('wd0.03', 'pred_LSTM_wd0.03.npz'),
        ('degwd', 'pred_LSTM_degwd.npz'),
        ('deg5', 'pred_LSTM_deg5.npz'),
        ('deg10', 'pred_LSTM_deg10.npz'),
        ('deg32', 'pred_LSTM_deg32.npz'),
        ('LSTM', 'pred_LSTM_staG.npz'),
        ('HYBNS', 'pred_HYBNS_staG.npz')]


def main():
    P('=== 58: per-configuration fingerprints of the damaged control vs persistence ===')
    d = np.load(os.path.join(BENCH, NPZ), allow_pickle=True)
    Y = d['Y']
    time = d['time']
    cells = d['cells']
    T, Nsub, _ = Y.shape

    z0 = np.load(os.path.join(BENCH, 'pred_HYBNS_staG.npz'))
    starts = z0['starts']
    K = int(z0['K'])
    s0 = Y[starts]
    PERS = np.repeat(s0[:, :, None, :], K, axis=2)      # (nw,N,K,2)

    sta = pd.read_csv(os.path.join(B, 'stations_in_windows.csv'), dtype={'id': str})
    sta = sta[sta.region == REGION].copy()
    obs = pd.read_csv(os.path.join(B, 'truth_chain_L1_monthly.csv'), dtype={'id': str})
    obs = obs[obs.region == REGION]
    cfull = np.load(os.path.join(MULTI, REGION, 'samples', 'cells.npz'))
    key2flat = {(int(cfull['iy'][n]), int(cfull['ix'][n])): n
                for n in range(len(cfull['iy']))}
    pos2flat = {p: int(cells[p]) for p in range(Nsub)}
    flat2pos = {v: k for k, v in pos2flat.items()}
    sta['flat'] = [key2flat.get((int(r.iy), int(r.ix)), -1) for _, r in sta.iterrows()]
    sta['pos'] = [flat2pos.get(int(f), -1) for f in sta['flat']]
    yy, mm = month_index(time)

    P('')
    P('--- (a) daily-scale grid-cell convention: distance between prediction and persistence ---')
    P('%-8s %10s %12s %12s' % ('configuration', 'pred-PERS', 'std along K', 'global std'))
    P('-' * 48)
    for nm, fn in CFGS:
        if fn is None:
            pr = PERS
        else:
            z = np.load(os.path.join(BENCH, fn))
            pr = z['pred'].reshape(len(z['starts']), Nsub, K, 2)
            if len(z['starts']) != len(starts):
                P('  !! %s has a different window count, daily-scale comparison skipped' % nm)
                continue
        dif = float(np.sqrt(np.nanmean((pr - PERS) ** 2)))
        sdk = float(np.nanmean(np.nanstd(pr, axis=2)))
        P('%-8s %10.6f %12.6f %12.6f' % (nm, dif, sdk, float(np.nanstd(pr))))

    P('')
    P('--- (b) monthly-scale within-station convention (same convention as main-text Table tab:mlbase) ---')
    P('%-8s %-6s %6s | %8s %8s %8s %8s %8s' %
      ('configuration', 'layer', 'n', 'E1', 'E3', 'rho', 'x', 'E2/E1'))
    P('-' * 74)
    rows = []
    for nm, fn in CFGS:
        if fn is None:
            pr = PERS
        else:
            z = np.load(os.path.join(BENCH, fn))
            pr = z['pred'].reshape(len(z['starts']), Nsub, K, 2)
            st = z['starts']
        daily, _ = daily_from_seq(pr.reshape(-1, K, 2),
                                  starts if fn is None else st, Nsub, T)
        recs = []
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
                s = j['p' + dep].values
                sm = j['smci' + dep].values
                k = np.isfinite(s) & np.isfinite(sm)
                if k.sum() < 3:
                    continue
                recs.append(pd.DataFrame(dict(
                    sid=str(r['id']), layer=dep + 'cm',
                    obs=j['obs'].values[k], smci=sm[k], model=s[k])))
        A = pd.concat(recs, ignore_index=True)
        # W convention: de-mean obs/smci/model each by (station, layer)
        for c in ['obs', 'smci', 'model']:
            A[c] = A[c] - A.groupby(['sid', 'layer'])[c].transform('mean')
        out = []
        for lay in ['10cm', '30cm']:
            v = A[A.layer == lay]
            b = budget(v.obs.values, v.smci.values, v.model.values)
            out.append((lay, len(v), b))
            rows.append(dict(cfg=nm, layer=lay, n=len(v), E1=b['E1'],
                             E3=b['E3'], rho=b['rho'], x=b['E3'] / b['E1'],
                             ratio=b['E2'] / b['E1']))
        for lay, n, b in out:
            P('%-8s %-6s %6d | %8.5f %8.5f %8.5f %8.5f %8.5f' %
              (nm, lay, n, b['E1'], b['E3'], b['rho'],
               b['E3'] / b['E1'], b['E2'] / b['E1']))
    P('')
    P('Note: the table above is the G convention (globally de-meaned) approximate diagnostic; the main-text primary convention W (de-meaned by station)')
    P('    exact values are in pers_truthchain.csv and paper_families.csv output by 57.')
    P('')
    P('--- Interpretation of the conclusion ---')
    P('If the (E3, rho) of degwd is almost identical to PERS while deg5/deg10/deg32 differ clearly,')
    P('then the "under-trained fingerprint" of the DAMAGED family is actually dominated by the weight-decay configuration,')
    P('whose physical meaning is "under-training makes the output degenerate to persistence"; this should be stated in the main text;')
    P('conversely, if all four are identical to PERS, then the damaged-control family as a whole degenerates to persistence.')
    report('58_damaged_vs_pers', LOG)


if __name__ == '__main__':
    main()
