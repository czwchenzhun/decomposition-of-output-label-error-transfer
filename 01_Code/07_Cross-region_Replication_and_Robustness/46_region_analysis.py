# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""46 Region-parameterised "model output vs training label" analysis (reused across extension regions to prevent convention drift)

Merge the core algorithms of 25/27/28/30/34 into one **region-parameterised** implementation,
so that R3 (85 stations) and the extension regions R2 (21 stations) / R5 (19 stations) / R4 (14 stations) follow exactly the same code path.
The same script also re-runs R3 (pattern=gpu*) to cross-check the existing numbers from 33/34.

Three error layers (within-station convention: de-mean by station first, then compute)
----------------------------------------
  e1 = label(SMCI) - truth        E1 = RMS(e1)
  e2 = output      - truth        E2 = RMS(e2)
  e3 = output      - label        E3 = RMS(e3)
  rho = Cov(e1,e3)/(E1*E3)

Identity: (E2/E1)^2 = (1-rho^2) + (x+rho)^2,  x = E3/E1
Calibration condition: m_lam = r + lam*(m-r)  ->  rho unchanged, lam* = -rho/x, lower bound sqrt(1-rho^2)

Usage
----
  python 46_region_analysis.py --region R0_northchina --pattern 'gpu*'
  python 46_region_analysis.py --region R1_yangtze    --pattern 'xr2_*'

Output (02_Results_Data/truth_chain/multi/)
----------------------------------------------
  withinstation_{region}.csv   three error layers and derived quantities per run x cfg x layer
  quadratic_{region}.csv       verification of the identity and the three predictions of the calibration condition
  price_{region}.csv           price of truth: how much gain calibration with k stations yields
  bootstrap_{region}.csv       family means and between-family difference CIs from station-clustered bootstrap
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: main-text Table 8 (cross-region replication) and the cross-region analysis in §4.8; note: this script is called by 37/49/52/39/54/55/56/59/69 and other scripts, so its name is unchanged
#

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

B = TRUTH
OD = os.path.join(B, 'multi')
os.makedirs(OD, exist_ok=True)
LAYERS = ['10cm', '30cm']
FAMS = ['FREE', 'SOFT', 'HYBL', 'HARD', 'DAMAGED']


# ----------------------------------------------------------------- family
def stratum_of(cfg):
    """Training-regime layer: NS = free-running-rollout training (no scheduled sampling), SS = with SS.

    The FREE family originally contained both LSTM(SS) and LSTMNS(NS), and the regime itself affects rho on the same order as
    the constraint form, so family comparison must be stratified within regime (see the chain decomposition in 37).
    """
    return 'NS' if cfg in ('LSTMNS', 'HYBNS', 'HYBLSTNS', 'LPMIXNS') else 'SS'


def fam_of(run, cfg):
    """Assign family from cfg (extension regions run only four classes, no need to guess hyperparameters from the run name)"""
    if cfg == 'HYBNS':
        return 'HARD'
    if cfg in ('HYBLST', 'HYBLSTNS'):
        # Intermediate: hard update + LSTM backbone (pivot for separating the form effect from the architecture effect)
        return 'HYBL'
    if cfg in ('LPMIX', 'LPMIXNS'):
        # LPMIXNS = LPMIX with scheduled sampling removed (within-regime control for M2)
        return 'SOFT'
    if re.search(r'wd', run):
        return 'DAMAGED'
    if cfg in ('LSTM', 'LSTMNS'):
        return 'FREE'
    return None


def within(sr, grp):
    return (sr - sr.groupby(grp).transform('mean')).values


def budget(e1, e3):
    """Compute the three error layers directly from e1(=r-o) and e3(=m-r) (e2 = e1+e3)"""
    E1 = float(np.sqrt(np.mean(e1 ** 2)))
    E3 = float(np.sqrt(np.mean(e3 ** 2)))
    e2 = e1 + e3
    E2 = float(np.sqrt(np.mean(e2 ** 2)))
    C = float(np.mean(e1 * e3))
    rho = C / (E1 * E3) if E1 * E3 > 0 else 0.0
    return E1, E2, E3, rho


# ----------------------------------------------------------------- load
def load_pairs(pattern):
    """Return {(run,cfg,layer): DataFrame(obs,smci,model,id)}"""
    files = sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_%s.csv' % pattern)))
    out = {}
    for f in files:
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '').replace('.csv', '')
        try:
            A = pd.read_csv(f, encoding='utf-8-sig', dtype={'id': str})
        except Exception as e:
            print('  skip %s (%s)' % (nm, e))
            continue
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        for cfg in sorted(set(A.cfg)):
            fam = fam_of(nm, cfg)
            if fam is None:
                continue
            for lay in LAYERS:
                v = A[(A.cfg == cfg) & (A.layer == lay)]
                if len(v) < 60:      # extension regions have fewer stations, so the threshold is lowered
                    continue
                out[(nm, cfg, lay)] = v[['id', 'obs', 'smci', 'model']].reset_index(drop=True)
    return out


# ----------------------------------------------------------------- 1 within-station convention
def stage_within(P, region):
    rows = []
    for (nm, cfg, lay), v in P.items():
        ow = within(v.obs, v.id)
        rw = within(v.smci, v.id)
        mw = within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        E1, E2, E3, rho = budget(e1, e3)
        rows.append(dict(region=region, run=nm, cfg=cfg, layer=lay,
                         strat=stratum_of(cfg), fam=fam_of(nm, cfg),
                         n=len(v), nsta=v.id.nunique(),
                         E1w=E1, E2w=E2, E3w=E3, rho_w=rho,
                         a_w=-np.mean(e1 * e3) / (E1 ** 2)))
    W = pd.DataFrame(rows)
    if not len(W):
        return W
    W['x'] = W.E3w / W.E1w
    W['ratio'] = W.E2w / W.E1w
    W['floor'] = np.sqrt(1.0 - W.rho_w ** 2)
    W['exc'] = W.x + W.rho_w
    W['chk'] = np.sqrt(W.floor ** 2 + W.exc ** 2)
    W['beat'] = (W.ratio < 1.0).astype(int)
    W.to_csv(os.path.join(OD, 'withinstation_%s.csv' % region),
             index=False, encoding='utf-8-sig')
    return W


# ----------------------------------------------------------------- 2 identity + calibration condition
def stage_quadratic(P, W, region):
    rows = []
    for (nm, cfg, lay), v in P.items():
        ow = within(v.obs, v.id)
        rw = within(v.smci, v.id)
        mw = within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        E1, E2, E3, rho = budget(e1, e3)
        x = E3 / E1
        lam_th = -rho / x if x > 0 else 1.0
        # Empirical scan
        lams = np.linspace(0.0, 2.0, 2001)
        vals = np.array([np.sqrt(np.mean((rw + L * (mw - rw) - ow) ** 2))
                         for L in lams])
        j = int(np.argmin(vals))
        lam_em, E2opt = float(lams[j]), float(vals[j])
        # P1: invariance of rho under scaling (rechecked at lam=0.5)
        _, _, _, rho_at = budget(e1, 0.5 * e3)
        rows.append(dict(region=region, run=nm, cfg=cfg, layer=lay,
                         fam=fam_of(nm, cfg), x=x, rho=rho, E2E1=E2 / E1,
                         lam_th=lam_th, lam_em=lam_em,
                         E2E1_opt=E2opt / E1,
                         # When rho>0 the theoretical lam*<0, i.e. "no deviation should be retained",
                         # the optimum is truncated by the [0,2] grid lower bound -> flag it, not counted as a prediction failure
                         lam_bound=(lam_th < 1e-4) or (lam_th > 2 - 1e-4),
                         floor=np.sqrt(max(0.0, 1 - rho ** 2)),
                         rho_check=rho_at))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OD, 'quadratic_%s.csv' % region),
             index=False, encoding='utf-8-sig')
    return D


# ----------------------------------------------------------------- 3 price of truth
def stage_price(P, region, nrep=200, ks=(1, 2, 3, 5, 10)):
    rows = []
    for (nm, cfg, lay), v in P.items():
        ids = v.id.values
        uid = np.unique(ids)
        if len(uid) < 6:
            continue
        ow = within(v.obs, v.id)
        rw = within(v.smci, v.id)
        mw = within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        E1, E2, E3, rho = budget(e1, e3)
        base = E2 / E1
        lam_or = float(np.clip(-rho / (E3 / E1), 0, 2))
        oracle = float(np.sqrt(np.mean((rw + lam_or * e3 - ow) ** 2)) / E1)
        idx = {s: np.where(ids == s)[0] for s in uid}
        re = dict(region=region, run=nm, cfg=cfg, layer=lay, fam=fam_of(nm, cfg),
                  nsta=len(uid), lam_or=lam_or, base=base, oracle=oracle)
        for k in ks:
            if k > len(uid) // 2:
                continue
            acc = []
            for rep in range(nrep):
                rng = np.random.RandomState(7 * rep + k)
                pick = set(rng.choice(uid, size=k, replace=False))
                comp = ~np.isin(ids, list(pick))
                vc = v[comp]
                oc = within(vc.obs, vc.id)
                rc = within(vc.smci, vc.id)
                mc = within(vc['model'], vc.id)
                E1c = float(np.sqrt(np.mean((rc - oc) ** 2)))
                vh = v[np.isin(ids, list(pick))]
                o1 = within(vh.obs, vh.id)
                r1 = within(vh.smci, vh.id)
                m1 = within(vh['model'], vh.id)
                E1k = float(np.sqrt(np.mean((r1 - o1) ** 2)))
                E3k = float(np.sqrt(np.mean((m1 - r1) ** 2)))
                rhok = float(np.mean((r1 - o1) * (m1 - r1)) / (E1k * E3k))
                lh = float(np.clip(-rhok / (E3k / E1k), 0, 2))
                acc.append((float(np.sqrt(np.mean((rc + lh * (mc - rc) - oc) ** 2)) / E1c),
                            float(np.sqrt(np.mean((mc - oc) ** 2)) / E1c)))
            aa = np.array(acc)
            re['k%d' % k] = aa[:, 0].mean()
            re['k%d_sd' % k] = aa[:, 0].std()
            re['k%d_base' % k] = aa[:, 1].mean()
        rows.append(re)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OD, 'price_%s.csv' % region),
             index=False, encoding='utf-8-sig')
    return R


# ----------------------------------------------------------------- 4 cluster bootstrap
def stage_bootstrap(P, region, nboot=2000, seed=20261006):
    """Resample with stations as cluster units; the same draw indices are used for all configurations in the layer.
    Precompute per-station (A=sum e1^2, Bb=sum e3^2, C=sum e1*e3), then sum after resampling
    (within-station de-meaning depends only on the station itself, so it is unchanged by resampling)."""
    raw = {}
    for (nm, cfg, lay), v in P.items():
        ids = v.id.values
        ow = within(v.obs, v.id)
        rw = within(v.smci, v.id)
        mw = within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(ids, return_inverse=True)
        k = len(u)
        raw.setdefault(lay, {})[(nm, cfg)] = dict(
            fam=fam_of(nm, cfg), ids=u,
            n=np.bincount(inv, minlength=k).astype(float),
            A=np.bincount(inv, weights=e1 ** 2, minlength=k),
            B=np.bincount(inv, weights=e3 ** 2, minlength=k),
            C=np.bincount(inv, weights=e1 * e3, minlength=k))

    rng = np.random.default_rng(seed)
    out = []
    for lay, cfgs in raw.items():
        keys = list(cfgs.keys())
        nst = [len(cfgs[k]['ids']) for k in keys]
        if max(nst) != min(nst):
            # Station sets may differ slightly across configurations (missing months); take the smallest common station set
            common = set(cfgs[keys[0]]['ids'])
            for k in keys[1:]:
                common &= set(cfgs[k]['ids'])
            common = np.sort(np.array(sorted(common)))
        else:
            common = cfgs[keys[0]]['ids']
        # Reorder each configuration to the common order
        prep = {}
        for k in keys:
            pos = {s: i for i, s in enumerate(cfgs[k]['ids'])}
            ii = np.array([pos[s] for s in common])
            prep[k] = dict(fam=cfgs[k]['fam'],
                           n=cfgs[k]['n'][ii], A=cfgs[k]['A'][ii],
                           B=cfgs[k]['B'][ii], C=cfgs[k]['C'][ii])
        ns = len(common)
        fams = sorted(set(v['fam'] for v in prep.values()))
        acc = {f: [] for f in fams}
        acc_r = {f: [] for f in fams}
        # Missing months differ across configurations -> each uses its own sample count, never mixed
        cnts = {k: prep[k]['n'] for k in keys}
        for b in range(nboot):
            ii = rng.integers(0, ns, ns)
            for f in fams:
                ks = [k for k in keys if prep[k]['fam'] == f]
                rs, rr = [], []
                for k in ks:
                    d = prep[k]
                    A = d['A'][ii].sum()
                    Bb = d['B'][ii].sum()
                    C = d['C'][ii].sum()
                    nn = cnts[k][ii].sum()
                    E1 = np.sqrt(A / nn)
                    E3 = np.sqrt(Bb / nn)
                    # Cov must likewise be divided by nn: rho = (C/nn)/(E1*E3) = C/sqrt(A*B)
                    rho = (C / nn) / (E1 * E3) if E1 * E3 > 0 else 0.0
                    E2 = np.sqrt((A + Bb + 2 * C) / nn)
                    rs.append(rho)
                    rr.append(E2 / E1)
                acc[f].append(np.mean(rs))
                acc_r[f].append(np.mean(rr))
        for f in fams:
            a = np.array(acc[f])
            ar = np.array(acc_r[f])
            nf = len([k for k in keys if prep[k]['fam'] == f])
            out.append(dict(region=region, layer=lay, fam=f, ncfg=nf,
                            nsta=ns,
                            rho=a.mean(), rho_lo=np.percentile(a, 2.5),
                            rho_hi=np.percentile(a, 97.5),
                            ratio=ar.mean(), ratio_lo=np.percentile(ar, 2.5),
                            ratio_hi=np.percentile(ar, 97.5)))
        for i, f1 in enumerate(fams):
            for f2 in fams[i + 1:]:
                d = np.array(acc[f1]) - np.array(acc[f2])
                dr = np.array(acc_r[f1]) - np.array(acc_r[f2])
                out.append(dict(region=region, layer=lay,
                                fam='%s-%s' % (f1, f2), ncfg=0, nsta=ns,
                                rho=d.mean(), rho_lo=np.percentile(d, 2.5),
                                rho_hi=np.percentile(d, 97.5),
                                ratio=dr.mean(), ratio_lo=np.percentile(dr, 2.5),
                                ratio_hi=np.percentile(dr, 97.5)))
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(OD, 'bootstrap_%s.csv' % region),
             index=False, encoding='utf-8-sig')
    return R


# ----------------------------------------------------------------- report
def report(region, W, D, PR, BS):
    o = []
    ap = o.append
    ap('=' * 96)
    ap('46 Region analysis -- %s' % region)
    ap('=' * 96)
    if not len(W):
        ap('!! no samples')
        return '\n'.join(o)
    ap('Configuration count %d (run x cfg x layer=%d)  station count %d'
       % (W.groupby(['run', 'cfg']).ngroups, len(W), int(W.nsta.max())))
    ap('')
    ap('[Identity check] max |chk - E2/E1| = %.3e' % np.abs(W.chk - W.ratio).max())
    ap('')
    ap('%-8s %-6s %3s  %8s %8s %8s %7s   %s'
       % ('fam', 'layer', 'n', 'E2/E1', 'sd', 'rho', 'floor', 'better than label'))
    for lay in LAYERS:
        s = W[W.layer == lay]
        for f in FAMS:
            g = s[s.fam == f]
            if not len(g):
                continue
            ap('%-8s %-6s %3d  %8.4f %8.4f %8.4f %7.4f   %d/%d'
               % (f, lay, len(g), g.ratio.mean(), g.ratio.std(),
                  g.rho_w.mean(), g.floor.mean(), int(g.beat.sum()), len(g)))
    ap('')
    ap('[Within one family, stratified by training regime] (SS = with scheduled sampling, NS = free-running-rollout training)')
    ap('%-5s %-8s %-6s %3s  %8s %8s %8s   %s'
       % ('strat', 'fam', 'layer', 'n', 'E2/E1', 'sd', 'rho', 'better than label'))
    for lay in LAYERS:
        s = W[W.layer == lay]
        for st in ('SS', 'NS'):
            for f in FAMS:
                g = s[(s.strat == st) & (s.fam == f)]
                if not len(g):
                    continue
                ap('%-5s %-8s %-6s %3d  %8.4f %8.4f %8.4f   %d/%d'
                   % (st, f, lay, len(g), g.ratio.mean(), g.ratio.std(),
                      g.rho_w.mean(), int(g.beat.sum()), len(g)))
    ap('')
    ap('[Three predictions of the calibration condition]')
    for lay in LAYERS:
        s = D[D.layer == lay]
        if not len(s):
            continue
        inner = s[~s.lam_bound]
        nb = int(s.lam_bound.sum())
        note = ('  A further %d configurations have rho>=0 -> lam*<=0 truncated by the grid lower bound (not counted as prediction failures)' % nb
                if nb else '')
        ap('  %s  P1 rho-unchanged max|dev|=%.3e   P2 lam difference max=%.3e   P3 optimum=FLOOR max=%.3e   (n=%d/%d)%s'
           % (lay, np.abs(s.rho_check - s.rho).max(),
              np.abs(inner.lam_th - inner.lam_em).max() if len(inner) else np.nan,
              np.abs(inner.E2E1_opt - inner.floor).max() if len(inner) else np.nan,
              len(inner), len(s), note))
    ap('')
    ap('[Price of truth] base -> oracle, and the capture fraction after calibration with k stations')
    for lay in LAYERS:
        s = PR[PR.layer == lay]
        if not len(s):
            continue
        for f in FAMS:
            g = s[s.fam == f]
            if not len(g):
                continue
            kc = sorted([c for c in g.columns if re.match(r'^k\d+$', c)],
                        key=lambda s: int(s[1:]))
            seg = []
            for c in kc:
                # The denominator of the capture fraction (base-oracle) is tiny for near-optimal families, so the percentage explodes;
                # therefore only the k3/k10 tiers are reported, together with the absolute E2/E1 after calibration
                if c in ('k3', 'k10'):
                    cap = (g.base - g[c]) / (g.base - g.oracle)
                    seg.append('%s=%.4f (capture %+.0f%%)'
                               % (c, g[c].mean(), 100 * cap.mean()))
            ap('  %-6s %-8s base %.4f -> oracle %.4f   %s'
               % (lay, f, g.base.mean(), g.oracle.mean(), '  '.join(seg)))
    ap('')
    ap('[Station-clustered bootstrap 95% CI]')
    for lay in LAYERS:
        s = BS[BS.layer == lay]
        for _, r in s.iterrows():
            ap('  %-6s %-12s rho %+7.4f [%+.4f,%+.4f]   E2/E1 %7.4f [%.4f,%.4f]'
               % (lay, r.fam, r.rho, r.rho_lo, r.rho_hi, r.ratio, r.ratio_lo, r.ratio_hi))
    return '\n'.join(o)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--region', default='R0_northchina')
    ap_.add_argument('--pattern', default='gpu*',
                     help="monthly filename-suffix wildcard, e.g. 'gpu*' / 'xr2_*'")
    ap_.add_argument('--nboot', type=int, default=2000)
    a = ap_.parse_args()

    print('46 %s  pattern=%s' % (a.region, a.pattern), flush=True)
    P = load_pairs(a.pattern)
    print('  Loaded %d (run,cfg,layer) combinations' % len(P), flush=True)
    if not P:
        print('!! no samples, exiting')
        return
    W = stage_within(P, a.region)
    D = stage_quadratic(P, W, a.region)
    PR = stage_price(P, a.region)
    BS = stage_bootstrap(P, a.region, nboot=a.nboot)
    txt = report(a.region, W, D, PR, BS)
    print(txt)
    with open(LOG + '/_395_%s.txt' % a.region, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\nOutput directory: %s' % OD)


if __name__ == '__main__':
    main()
