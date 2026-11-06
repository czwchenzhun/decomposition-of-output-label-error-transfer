# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""55: Alternative inference when the number of clusters is small -- wild cluster bootstrap and BCa

Addresses the fourth-round review M4:
  * The Northeast China black-soil region has only 14 cluster units, so percentile-bootstrap coverage is low
    -> add two alternative intervals: wild cluster bootstrap (Rademacher weights) and BCa
  * The "cluster-robust p<=0.063" in Table \ref{tab:wscan} rests on 7 w levels,
    the cluster-robust standard error with 7 clusters is itself underestimated -> recheck with wild cluster bootstrap

Method
----
Let the statistic be a smooth function of several "sample-equally-weighted moments": theta = f(mu),
mu = (1/N) sum_s t_s (t_s is the moment total of the s-th cluster, N is the sample size).
wild cluster bootstrap assigns Rademacher weights to the cluster score u_s = t_s - n_s*mu:

    mu* = mu + (1/N) * sum_s eps_s * u_s,  eps_s in {-1, +1}
    theta* = f(mu*)

The same draw of eps is used for all configurations under that (region, layer), to preserve dependence among configurations.
BCa: the acceleration constant a comes from the cluster jackknife, the bias correction z0 from percentile bootstrap.

Output: 02_Results_Data/truth_chain/multi/wild_bootstrap.csv
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: the three interval conventions wild cluster / BCa / percentile in main-text Table 7
#

import os
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    'm395', os.path.join(HERE, '46_region_analysis.py'))
M395 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M395)

B = M395.B
OD = M395.OD
LAYERS = M395.LAYERS
REGIONS = [('R0_northchina', 'gpu*'), ('R1_yangtze', 'xr2_*'),
           ('R2_loess', 'xr5_*'), ('R3_northeast', 'xr4_*'),
           ('R4_longzhong', 'xr7_*'), ('R5_sichuan', 'xr8_*')]
REGCN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
         'R2_loess': 'Loess Plateau', 'R3_northeast': 'Northeast China black soil',
         'R4_longzhong': 'Longzhong loess', 'R5_sichuan': 'Sichuan Basin'}


# ----------------------------------------------------------------- low-level utilities
def cluster_moments(P, cfgs, lay):
    """Return (stations, {cfg: (n_s, A_s, B_s, C_s)}); A=e1^2, B=e3^2, C=e1*e3"""
    out = {}
    stas = None
    for (nm, cfg, ly), v in P.items():
        if ly != lay or cfg not in cfgs:
            continue
        ow = M395.within(v.obs, v.id)
        rw = M395.within(v.smci, v.id)
        mw = M395.within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(v.id.values, return_inverse=True)
        k = len(u)
        d = dict(ids=u,
                 n=np.bincount(inv, minlength=k).astype(float),
                 A=np.bincount(inv, weights=e1 ** 2, minlength=k),
                 B=np.bincount(inv, weights=e3 ** 2, minlength=k),
                 C=np.bincount(inv, weights=e1 * e3, minlength=k))
        out.setdefault(cfg, []).append(d)
        stas = set(u) if stas is None else (stas & set(u))
    if stas is None:
        return None, None
    stas = np.sort(np.array(sorted(stas)))
    # Align to the common station set
    al = {}
    for cfg, lst in out.items():
        ns, As, Bs, Cs = [], [], [], []
        for d in lst:
            pos = {s: i for i, s in enumerate(d['ids'])}
            ii = np.array([pos[s] for s in stas])
            ns.append(d['n'][ii]); As.append(d['A'][ii])
            Bs.append(d['B'][ii]); Cs.append(d['C'][ii])
        al[cfg] = (np.array(ns), np.array(As), np.array(Bs), np.array(Cs))
    return stas, al


def fam_stat(mu):
    """mu = (Abar, Bbar, Cbar) -> (rho, x, ratio)"""
    A, Bb, C = mu
    E1 = np.sqrt(np.maximum(A, 1e-30))
    E3 = np.sqrt(np.maximum(Bb, 1e-30))
    E2 = np.sqrt(np.maximum(A + Bb + 2 * C, 1e-30))
    rho = C / (E1 * E3)
    return rho, E3 / E1, E2 / E1


def theta_of(al, cfgs):
    """Family statistic: compute (rho,x,ratio) per configuration, then equal-weight average"""
    out = []
    for cfg in cfgs:
        if cfg not in al:
            continue
        n, A, Bb, C = al[cfg]
        N = n.sum(axis=1)
        mu = np.stack([A.sum(axis=1) / N, Bb.sum(axis=1) / N,
                       C.sum(axis=1) / N], axis=0)
        out.append(np.array(fam_stat(mu)))
    return np.mean(np.concatenate(out, axis=1), axis=1) if out else None


def theta_of_perturbed(al, cfgs, eps):
    out = []
    for cfg in cfgs:
        if cfg not in al:
            continue
        n, A, Bb, C = al[cfg]
        N = n.sum(axis=1)
        Mu = np.stack([A.sum(axis=1), Bb.sum(axis=1), C.sum(axis=1)], axis=0)
        mu = Mu / N
        # u_s = t_s - n_s * mu   (t_s is the moment total of the s-th station, shape (3, S))
        U = np.stack([A, Bb, C], axis=0) - n[None, :, :] * mu[:, :, None]
        pert = (U * eps[None, None, :]).sum(axis=2) / N
        out.append(np.array(fam_stat(mu + pert)))
    return np.mean(np.concatenate(out, axis=1), axis=1) if out else None


def diff_theta(al, cfgs_hi, cfgs_lo, eps=None):
    f = (lambda c: theta_of_perturbed(al, c, eps)) if eps is not None else \
        (lambda c: theta_of(al, c))
    a, b = f(cfgs_hi), f(cfgs_lo)
    if a is None or b is None:
        return None
    return a - b


def bca(theta_hat, boot, jack):
    """BCa interval: z0 from percentile bootstrap, a from cluster jackknife"""
    boot = np.asarray(boot)
    z0 = norm.ppf(np.mean(boot < theta_hat))
    jm = np.mean(jack)
    d = jm - jack
    den = np.sum(d ** 2)
    a = np.sum(d ** 3) / (6 * den ** 1.5) if den > 0 else 0.0
    out = []
    for al in (0.025, 0.975):
        z = norm.ppf(al)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(np.percentile(boot, 100 * adj))
    return out[0], out[1], z0, a


def _slope(z, y):
    """OLS slope"""
    zz = z - z.mean()
    yy = y - y.mean()
    return float((zz * yy).sum() / (zz ** 2).sum())


def _slope_from_moments(mu):
    """mu = (Ez, Ey, Ezy, Ezz) -> OLS slope"""
    ez, ey, ezy, ezz = mu
    return (ezy - ez * ey) / (ezz - ez ** 2)


def _one(rng, a, o, rows, region, lay, stas, al, hard, soft, tag):
    """Compute three confidence intervals for a single (region, layer, contrast)"""
    S = len(stas)
    d0 = diff_theta(al, hard, soft)
    # ---- percentile (station non-parametric bootstrap) ----
    pb = []
    for _ in range(a.nboot):
        idx = rng.integers(0, S, S)
        al2 = {c: (n[:, idx], A[:, idx], Bb[:, idx], C[:, idx])
               for c, (n, A, Bb, C) in al.items()}
        pb.append(diff_theta(al2, hard, soft))
    PB = np.array(pb)
    # ---- wild (Rademacher weights) ----
    wb = []
    for _ in range(a.nboot):
        eps = rng.choice([-1.0, 1.0], size=S)
        wb.append(diff_theta(al, hard, soft, eps=eps))
    WB = np.array(wb)
    # ---- cluster jackknife (acceleration constant of BCa) ----
    jk = []
    for s in range(S):
        keep = np.array([i for i in range(S) if i != s])
        al3 = {c: (n[:, keep], A[:, keep], Bb[:, keep], C[:, keep])
               for c, (n, A, Bb, C) in al.items()}
        jk.append(diff_theta(al3, hard, soft))
    JK = np.array(jk)
    for qi, qn in enumerate(['rho', 'x', 'ratio']):
        th = d0[qi]
        pl, ph = np.percentile(PB[:, qi], [2.5, 97.5])
        wl, wh = th + np.percentile(WB[:, qi] - th, [2.5, 97.5])
        bl, bh, z0, acc = bca(th, PB[:, qi], JK[:, qi])
        rows.append(dict(region=region, region_cn=REGCN[region], layer=lay,
                         contrast=tag, quantity=qn, nclu=S, theta=th,
                         perc_lo=pl, perc_hi=ph, wild_lo=wl, wild_hi=wh,
                         bca_lo=bl, bca_hi=bh, z0=z0, a=acc))
        o.append('%s %-5s %-14s %-6s S=%2d  theta=%+.4f | perc [%+.4f,%+.4f] '
                 '| wild [%+.4f,%+.4f] | BCa [%+.4f,%+.4f]'
                 % (REGCN[region], lay, tag, qn, S, th, pl, ph, wl, wh, bl, bh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261009)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    o = ['=' * 100,
         '55 wild cluster bootstrap / BCa -- alternative inference when the number of clusters is small',
         '=' * 100]
    rows = []
    for region, pat in REGIONS:
        P = M395.load_pairs(pat)
        for lay in LAYERS:
            cfgs = sorted(set(cfg for (nm, cfg, ly) in P if ly == lay))
            stas, al = cluster_moments(P, cfgs, lay)
            if al is None:
                continue
            S = len(stas)
            for tag, soft in (('cross-regime(SOFT-SS)', [c for c in cfgs if c == 'LPMIX']),
                              ('within-regime(SOFT-NS)', [c for c in cfgs if c == 'LPMIXNS'])):
                hard = [c for c in cfgs if c == 'HYBNS']
                if not soft or not hard:
                    continue
                _one(rng, a, o, rows, region, lay, stas, al, hard, soft, tag)
    DF = pd.DataFrame(rows)
    op = os.path.join(OD, 'wild_bootstrap.csv')
    DF.to_csv(op, index=False, encoding='utf-8-sig')

    # ---------------- wild cluster bootstrap of the strength scan (clusters = w levels) ----
    o.append('')
    o.append('[Strength scan (North China primary region): clusters = w levels, wild cluster bootstrap recheck of the delta_x slope]')
    W = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'),
                    encoding='utf-8-sig')
    W = W[W.cfg == 'LPMIX'] if 'cfg' in W.columns else W
    import re as _re
    wv, sd = [], []
    for r in W.run.astype(str):
        m = _re.match(r'gpu_wbw([0-9.]+)(?:s(\d+))?$', r)
        wv.append(float(m.group(1)) if m else np.nan)
        sd.append(m.group(2) if m else None)
    W['w'] = wv
    W = W[np.isfinite(W.w)]
    for lay in LAYERS:
        g = W[W.layer == lay]
        if len(g) < 3:
            continue
        z = np.log10(g.w.values)
        for q, cn in (('x_w', 'x'), ('rho_w', 'rho')):
            if q == 'x_w':
                y = (g.E3w / g.E1w).values
            else:
                y = g[q].values
            # clusters = w levels
            lev = np.unique(z)
            sl0 = _slope(z, y)
            # Perturb cluster scores: mu = (Ez, Ey, Ezy, Ezz)
            M = np.stack([z, y, z * y, z * z], axis=0)
            mu = M.mean(axis=1)
            U = []
            for L in lev:
                m_ = z == L
                U.append(M[:, m_].sum(axis=1) - m_.sum() * mu)
            U = np.stack(U, axis=0)          # (nlev, 4)
            nlev = len(lev)
            bs = []
            for _ in range(4000):
                eps = rng.choice([-1.0, 1.0], size=nlev)
                mu2 = mu + (eps @ U) / len(z)
                bs.append(_slope_from_moments(mu2))
            bs = np.array(bs)
            lo, hi = sl0 + np.percentile(bs - sl0, [2.5, 97.5])
            frac_neg = float(np.mean(bs <= 0))
            o.append('  %-5s %-4s slope %+.4f  clusters %d  wild 95%% [%+.4f,%+.4f]  '
                     'P(slope<=0)=%.3f' % (lay, cn, sl0, nlev, lo, hi, frac_neg))
            rows.append(dict(region='R0_northchina', region_cn='North China Plain',
                             layer=lay, contrast='w scan(%s)' % cn, quantity='slope',
                             nclu=nlev, theta=sl0, perc_lo=np.nan, perc_hi=np.nan,
                             wild_lo=lo, wild_hi=hi, bca_lo=np.nan, bca_hi=np.nan,
                             z0=np.nan, a=frac_neg))

    txt = '\n'.join(o)
    with open(os.path.join(HERE, '_408.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print()
    print('written ->', op)


if __name__ == '__main__':
    main()
