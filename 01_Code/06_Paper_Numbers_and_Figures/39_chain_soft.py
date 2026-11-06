# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""39: Chain decomposition left endpoint changed to LPMIXNS (soft penalty NS) -- aligned with the SOFT--HARD primary endpoint

Background (fourth-round review, M2)
--------------------
Section \ref{sec:fams} promised that LPMIX -> HYBLST is a single-variable control that "changes the update rule only", 
But Section \ref{sec:confound} the NS chain actually executed in the section is

    LSTMNS --> HYBLSTNS --> HYBNS

its left endpoint LSTMNS is a bare LSTM (no flux head, $7.96\\times10^4$ parameters), so the FORM step
also mixes in "enabling the flux head" and is not single-variable; more importantly the primary endpoint is SOFT--HARD,
whereas this chain decomposes FREE--HARD.

This script replaces the left endpoint with LPMIXNS (same family and regime as the primary endpoint):

    LPMIXNS --FORM(change update rule only)--> HYBLSTNS --BACKBONE(change backbone/residual only)--> HYBNS
    TOTAL = HYBNS - LPMIXNS   ==  Section \ref{sec:regime} "within-regime soft--hard gap" of the section

LPMIXNS and HYBLSTNS have the same parameter count of $2.12\\times10^5$ (component-wise identical), so the FORM step
is a true single-variable control; the capacity gap in the BACKBONE step (2.08e5 -> 9.25e4) is a known confound.

Both chains are output; the main text uses the new chain and the old chain is for comparison only.

Usage
----
    python 39_chain_soft.py --nboot 2000
Output 02_Results_Data/truth_chain/multi/chain_soft_ALL.csv
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: §4.3 chain decomposition (FORM / BACKBONE / TOTAL) and the data for Fig. 6
#

import os
import re
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    'm395', os.path.join(HERE, '46_region_analysis.py'))
M395 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M395)

B = M395.B
OD = M395.OD
LAYERS = M395.LAYERS

REGIONS = [('R1_yangtze', 'xr2_*'), ('R0_northchina', 'gpu*'),
           ('R3_northeast', 'xr4_*'), ('R2_loess', 'xr5_*'),
           ('R4_longzhong', 'xr7_*'), ('R5_sichuan', 'xr8_*')]

PRETTY = {'LSTMNS': 'unconstrained NS', 'LPMIXNS': 'soft penalty NS',
          'HYBLSTNS': 'intermediate NS', 'HYBNS': 'hard skeleton NS'}
LEVELS = ['LPMIXNS', 'HYBLSTNS', 'HYBNS', 'LSTMNS']

# (variable name, start, end): end minus start
NEW_CHAIN = [('FORM(new chain)', 'LPMIXNS', 'HYBLSTNS'),
             ('BACKBONE', 'HYBLSTNS', 'HYBNS'),
             ('TOTAL(new chain)', 'LPMIXNS', 'HYBNS')]
OLD_CHAIN = [('FORM(old chain)', 'LSTMNS', 'HYBLSTNS'),
             ('TOTAL(old chain)', 'LSTMNS', 'HYBNS')]


def lab_of(nm, cfg):
    """Same convention as 37: HYBLSTNS takes only the intermediate itself, excluding the hblg(GRU)/hblf(softplus) probes"""
    if cfg == 'HYBLSTNS':
        if re.search(r'hblg', nm) or re.search(r'hblf', nm):
            return None
    return cfg if cfg in LEVELS else None


def boot(P, labels, nboot=2000, seed=20261006):
    """station-clustered bootstrap (same convention as 37: one resample index used for all configurations of that layer)"""
    raw = {}
    for (nm, cfg, lay), v in P.items():
        if lab_of(nm, cfg) not in labels:
            continue
        ids = v.id.values
        ow = M395.within(v.obs, v.id)
        rw = M395.within(v.smci, v.id)
        mw = M395.within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(ids, return_inverse=True)
        k = len(u)
        raw.setdefault(lay, {}).setdefault(cfg, {})[(nm, cfg)] = dict(
            ids=u,
            n=np.bincount(inv, minlength=k).astype(float),
            A=np.bincount(inv, weights=e1 ** 2, minlength=k),
            B=np.bincount(inv, weights=e3 ** 2, minlength=k),
            C=np.bincount(inv, weights=e1 * e3, minlength=k))

    rng = np.random.default_rng(seed)
    res = {}
    for lay, labs in raw.items():
        common = None
        for cfg, runs in labs.items():
            for key, d in runs.items():
                s = d['ids']
                common = set(s) if common is None else (common & set(s))
        common = np.sort(np.array(sorted(common)))
        ns = len(common)
        prep = {}
        for cfg, runs in labs.items():
            for key, d in runs.items():
                pos = {s: i for i, s in enumerate(d['ids'])}
                ii = np.array([pos[s] for s in common])
                prep.setdefault(cfg, []).append(
                    (d['n'][ii], d['A'][ii], d['B'][ii], d['C'][ii]))
        acc = {cfg: {'rho': [], 'x': [], 'ratio': []} for cfg in labs}
        for _ in range(nboot):
            idx = rng.integers(0, ns, ns)
            for cfg, lst in prep.items():
                rs, xs, rr = [], [], []
                for n, A, Bb, C in lst:
                    n = n[idx]; A = A[idx]; Bb = Bb[idx]; C = C[idx]
                    nn = n.sum()
                    E1 = np.sqrt(A.sum() / nn)
                    E3 = np.sqrt(Bb.sum() / nn)
                    Cov = C.sum() / nn
                    rho = Cov / (E1 * E3) if E1 * E3 > 0 else 0.0
                    E2 = np.sqrt((A.sum() + Bb.sum() + 2 * C.sum()) / nn)
                    rs.append(rho); xs.append(E3 / E1); rr.append(E2 / E1)
                acc[cfg]['rho'].append(np.mean(rs))
                acc[cfg]['x'].append(np.mean(xs))
                acc[cfg]['ratio'].append(np.mean(rr))
        res[lay] = {cfg: {k: np.array(v) for k, v in d.items()}
                    for cfg, d in acc.items()}
        res[lay]['_nsta'] = np.array([ns])
        res[lay]['_ncfg'] = {cfg: len(labs[cfg]) for cfg in labs}
    return res


def ci(a, lo=2.5, hi=97.5):
    return float(np.mean(a)), float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()

    o = ['=' * 96,
         '39 chain decomposition (left endpoint LPMIXNS) -- aligned with the SOFT--HARD primary endpoint',
         '=' * 96]
    rows = []
    for region, pat in REGIONS:
        P = M395.load_pairs(pat)
        if not len(P):
            o.append('!! no samples %s' % region)
            continue
        ncfg = {}
        for (nm, cfg, lay) in P:
            if lay == '10cm' and lab_of(nm, cfg) in LEVELS:
                ncfg[lab_of(nm, cfg)] = ncfg.get(lab_of(nm, cfg), 0) + 1
        o.append('')
        o.append('---- %s  configuration count %s' % (region, ncfg))
        BS = boot(P, LEVELS, nboot=a.nboot)
        for lay in LAYERS:
            if lay not in BS:
                continue
            d = BS[lay]
            ns = int(d['_nsta'][0])
            o.append('')
            o.append('  [%s] station %d' % (lay, ns))
            o.append('    %-14s %-24s %-10s %-24s' %
                     ('label', 'rho [95% CI]', 'x', 'E2/E1 [95% CI]'))
            for cfg in LEVELS:
                if cfg not in d:
                    continue
                rm, rl, rh = ci(d[cfg]['rho'])
                xm, _, _ = ci(d[cfg]['x'])
                tm, tl, th = ci(d[cfg]['ratio'])
                o.append('    %-14s %+7.4f[%+.4f,%+.4f] %8.4f   %.4f[%.4f,%.4f]' %
                         (PRETTY.get(cfg, cfg), rm, rl, rh, xm, tm, tl, th))
                rows.append(dict(region=region, layer=lay, kind='level',
                                 label=cfg, ncfg=d['_ncfg'].get(cfg, 0),
                                 nsta=ns, rho=rm, rho_lo=rl, rho_hi=rh,
                                 x=xm, ratio=tm, ratio_lo=tl, ratio_hi=th))
            o.append('    New chain LPMIXNS -> HYBLSTNS -> HYBNS :')
            for tag, l1, l2 in NEW_CHAIN:
                if l1 not in d or l2 not in d:
                    continue
                out = []
                rec = dict(region=region, layer=lay, kind=tag,
                           label='%s-%s' % (l2, l1), nsta=ns)
                for q in ('rho', 'x', 'ratio'):
                    m, lo, hi = ci(d[l2][q] - d[l1][q])
                    star = '*' if (lo * hi > 0) else ' '
                    out.append('%+.4f[%+.4f,%+.4f]%s' % (m, lo, hi, star))
                    rec[q] = m
                    rec[q + '_lo'] = lo
                    rec[q + '_hi'] = hi
                o.append('      %-16s rho %s  x %s  E2/E1 %s' %
                         (tag, out[0], out[1], out[2]))
                rows.append(rec)
            o.append('    Old chain LSTMNS -> HYBLSTNS -> HYBNS (comparison, not used):')
            for tag, l1, l2 in OLD_CHAIN:
                if l1 not in d or l2 not in d:
                    continue
                out = []
                rec = dict(region=region, layer=lay, kind=tag,
                           label='%s-%s' % (l2, l1), nsta=ns)
                for q in ('rho', 'x', 'ratio'):
                    m, lo, hi = ci(d[l2][q] - d[l1][q])
                    star = '*' if (lo * hi > 0) else ' '
                    out.append('%+.4f[%+.4f,%+.4f]%s' % (m, lo, hi, star))
                    rec[q] = m
                    rec[q + '_lo'] = lo
                    rec[q + '_hi'] = hi
                o.append('      %-16s rho %s  x %s  E2/E1 %s' %
                         (tag, out[0], out[1], out[2]))
                rows.append(rec)

    df = pd.DataFrame(rows)
    op = os.path.join(OD, 'chain_soft_ALL.csv')
    df.to_csv(op, index=False, encoding='utf-8-sig')
    o.append('')
    o.append('written -> %s' % op)

    # Summary: sign consistency across the 8 (region x layer) combinations
    o.append('')
    o.append('=' * 96)
    o.append('sign consistency across the 8 (region x layer) combinations')
    for tag in [t for t, _, _ in NEW_CHAIN] + [t for t, _, _ in OLD_CHAIN]:
        s = df[(df.kind == tag) & (df.label.str.contains('-'))]
        if not len(s):
            continue
        for q in ('rho', 'ratio'):
            v = s[q].values
            npos = int((v > 0).sum())
            nci = int(((s[q + '_lo'].values > 0) | (s[q + '_hi'].values < 0)).sum())
            o.append('  %-16s %-6s positive %d/%d  CI excludes zero %d/%d  mean %+.4f  range [%+.4f,%+.4f]'
                     % (tag, q, npos, len(v), nci, len(v), v.mean(),
                        v.min(), v.max()))
    txt = '\n'.join(o)
    with open(os.path.join(HERE, '_406.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
