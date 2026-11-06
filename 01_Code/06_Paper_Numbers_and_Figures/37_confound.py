# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""37: Architectural-confound decomposition -- "hard update + LSTM backbone" chain control enabled by the intermediate

Background
----
In the original design, HYBNS (hard water-balance skeleton) and LSTM (free increment) are entangled by at least three differences:
    (1) whether the state update is imposed by the water-balance equation (the "form" of interest here)
    (2) whether memory comes from an LSTM cell or from a "memoryless feed-forward flux head + GRU residual" (the "backbone")
    (3) parameterisation of the irrigation term and percolation flux (residual difference)
Therefore a "collapse of rho in HYBNS" (if present) cannot be attributed to (1).

The intermediate HYBLST makes a chain decomposition possible (under the same NS training regime, each step differs in one factor only):
    LSTMNS   -- change update rule only -->   HYBLSTNS   : pure "form" effect FORM
    HYBLSTNS -- change backbone/residual only -->  HYBNS      : pure "backbone and design" effect BACKBONE
    LSTMNS   -- the two ends of the chain ------->  HYBNS      : total effect TOTAL = FORM + BACKBONE (telescoping identity)

Residual-difference probes (primary region only):
    HYBLSTNS --cell gru   : backbone replaced by GRUCell            -> CELL
    HYBLSTNS --hxd free   : percolation replaced by the HYBNS-style softplus  -> DFORM

Usage
----
    python 37_confound.py --region R0_northchina --pattern 'gpu*' --nboot 2000
Output 02_Results_Data/truth_chain/multi/confound_{region}.csv
"""
import os
import re
import sys
import glob
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

PRETTY = {
    'LSTMNS': 'unconstrained NS', 'LSTM': 'unconstrained SS', 'LPMIX': 'soft penalty SS',
    'LPMIXNS': 'soft penalty NS',
    'HYBLSTNS': 'intermediate NS', 'HYBLST': 'intermediate SS', 'HYBNS': 'hard skeleton NS',
    'HYBLSTNS_GRU': 'intermediate NS (GRU backbone)',
    'HYBLSTNS_FREED': 'intermediate NS (softplus percolation)',
    'HYBLST_GRU': 'intermediate SS (GRU backbone)',
    'HYBLST_FREED': 'intermediate SS (softplus percolation)',
}


def lab_of(nm, cfg):
    """map (run, cfg) to a controlled label; return None to exclude from this decomposition"""
    if cfg in ('HYBLSTNS', 'HYBLST'):
        base = cfg
        if re.search(r'hblg', nm):
            return base + '_GRU'
        if re.search(r'hblf', nm):
            return base + '_FREED'
        return base
    if cfg in ('LSTMNS', 'LSTM', 'LPMIX', 'LPMIXNS', 'HYBNS'):
        return cfg
    return None


# (variable name, start label, end label) -- end minus start
CONTRASTS_NS = [
    ('FORM', 'LSTMNS', 'HYBLSTNS'),          # pure form effect (backbone balanced)
    ('BACKBONE', 'HYBLSTNS', 'HYBNS'),       # pure backbone/design effect (update rule balanced)
    ('TOTAL', 'LSTMNS', 'HYBNS'),            # total effect (= FORM + BACKBONE)
    ('SOFTONLY', 'LSTMNS', 'LPMIX'),         # soft vs none (cross-regime, for reference only)
]
CONTRASTS_SS = [
    ('FORM', 'LSTM', 'HYBLST'),
    ('SOFTONLY', 'LSTM', 'LPMIX'),
]
PROBES = [
    ('CELL(backbone lstm to gru)', 'HYBLSTNS', 'HYBLSTNS_GRU'),
    ('CELL_SS(backbone lstm to gru)', 'HYBLST', 'HYBLST_GRU'),
    ('DFORM(percolation sat to free)', 'HYBLSTNS', 'HYBLSTNS_FREED'),
]
# The training regime (scheduled sampling) is itself a factor previously confounded into the FREE family
STAGE = [
    ('SSEFFECT unconstrained(NS→SS)', 'LSTMNS', 'LSTM'),
    ('SSEFFECT intermediate(NS→SS)', 'HYBLSTNS', 'HYBLST'),
]


def boot(P, labels, nboot=2000, seed=20261006):
    """station-clustered bootstrap; returns {layer: {label: dict(sample arrays of rho/x/ratio)}}"""
    raw = {}
    for (nm, cfg, lay), v in P.items():
        lab = lab_of(nm, cfg)
        if lab is None or lab not in labels:
            continue
        ids = v.id.values
        ow = M395.within(v.obs, v.id)
        rw = M395.within(v.smci, v.id)
        mw = M395.within(v['model'], v.id)
        e1, e3 = rw - ow, mw - rw
        u, inv = np.unique(ids, return_inverse=True)
        k = len(u)
        raw.setdefault(lay, {}).setdefault(lab, {})[(nm, cfg)] = dict(
            ids=u,
            n=np.bincount(inv, minlength=k).astype(float),
            A=np.bincount(inv, weights=e1 ** 2, minlength=k),
            B=np.bincount(inv, weights=e3 ** 2, minlength=k),
            C=np.bincount(inv, weights=e1 * e3, minlength=k))

    rng = np.random.default_rng(seed)
    res = {}
    for lay, labs in raw.items():
        # common station set
        common = None
        for lab, runs in labs.items():
            for key, d in runs.items():
                s = d['ids']
                common = set(s) if common is None else (common & set(s))
        common = np.sort(np.array(sorted(common)))
        ns = len(common)
        prep = {}
        for lab, runs in labs.items():
            for key, d in runs.items():
                pos = {s: i for i, s in enumerate(d['ids'])}
                ii = np.array([pos[s] for s in common])
                prep.setdefault(lab, []).append(
                    (d['n'][ii], d['A'][ii], d['B'][ii], d['C'][ii]))
        acc = {lab: {'rho': [], 'x': [], 'ratio': []} for lab in labs}
        for _ in range(nboot):
            idx = rng.integers(0, ns, ns)
            for lab, lst in prep.items():
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
                acc[lab]['rho'].append(np.mean(rs))
                acc[lab]['x'].append(np.mean(xs))
                acc[lab]['ratio'].append(np.mean(rr))
        res[lay] = {lab: {k: np.array(v) for k, v in d.items()}
                    for lab, d in acc.items()}
        res[lay]['_nsta'] = np.array([ns])
    return res


def ci(a, lo=2.5, hi=97.5):
    return float(np.mean(a)), float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True)
    ap.add_argument('--pattern', required=True, help="glob for the monthly-pair file suffix, e.g. 'gpu*' 'xr2_*'")
    ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()

    P = M395.load_pairs(a.pattern)
    if not len(P):
        print('!! no samples (%s)' % a.pattern)
        return
    labs = sorted(set(l for (nm, cfg, lay) in P
                      if (l := lab_of(nm, cfg)) is not None))
    o = []
    app = o.append
    app('=' * 100)
    app('37 Architectural-confound decomposition -- %s' % a.region)
    app('=' * 100)
    app('Controlled labels included: ' + ', '.join('%s(%s)' % (PRETTY.get(l, l), l) for l in labs))
    ncfg = {}
    for (nm, cfg, lay), v in P.items():
        ncfg[(lab_of(nm, cfg), lay)] = ncfg.get((lab_of(nm, cfg), lay), 0) + 1
    app('Configuration count per label (both layers): ' + '  '.join(
        '%s:%d' % (PRETTY.get(k[0], k[0]), v) for k, v in sorted(ncfg.items())
        if k[1] == '10cm'))

    BS = boot(P, labs, nboot=a.nboot)
    rows = []
    for lay in LAYERS:
        if lay not in BS:
            continue
        d = BS[lay]
        ns = int(d['_nsta'][0])
        app('')
        app('---------------- %s   station count %d   bootstrap %d resamples ----------------' %
            (lay, ns, a.nboot))
        app('%-22s %-16s %-16s %-20s' % ('label', 'rho [95% CI]', 'x=E3/E1', 'E2/E1 [95% CI]'))
        for lab in labs:
            if lab not in d:
                continue
            rm, rl, rh = ci(d[lab]['rho'])
            xm, _, _ = ci(d[lab]['x'])
            tm, tl, th = ci(d[lab]['ratio'])
            app('%-22s %+7.4f[%+.4f,%+.4f] %8.4f        %.4f[%.4f,%.4f]' %
                (PRETTY.get(lab, lab), rm, rl, rh, xm, tm, tl, th))
            rows.append(dict(region=a.region, layer=lay, kind='level', label=lab,
                             rho=rm, rho_lo=rl, rho_hi=rh, x=xm, ratio=tm,
                             ratio_lo=tl, ratio_hi=th, nsta=ns))
        for tag, l1, l2 in CONTRASTS_NS + CONTRASTS_SS + PROBES + STAGE:
            if l1 not in d or l2 not in d:
                continue
            for q in ('rho', 'x', 'ratio'):
                m, lo, hi = ci(d[l2][q] - d[l1][q])
                rows.append(dict(region=a.region, layer=lay, kind=tag, label=q,
                                 rho=np.nan, rho_lo=np.nan, rho_hi=np.nan,
                                 x=np.nan, ratio=np.nan, ratio_lo=np.nan,
                                 ratio_hi=np.nan, nsta=ns))
                rows[-1][q] = m
                rows[-1][q + '_lo'] = lo
                rows[-1][q + '_hi'] = hi
        # print the three NS-regime chains
        app('')
        app('  chain decomposition (NS regime): ρ            x          E2/E1')
        for tag, l1, l2 in CONTRASTS_NS + PROBES:
            if l1 not in d or l2 not in d:
                continue
            out = []
            for q in ('rho', 'x', 'ratio'):
                m, lo, hi = ci(d[l2][q] - d[l1][q])
                star = '*' if (lo * hi > 0) else ' '
                out.append('%+.4f[%+.4f,%+.4f]%s' % (m, lo, hi, star))
            app('  %-28s %s  %s  %s' % (tag, out[0], out[1], out[2]))
        # SS regime
        if 'LSTM' in d and 'HYBLST' in d:
            app('')
            app('  chain decomposition (SS regime): ρ            x          E2/E1')
            for tag, l1, l2 in CONTRASTS_SS + STAGE:
                if l1 not in d or l2 not in d:
                    continue
                out = []
                for q in ('rho', 'x', 'ratio'):
                    m, lo, hi = ci(d[l2][q] - d[l1][q])
                    star = '*' if (lo * hi > 0) else ' '
                    out.append('%+.4f[%+.4f,%+.4f]%s' % (m, lo, hi, star))
                app('  %-28s %s  %s  %s' % (tag, out[0], out[1], out[2]))

    df = pd.DataFrame(rows)
    op = os.path.join(OD, 'confound_%s.csv' % a.region)
    df.to_csv(op, index=False, encoding='utf-8-sig')
    app('')
    app('written -> %s' % op)
    txt = '\n'.join(o)
    with open(os.path.join(HERE, '_398_%s.txt' % a.region), 'w',
              encoding='utf-8') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
