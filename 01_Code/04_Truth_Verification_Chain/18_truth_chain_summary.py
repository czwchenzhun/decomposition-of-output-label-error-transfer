# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""18_truth_chain_summary.py -- merge the three-layer "model vs truth" verification chain into a single manuscript table

Three layers (missing any one leads to misreading)
----------------------------
  L1  reference field ↔ station   (15)  how accurate the reference field itself is -- determines the applicability boundary of the extrapolated output
  L2  model   ↔ station   (16/17)  how accurate the extrapolated output is for true soil moisture
  L3  model   ↔ reference field (already in 13)  whether the model faithfully reproduces the reference field

Core criterion
--------
  * If L2 ≈ L1, **extrapolation has introduced no error beyond that of the reference field itself** -- a strong statement that can go into the conclusion.
  * If L2 is clearly > L1, only the excess is "the model's fault".
  * L3, however good, cannot alone serve as accuracy evidence (it only shows whether it resembles the reference field).

Output
----
  09_stations/truth_chain/truth_chain_table.csv     Main manuscript table (long form)
  09_stations/truth_chain/truth_chain_table.md      ready-to-paste Markdown
  scripts/_18_run.txt                              Log
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

from common import ROOT, OUT

OUTDIR = TRUTH

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fmt(v, nd=4):
    try:
        if not np.isfinite(float(v)):
            return '-'
        return ('%.*f' % (nd, float(v)))
    except Exception:
        return '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='R0_northchina')
    args = ap.parse_args()

    P('=== 18 three-layer verification chain summary ===')

    rows = []

    # ---------- L1: reference field vs station (full period 2000-2013, nationwide + by region)
    f1 = os.path.join(OUTDIR, 'truth_chain_L1_region.csv')
    if os.path.exists(f1):
        L1 = pd.read_csv(f1)
        for _, r in L1.iterrows():
            rows.append(dict(
                layer='L1 reference field↔station', scope=r['region'],
                target=r['layer'], lead='-', source='SMCI reference field',
                nsta=int(r['nsta']), R=r['R_med'], bias=r['bias_med'],
                rmse=r['rmse_med'], ubRMSE=r['ub_med'], n=int(r['pooled_n'])))
        P('L1 loaded %d rows' % len(L1))
    else:
        P('!! missing L1: %s' % f1)

    # ---------- L2: model vs station (monthly scale, multiple stations)
    f2 = os.path.join(OUTDIR, 'truth_chain_L2_region.csv')
    if os.path.exists(f2):
        L2 = pd.read_csv(f2)
        for _, r in L2.iterrows():
            rows.append(dict(
                layer='L2 model↔station', scope=r['region'],
                target=r['layer'], lead='%d days' % int(r['horizon']),
                source=r['source'], nsta=int(r['nsta']),
                R=r['R_med'], bias=r['bias_med'], rmse=r['rmse_med'],
                ubRMSE=r['ub_med'], n=int(r['nobs'])))
        P('L2 loaded %d rows' % len(L2))
    else:
        P('!! missing L2: %s (run 16 first)' % f2)

    # ---------- L3: model vs reference field (existing json in 13)
    for fp in sorted(glob.glob(os.path.join(BENCH, 'dl_hybrid_result*.json'))):
        try:
            J = json.load(open(fp, encoding='utf-8'))
        except Exception:
            continue
        tag = os.path.basename(fp).replace('dl_hybrid_result', '').replace('.json', '')
        for rec in (J if isinstance(J, list) else J.get('results', [])):
            if 'test' not in rec:
                continue
            t = rec['test']
            for lay, key in (('10cm', 'test_l1'), ('30cm', 'test_l2')):
                if key not in rec:
                    continue
                rows.append(dict(
                    layer='L3 model↔reference field', scope='in/out-of-sample experiment%s' % tag,
                    target=lay, lead='whole-period mean',
                    source=rec.get('cfg', '?'), nsta=np.nan,
                    R=np.nan, bias=np.nan,
                    rmse=rec[key].get('all', np.nan), ubRMSE=np.nan,
                    n=np.nan))
        P('L3 loaded %s' % os.path.basename(fp))

    T = pd.DataFrame(rows)
    if len(T) == 0:
        P('!! no data for any layer')
        from common import report
        report('18_truth_chain_summary', LOG)
        return
    T.to_csv(os.path.join(OUTDIR, 'truth_chain_table.csv'), index=False,
             encoding='utf-8-sig')

    # ---------- Markdown
    md = []
    md.append('### Table: three-layer verification chain (station observations never entered training; the whole chain is strictly out of sample)')
    md.append('')
    md.append('| Layer | Scope | Target layer | Lead days | Convention | Station count | R | bias | RMSE | ubRMSE |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    for lay in ['L1 reference field↔station', 'L2 model↔station', 'L2b model↔station(daily)', 'L3 model↔reference field']:
        g = T[T.layer == lay]
        if len(g) == 0:
            continue
        for _, r in g.iterrows():
            md.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                lay, r['scope'], r['target'], r['lead'], r['source'],
                ('-' if not np.isfinite(r['nsta']) else int(r['nsta'])),
                fmt(r['R'], 3), fmt(r['bias'], 4), fmt(r['rmse'], 4),
                fmt(r['ubRMSE'], 4)))
        md.append('')

    # ---------- Criterion
    P('')
    P('--- Criterion: excess error of L2 (model↔truth) relative to "concurrent reference field↔truth" ---')
    md.append('**Criterion**:')
    if os.path.exists(f2):
        L2 = pd.read_csv(f2)
        L1 = pd.read_csv(f1) if os.path.exists(f1) else None
        for lay in ('10cm', '30cm'):
            # The baseline must be the **SMCI row in the L2 table** (same test period and same set of days as the model),
            # not L1 (full period 2000-2013) -- mixing the two periods in the comparison gives a false conclusion
            b = L2[(L2.source == 'SMCI') & (L2.layer == lay)]
            if len(b) == 0 and L1 is not None:
                b = L1[(L1.region == args.region) & (L1.layer == lay)]
            if len(b) == 0:
                continue
            b0 = float(b.ub_med.iloc[0])
            bR = float(b.R_med.iloc[0])
            for src in sorted(set(L2.source)):
                if src == 'SMCI':
                    continue
                g = L2[(L2.source == src) & (L2.layer == lay)].sort_values('horizon')
                for _, r in g.iterrows():
                    du = float(r['ub_med']) - b0
                    dR = float(r['R_med']) - bR
                    P('  %-5s %-8s h=%-3d  ubRMSE %+0.4f (%+6.1f%% vs reference-field error %.4f)  R %+.3f'
                      % (lay, src.replace('MODEL:', ''), int(r['horizon']), du,
                         100 * du / b0, b0, dR))
                    md.append('- %s %s lead %d days: ubRMSE relative to reference-field error %+0.4f (%+6.1f%%), R %+.3f'
                              % (lay, src.replace('MODEL:', ''), int(r['horizon']),
                                 du, 100 * du / b0, dR))

    fp = os.path.join(OUTDIR, 'truth_chain_table.md')
    open(fp, 'w', encoding='utf-8').write('\n'.join(md))
    P('')
    P('WROTE %s (%d rows)' % (os.path.join(OUTDIR, 'truth_chain_table.csv'), len(T)))
    P('WROTE %s' % fp)

    from common import report
    report('18_truth_chain_summary', LOG)


if __name__ == '__main__':
    main()
