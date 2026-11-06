# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""56: Capacity contrast and seasonal sensitivity (fourth-round review M1 / I7)

M1  Capacity was never manipulated independently across families:
    (a) FREE(LSTM 5.27e4) and SOFT(LPMIX 2.12e5) differ by 4.0 times, but the extra
        1.6e5 parameters all sit in the flux head; as w->0 the flux head does not take part in the state update, so LPMIX
        degenerates to the same-seed LSTM. Here the empirical distance at the smallest weight w=0.03 is given.
    (b) The FREE family already contains capacity variants: hidden size 128 -> 32 (deg32); give its rho sensitivity,
        as a quantitative reference for the "magnitude of the capacity effect".
I7  Station observations are concentrated in months 3--11 with none in winter: give the warm-season subsample (months 5--10) and
    the all-sample rho / E2/E1 comparison.

Output: 02_Results_Data/truth_chain/multi/capacity_season.csv
"""
import os
import sys
import re
import glob
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
REGIONS = [('R0_northchina', 'gpu*'), ('R1_yangtze', 'xr2_*'),
           ('R2_loess', 'xr5_*'), ('R3_northeast', 'xr4_*'),
           ('R4_longzhong', 'xr7_*'), ('R5_sichuan', 'xr8_*')]
REGCN = {'R0_northchina': 'North China Plain', 'R1_yangtze': 'middle--lower Yangtze',
         'R2_loess': 'Loess Plateau', 'R3_northeast': 'Northeast China black soil',
         'R4_longzhong': 'Longzhong loess', 'R5_sichuan': 'Sichuan Basin'}


def load_all(pattern):
    """Read monthly pairs with a time column (for seasonal stratification)"""
    out = {}
    for f in sorted(glob.glob(os.path.join(B, 'truth_chain_L2_monthly_%s.csv'
                                           % pattern))):
        nm = os.path.basename(f).replace('truth_chain_L2_monthly_', '') \
            .replace('.csv', '')
        A = pd.read_csv(f, encoding='utf-8-sig', dtype={'id': str})
        if 'horizon' in A.columns:
            A = A[A.horizon == 0]
        # The monthly pairing file has two columns, year / month (integers); stratify directly by month
        tcol = 'month' if 'month' in A.columns else None
        out[nm] = (A, tcol)
    return out


def stat(v, ids=None):
    ow = M395.within(v.obs, v.id)
    rw = M395.within(v.smci, v.id)
    mw = M395.within(v['model'], v.id)
    e1, e3 = rw - ow, mw - rw
    e2 = e1 + e3
    E1 = float(np.sqrt(np.mean(e1 ** 2)))
    E3 = float(np.sqrt(np.mean(e3 ** 2)))
    E2 = float(np.sqrt(np.mean(e2 ** 2)))
    rho = float(np.mean(e1 * e3) / (E1 * E3))
    return dict(E1=E1, E3=E3, E2=E2, rho=rho, x=E3 / E1, ratio=E2 / E1)


def main():
    o = ['=' * 96, '56 Capacity contrast and seasonal sensitivity', '=' * 96]
    rows = []

    # ---------------- (a) w->0 degeneration + (b) hidden size 32 ----------------
    W = pd.read_csv(os.path.join(OD, 'withinstation_R0_northchina.csv'),
                    encoding='utf-8-sig')
    o.append('')
    o.append('[M1-a capacity contrast: are the extra 1.6e5 parameters of the soft-penalty family inert]')
    o.append('  %-16s %-5s %8s %8s %8s' % ('configuration', 'layer', 'rho', 'x', 'E2/E1'))
    pick = [('gpu', 'LSTM'), ('gpu_wbw0.03', 'LPMIX'), ('gpu_wbw0.1', 'LPMIX'),
            ('gpu_wbw0.3', 'LPMIX'), ('gpu_wbw1', 'LPMIX'), ('gpu_wbw3', 'LPMIX'),
            ('gpu_wbw10s42', 'LPMIX'), ('gpu_wbw30s42', 'LPMIX'),
            ('gpu_deg32', 'LSTM'), ('gpu_deg5', 'LSTM'), ('gpu_deg10', 'LSTM')]
    for run, cfg in pick:
        for lay in ['10cm']:
            g = W[(W.run == run) & (W.cfg == cfg) & (W.layer == lay)]
            if not len(g):
                continue
            r = g.iloc[0]
            o.append('  %-16s %-5s %+8.4f %8.4f %8.4f'
                     % (run, lay, r.rho_w, r.x, r.ratio))
            rows.append(dict(kind='capacity', item=run, layer=lay,
                             rho=float(r.rho_w), x=float(r.x),
                             ratio=float(r.ratio)))
    lst = W[(W.run == 'gpu') & (W.cfg == 'LSTM') & (W.layer == '10cm')].iloc[0]
    w003 = W[(W.run == 'gpu_wbw0.03') & (W.cfg == 'LPMIX')
             & (W.layer == '10cm')].iloc[0]
    soft = W[(W.cfg == 'LPMIX') & (W.layer == '10cm') & (W.run != 'LPMIXNS')]
    soft = soft[~soft.run.str.contains('lpmixns', case=False, na=False)]
    free = W[(W.fam == 'FREE') & (W.layer == '10cm')]
    o.append('  -> Same-seed LSTM(5.27e4) and LPMIX w=0.03(2.12e5):'
             'delta_rho=%+.4f, delta_x=%+.4f' % (float(w003.rho_w) - float(lst.rho_w),
                                       float(w003.x) - float(lst.x)))
    o.append('  -> Between-family difference of North China SOFT(15) and FREE(7): delta_rho=%+.4f, delta_x=%+.4f'
             % (soft.rho_w.mean() - free.rho_w.mean(),
                soft.x.mean() - free.x.mean()))
    d32 = W[(W.run == 'gpu_deg32') & (W.layer == '10cm')].iloc[0]
    o.append('')
    o.append('[M1-b capacity sensitivity (within the unconstrained family, North China 10cm)]')
    o.append('  hidden size 128 (gpu, seed42) rho=%+.4f -> hidden size 32 (gpu_deg32) rho=%+.4f'
             '  Δrho=%+.4f'
             % (float(lst.rho_w), float(d32.rho_w),
                float(d32.rho_w) - float(lst.rho_w)))
    o.append('  -> the rho change (%+.4f) caused by reducing capacity to about 1/4 and the SOFT/FREE between-family difference'
             '(%+.4f) are of the same order, so capacity was not independently controlled in this paper.'
             % (float(d32.rho_w) - float(lst.rho_w),
                soft.rho_w.mean() - free.rho_w.mean()))

    # ---------------- (c) seasonal sensitivity ----------------
    o.append('')
    o.append('[I7 seasonal sensitivity: warm-season (months 5--10) subsample vs all samples]')
    for region, pat in REGIONS:
        D = load_all(pat)
        if not D:
            continue
        _, tcol0 = next(iter(D.values()))
        if tcol0 is None:
            o.append('  %s: pairing file has no time column, skipped' % region)
            break
        for lay in LAYERS:
            for fam in ['FREE', 'SOFT', 'HARD']:
                acc_all = []
                acc_warm = []
                nmonth = {'all': 0, 'warm': 0}
                for nm, (A, tcol) in D.items():
                    v = A[(A.layer == lay) & (A.cfg.apply(lambda c: True))]
                    for cfg in sorted(set(A.cfg)):
                        if M395.fam_of(nm, cfg) != fam:
                            continue
                        if cfg == 'LPMIXNS':
                            continue
                        vv = A[(A.cfg == cfg) & (A.layer == lay)]
                        if len(vv) < 60:
                            continue
                        mon = pd.to_numeric(vv[tcol], errors='coerce')
                        acc_all.append(stat(vv))
                        sel = mon.isin([5, 6, 7, 8, 9, 10]).values
                        nmonth['all'] += len(vv)
                        nmonth['warm'] += int(sel.sum())
                        if sel.sum() >= 40:
                            acc_warm.append(stat(vv[sel]))
                if not acc_all or not acc_warm:
                    continue
                Aa = pd.DataFrame(acc_all)
                Aw = pd.DataFrame(acc_warm)
                o.append('  %-10s %-5s %-5s n=%2d  rho all %+.4f / warm %+.4f  '
                         'E2/E1 all %.4f / warm %.4f  warm-season share %.0f%%'
                         % (REGCN[region], lay, fam, len(Aa), Aa.rho.mean(),
                            Aw.rho.mean(), Aa.ratio.mean(), Aw.ratio.mean(),
                            100.0 * nmonth['warm'] / max(nmonth['all'], 1)))
                rows.append(dict(kind='season', region=region, layer=lay,
                                 fam=fam, n=len(Aa),
                                 rho_all=Aa.rho.mean(), rho_warm=Aw.rho.mean(),
                                 ratio_all=Aa.ratio.mean(),
                                 ratio_warm=Aw.ratio.mean(),
                                 warm_frac=nmonth['warm'] / max(nmonth['all'], 1)))
    DF = pd.DataFrame(rows)
    op = os.path.join(OD, 'capacity_season.csv')
    DF.to_csv(op, index=False, encoding='utf-8-sig')
    txt = '\n'.join(o)
    with open(os.path.join(HERE, '_409.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print()
    print('written ->', op)


if __name__ == '__main__':
    main()
