# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
33 Final paper-number recheck (fixing the fam-classification bug in 27/29/30)

bug: 27.fam() assigned the family from the run name, but run='gpu' produced both HYBNS and LSTM rows,
     so gpu/LSTM (the standard-hyperparameter LSTM) was wrongly merged into HARD.
     After the fix: FREE n=7 (including gpu/LSTM), HARD n=9.

Output: 
  02_Results_Data/truth_chain/paper_families.csv   family x layer summary
  02_Results_Data/truth_chain/paper_tests.csv      between-group Welch test
  02_Results_Data/truth_chain/paper_price.csv      price of truth (corrected family)
  02_Results_Data/truth_chain/paper_oos.csv        hold-out calibration (corrected family)
  scripts/_33_final_numbers.txt                        readable report
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: main-text Table 5 (four-family outcome summary) and other family-level summary numbers
#

import os
import io
import numpy as np
import pandas as pd
from scipy import stats

B = TRUTH
TXT = LOG + '/_33_final_numbers.txt'

# ---- correct family definition (by run x cfg, not run alone) ----
FREE_RUNCFG = [
    ('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'), ('gpu_LSTMs21', 'LSTM'),
    ('gpu_lstmns', 'LSTMNS'),
    ('gpu_deg5', 'LSTM'), ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM'),
]
# the "standard configuration" subset of the unconstrained family: LSTM/LSTMNS without capacity/epoch reduction
FREE_PROPER = [('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'),
               ('gpu_LSTMs21', 'LSTM'), ('gpu_lstmns', 'LSTMNS')]
DEGRADED_CAP = [('gpu_deg5', 'LSTM'), ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM')]
DAMAGED = [('gpu_degwd', 'LSTM'), ('gpu_wd0.01', 'LSTM'), ('gpu_wd0.03', 'LSTM')]

FAMS = ['FREE', 'SOFT', 'HARD', 'DAMAGED']
FAMNAME = {'FREE': 'unconstrained LSTM/LSTMNS', 'SOFT': 'soft penalty LPMIX',
           'HARD': 'hard skeleton HYBNS', 'DAMAGED': 'training-damaged LSTM'}


def assign(row):
    key = (row['run'], row['cfg'])
    if key in DAMAGED:
        return 'DAMAGED'
    if key in FREE_RUNCFG:
        return 'FREE'
    if row['cfg'] in ('LPMIX', 'LPMIXNS'):
        return 'SOFT'
    if row['cfg'] in ('RIDGE', 'LGBM'):
        return 'ML'      # conventional supervised-learning baseline (M3, 49/16/19/25)
    if row['cfg'] in ('HYBLST', 'HYBLSTNS'):
        return 'HYBL'
    if row['cfg'] == 'HYBNS':
        return 'HARD'
    return 'OTHER'


def main():
    W = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
    W['fam'] = W.apply(assign, axis=1)
    assert (W.fam == 'OTHER').sum() == 0, W[W.fam == 'OTHER']

    out = []

    def ap(s=''):
        out.append(s)
        print(s)

    ap('=' * 96)
    ap('33 Final paper-number recheck -- fixing the fam classification (27/29/30 wrongly merged gpu/LSTM into HARD)')
    ap('=' * 96)
    ap('')
    ap('[Family sizes]')
    for lay in ['10cm', '30cm']:
        s = W[W.layer == lay]
        ap('  %s : ' % lay + '  '.join('%s=%d' % (k, int((s.fam == k).sum())) for k in FAMS))
    ap('')

    # ---------- 1. Derived quantities ----------
    W = W.copy()
    W['x'] = W.E3w / W.E1w
    W['ratio'] = W.E2w / W.E1w
    W['floor'] = np.sqrt(1.0 - W.rho_w ** 2)
    W['exc'] = W.x + W.rho_w
    W['chk'] = np.sqrt(W.floor ** 2 + W.exc ** 2)
    W['beat'] = (W.ratio < 1.0).astype(int)
    ap('[Identity check] max |chk - E2/E1| = %.3e' % np.abs(W.chk - W.ratio).max())
    ap('')

    # ---------- 2. Family summary ----------
    rows = []
    for lay in ['10cm', '30cm']:
        for k in FAMS:
            s = W[(W.layer == lay) & (W.fam == k)].sort_values('run')
            rows.append(dict(layer=lay, fam=k, fam_cn=FAMNAME[k], n=len(s),
                             E1w=s.E1w.mean(), E3w=s.E3w.mean(), E2w=s.E2w.mean(),
                             x_mean=s.x.mean(), x_sd=s.x.std(ddof=1),
                             rho_mean=s.rho_w.mean(), rho_sd=s.rho_w.std(ddof=1),
                             floor_mean=s.floor.mean(), exc_mean=s.exc.mean(),
                             exc_sd=s.exc.std(ddof=1),
                             ratio_mean=s.ratio.mean(), ratio_sd=s.ratio.std(ddof=1),
                             ratio_min=s.ratio.min(), ratio_max=s.ratio.max(),
                             beat=int(s.beat.sum())))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(B, 'paper_families.csv'), index=False, encoding='utf-8-sig')

    ap('[Table: family summary (within-station anomaly convention)]')
    for lay in ['10cm', '30cm']:
        ap('  --- %s ---' % lay)
        ap('  %-8s %3s %8s %8s %8s %9s %9s %9s %6s' %
           ('family', 'n', 'x=E3/E1', 'rho', 'FLOOR', 'EXC=x+rho', 'E2/E1', 'sd', 'beats label'))
        for _, r in F[F.layer == lay].iterrows():
            ap('  %-8s %3d %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %3d/%d' %
               (r.fam, r.n, r.x_mean, r.rho_mean, r.floor_mean, r.exc_mean,
                r.ratio_mean, r.ratio_sd, r.beat, r.n))
        ap('')

    # excess ratio beyond the optimum
    ap('[Deviation from optimum] EXC / (-rho)  =  x/(-rho) - 1')
    for lay in ['10cm', '30cm']:
        for k in FAMS:
            s = W[(W.layer == lay) & (W.fam == k)]
            ap('  %s %-8s  x=%.4f  -rho=%.4f  excess %.0f%%' %
               (lay, k, s.x.mean(), -s.rho_w.mean(),
                100 * (s.x.mean() / (-s.rho_w.mean()) - 1)))
    ap('')

    # ---------- 3. Between-group Welch test ----------
    tests = []
    pairs = [('FREE', 'SOFT'), ('FREE', 'HARD'), ('SOFT', 'HARD'), ('HARD', 'DAMAGED')]
    for lay in ['10cm', '30cm']:
        for var, cn in [('rho_w', 'rho'), ('x', 'x'), ('exc', 'EXC'), ('ratio', 'E2E1')]:
            for a, b in pairs:
                va = W[(W.layer == lay) & (W.fam == a)][var].values
                vb = W[(W.layer == lay) & (W.fam == b)][var].values
                if len(va) < 2 or len(vb) < 2:
                    continue
                t, p = stats.ttest_ind(va, vb, equal_var=False)
                tests.append(dict(layer=lay, var=cn, a=a, b=b,
                                  mean_a=va.mean(), mean_b=vb.mean(),
                                  sd_a=va.std(ddof=1), sd_b=vb.std(ddof=1),
                                  t=t, p=p))
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(B, 'paper_tests.csv'), index=False, encoding='utf-8-sig')

    ap('[Table: between-group Welch t-test (unequal variance)]')
    for lay in ['10cm', '30cm']:
        ap('  --- %s ---' % lay)
        for _, r in T[T.layer == lay].iterrows():
            ap('  %-5s %-6s vs %-6s : %+.4f±%.4f vs %+.4f±%.4f  t=%+.2f  p=%.3g' %
               (r['var'], r.a, r.b, r.mean_a, r.sd_a, r.mean_b, r.sd_b, r.t, r.p))
        ap('')

    # FREE-proper subset
    ap('[Sensitivity: within the unconstrained family] proper(4) vs capacity/epoch-reduced(3)')
    for lay in ['10cm', '30cm']:
        for nm, sel in [('proper', FREE_PROPER), ('degraded', DEGRADED_CAP)]:
            s = W[(W.layer == lay) &
                  W.apply(lambda r: (r['run'], r['cfg']) in sel, axis=1)]
            ap('  %s %-9s n=%d rho=%.4f±%.4f x=%.4f E2/E1=%.4f±%.4f' %
               (lay, nm, len(s), s.rho_w.mean(), s.rho_w.std(ddof=1),
                s.x.mean(), s.ratio.mean(), s.ratio.std(ddof=1)))
        a = W[(W.layer == lay) & W.apply(lambda r: (r['run'], r['cfg']) in FREE_PROPER, axis=1)].rho_w.values
        b = W[(W.layer == lay) & W.apply(lambda r: (r['run'], r['cfg']) in DEGRADED_CAP, axis=1)].rho_w.values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        ap('  %s proper vs degraded rho: p=%.3g' % (lay, p))
    ap('')

    # ---------- 4. Within SOFT: rho vs log(w) ----------
    import re
    P = pd.read_csv(os.path.join(B, 'shrink_optimality.csv'), encoding='utf-8-sig')
    PALL = P.copy()
    P = P[P.cfg == 'LPMIX'].copy()
    wmap = {}
    for r in P.run.unique():
        m = re.match(r'gpu_wbw([0-9.]+)(?:s\d+)?$', r)
        wmap[r] = float(m.group(1)) if m else np.nan
    P['w'] = P.run.map(wmap)
    assert P.w.notna().all(), P[P.w.isna()].run.unique()
    ap('[Within SOFT: regression of rho / EXC on log10(w)]')
    for lay in ['10cm', '30cm']:
        s = P[P.layer == lay].dropna(subset=['w'])
        for var in ['rho', 'x']:
            lw = np.log10(s.w.values)
            v = s[var].values
            sl, ic, rv, pv, se = stats.linregress(lw, v)
            ap('  %s %-4s slope=%+.4f /decade-of-w  p=%.3g  R2=%.2f  (n=%d)' %
               (lay, var, sl, pv, rv ** 2, len(s)))
    ap('')

    # ---------- 5. Price of truth (corrected family) ----------
    PR = pd.read_csv(os.path.join(B, 'shrink_price.csv'), encoding='utf-8-sig')
    PR['fam2'] = PR.apply(assign, axis=1)
    ap('[Table: price of truth -- calibrate lambda* on k stations, evaluate on the complement (E2/E1, lower is better)]')
    prows = []
    for lay in ['10cm', '30cm']:
        ap('  --- %s ---' % lay)
        ap('  %-8s %3s %8s %8s %8s %8s %8s %8s %8s' %
           ('family', 'n', 'raw', 'oracle', 'k=3', 'k=10', 'k=20', 'k=42', 'capture fraction k3'))
        for k in FAMS:
            s = PR[(PR.layer == lay) & (PR.fam2 == k)]
            if not len(s):
                continue
            cr = (s.base - s.k3) / (s.base - s.oracle)
            prows.append(dict(layer=lay, fam=k, n=len(s),
                              base=s.base.mean(), oracle=s.oracle.mean(),
                              k3=s.k3.mean(), k10=s.k10.mean(),
                              k20=s.k20.mean(), k42=s.k42.mean(),
                              capture_k3=cr.mean(),
                              k3_sd=s.k3_sd.mean()))
            ap('  %-8s %3d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %+7.0f%%' %
               (k, len(s), s.base.mean(), s.oracle.mean(), s.k3.mean(),
                s.k10.mean(), s.k20.mean(), s.k42.mean(), 100 * cr.mean()))
        ap('')
    PD = pd.DataFrame(prows)
    PD.to_csv(os.path.join(B, 'paper_price.csv'), index=False, encoding='utf-8-sig')

    # ---------- 6. Hold-out calibration (corrected family) ----------
    OS = pd.read_csv(os.path.join(B, 'shrink_oos.csv'), encoding='utf-8-sig')
    OS['fam2'] = OS.apply(assign, axis=1)
    ap('[Table: hold-out calibration (two hold-out schemes A/B)]')
    orows = []
    for lay in ['10cm', '30cm']:
        ap('  --- %s ---' % lay)
        ap('  %-8s %3s %8s %8s %8s %8s %8s' %
           ('family', 'n', 'raw', 'shrink_A', 'shrink_B', 'lam_A', 'lam_B'))
        for k in FAMS:
            s = OS[(OS.layer == lay) & (OS.fam2 == k)]
            if not len(s):
                continue
            orows.append(dict(layer=lay, fam=k, n=len(s), base=s.base.mean(),
                              A=s.shrink_A.mean(), B=s.shrink_B.mean(),
                              lam_A=s.lam_A.mean(), lam_B=s.lam_B.mean(),
                              floor=s.floor.mean()))
            ap('  %-8s %3d %8.4f %8.4f %8.4f %8.4f %8.4f' %
               (k, len(s), s.base.mean(), s.shrink_A.mean(), s.shrink_B.mean(),
                s.lam_A.mean(), s.lam_B.mean()))
        ap('')
    OD = pd.DataFrame(orows)
    OD.to_csv(os.path.join(B, 'paper_oos.csv'), index=False, encoding='utf-8-sig')

    # ---------- 7. E1 structure ----------
    ap('[Between-/within-station decomposition of E1]')
    for lay in ['10cm', '30cm']:
        s = W[W.layer == lay]
        e1f = s.E1f.iloc[0] ** 2
        e1w = s.E1w.iloc[0] ** 2
        ap('  %s  MSE_full=%.6f  MSE_within=%.6f  between-station static share %.1f%%' %
           (lay, e1f, e1w, 100 * (1 - e1w / e1f)))
    ap('')

    # ---------- 8. Machine-precision check of the calibration condition ----------
    ap('[Check of the three predictions of the calibration condition] all %d configuration x layer' % len(PALL))
    d_rho = np.abs(PALL.rho_check - PALL.rho).max()
    d_lam = np.abs(PALL.lam_th - PALL.lam_em).max()
    d_opt = np.abs(PALL.E2E1_opt - PALL.floor).max()
    ap('  P1 rho scale-invariant        max dev = %.3e' % d_rho)
    ap('  P2 empirical lambda* = theoretical -rho/x max dev = %.3e' % d_lam)
    ap('  P3 E2/E1=FLOOR at optimum   max dev = %.3e' % d_opt)
    ap('  P2 mean |dev| = %.2e ; P3 mean |dev| = %.2e'
       % (np.abs(PALL.lam_th - PALL.lam_em).mean(),
          np.abs(PALL.E2E1_opt - PALL.floor).mean()))

    # proper subset vs SOFT
    ap('')
    ap('[Supplement: unconstrained proper subset (n=4) vs SOFT (n=15)]')
    for lay in ['10cm', '30cm']:
        a = W[(W.layer == lay) &
              W.apply(lambda r: (r['run'], r['cfg']) in FREE_PROPER, axis=1)]
        b = W[(W.layer == lay) & (W.fam == 'SOFT')]
        for var, cn in [('rho_w', 'rho'), ('x', 'x'), ('ratio', 'E2/E1')]:
            t, p = stats.ttest_ind(a[var].values, b[var].values, equal_var=False)
            ap('  %s %-6s %.4f±%.4f vs %.4f±%.4f  p=%.3g'
               % (lay, cn, a[var].mean(), a[var].std(ddof=1),
                  b[var].mean(), b[var].std(ddof=1), p))

    with io.open(TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print('\nwrote', TXT)


if __name__ == '__main__':
    main()
