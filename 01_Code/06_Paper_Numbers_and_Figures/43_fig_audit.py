# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""43: Figure-number audit -- recompute the statistics annotated in each figure and compare with main-text/script hard-coded values.

Read-only. Output to 02_Results_Data/truth_chain/fig_audit.txt
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

B = TRUTH
M = os.path.join(B, 'multi')
L = []

FREE_RUNCFG = [('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'), ('gpu_LSTMs21', 'LSTM'),
               ('gpu_lstmns', 'LSTMNS'), ('gpu_deg5', 'LSTM'),
               ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM')]
DAMAGED = [('gpu_degwd', 'LSTM'), ('gpu_wd0.01', 'LSTM'), ('gpu_wd0.03', 'LSTM')]


def assign(row):
    key = (row['run'], row['cfg'])
    if key in DAMAGED:
        return 'DAMAGED'
    if key in FREE_RUNCFG:
        return 'FREE'
    if row['cfg'] == 'LPMIX':
        return 'SOFT'
    if row['cfg'] == 'HYBNS':
        return 'HARD'
    return None


def ap(s=''):
    L.append(s)
    print(s, flush=True)


W = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
W['fam'] = W.apply(assign, axis=1)
W['x'] = W.E3w / W.E1w
W['ratio'] = W.E2w / W.E1w
W = W.dropna(subset=['fam'])
W['w'] = [float(m.group(1)) if (m := re.match(r'gpu_wbw([0-9.]+)(?:s\d+)?$', r)) else np.nan
          for r in W.run]

# ---------------------------------------------------------------- Fig. 4
ap('=' * 78)
ap('[Fig. 04 e1_decomposition] in-figure note "dilution of about 3x"')
ap('-' * 78)
for lay in ['10cm', '30cm']:
    for cfg in ['LSTM', 'HYBNS']:
        s = W[(W.run == 'gpu') & (W.cfg == cfg) & (W.layer == lay)]
        if not len(s):
            continue
        af = float(s.a_f.iloc[0]) * 100
        aw = float(s.a_w.iloc[0]) * 100
        ap('  %s %-6s full-convention a_f=%.3f%%   within-station a_w=%.3f%%   total dilution=%.2fx'
           % (lay, cfg, af, aw, aw / af))
# denominator/numerator decomposition (v17 main-text convention)
for lay in ['10cm', '30cm']:
    for cfg in ['LSTM', 'HYBNS']:
        v = W[(W.cfg == cfg) & (W.layer == lay)]
        if not len(v):
            continue
        af = float(v.a_f.mean())
        aw = float(v.a_w.mean())
        E1f = float(v.E1f.mean())
        E1w = float(v.E1w.mean())
        denom = (E1f / E1w) ** 2
        ap('  %s %-6s family mean a_f=%.5f a_w=%.5f dilution=%.2fx | denominator=%.3fx numerator=%.3fx'
           % (lay, cfg, af, aw, aw / af, denom, (aw / af) / denom))
ap('  Note: panel (b) plots only run=gpu LSTM and HYBNS; the in-figure text "dilution of about 3x" is a manual annotation.')

# ---------------------------------------------------------------- Fig. 6
ap()
ap('=' * 78)
ap('[Fig. 06 form_vs_strength] in-figure annotation: slope, p=4.5e-7')
ap('-' * 78)
for q, lab in (('rho_w', 'rho'), ('x', 'x')):
    for lay in ['10cm', '30cm']:
        s = W[(W.fam == 'SOFT') & (W.layer == lay)].dropna(subset=['w'])
        if len(s) < 3:
            continue
        sl, ic, rv, pv, se = stats.linregress(np.log10(s.w.values), s[q].values)
        ap('  (a/b) SOFT %s %s vs log10 w: slope %+.4f  p=%.3g  n=%d'
           % (lay, lab, sl, pv, len(s)))
a = W[(W.fam == 'FREE') & (W.layer == '10cm')].rho_w.values
h = W[(W.fam == 'HARD') & (W.layer == '10cm')].rho_w.values
t, pv = stats.ttest_ind(a, h, equal_var=False)
ap('  (c) FREE vs HARD rho 10 cm: Welch t=%.3f p=%.3g  (in-figure annotation 4.5e-7)  n=%d/%d'
   % (t, pv, len(a), len(h)))
u, pu = stats.mannwhitneyu(a, h, alternative='two-sided')
ap('      Mann-Whitney U p=%.3g' % pu)
for lay in ['10cm', '30cm']:
    for f in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
        v = W[(W.fam == f) & (W.layer == lay)].rho_w
        ap('      %s %-8s rho mean %+.4f  n=%d' % (lay, f, v.mean(), len(v)))

# ---------------------------------------------------------------- Fig. 9
ap()
ap('=' * 78)
ap('[Fig. 09 calibration] in-figure annotation "max deviation 4.9e-4 (n=34x2)"')
ap('-' * 78)
OP = pd.read_csv(os.path.join(B, 'shrink_optimality.csv'), encoding='utf-8-sig')
OP['fam'] = OP.apply(assign, axis=1)
d = (OP.lam_th - OP.lam_em).abs()
ap('  n=%d  max deviation=%.3e  mean=%.3e' % (len(OP), d.max(), d.mean()))
for lay in ['10cm', '30cm']:
    s = OP[OP.layer == lay]
    dd = (s.lam_th - s.lam_em).abs()
    ap('    %s n=%d max=%.3e mean=%.3e' % (lay, len(s), dd.max(), dd.mean()))
ap('  Note: panel (b) plots only lam_th vs lam_em points, so n should be %d (not 34x2=68).' % len(OP))

# ---------------------------------------------------------------- Fig. 11
ap()
ap('=' * 78)
ap('[Fig. 11 trap] main-text caption states alpha +0.99 pp/decade, p=1.2e-4')
ap('-' * 78)
s = W[(W.fam == 'SOFT') & (W.layer == '10cm')].dropna(subset=['w'])
sl1, ic1, r1, p1, _ = stats.linregress(np.log10(s.w.values), s.a_w.values * 100)
ap('  (a) configuration-level alpha: slope %+.3f pp/decade  p=%.3g  n=%d' % (sl1, p1, len(s)))
g = s.groupby('w').agg(alpha=('a_w', 'mean')).reset_index()
slg, _, _, pg, _ = stats.linregress(np.log10(g.w.values), g.alpha.values * 100)
ap('  (a) aggregated-by-w level:   slope %+.3f pp/decade  p=%.3g  n=%d' % (slg, pg, len(g)))
sl3, ic3, r3, p3, _ = stats.linregress(np.log10(s.w.values), s.ratio.values)
ap('  (b) configuration-level E2/E1:  slope %+.5f /decade  p=%.3g' % (sl3, p3))
ap('  Note: the trend line uses the configuration-level regression (lines 38-53).')

# ---------------------------------------------------------------- Fig. 12
ap()
ap('=' * 78)
ap('[Fig. 12a/12b] suptitle "consistent family ranking" and "most negative in 6 of 8"')
ap('-' * 78)
rows = []
for reg, f in (('R0_northchina', 'bootstrap_R0_northchina.csv'),
               ('R1_yangtze', 'bootstrap_R1_yangtze.csv'),
               ('R2_loess', 'bootstrap_R2_loess.csv'),
               ('R3_northeast', 'bootstrap_R3_northeast.csv')):
    p = os.path.join(M, f)
    if not os.path.exists(p):
        ap('  !! missing %s' % f)
        continue
    b = pd.read_csv(p, encoding='utf-8-sig')
    for lay in ['10cm', '30cm']:
        s = b[(b.layer == lay)]
        rs = s.set_index('fam').ratio
        rr = s.set_index('fam').rho
        order = sorted(rs.index, key=lambda k: rs[k])
        rorder = sorted(rr.index, key=lambda k: rr[k])
        rows.append((reg, lay, '/'.join(order), rorder[0],
                     ''.join(['%.3f ' % rs[k] for k in ['FREE', 'SOFT', 'HARD'] if k in rs])))
        ap('  %-14s %-5s E2/E1 ascending: %-22s | most negative rho: %-6s | F/S/H=%.3f/%.3f/%.3f'
           % (reg, lay, '/'.join(order), rorder[0],
              rs.get('FREE', np.nan), rs.get('SOFT', np.nan), rs.get('HARD', np.nan)))
# whether SOFT is most negative in all 8 combinations (including HYBL)
ap()
ap('  -- whether SOFT is most negative in all 8 (region x layer) combinations --')
for reg, f in (('R0_northchina', 'bootstrap_R0_northchina.csv'),
               ('R1_yangtze', 'bootstrap_R1_yangtze.csv'),
               ('R2_loess', 'bootstrap_R2_loess.csv'),
               ('R3_northeast', 'bootstrap_R3_northeast.csv')):
    b = pd.read_csv(os.path.join(M, f), encoding='utf-8-sig')
    for lay in ['10cm', '30cm']:
        s = b[(b.layer == lay)].set_index('fam').rho
        tri = {k: s[k] for k in ['FREE', 'SOFT', 'HARD'] if k in s.index}
        allf = dict(s)
        who3 = min(tri, key=lambda k: tri[k])
        whoa = min(allf, key=lambda k: allf[k])
        ap('  %-14s %-5s most negative among three families=%-5s  most negative among all families=%-5s (%s)'
           % (reg, lay, who3, whoa,
              'consistent' if who3 == whoa == 'SOFT' else ('SOFT surpassed by intermediate' if whoa == 'HYBL' else 'exception')))

# ---------------------------------------------------------------- Fig. 14
ap()
ap('=' * 78)
ap('[Fig. 14 price_cross] benefit capture fraction for calibration on k=3 stations')
ap('-' * 78)
for reg in ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast']:
    p = os.path.join(M, 'price_%s.csv' % reg)
    if not os.path.exists(p):
        continue
    PP = pd.read_csv(p, encoding='utf-8-sig')
    if 'k3' not in PP.columns:
        continue
    for lay in ['10cm', '30cm']:
        s = PP[PP.layer == lay]
        for fam in ['FREE', 'SOFT', 'HYBL', 'HARD']:
            g = s[s.fam == fam]
            if not len(g):
                continue
            cap = (g.base - g.k3) / (g.base - g.oracle)
            cap = np.where(np.isfinite(cap), cap, np.nan)
            ap('  %-14s %-5s %-6s mean capture fraction %+7.1f%%  n=%d  mismatch base-1=%+.4f'
               % (reg, lay, fam, 100 * np.nanmean(cap), len(g), (g.base - 1.0).mean()))

out = os.path.join(B, 'fig_audit.txt')
open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n-> ' + out)
