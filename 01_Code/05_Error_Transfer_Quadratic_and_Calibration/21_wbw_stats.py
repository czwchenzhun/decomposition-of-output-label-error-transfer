# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
21 Formal statistical tests of the dose response of the soft water-balance constraint

Criteria (declared in advance, to avoid post hoc rationalisation):
  T1 axis validity: E3 (fidelity error) rises monotonically with w -- proving that w really is "constraint strength"
  T2 within-seed monotonicity: α is monotonic in w within each seed (paired trend test over 3 points × 3 seeds)
  T3 between-group difference: the α difference w=3 vs w=0 exceeds the seed sd (Welch t)
  T4 form vs strength: HYBNS (hard skeleton) against LPMIX w=30 (soft, same seed, comparable E3)
"""
import os
import numpy as np
import pandas as pd

B = TRUTH + '/'

SPEC = []
SPEC.append((0.0, 42, 'LSTM', 'err_transfer_region_gpu.csv'))
for sd in (7, 21):
    SPEC.append((0.0, sd, 'LSTM', 'err_transfer_region_gpu_LSTMs%d.csv' % sd))
SPEC.append((0.0, 42, 'LSTMNS', 'err_transfer_region_gpu_lstmns.csv'))
for s, t in [('0.03', 0.03), ('0.1', 0.1), ('0.3', 0.3), ('1', 1.0), ('3', 3.0)]:
    SPEC.append((t, 42, 'LPMIX', 'err_transfer_region_gpu_wbw%s.csv' % s))
for s, t in [('10', 10.0), ('30', 30.0)]:
    SPEC.append((t, 42, 'LPMIX', 'err_transfer_region_gpu_wbw%ss42.csv' % s))
for s in ['0.3', '3', '10', '30']:
    for sd in (7, 21):
        SPEC.append((float(s), sd, 'LPMIX',
                     'err_transfer_region_gpu_wbw%ss%d.csv' % (s, sd)))

# HYBNS three seeds (λ=5) as the "hard skeleton" control
SPEC.append((np.nan, 42, 'HYBNS', 'err_transfer_region_gpu.csv'))
for sd in (7, 21):
    SPEC.append((np.nan, sd, 'HYBNS', 'err_transfer_region_gpu_lam5_s%d.csv' % sd))


def load():
    rows = []
    miss = []
    for w, sd, cfg, f in SPEC:
        p = B + f
        if not os.path.exists(p):
            miss.append(f)
            continue
        d = pd.read_csv(p)
        d = d[(d.scope == 'pooled') & (~d.debias)]
        if cfg != 'ALL':
            d = d[d.cfg == cfg]
        for _, r in d.iterrows():
            rows.append(dict(w=w, seed=sd, cfg=r.cfg, layer=r.layer,
                             E1=r.E1, E2=r.E2, E3=r.E3,
                             rho=r.rho, rho_lo=r.rho_lo, rho_hi=r.rho_hi,
                             alpha=r.alpha))
    A = pd.DataFrame(rows)
    if len(miss):
        print('!! missing files:', miss)
    return A


def main():
    A = load()
    A.to_csv(B + 'err_transfer_wbw_full.csv', index=False, encoding='utf-8-sig')
    L = A[(~A.w.isna()) & (A.cfg != 'LSTMNS')]   # dose-response series (LSTM + LPMIX)
    H = A[A.w.isna()]                            # HYBNS (hard skeleton, no w)
    NS = A[A.cfg == 'LSTMNS']                    # SS control, not in the series

    P = []
    def p(s=''):
        print(s)
        P.append(s)

    p('=' * 96)
    p('Dose-response statistics of the soft water-balance constraint (LPMIX, --wbw w)   R3 station period  85 stations  monthly scale')
    p('=' * 96)

    # ---------- T1 axis validity: E3 monotonic in w ----------
    p('')
    p('### T1 Axis validity: does the fidelity error E3 deteriorate monotonically with w')
    for lay in ['10cm', '30cm']:
        s = L[L.layer == lay].groupby('w').E3.agg(['mean', 'min', 'max', 'count'])
        p('  %s:' % lay)
        for w, r in s.iterrows():
            p('    w=%-6g E3 %.5f  (n=%d, range %.5f-%.5f)' %
              (w, r['mean'], int(r['count']), r['min'], r['max']))
        x = np.log10(L[L.layer == lay].w.values + 0.01)
        y = L[L.layer == lay].E3.values
        sl, ic, rv, pv, se = __import__('scipy.stats', fromlist=['x']).linregress(x, y)
        p('    log(w) regression slope=%+.5f  r=%.3f  p=%.2g  -> E3 %s' %
          (sl, rv, pv, 'worsens with constraint (axis valid)' if sl > 0 and pv < 0.05 else 'no trend'))

    # ---------- T2 within-seed monotonicity ----------
    p('')
    p('### T2 Within-seed monotonic trend (w = 0 / 0.3 / 3, three points per seed)')
    for lay in ['10cm', '30cm']:
        s = L[(L.layer == lay) & (L.w.isin([0.0, 0.3, 3.0]))]
        up = dn = 0
        tot = 0
        for sd in sorted(set(s.seed)):
            v = s[s.seed == sd].sort_values('w').alpha.values
            if len(v) != 3:
                continue
            tot += 1
            inc = bool(v[0] < v[1] < v[2])
            dec = bool(v[0] > v[1] > v[2])
            up += inc
            dn += dec
            p('    seed %-3d  %.2f%% -> %.2f%% -> %.2f%%   %s' %
              (sd, 100 * v[0], 100 * v[1], 100 * v[2],
               'monotonically increasing' if inc else ('monotonically decreasing' if dec else 'non-monotonic')))
        # The probability of monotonicity under a random three-point permutation is 1/6
        from scipy import stats as st
        pb = lambda k, n: sum(st.binom.pmf(i, n, 1 / 6.) for i in range(k, n + 1))
        p('    increasing %d/%d (p=%.4f)   decreasing %d/%d (p=%.4f)' %
          (up, tot, pb(up, tot), dn, tot, pb(dn, tot)))
        p('')

    # ---------- T3 between-group difference ----------
    p('### T3 Between-group difference w=3 vs w=0 (Welch t, seed as unit)')
    for lay in ['10cm', '30cm']:
        s = L[L.layer == lay]
        x = s[s.w == 3.0].alpha.values
        y = s[s.w == 0.0].alpha.values
        from scipy import stats as st
        t, pv = st.ttest_ind(x, y, equal_var=False)
        p('  %s: w=3 %.2f%% (n=%d, sd %.2f) vs w=0 %.2f%% (n=%d, sd %.2f)  '
          'Δ=%+.2f pp  t=%.2f p=%.4f' %
          (lay, 100 * x.mean(), len(x), 100 * x.std(ddof=1),
           100 * y.mean(), len(y), 100 * y.std(ddof=1),
           100 * (x.mean() - y.mean()), t, pv))

    # ---------- T4 form vs strength ----------
    p('')
    p('### T4 The "form" vs "strength" of the constraint: hard skeleton HYBNS vs soft penalty LPMIX (matched seed / matched E3)')
    for lay in ['10cm', '30cm']:
        s42h = H[(H.layer == lay) & (H.seed == 42)]
        s42s = L[(L.layer == lay) & (L.seed == 42) & (L.w == 30.0)]
        hm = H[H.layer == lay]
        p('  %s:' % lay)
        p('    HYBNS (hard skeleton)   α %.2f%% ± %.2f (3 seed)  | seed42: E3 %.5f ρ %+.3f α %.2f%%' %
          (100 * hm.alpha.mean(), 100 * hm.alpha.std(ddof=1),
           s42h.E3.iloc[0], s42h.rho.iloc[0], 100 * s42h.alpha.iloc[0]))
        p('    LPMIX w=30 (soft)  seed42: E3 %.5f ρ %+.3f α %.2f%%' %
          (s42s.E3.iloc[0], s42s.rho.iloc[0], 100 * s42s.alpha.iloc[0]))
        p('    ceiling α_max=E3/E1: HYBNS %.1f%%  vs  LPMIX w=30 %.1f%%' %
          (100 * s42h.E3.iloc[0] / s42h.E1.iloc[0],
           100 * s42s.E3.iloc[0] / s42s.E1.iloc[0]))

    # ---------- T5 plateau / decline ----------
    p('')
    p('### T5 High-strength end: plateau or decline? (w = 3 / 10 / 30, 3 seeds each)')
    for lay in ['10cm', '30cm']:
        s = L[L.layer == lay]
        from scipy import stats as st
        out = []
        for w in [3.0, 10.0, 30.0]:
            v = s[s.w == w].alpha.values
            out.append((w, 100 * v.mean(), 100 * v.std(ddof=1), len(v)))
        p('  %s:' % lay)
        for w, m, sd, n in out:
            p('    w=%-5g  α %.2f%% ± %.2f (n=%d)' % (w, m, sd, n))
        t, pv = st.ttest_ind(s[s.w == 30.0].alpha.values,
                             s[s.w == 3.0].alpha.values, equal_var=False)
        p('    w=30 vs w=3: Δ=%+.2f pp  t=%.2f p=%.4f -> %s' %
          (100 * (s[s.w == 30.0].alpha.mean() - s[s.w == 3.0].alpha.mean()), t, pv,
           'plateau (no significant decline)' if pv > 0.05 else 'significant change'))
        # Five-point monotonicity (3 seeds, full series)
        up = dn = tot = 0
        for sd in sorted(set(s.seed)):
            v = s[s.seed == sd].sort_values('w')
            v = v[v.w.isin([0.0, 0.3, 3.0, 10.0, 30.0])].alpha.values
            if len(v) != 5:
                continue
            tot += 1
            up += bool(all(np.diff(v) > 0))
            dn += bool(all(np.diff(v) < 0))
        p('    Five-point full series: increasing %d/%d  decreasing %d/%d' % (up, tot, dn, tot))
        p('')

    # ---------- T6 full-range trend (avoiding cherry-picking only three favourable points) ----------
    p('')
    p('### T6 Full-range trend: Spearman on the five points w=0/0.3/3/10/30 per seed, combined by the Fisher method')
    from scipy import stats as st
    for lay in ['10cm', '30cm']:
        s = L[L.layer == lay]
        ps = []
        for sd in sorted(set(s.seed)):
            v = s[s.seed == sd].sort_values('w')
            v = v[v.w.isin([0.0, 0.3, 3.0, 10.0, 30.0])]
            if len(v) != 5:
                continue
            r, _ = st.spearmanr(v.w.values, v.alpha.values)
            # n=5 exact permutation test (120 permutations), to avoid scipy returning 0 at ρ=±1 and inflating Fisher's statistic
            from itertools import permutations
            obs = st.spearmanr(v.w.values, v.alpha.values)[0]
            null = [st.spearmanr(v.w.values, np.array(q))[0]
                    for q in permutations(v.alpha.values)]
            null = np.array(null)
            p_up = float(np.mean(null >= obs - 1e-12))   # one-sided: increasing direction
            ps.append(p_up)
            p('    seed %-3d  Spearman ρ=%+.3f  α: %s  one-sided exact p=%.4f' %
              (sd, r, ' '.join('%.2f' % (100 * x) for x in v.alpha.values), p_up))
        if ps:
            chi = -2 * np.sum(np.log(np.clip(ps, 1e-12, 1)))
            pcomb = st.chi2.sf(chi, 2 * len(ps))
            p('    Fisher combined χ²=%.2f (df=%d)  p=%.5f  -> %s' %
              (chi, 2 * len(ps), pcomb,
               'significant increase over the full range' if pcomb < 0.05 else 'no significant trend'))
        p('')

    # ---------- summary table ----------
    p('')
    p('### summary: α(%) by w × seed')
    p('%-6s | %-22s | %-22s' % ('w', '10cm', '30cm'))
    for w in [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        r = ['%-6g' % w]
        for lay in ['10cm', '30cm']:
            v = L[(L.layer == lay) & (L.w == w)].sort_values('seed').alpha.values
            if len(v) == 0:
                r.append('%-22s' % '-')
            elif len(v) == 1:
                r.append('%-22s' % ('%.2f (n=1)' % (100 * v[0])))
            else:
                r.append('%-22s' % ('%.2f ± %.2f (n=%d)' %
                                    (100 * v.mean(), 100 * v.std(ddof=1), len(v))))
        p(' | '.join(r))
    hm10 = H[H.layer == '10cm'].alpha
    hm30 = H[H.layer == '30cm'].alpha
    p('%-6s | %-22s | %-22s' % ('HARD', '%.2f ± %.2f (n=%d)' % (100 * hm10.mean(), 100 * hm10.std(ddof=1), len(hm10)),
                                '%.2f ± %.2f (n=%d)' % (100 * hm30.mean(), 100 * hm30.std(ddof=1), len(hm30))))
    ns10 = NS[NS.layer == '10cm'].alpha
    ns30 = NS[NS.layer == '30cm'].alpha
    p('%-6s | %-22s | %-22s' % ('noSS', '%.2f (n=%d)' % (100 * ns10.mean(), len(ns10)),
                                '%.2f (n=%d)' % (100 * ns30.mean(), len(ns30))))

    with open(LOG + '/_21_wbw_stats.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(P))
    print()
    print('saved err_transfer_wbw_full.csv / _21_wbw_stats.txt')


if __name__ == '__main__':
    main()
