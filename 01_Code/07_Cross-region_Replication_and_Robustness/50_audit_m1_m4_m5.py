# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""50: Supplementary audit for the fourth-round review (all based on existing data products, no new training)

Output (all written to 02_Results_Data/truth_chain/):
  50_variance_budget.csv   M1: variance-budget decomposition (σ_u²/E1², σ_v²/E1²) and skill diagnostics
  50_lambda_layers.csv     M5: hard-skeleton family stratified by auxiliary evapotranspiration loss weight lambda in {0,5,20}
  50_standard_capacity.csv M4: cross-region family means -- all FREE vs standard FREE with degraded configurations removed
  50_missingness.csv       M7: missingness rate, month coverage, per-station sample-size distribution
  50_calibration_affine.csv M8: no-intercept vs affine (with intercept) calibration; station-clustered bootstrap of lambda*
  50_skill_table.csv       M12: parameter count per family, fidelity to the label E3, h=1 RMSE

Usage (geoenv):
  python 50_audit_m1_m4_m5.py
"""
import os, re, json, glob
import numpy as np
import pandas as pd

import common

TC = TRUTH
MULTI = os.path.join(TC, 'multi')
BENCH = BENCH

RNG = np.random.default_rng(20261008)


def P(*a):
    print(*a, flush=True)


# ============================================================ M1 variance budget
def variance_budget():
    """Under the orthogonal-decomposition assumption of M1 (u⊥v, o⊥u,v):
        σ_u² = -Cov(e1,e3) = α·E1² ;  σ_v² = E3² - σ_u²
       hence σ_u²/E1² = α and σ_v²/E1² = x² - α.
       Also report Var(o)/Var(r)/Var(m) directly as raw quantities that do not rely on this assumption.
    """
    w = pd.read_csv(os.path.join(TC, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
    fam = pd.read_csv(os.path.join(TC, 'paper_families.csv'), encoding='utf-8-sig')
    rows = []
    for _, r in w.iterrows():
        E1, E3, rho, a = r['E1w'], r['E3w'], r['rho_w'], r['a_w']
        x = E3 / E1
        su = a                 # σ_u² / E1²  (= α)
        sv = x * x - a         # σ_v² / E1²  (= x² − α)
        rows.append(dict(run=r['run'], cfg=r['cfg'], layer=r['layer'], n=r['n'],
                         E1=E1, E3=E3, x=x, rho=rho, alpha=a,
                         su2_over_E12=su, sv2_over_E12=sv,
                         win=(su > sv),          # whether "removed noise > own error"
                         lam_star=(a / (x * x) if x > 0 else np.nan),
                         beats=(a / (x * x) > 0.5 if x > 0 else False)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TC, '50_variance_budget.csv'), index=False, encoding='utf-8-sig')

    # Family level (consistent with the family definitions in paper_families)
    def fam_of(run, cfg):
        if re.search(r'wd', run):
            return 'DAMAGED'
        if cfg == 'HYBNS':
            return 'HARD'
        if cfg in ('HYBLST', 'HYBLSTNS'):
            return 'HYBL'
        if cfg in ('LPMIX', 'LPMIXNS'):
            return 'SOFT'
        if cfg in ('LSTM', 'LSTMNS'):
            return 'FREE'
        return None

    d['fam'] = [fam_of(r.run, r.cfg) for r in d.itertuples()]
    g = d.groupby(['fam', 'layer']).agg(
        n_cfg=('x', 'size'), alpha=('alpha', 'mean'), x=('x', 'mean'),
        su2=('su2_over_E12', 'mean'), sv2=('sv2_over_E12', 'mean'),
        win_rate=('win', 'mean'), lam_star=('lam_star', 'mean')).reset_index()
    g['sv2_note'] = 'Under the u⊥v assumption; alpha inconsistent across families indicates the assumption fails for every family'
    P('[M1 variance budget (family level, within-station convention)]')
    P(g.to_string(index=False))
    g.to_csv(os.path.join(TC, '50_variance_budget_fam.csv'), index=False, encoding='utf-8-sig')
    return d


# ============================================================ M5 lambda stratification
def lambda_layers():
    w = pd.read_csv(os.path.join(TC, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
    h = w[w.cfg == 'HYBNS'].copy()

    def lam_of(run):
        if 'lam0' in run:
            return 0
        if 'lam20' in run:
            return 20
        return 5          # gpu / gpu_lam5_*

    h['lamet'] = [lam_of(r) for r in h.run]
    # ratio = E2/E1 must be computed row by row before averaging
    h['ratio'] = h['E2w'] / h['E1w']
    h['x'] = h['E3w'] / h['E1w']
    g = h.groupby(['lamet', 'layer']).agg(
        n=('x', 'size'), E1=('E1w', 'mean'), E3=('E3w', 'mean'), x=('x', 'mean'),
        rho=('rho_w', 'mean'), alpha=('a_w', 'mean'), ratio=('ratio', 'mean'),
        beat=('ratio', lambda s: int((s < 1).sum()))).reset_index()
    g.to_csv(os.path.join(TC, '50_lambda_layers.csv'), index=False, encoding='utf-8-sig')
    P('[M5 hard-skeleton family stratified by auxiliary evapotranspiration loss weight lambda (North China primary region, within-station convention)]')
    P(g.to_string(index=False))
    return g


# ============================================================ M4 standard-capacity subset
DEGRADED = re.compile(r'deg(5|10|32)')     # 5/10 epochs, hidden size 32: explicitly flagged as degraded variants


def standard_capacity():
    out = []
    for reg in ['R1_yangtze', 'R0_northchina', 'R4_northheast'.replace('hea', 'hea'),
                'R3_northeast', 'R2_loess']:
        fp = os.path.join(MULTI, 'withinstation_%s.csv' % reg)
        if not os.path.exists(fp):
            continue
        w = pd.read_csv(fp, encoding='utf-8-sig')
        if 'fam' not in w.columns:
            continue
        for layer in ['10cm', '30cm']:
            s = w[(w.layer == layer)]
            for tag, keep in [('all', s), ('std', s[~s.run.astype(str).str.contains(DEGRADED)])]:
                for fam in ['FREE', 'SOFT', 'HARD', 'HYBL']:
                    v = keep[keep.fam == fam]
                    if len(v) == 0:
                        continue
                    out.append(dict(region=reg, layer=layer, subset=tag, fam=fam,
                                    n_cfg=len(v),
                                    n_sta=int(v['nsta'].max()) if 'nsta' in v else np.nan,
                                    E1=v['E1w'].mean(), E3=v['E3w'].mean(),
                                    x=v['x'].mean() if 'x' in v else np.nan,
                                    rho=v['rho_w'].mean(),
                                    ratio=v['ratio'].mean() if 'ratio' in v else np.nan,
                                    beat=int((v['beat'] == 1).sum()) if 'beat' in v else np.nan))
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(TC, '50_standard_capacity.csv'), index=False, encoding='utf-8-sig')
    P('[M4 cross-region family means: all FREE vs standard FREE with degraded configurations removed]')
    piv = d.pivot_table(index=['region', 'layer', 'fam'], columns='subset',
                        values=['n_cfg', 'rho', 'ratio'], aggfunc='first')
    P(piv.to_string())
    return d


# ============================================================ M7 missingness and season
def missingness():
    fp = os.path.join(TC, 'truth_chain_L2_monthly_gpu.csv')
    if not os.path.exists(fp):
        P('!! missing %s' % fp)
        return None
    d = pd.read_csv(fp, encoding='utf-8-sig')
    rows = []
    for (layer, cfg), v in d.groupby(['layer', 'cfg']):
        nsta = v['id'].nunique()
        nmon = v[['year', 'month']].drop_duplicates().shape[0]
        yrs = sorted(v['year'].unique())
        theo = nsta * len(yrs) * 12
        per_sta = v.groupby('id').size()
        rows.append(dict(layer=layer, cfg=cfg, n_pair=len(v), n_sta=nsta,
                         n_calendar_months=nmon, year_min=min(yrs), year_max=max(yrs),
                         theo_full_year=nsta * len(yrs) * 12,
                         missing_rate_full=1 - len(v) / max(1, theo),
                         theo_obs_months=nsta * nmon,
                         missing_rate_inseason=1 - len(v) / max(1, nsta * nmon),
                         per_station_min=int(per_sta.min()),
                         per_station_median=float(per_sta.median()),
                         per_station_max=int(per_sta.max()),
                         months=sorted(v['month'].unique())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TC, '50_missingness.csv'), index=False, encoding='utf-8-sig')
    P('[M7 data completeness (North China primary region, monthly pairs)]')
    P(out[['layer', 'cfg', 'n_pair', 'n_sta', 'n_calendar_months', 'missing_rate_full',
           'missing_rate_inseason', 'per_station_min', 'per_station_max']].to_string(index=False))
    P('Covered months: %s' % out['months'].iloc[0])
    return out


# ============================================================ M8 affine calibration
def calibration_affine(nboot=2000):
    """Comparison of no-intercept lambda*=-rho/x and affine (with intercept) calibration; station-clustered bootstrap CI of lambda*."""
    files = {
        'HARD/HYBNS': 'gpu', 'FREE/LSTM': 'gpu', 'SOFT/LPMIX(w=1)': 'gpu_wbw1',
        'DAMAGED': 'gpu_degwd', 'HYBL': 'gpu_hbl_s42',
        'HARD λ=0': 'gpu_lam0', 'HARD λ=20': 'gpu_lam20',
    }
    rows = []
    for tag, suf in files.items():
        fp = os.path.join(TC, 'truth_chain_L2_monthly_%s.csv' % suf)
        if not os.path.exists(fp):
            P('  skip %s (missing %s)' % (tag, fp))
            continue
        d = pd.read_csv(fp, encoding='utf-8-sig')
        for layer in ['10cm', '30cm']:
            v = d[d.layer == layer].dropna(subset=['obs', 'model', 'smci'])
            if len(v) < 200:
                continue
            e1 = (v['smci'] - v['obs']).values
            e3 = (v['model'] - v['smci']).values
            # No intercept
            lam0 = -np.cov(e1, e3)[0, 1] / np.var(e3)
            E1 = float(np.sqrt(np.mean(e1 ** 2)))
            E2_0 = float(np.sqrt(np.mean((e1 + lam0 * e3) ** 2))) / E1
            # Within-station (de-meaned by station) convention
            g = v.groupby('id')
            e1w = g['smci'].transform(lambda s: s - s.mean()).sub(
                g['obs'].transform(lambda s: s - s.mean())).values
            e3w = g['model'].transform(lambda s: s - s.mean()).sub(
                g['smci'].transform(lambda s: s - s.mean())).values
            lamw = -np.cov(e1w, e3w)[0, 1] / np.var(e3w)
            E1w = float(np.sqrt(np.mean(e1w ** 2)))
            E2w_0 = float(np.sqrt(np.mean((e1w + lamw * e3w) ** 2))) / E1w
            # Affine (with intercept): least-squares fit o ≈ a + λ·(m−r) + r  ⟺ e1 ≈ −a − λ e3 with intercept
            Xa = np.column_stack([np.ones_like(e3), e3])
            beta, *_ = np.linalg.lstsq(Xa, -e1, rcond=None)
            a_hat, lam_aff = float(beta[0]), float(beta[1])
            E2_aff = float(np.sqrt(np.mean((e1 + a_hat + lam_aff * e3) ** 2))) / E1
            Xw = np.column_stack([np.ones_like(e3w), e3w])
            bw, *_ = np.linalg.lstsq(Xw, -e1w, rcond=None)
            E2w_aff = float(np.sqrt(np.mean((e1w + bw[0] + bw[1] * e3w) ** 2))) / E1w
            # Station-clustered bootstrap of lambda* (within-station convention)
            ids = v['id'].values
            uid = np.unique(ids)
            idxby = {s: np.where(ids == s)[0] for s in uid}
            boots = np.empty(nboot)
            for b in range(nboot):
                pick = RNG.choice(uid, size=len(uid), replace=True)
                ii = np.concatenate([idxby[s] for s in pick])
                e1b, e3b = e1w[ii], e3w[ii]
                boots[b] = -np.cov(e1b, e3b)[0, 1] / np.var(e3b)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(tag=tag, file=suf, layer=layer, n=len(v), n_sta=len(uid),
                             lam_star_full=lam0, lam_star_within=lamw,
                             lam_lo=lo, lam_hi=hi,
                             reject_lam_eq_1=bool(not (lo <= 1.0 <= hi)),
                             E2E1_nointer=1.0, E2E1_lam0_full=E2_0,
                             E2E1_lam0_within=E2w_0,
                             aff_a=a_hat, aff_lam=lam_aff,
                             E2E1_affine_full=E2_aff, E2E1_affine_within=E2w_aff,
                             gain_affine_pp=(E2w_0 - E2w_aff) * 100))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TC, '50_calibration_affine.csv'), index=False, encoding='utf-8-sig')
    P('[M8 no-intercept vs affine calibration; station-clustered bootstrap of lambda* (2000 draws)]')
    P(out[['tag', 'layer', 'lam_star_within', 'lam_lo', 'lam_hi', 'reject_lam_eq_1',
           'E2E1_lam0_within', 'E2E1_affine_within', 'gain_affine_pp']].to_string(index=False))
    return out


# ============================================================ M12 skill and capacity
def skill_table():
    """Parameter count (json nparam) and fidelity to the label (E3, within-station convention) per family."""
    w = pd.read_csv(os.path.join(TC, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
    rows = []
    for jf in sorted(glob.glob(os.path.join(BENCH, 'dl_hybrid_result_*.json'))):
        try:
            d = json.load(open(jf, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        tag = os.path.basename(jf)[len('dl_hybrid_result_'):-5]
        for e in d:
            if not isinstance(e, dict) or 'cfg' not in e:
                continue
            rows.append(dict(json=tag, cfg=e.get('cfg'), nparam=e.get('nparam'),
                             h1=(e.get('test') or {}).get('1'),
                             all=(e.get('test') or {}).get('all', np.nan),
                             et_rmse=e.get('et_rmse')))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(TC, '50_skill_table.csv'), index=False, encoding='utf-8-sig')
    P('[M12 parameter count (json nparam) and h=1 RMSE, aggregated by family]')

    def fam_of(cfg):
        return {'HYBNS': 'HARD', 'HYBLST': 'HYBL', 'HYBLSTNS': 'HYBL',
                'LPMIX': 'SOFT', 'LPMIXNS': 'SOFT', 'LSTM': 'FREE',
                'LSTMNS': 'FREE', 'PHY': 'PHY'}.get(cfg, cfg)

    s['fam'] = s.cfg.map(fam_of)
    g = s.groupby('fam').agg(n_run=('nparam', 'size'), nparam=('nparam', 'mean'),
                             h1=('h1', 'mean')).reset_index()
    P(g.to_string(index=False))
    return s


if __name__ == '__main__':
    P('===== 50 Supplementary audit for the fourth-round review =====')
    variance_budget()
    lambda_layers()
    standard_capacity()
    missingness()
    calibration_affine()
    skill_table()
    P('===== DONE =====')
    try:
        common.report('402_audit', [])
    except Exception:
        pass
