# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""08: multi-region input completeness gate  --  checks whether a new region has all variables required by the single-region 78-dimensional features

Compare item by item with the single-region 80_build_samples.py feature list against the new region cube actual variable names in the new cube,
Any missing item must be fixed before entering 09 fixed before sample construction.
"""
import os, json
import numpy as np, xarray as xr

OUT = REPRO + '/_Upstream_Intermediate_Products/out'
MULTI = REPRO + '/_Upstream_Intermediate_Products/out/04_ready/multi'
STATIC = REPRO + '/_Upstream_Intermediate_Products/out/01_static/multi'

STATIC_VARS = ['dem_mean', 'dem_std', 'slope', 'northness', 'eastness', 'tpi']
SOIL_SETS = [
    ('0_5cm', ['sand', 'silt', 'clay', 'BD', 'OC', 'pH', 'CEC', 'gravel', 'Wet_R']),
    ('15_30cm', ['sand', 'silt', 'clay', 'BD', 'OC', 'pH', 'CEC', 'gravel', 'Wet_R']),
    ('30_60cm', ['sand', 'silt', 'clay', 'BD']),
]
CMFD_VARS = ['t2m', 'rhum', 'wspeed', 'spressure', 'swrad', 'lwrad', 'netrad', 'vpd', 'pet', 'precip']
CHM_VARS = ['precip_daily', 'precip_sum03d', 'precip_sum07d', 'precip_sum15d',
            'precip_sum30d', 'precip_sum60d', 'API_k85', 'API_k95']

need_static = STATIC_VARS + ['%s_%s' % (p, layer) for layer, props in SOIL_SETS for p in props]

lines = []
def P(s=''):
    print(s); lines.append(s)

def load(region, fname):
    for b in (MULTI, STATIC):
        p = os.path.join(b, region, fname)
        if os.path.exists(p):
            return xr.open_dataset(p), p
    return None, None

P('=' * 78)
P('multi-region input completeness gate (against the single-region 80_build_samples.py feature list)')
P('Required: static %d (terrain %d + soil %d)+ longitude/latitude 2 + CMFD %d + CHM_PRE %d + time2 + lag12 = 78' % (
    len(need_static), len(STATIC_VARS), len(need_static) - len(STATIC_VARS),
    len(CMFD_VARS), len(CHM_VARS)))
P('=' * 78)

_sel = None
import sys as _sys
for _a in _sys.argv[1:]:
    if _a.startswith('--region='):
        _sel = _a.split('=', 1)[1].split(',')
regions = sorted(d for d in os.listdir(MULTI) if os.path.isdir(os.path.join(MULTI, d)))
regions = [d for d in regions if d in _sel] if _sel else regions
ok_all = True
summary = {}

for r in regions:
    P('')
    P('---------- %s ----------' % r)
    ok = True

    st, stp = load(r, 'static_covariates.nc')
    if st is None:
        P('  !! missing static_covariates.nc'); ok = False; continue
    have = set(st.data_vars)
    miss = [v for v in need_static if v not in have]
    P('  [static] actual %d  variables, required %d entries -> missing %s' % (len(have), len(need_static), miss or 'none'))
    if miss:
        ok = False
        # suggest similar candidates
        for m in miss:
            key = m.split('_')[0]
            cand = sorted(h for h in have if key in h)[:6]
            P('        %s candidates: %s' % (m, cand or '(none)'))
    else:
        rng = {v: (float(np.nanmin(st[v].values)), float(np.nanmax(st[v].values))) for v in need_static}
        bad = {k: v for k, v in rng.items() if not np.isfinite(v[0]) or not np.isfinite(v[1])}
        P('        terrain 6  quantities: ' + '  '.join('%s[%.1f,%.1f]' % (v, *rng[v]) for v in STATIC_VARS))
        P('        soil examples: ' + '  '.join('%s[%.2f,%.2f]' % (v, *rng[v]) for v in need_static[6:10]))
        if bad:
            P('        !! contains non-finite values: %s' % list(bad)); ok = False
    if 'lat' not in st.coords or 'lon' not in st.coords:
        P('  !! missing lat/lon coordinates'); ok = False
    else:
        P('        Grid %d x %d  lat[%.2f,%.2f] lon[%.2f,%.2f]' % (
            st.sizes['lat'], st.sizes['lon'], float(st.lat.min()), float(st.lat.max()),
            float(st.lon.min()), float(st.lon.max())))
    st.close()

    cm, cmp_ = load(r, 'cmfd_meteo_daily.nc')
    if cm is None:
        P('  !! missing cmfd_meteo_daily.nc'); ok = False
    else:
        miss = [v for v in CMFD_VARS if v not in cm.data_vars]
        P('  [CMFD] actual %d  variables -> missing %s' % (len(cm.data_vars), miss or 'none'))
        if miss: ok = False
        else:
            P('        climatology: t2m %.1fC  rhum %.0f%%  wspeed %.1f m/s  swrad %.1f MJ  precip %.2f mm/d' % (
                float(cm['t2m'].mean()), float(cm['rhum'].mean()), float(cm['wspeed'].mean()),
                float(cm['swrad'].mean()), float(cm['precip'].mean())))
        cm.close()

    pv, pvp = load(r, 'cube_precip_dynamic.nc')
    if pv is None:
        P('  !! missing cube_precip_dynamic.nc'); ok = False
    else:
        miss = [v for v in CHM_VARS if v not in pv.data_vars]
        P('  [CHM_PRE] actual %d  variables -> missing %s' % (len(pv.data_vars), miss or 'none'))
        if miss: ok = False
        else:
            P('        precip_daily annual mean %.2f mm/d  API_k95 median %.2f' % (
                float(pv['precip_daily'].mean()), float(np.nanmedian(pv['API_k95'].values))))
        pv.close()

    tg, tgp = load(r, 'cube_target_SM.nc')
    if tg is None:
        P('  !! missing cube_target_SM.nc'); ok = False
    else:
        miss = [v for v in ('SM_10cm', 'SM_30cm') if v not in tg.data_vars]
        P('  [Target] actual %s -> missing %s' % (list(tg.data_vars), miss or 'none'))
        if miss: ok = False
        else:
            for v in ('SM_10cm', 'SM_30cm'):
                a = tg[v].values
                P('        %s  median %.4f  p1 %.4f  p99 %.4f  nan%% %.3f' % (
                    v, float(np.nanmedian(a)), float(np.nanpercentile(a, 1)),
                    float(np.nanpercentile(a, 99)), 100 * float(np.isnan(a).mean())))
        # time-axis consistency
        nT = tg.sizes['time']
        P('        time axis %d days  %s ~ %s' % (nT, str(tg.time.values[0])[:10], str(tg.time.values[-1])[:10]))
        tg.close()

    P('  => %s' % ('PASS' if ok else '** FAIL **'))
    ok_all &= ok
    summary[r] = ok

P('')
P('=' * 78)
P('Total: ' + '  '.join('%s=%s' % (k, 'OK' if v else 'FAIL') for k, v in summary.items()))
P('Conclusion: %s' % ('All passed, may proceed to 09 sample construction' if ok_all else 'There are missing items; complete them before continuing'))
P('=' * 78)

open(LOG + '/_08_gate_report.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n[written] _08_gate_report.txt')
