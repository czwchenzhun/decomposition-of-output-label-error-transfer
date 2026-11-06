# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""15_station_truth_chain.py -- the "model vs truth" verification chain (station observations)

Background
----
Previously the evaluation baseline for the whole chain was the SMCI reference field (the term "truth" is banned in the paper). This chain adds a genuinely independent
in situ observation control, and splits the evaluation into three layers, avoiding mistaking "whether the model resembles SMCI" for "whether the model is accurate":

    L1  reference field vs truth      SMCI   vs station       -> defines the credible boundary of the reference field
    L2  model   vs truth      model    vs station       -> error of the extrapolated output against true soil moisture
    L3  model   vs reference field    model    vs SMCI       -> already available (07_models/multi)

This script handles L1 (pure data, no model needed) and writes the station--grid-cell pairing table to disk for reuse by L2.

Data source
------
1) China 732 stations, oven-drying observations (monthly values, 1992-2013)
   Raw dataset/China_732_Station_Multilayer_Soil_Moisture_Dataset/
     sm_stninfor.txt        id lat lon elev
     stn_Mon_sm_vol/*.txt   id year month sm10 sm20 sm50 sm70 sm100 (-99 missing)
   Depths: 10 / 20 / 50 / 70 / 100 cm
2) Independent validation station ChinaFLUX (daily values, 2003-2010) -- handled separately by script 16 (daily scale)

Depth pairing (consistent with 100_station732_validate.py)
---------------------------------------------
    SMCI 10cm  <-> obs sm10
    SMCI 30cm  <-> 0.5*(sm20 + sm50)        centre 20/50cm -> weighted to 30cm
                    (linear interpolation to 30cm is more accurate: sm20 + (sm50-sm20)*(30-20)/(50-20) = (2*sm20+sm50)/3
                     Here we keep the convention of script 100, 0.5*(20+50), to stay comparable with the archived results;
                     --interp30 can switch to the linear-interpolation convention for a sensitivity check)

Region assignment
--------
Assigned by spatial containment in the six R0--R5 3°×3° experimental windows (consistent with the 02-series REGIONS):
    R0_northchina 114-117E / 36-39N       R1_yangtze  114-117E / 30-33N
    R2_loess      108-111E / 35-38N       R3_northeast 125-128E / 44-47N
    R4_longzhong  103-106E / 34-37N       R5_sichuan   105-108E / 29-32N
Stations outside the windows do not enter the regional statistics (recorded separately as OUT, to avoid forcing out-of-domain stations in and creating spurious bias --
a pitfall encountered in this project: the 68 modelling grid cells cover only one region's municipal area).

Output (02_Results_Data/truth_chain/)
----------------------------------------
  stations_in_windows.csv        List of stations inside the windows (with nearest grid-cell distance and grid-cell index)
  truth_chain_L1_station.csv     Per-station metrics
  truth_chain_L1_region.csv      Regional summary (median / IQR / station count)
  truth_chain_L1_monthly.csv     Monthly paired samples (for direct join by L2)
  _15_run.txt                   Log
"""
import os
import io
import argparse
import numpy as np
import pandas as pd

from common import RAW, OUT

# ----------------------------------------------------------------- Configuration
REGIONS = {
    'R1_yangtze':    (114.0, 30.0, 117.0, 33.0),
    'R0_northchina': (114.0, 36.0, 117.0, 39.0),
    'R3_northeast':  (125.0, 44.0, 128.0, 47.0),
    'R2_loess':      (108.0, 35.0, 111.0, 38.0),
    # v20 Extended regions: added in sync with the 02-series REGIONS (never overlapping, see 62_new_regions.py)
    'R4_longzhong':  (103.0, 34.0, 106.0, 37.0),
    'R5_sichuan':    (105.0, 29.0, 108.0, 32.0),
}

STADIR = os.path.join(RAW, 'China_732_Station_Multilayer_Soil_Moisture_Dataset')
STAF = os.path.join(STADIR, 'stn_Mon_sm_vol')
SMCI_AT_STA = os.path.join(STATION, 'smci_at_732stations_monthly.csv')

OUTDIR = TRUTH
os.makedirs(OUTDIR, exist_ok=True)

# The raw monthly files have 9 columns: id year month drythk sm10 sm20 sm50 sm70 sm100
#  (drythk = dry-soil-layer thickness, -99 is missing; omitting it makes pandas treat id as the index and shift the whole table one column left)
OBS_COLS = ['id', 'year', 'month', 'drythk', 'sm10', 'sm20', 'sm50', 'sm70', 'sm100']

LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------- Metrics
def metrics(o, s):
    """o/s are equal-length one-dimensional arrays; returns a dict. ubRMSE first removes the per-station bias (consistent with the official convention)"""
    o = np.asarray(o, float)
    s = np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    n = len(o)
    if n < 3:
        return dict(n=n, R=np.nan, bias=np.nan, rmse=np.nan, ubRMSE=np.nan,
                    obs_mean=np.nan, sim_mean=np.nan, obs_std=np.nan, sim_std=np.nan)
    b = float(np.mean(s - o))
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    ub = float(np.sqrt(np.mean(((s - b) - o) ** 2)))
    sd_o, sd_s = float(np.std(o)), float(np.std(s))
    R = float(np.corrcoef(o, s)[0, 1]) if (sd_o > 0 and sd_s > 0) else np.nan
    return dict(n=n, R=R, bias=b, rmse=rmse, ubRMSE=ub,
                obs_mean=float(np.mean(o)), sim_mean=float(np.mean(s)),
                obs_std=sd_o, sim_std=sd_s)


def grid_centers(lon0, lat0, lon1, lat1):
    """Central window 3°×3° expanded by 0.4°, cell centres aligned with .X0 (consistent with the 02-series grid_for)"""
    import math
    lat = np.arange(math.ceil((lat0 - 0.4 + 0.05) * 10) / 10,
                    math.floor((lat1 + 0.4 - 0.05) * 10) / 10 + 1e-9, 0.1)
    lon = np.arange(math.ceil((lon0 - 0.4 + 0.05) * 10) / 10,
                    math.floor((lon1 + 0.4 - 0.05) * 10) / 10 + 1e-9, 0.1)
    return np.round(lat, 4), np.round(lon, 4)


# ----------------------------------------------------------------- Read data
def read_station_info():
    rows = []
    with open(os.path.join(STADIR, 'sm_stninfor.txt'), encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                try:
                    rows.append((p[0], float(p[1]), float(p[2]), float(p[3])))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=['id', 'lat', 'lon', 'elev'])
    df['id'] = df['id'].astype(str)
    return df


def read_obs(y0=1992, y1=2013):
    """Read all station monthly values, missing -99 → NaN"""
    frames = []
    for fn in sorted(os.listdir(STAF)):
        if not fn.endswith('.txt'):
            continue
        try:
            df = pd.read_csv(os.path.join(STAF, fn), sep=r'\s+', header=None,
                             names=OBS_COLS, dtype={'id': str})
        except Exception:
            continue
        df = df.dropna(subset=['year', 'month'])
        if len(df) == 0:
            continue            # The directory contains empty/summary placeholder files (e.g. stn732_sm_vol_Apr2Sep_*.txt is empty)
        frames.append(df)
    obs = pd.concat(frames, ignore_index=True)
    obs['year'] = obs['year'].astype(int)
    obs['month'] = obs['month'].astype(int)
    for c in ('sm10', 'sm20', 'sm50', 'sm70', 'sm100'):
        obs[c] = pd.to_numeric(obs[c], errors='coerce')
        obs.loc[obs[c] <= -90, c] = np.nan
    obs = obs[(obs.year >= y0) & (obs.year <= y1)]
    obs = obs.dropna(subset=['sm10', 'sm20', 'sm50'], how='all')
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interp30', action='store_true',
                    help='SMCI30 pairing uses linear interpolation (2*sm20+sm50)/3, otherwise keeps 0.5*(sm20+sm50)')
    ap.add_argument('--min_months', type=int, default=12, help='Minimum number of valid months for per-station statistics')
    args = ap.parse_args()

    P('=== 15 Station Truth_Verification_Chain · L1 reference field vs truth ===')
    P('Pairing convention: SMCI30 <- %s' % ('(2*sm20+sm50)/3' if args.interp30 else '0.5*(sm20+sm50)'))

    info = read_station_info()
    obs = read_obs()
    P('station count %d; monthly records %d rows (%d-%d)' % (
        len(info), len(obs), int(obs.year.min()), int(obs.year.max())))

    # ---- Observation-layer synthesis
    obs['o10'] = obs['sm10']
    obs['o30'] = (2 * obs['sm20'] + obs['sm50']) / 3.0 if args.interp30 \
        else 0.5 * (obs['sm20'] + obs['sm50'])

    # ---- Region assignment + nearest grid cell
    recs = []
    for _, r in info.iterrows():
        hit = None
        for reg, (lon0, lat0, lon1, lat1) in REGIONS.items():
            if lon0 <= r['lon'] <= lon1 and lat0 <= r['lat'] <= lat1:
                hit = reg
                break
        if hit is None:
            recs.append(dict(id=r['id'], lat=r['lat'], lon=r['lon'], elev=r['elev'],
                             region='OUT', gdist=np.nan, iy=-1, ix=-1,
                             glat=np.nan, glon=np.nan))
            continue
        lon0, lat0, lon1, lat1 = REGIONS[hit]
        LATg, LONg = grid_centers(lon0, lat0, lon1, lat1)
        iy = int(np.argmin(np.abs(LATg - r['lat'])))
        ix = int(np.argmin(np.abs(LONg - r['lon'])))
        glat, glon = float(LATg[iy]), float(LONg[ix])
        d = float(np.hypot(glat - r['lat'], glon - r['lon']))
        recs.append(dict(id=r['id'], lat=r['lat'], lon=r['lon'], elev=r['elev'],
                         region=hit, gdist=d, iy=iy, ix=ix, glat=glat, glon=glon))
    sta = pd.DataFrame(recs)

    cnt = sta.groupby('region').size().sort_values(ascending=False)
    P('')
    P('--- Number of stations in each experimental window ---')
    for k, v in cnt.items():
        P('  %-14s %4d' % (k, v))

    ins = sta[sta.region != 'OUT']
    P('')
    P('stations inside windows: %d; station--nearest grid-cell distance: median %.4f° (P90 %.4f°, max %.4f°)' % (
        len(ins), ins.gdist.median(), ins.gdist.quantile(0.9), ins.gdist.max()))

    sta.to_csv(os.path.join(OUTDIR, 'stations_in_windows.csv'),
               index=False, encoding='utf-8-sig')

    # ---- Merge SMCI (already extracted to stations monthly by script 99)
    P('')
    if not os.path.exists(SMCI_AT_STA):
        P('!! missing %s, please run 99_smci_station_extract.py' % SMCI_AT_STA)
        from common import report
        report('15_station_truth_chain', LOG)
        return
    sm = pd.read_csv(SMCI_AT_STA, dtype={'id': str})
    P('SMCI@station monthly table %d rows (%d-%d)' % (len(sm), int(sm.year.min()), int(sm.year.max())))

    m = obs.merge(sm, on=['id', 'year', 'month'], how='inner')
    m = m.merge(sta[['id', 'region', 'gdist', 'iy', 'ix', 'glat', 'glon', 'elev']],
                on='id', how='left', suffixes=('', '_s'))
    m = m[m.region.notna()]
    P('after three-way merge: %d rows; inside windows: %d rows' % (len(m), int((m.region != 'OUT').sum())))

    long = []
    for dep in ('10', '30'):
        sub = m[['id', 'year', 'month', 'region', 'gdist', 'iy', 'ix',
                 'o' + dep, 'smci_' + dep]].copy()
        sub.columns = ['id', 'year', 'month', 'region', 'gdist', 'iy', 'ix',
                       'obs', 'smci']
        sub['layer'] = dep + 'cm'
        long.append(sub)
    pair = pd.concat(long, ignore_index=True)
    pair = pair[np.isfinite(pair['obs']) & np.isfinite(pair['smci'])]
    pair['date'] = pd.to_datetime(dict(year=pair.year, month=pair.month, day=1))
    pair.to_csv(os.path.join(OUTDIR, 'truth_chain_L1_monthly.csv'),
                index=False, encoding='utf-8-sig')
    P('valid paired samples: %d rows (10cm/30cm combined)' % len(pair))

    # ---- Per-station metrics
    rows = []
    for (sid, lay), g in pair.groupby(['id', 'layer']):
        d = metrics(g['obs'], g['smci'])
        d.update(id=sid, layer=lay, region=g['region'].iloc[0],
                 gdist=float(g['gdist'].iloc[0]),
                 iy=int(g['iy'].iloc[0]), ix=int(g['ix'].iloc[0]))
        rows.append(d)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(OUTDIR, 'truth_chain_L1_station.csv'),
              index=False, encoding='utf-8-sig')

    # ---- Regional summary
    P('')
    P('--- L1: SMCI reference field vs station observations (by region, median of per-station metrics)---')
    P('%-14s %-5s %4s %6s %8s %8s %8s %8s' % (
        'region', 'layer', 'nsta', 'R', 'bias', 'RMSE', 'ubRMSE', 'σ_obs'))
    summ = []
    for reg in [r for r in REGIONS if r in set(st.region)] + (['ALL'] if len(st) else []):
        for lay in ('10cm', '30cm'):
            g = st[st.layer == lay] if reg == 'ALL' else st[(st.region == reg) & (st.layer == lay)]
            g = g[g.n >= args.min_months]
            if len(g) == 0:
                continue
            # pooled (remove per-station bias before pooling, to avoid between-station climatological differences inflating ubRMSE)
            sub = pair[pair.layer == lay] if reg == 'ALL' else \
                pair[(pair.region == reg) & (pair.layer == lay)]
            sub = sub[sub.id.isin(set(g.id))]
            pm = metrics(sub['obs'], sub['smci'])
            row = dict(region=reg, layer=lay, nsta=len(g),
                       R_med=float(g.R.median()), R_iqr=float(g.R.quantile(.75) - g.R.quantile(.25)),
                       bias_med=float(g.bias.median()),
                       rmse_med=float(g.rmse.median()),
                       ub_med=float(g.ubRMSE.median()),
                       ub_iqr=float(g.ubRMSE.quantile(.75) - g.ubRMSE.quantile(.25)),
                       obs_std_med=float(g.obs_std.median()),
                       smci_std_med=float(g.sim_std.median()),
                       pooled_R=pm['R'], pooled_bias=pm['bias'],
                       pooled_rmse=pm['rmse'], pooled_ub=pm['ubRMSE'],
                       pooled_n=pm['n'])
            summ.append(row)
            P('%-14s %-5s %4d %6.3f %+8.4f %8.4f %8.4f %8.4f' % (
                reg, lay, row['nsta'], row['R_med'], row['bias_med'],
                row['rmse_med'], row['ub_med'], row['obs_std_med']))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUTDIR, 'truth_chain_L1_region.csv'),
             index=False, encoding='utf-8-sig')

    P('')
    P('--- Convention notes ---')
    P('  * R_med / ub_med are "computed separately per station then taking the median", so between-station climatological differences do not contaminate each other')
    P('  * pooled_* are computed once after pooling all station samples, mixing in between-station spatial variability, so values usually look better')
    P('  * The primary convention in the paper uses the per-station median (a convention comparable to the official SMCI evaluation ubRMSE≈0.02)')
    P('  * In the table σ_obs is the temporal standard deviation of station observations, used to judge the attainable upper bound of R')

    from common import report
    report('15_station_truth_chain', LOG)


if __name__ == '__main__':
    main()
