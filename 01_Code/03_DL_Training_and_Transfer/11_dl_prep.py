# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""11: Prepare a compact data subset for deep learning (physically constrained water-balance model)

From 04_ready/multi/{region}/samples/X.dat (T,1369,78) extract only the few variables the physical model needs from
together with the two-layer SM targets, saving a small npz to workspace/_bench/, for torch environment can read directly.
Avoid repeated torch memmap of the 3.56 GB large file.

Usage:
  python 11_dl_prep.py --region R0_northchina --ncells 256
  python 11_dl_prep.py --region R0_northchina --ncells 64 --t0 2016-01-01 --t1 2019-12-31
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = PKG
SAMP = os.path.join(REPRO + '/_Upstream_Intermediate_Products/out', '04_ready', 'multi')
BENCH = BENCH
LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


# Variables needed by the physical model
# Note:cmfd_pet observed annual total is only ~520 mm/yr (actual North China PET approx. 900-1100), systematically too low,
#      so derive an extra column Jensen-Haise -style PET as a control prior (cmfd_swrad units verified as MJ/m2/day)
FEATS = ['chm_precip_daily', 'cmfd_precip', 'cmfd_pet', 'cmfd_t2m',
         'cmfd_vpd', 'cmfd_swrad', 'doy_sin', 'doy_cos',
         'clay_0_5cm', 'sand_0_5cm', 'BD_0_5cm',
         'clay_15_30cm', 'sand_15_30cm', 'BD_15_30cm', 'slope']
DERIVED = ['pet_jh']

# Antecedent meteorological history (antecedent conditions)
# infiltration/Infiltration/runoff partitioning depends strongly on antecedent wetness; same-day precipitation alone cannot fix the runoff coefficient.
# Note: use only **meteorological** history here, not SM history  -- 
#   During free-running rollout, SM is the model own state; feeding observed SM lags would cause information leakage.
LAGD = [1, 2, 3]        # single-day lagged precipitation (mm/day)
LAGW = [3, 7, 15]       # antecedent accumulated precipitation (mm) and antecedent mean PET (mm/day)
MAXLAG = max(LAGD + LAGW)


def shift_back(v, d):
    """v[t] = original series at t-d value at t-d; pad the start with the first value"""
    o = np.empty_like(v)
    o[d:] = v[:-d]
    o[:d] = v[0]
    return o


def roll_sum(v, w):
    """v[t] = cumulative sum of the original series [t-w+1, t]; at the start, when fewer than w days remain, sum over the actual days"""
    n = v.shape[0]
    c = np.concatenate([np.zeros((1, v.shape[1]), v.dtype), np.cumsum(v, axis=0)], 0)
    hi = np.arange(n) + 1
    lo = np.maximum(hi - w, 0)
    return c[hi] - c[lo], (hi - lo).astype(v.dtype)


def roll_mean(v, w):
    s, cnt = roll_sum(v, w)
    return s / np.maximum(cnt, 1.0)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='R0_northchina')
    ap.add_argument('--ncells', type=int, default=256)
    ap.add_argument('--t0', default='2015-01-01')
    ap.add_argument('--t1', default='2020-12-31')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')   # output filename suffix, to distinguish feature versions
    ap.add_argument('--station_cells', default='',
                    help='csv path containing iy,ix columns: these grid cells must be included in the subset (station-matched grid cells)')
    a = ap.parse_args()

    d = os.path.join(SAMP, a.region, 'samples')
    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
    T, N, F = meta['T'], meta['N'], meta['F']
    names = meta['features']
    P('Region %s  T=%d N=%d F=%d' % (a.region, T, N, F))

    # feature column indices
    idx = []
    for f in FEATS:
        if f not in names:
            raise SystemExit('missing features: %s' % f)
        idx.append(names.index(f))
    P('Extracted %d  variables: %s' % (len(idx), ','.join(FEATS)))

    # time indices
    tm = np.load(os.path.join(d, 'time.npy'))
    P('time dtype=%s  first=%s last=%s' % (tm.dtype, tm[0], tm[-1]))
    tmv = tm.astype('datetime64[D]')
    i0 = int(np.searchsorted(tmv, np.datetime64(a.t0)))
    i1 = int(np.searchsorted(tmv, np.datetime64(a.t1), side='right'))
    i0 = max(0, i0)
    i1 = min(T, i1)
    nt = i1 - i0
    P('time window %s ~ %s -> indices [%d,%d)  total %d days' % (a.t0, a.t1, i0, i1, nt))
    if nt <= 0:
        raise SystemExit('time window is empty')
    # To compute antecedent meteorological history, read an extra lead-in period (trimmed afterwards)
    lead = min(MAXLAG, i0)
    ri0 = i0 - lead
    nt_pad = i1 - ri0
    P('pre-read for the lag features %d days -> actual read indices [%d,%d) total %d days' % (lead, ri0, i1, nt_pad))

    # grid-cell subset
    rng = np.random.default_rng(a.seed)
    nc = min(a.ncells, N)
    cs = np.load(os.path.join(d, 'cells.npz'))
    key = {}
    for n in range(len(cs['iy'])):
        key[(int(cs['iy'][n]), int(cs['ix'][n]))] = n
    forced = []
    if a.station_cells:
        import pandas as _pd
        sc = _pd.read_csv(a.station_cells)
        for _, r in sc.iterrows():
            k = (int(r['iy']), int(r['ix']))
            if k in key and key[k] not in forced:
                forced.append(key[k])
        P('forcibly include station-matched grid cells %d  (after deduplication)' % len(forced))
    cells = list(forced)
    if nc > len(cells):
        rest = [n for n in range(N) if n not in set(cells)]
        pick = rng.choice(len(rest), nc - len(cells), replace=False)
        cells.extend([rest[i] for i in pick])
    cells = np.sort(np.asarray(cells[:nc], dtype=np.int64))
    P('grid-cell subset n=%d (including the forced station grid cells %d , the rest from %d at random)' % (len(cells), len(forced), N))

    # read in chunks in order X.dat (only the required columns and grid cells)
    X = np.memmap(os.path.join(d, 'X.dat'), dtype='float32', mode='r',
                  shape=(T, N, F))
    out = np.empty((nt_pad, nc, len(idx)), dtype='float32')
    CH = 200
    t_all = time.time()
    for s in range(0, nt_pad, CH):
        e = min(nt_pad, s + CH)
        blk = np.asarray(X[ri0 + s:ri0 + e])          # (chunk,N,F)
        out[s:e] = blk[:, cells][:, :, idx]
        if (s // CH) % 5 == 0:
            P('  read chunk %d/%d' % (e, nt_pad))
    del X
    P('X subset read complete  %.1f s  shape=%s' % (time.time() - t_all, out.shape))

    # derived Jensen-Haise -style PET (mm/day)
    #   Rs_mm = swrad(MJ/m2/day) / 2.45 ; PET = 0.025*(T+3)*Rs_mm
    if 'pet_jh' in DERIVED:
        i_t2 = FEATS.index('cmfd_t2m')
        i_sr = FEATS.index('cmfd_swrad')
        rs_mm = out[:, :, i_sr] / 2.45
        pet_jh = np.clip(0.025 * (out[:, :, i_t2] + 3.0) * rs_mm, 0.0, None)
        out = np.concatenate([out, pet_jh[:, :, None]], axis=2)
        FEATS.append('pet_jh')

    # Antecedent meteorological history (computed on the padded array with lead-in days padded array, then trim the lead-in segment)
    i_p = FEATS.index('chm_precip_daily')
    i_pet = FEATS.index('pet_jh')
    Pv = out[:, :, i_p]
    PETv = out[:, :, i_pet]
    newcols, newnames = [], []
    for dd in LAGD:
        newcols.append(shift_back(Pv, dd))
        newnames.append('P_lag%d' % dd)
    for w in LAGW:
        s, _ = roll_sum(Pv, w)
        newcols.append(s)
        newnames.append('P_sum%d' % w)
    for w in LAGW:
        newcols.append(roll_mean(PETv, w))
        newnames.append('pet_m%d' % w)
    out = np.concatenate([out] + [c[:, :, None] for c in newcols], axis=2)
    FEATS.extend(newnames)
    P('new antecedent meteorological features %d entries: %s' % (len(newnames), ','.join(newnames)))

    out = out[lead:]
    assert out.shape[0] == nt, 'length after cropping does not match: %d vs %d' % (out.shape[0], nt)

    # Target y (2,T,N) -> (T,nc,2)
    Y = np.load(os.path.join(d, 'y.npy'), mmap_mode='r')
    Ys = np.empty((nt, nc, 2), dtype='float32')
    CH = 500
    for s in range(0, nt, CH):
        e = min(nt, s + CH)
        b = np.asarray(Y[:, i0 + s:i0 + e])          # (2,chunk,N)
        Ys[s:e] = np.moveaxis(b[:, :, cells], 0, -1)  # (chunk,nc,2)
    del Y
    P('y subset read complete  shape=%s' % (Ys.shape,))

    # sanity check
    P('')
    P('=== Variable sanity check (median / range)===')
    for k, f in enumerate(FEATS):
        v = out[:, :, k].ravel()
        v = v[np.isfinite(v)]
        P('  %-18s p50=%10.3f  [%9.3f, %9.3f]  missing%.2f%%'
          % (f, np.median(v), v.min(), v.max(),
             100.0 * (1 - v.size / out[:, :, k].size)))
    for k, nm in enumerate(['SM_10cm', 'SM_30cm']):
        v = Ys[:, :, k].ravel()
        v = v[np.isfinite(v)]
        P('  %-18s p50=%10.4f  [%9.4f, %9.4f]'
          % (nm, np.median(v), v.min(), v.max()))

    # Annual-total check (whether PET and whether the precipitation convention is reasonable)
    tv = tmv[i0:i1]
    yr = tv.astype('datetime64[Y]').astype(int) + 1970
    P('')
    P('=== Annual-total check (mm/yr)===')
    P('  Years   cmfd_pet   pet_jh   chm_precip   cmfd_precip')
    i_pet = FEATS.index('cmfd_pet')
    i_jh = FEATS.index('pet_jh')
    i_c1 = FEATS.index('chm_precip_daily')
    i_c2 = FEATS.index('cmfd_precip')
    for y in sorted(set(yr.tolist())):
        m = yr == y
        n = int(m.sum())
        P('  %4d  %9.1f %8.1f %11.1f %12.1f'
          % (y, out[m, :, i_pet].mean() * 365, out[m, :, i_jh].mean() * 365,
             out[m, :, i_c1].mean() * 365, out[m, :, i_c2].mean() * 365))
    P('  Reference: North China Plain annual PET approx. 900-1100 mm/yr, annual precipitation approx. 500-700 mm/yr')

    if not np.isfinite(out).all():
        P('!! WARNING:X subset contains non-finite values')
    if not np.isfinite(Ys).all():
        P('!! WARNING:y subset contains non-finite values')

    # Wrote
    os.makedirs(BENCH, exist_ok=True)
    op = os.path.join(BENCH, 'dl_%s_nc%d_%s_%s%s.npz'
                      % (a.region, nc, a.t0.replace('-', ''), a.t1.replace('-', ''), a.tag))
    np.savez_compressed(op, X=out, Y=Ys, time=tmv[i0:i1].astype('datetime64[D]'),
                        cells=cells, feats=np.array(FEATS), region=a.region)
    P('')
    P('WROTE %s  (%.1f MB)' % (op, os.path.getsize(op) / 1e6))

    lp = os.path.join(LOG, '_300_dl_prep_%s.txt' % a.region)
    open(lp, 'w', encoding='utf-8').write('\n'.join(LOG))
    P('WROTE log %s' % lp)


if __name__ == '__main__':
    main()
