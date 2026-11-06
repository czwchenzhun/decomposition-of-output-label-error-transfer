# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""Stage A probe: multi-region data availability + grid definition

Confirm SMCI / CMFD / CSDLv2 soil / DEM tiles are available in each target window and whether coverage is sufficient.
Output:scripts/_01_multiregion_probe.txt
"""
import os, glob, zipfile, re
import numpy as np

RAW = RAW
DEM_DIR = RAW + '/dem/DEMtile data'

# ---- Region definition: central window 3°×3°, expanded by 0.4° ----
# (lon0, lat0, lon1, lat1) central window; modelling grid = expanded cell centres (aligned with SMCI .X0 cell centres aligned)
BUF = 0.4
REGIONS = {
    'R0_northchina': (114.0, 36.0, 117.0, 39.0),
}

L = []
def P(s=''):
    L.append(str(s)); print(s, flush=True)


def grid_for(lon0, lat0, lon1, lat1, buf=BUF):
    """expanded buf after 0.1° cell-centre sequence (aligned with .X0, taking cell centres within [min+0.05, max-0.05])"""
    def axis(a, b):
        lo, hi = a - buf, b + buf
        # Cell centres must keep the whole cell [c-0.05, c+0.05] fall inside the expanded box
        c0 = np.ceil((lo + 0.05) * 10) / 10.0
        c1 = np.floor((hi - 0.05) * 10) / 10.0
        n = int(round((c1 - c0) / 0.1)) + 1
        return np.round(c0 + 0.1 * np.arange(n), 4)
    return axis(lat0, lat1), axis(lon0, lon1)


P('=== Stage A multi-region data-availability probe ===')
P('')

# ---------- 1. Regions and grid ----------
P('--- 1. Regions and modelling grid (expanded by %.1f°)---' % BUF)
grids = {}
for k, (lo0, la0, lo1, la1) in REGIONS.items():
    LATg, LONg = grid_for(lo0, la0, lo1, la1)
    grids[k] = (LATg, LONg)
    P('  %-15s central window lon %.1f-%.1f lat %.1f-%.1f' % (k, lo0, lo1, la0, la1))
    P('      Grid LAT %.1f..%.1f (%d)  LON %.1f..%.1f (%d)  -> %d grid cells' % (
        LATg[0], LATg[-1], len(LATg), LONg[0], LONg[-1], len(LONg), len(LATg) * len(LONg)))
P('')

# ---------- 2. SMCI ----------
P('--- 2. SMCI target field (data5/SMCI_9km)---')
for dep in ('10cm', '30cm'):
    d = os.path.join(RAW, 'data5', 'SMCI_9km', dep)
    ok = os.path.isdir(d)
    zs = sorted(glob.glob(os.path.join(d, '*.zip'))) if ok else []
    P('  %s: directory=%s  zipcount=%d' % (dep, ok, len(zs)))
    if zs:
        yrs = sorted(int(re.search(r'_(\d{4})_', os.path.basename(z)).group(1)) for z in zs
                     if re.search(r'_(\d{4})_', os.path.basename(z)))
        P('      Years %d-%d (%d entries)' % (min(yrs), max(yrs), len(yrs)))

z = os.path.join(RAW, 'data5', 'SMCI_9km', '10cm', 'SMCI_9km_2020_10cm.zip')
if os.path.exists(z):
    import netCDF4
    with zipfile.ZipFile(z) as zf:
        nm = [n for n in zf.namelist() if n.endswith('.nc')][0]
        buf = zf.open(nm).read()
    ncd = netCDF4.Dataset('mem', memory=buf)
    lat = np.asarray(ncd.variables['lat'][:]); lon = np.asarray(ncd.variables['lon'][:])
    v = [x for x in ncd.variables if x not in ('lat', 'lon', 'time')][0]
    P('  Sample 2020 10cm: variables=%s shape=%s' % (v, ncd.variables[v].shape))
    P('      lat %.2f..%.2f (%d)  lon %.2f..%.2f (%d)' % (
        lat.min(), lat.max(), len(lat), lon.min(), lon.max(), len(lon)))
    for k, (LATg, LONg) in grids.items():
        inside = (lat.min() <= LATg[0]) and (lat.max() >= LATg[-1]) and \
                 (lon.min() <= LONg[0]) and (lon.max() >= LONg[-1])
        P('      %-15s grid covered by SMCI covered by: %s' % (k, 'Yes' if inside else 'No'))
    ncd.close()
else:
    P('  !! sample file does not exist: %s' % z)
P('')

# ---------- 3. CMFD ----------
P('--- 3. CMFD meteorological forcing (data7)---')
fd = RAW + '/data7/Data_forcing_01dy_010deg'
if os.path.isdir(fd):
    ns = sorted(glob.glob(os.path.join(fd, '*.nc')))
    P('  directory exists, nc count=%d' % len(ns))
    if ns:
        P('      sample: %s' % os.path.basename(ns[0]))
    # sample one variable to inspect its range
    samp = os.path.join(fd, 'temp_CMFD_V0200_B-01_01dy_010deg_202001-202012.nc')
    if os.path.exists(samp):
        import netCDF4
        ncd = netCDF4.Dataset(samp)
        lat = np.asarray(ncd.variables['lat'][:]); lon = np.asarray(ncd.variables['lon'][:])
        P('      temp 2020: lat %.2f..%.2f (%d)  lon %.2f..%.2f (%d)' % (
            lat.min(), lat.max(), len(lat), lon.min(), lon.max(), len(lon)))
        for k, (LATg, LONg) in grids.items():
            inside = (lat.min() <= LATg[0]) and (lat.max() >= LATg[-1]) and \
                     (lon.min() <= LONg[0]) and (lon.max() >= LONg[-1])
            P('      %-15s by CMFD covered by: %s' % (k, 'Yes' if inside else 'No'))
        ncd.close()
    else:
        P('      !! sample file does not exist')
    yrs = sorted(set(int(re.search(r'_(\d{4})01-', os.path.basename(n)).group(1))
                     for n in ns if re.search(r'_(\d{4})01-', os.path.basename(n))))
    if yrs:
        P('      Years %d-%d' % (min(yrs), max(yrs)))
else:
    P('  !! CMFD directory does not exist: %s' % fd)
P('')

# ---------- 4. soil properties data6 ----------
P('--- 4. CSDLv2 soil properties (data6)---')
d6 = os.path.join(RAW, 'data6')
if os.path.isdir(d6):
    props = sorted(os.listdir(d6))
    P('  attribute directories %d entries: %s' % (len(props), ', '.join(props)))
    p1 = os.path.join(d6, props[0]) if props else None
    if p1 and os.path.isdir(p1):
        zs = sorted(glob.glob(os.path.join(p1, '*.zip')))
        P('  %s: zip count=%d  sample=%s' % (props[0], len(zs),
                                        os.path.basename(zs[0]) if zs else '-'))
        if zs:
            import netCDF4
            with zipfile.ZipFile(zs[0]) as zf:
                nm = [n for n in zf.namelist() if n.endswith('.nc')][0]
                buf = zf.open(nm).read()
            ncd = netCDF4.Dataset('mem', memory=buf)
            lat = np.asarray(ncd.variables['latitude'][:]); lon = np.asarray(ncd.variables['longitude'][:])
            P('      range lat %.2f..%.2f (%d)  lon %.2f..%.2f (%d)' % (
                lat.min(), lat.max(), len(lat), lon.min(), lon.max(), len(lon)))
            for k, (LATg, LONg) in grids.items():
                inside = (lat.min() <= LATg[0]) and (lat.max() >= LATg[-1]) and \
                         (lon.min() <= LONg[0]) and (lon.max() >= LONg[-1])
                P('      %-15s covered: %s' % (k, 'Yes' if inside else 'No'))
            ncd.close()
else:
    P('  !! data6 does not exist')
P('')

# ---------- 5. DEM tile coverage ----------
P('--- 5. DEM tile coverage (unified directory)---')
if os.path.isdir(DEM_DIR):
    tifs = [f for f in os.listdir(DEM_DIR) if f.lower().endswith('.tif')]
    pat = re.compile(r'Copernicus_DSM_COG_10_N(\d{2})_00_E(\d{3})_00_DEM\.tif')
    have = set()
    for f in tifs:
        m = pat.match(f)
        if m:
            have.add((int(m.group(1)), int(m.group(2))))
    P('  unified directory tif=%d  parseable tiles=%d' % (len(tifs), len(have)))
    for k, (LATg, LONg) in grids.items():
        need = set()
        for la in LATg:
            for lo in LONg:
                need.add((int(np.floor(la)), int(np.floor(lo))))
        miss = need - have
        P('  %-15s need %d tiles, missing %d %s' % (k, len(need), len(miss),
                                          '' if not miss else sorted(miss)[:8]))
else:
    P('  !! DEM directory does not exist')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_01_multiregion_probe.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('WROTE', out)
