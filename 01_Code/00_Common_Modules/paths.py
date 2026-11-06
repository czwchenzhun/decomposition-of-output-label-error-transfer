# -*- coding: utf-8 -*-
"""Package path configuration: everything is relative to this package;
no data outside the package is referenced.

Every script in the package adds 01_Code/00_Common_Modules to sys.path at the top
of the file before importing this module.

So that running a script never modifies the archived originals, the first import
copies 02_Results_Data/truth_chain/ to 06_Reproduction_Outputs/truth_chain/ and
02_Results_Data/Training_Inputs/ to
06_Reproduction_Outputs/_Upstream_Intermediate_Products/_bench/. From then on,
scripts read and write only those runtime copies; the archived data in 02_Results_Data/ stays read-only.
"""
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))       # 01_Code/00_Common_Modules
CODE = os.path.dirname(_HERE)                             # 01_Code
PKG = os.path.dirname(CODE)                               # package root directory

# --- Archived data inside the package (included, read-only)---
DATA = os.path.join(PKG, '02_Results_Data')
TRUTH_ARCHIVE = os.path.join(DATA, 'truth_chain')
STATION = os.path.join(DATA, 'Station_and_Truth_Base_Tables')
BENCH_JSON = os.path.join(DATA, 'bench_json')
BENCH_ARCHIVE = os.path.join(DATA, 'Training_Inputs')      # training-input npz / prediction samples included in the package
TRAIN_SUM = os.path.join(DATA, 'Training_and_Baseline_Summaries')

# --- Figures and manuscript ---
FIG = os.path.join(PKG, '03_Figures', 'Paper_Figures_Main_and_Supplementary')
MANU = os.path.join(PKG, '00_Manuscript')
MANU_FIG = os.path.join(MANU, 'figures')

# --- Runtime outputs ---
LOG = os.path.join(PKG, '04_Runtime_Logs')
REPRO = os.path.join(PKG, '06_Reproduction_Outputs')

# --- Upstream chain: requires raw data, see Data_Acquisition_and_Reproduction_Guide.md ---
RAW = os.environ.get('RAW_ROOT', os.path.join(PKG, '_Raw_Data_To_Be_Downloaded'))
WORK = os.path.join(REPRO, '_Upstream_Intermediate_Products')
BENCH = os.path.join(WORK, '_bench')
OUT_COMPAT = os.path.join(WORK, 'out')                    # backward-compatible with the old os.path.join(OUT, ...)

# --- Runtime copy: training inputs are also included in the package; on first
#     import they are copied to WORK/_bench ---
# The archived originals are in 02_Results_Data/Training_Inputs/; scripts read and
# write 06_Reproduction_Outputs/_Upstream_Intermediate_Products/_bench/, so running
# a script (including training scripts writing best_*.pt / pred_*.npz) does not
# modify the archived originals.
_BENCH_SENTINEL = 'dl_R0_northchina_nc256_20020101_20131231_sta.npz'
if os.path.isdir(BENCH_ARCHIVE) and not os.path.isfile(os.path.join(BENCH, _BENCH_SENTINEL)):
    try:
        os.makedirs(BENCH, exist_ok=True)
        for _x in os.listdir(BENCH_ARCHIVE):
            _s = os.path.join(BENCH_ARCHIVE, _x)
            _d = os.path.join(BENCH, _x)
            if os.path.isfile(_s) and not os.path.isfile(_d):
                shutil.copy2(_s, _d)
    except OSError:
        BENCH = BENCH_ARCHIVE

for _d in (LOG, REPRO, WORK):
    os.makedirs(_d, exist_ok=True)

# --- Runtime copy: TRUTH/MULTI, read and written by the scripts, both point here ---
TRUTH = os.path.join(REPRO, 'truth_chain')
MULTI = os.path.join(TRUTH, 'multi')
if not os.path.isdir(TRUTH):
    try:
        shutil.copytree(TRUTH_ARCHIVE, TRUTH)
    except OSError:
        # Fallback for read-only media: read the archived originals directly (writes would then overwrite them, see the guide)
        TRUTH = TRUTH_ARCHIVE
        MULTI = os.path.join(TRUTH, 'multi')
