# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Generate 20-year (2000-2019) full-grid DL data + corresponding PSP evapotranspiration for the specified regions
# Usage: bash _prep20_regions.sh R1_yangtze ...
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate geoenv
cd ${PKG_CODE}

for R in "$@"; do
 NPZ="${BENCH_DIR}/dl_${R}_nc1369_20000101_20191231_full20.npz"
 ET="${BENCH_DIR}/et_${R}_2000_2019_full20.npz"
 echo "########## 300 $R $(date +%H:%M:%S) ##########"
 if [ ! -f "$NPZ" ]; then
 python 11_dl_prep.py --region $R --ncells 9999 \
 --t0 2000-01-01 --t1 2019-12-31 --tag _full20 2>&1 | tail -6
 else
 echo "npz already exists, skipping 300"
 fi
 echo "########## 310 $R $(date +%H:%M:%S) ##########"
 if [ ! -f "$ET" ]; then
 python 14_prep_et.py --npz "$NPZ" --tag _full20 2>&1 | tail -8
 else
 echo "et already exists, skipping 310"
 fi
done
echo "PREP_DONE: $*"
