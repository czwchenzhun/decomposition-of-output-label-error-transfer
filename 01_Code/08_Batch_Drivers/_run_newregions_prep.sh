# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Two new regions (v20 expansion): R4_longzhong Longzhong (32 stations) / R5_sichuan Sichuan Basin (24 stations)
# Period strictly matches the existing four regions: 2002-01-01 ~ 2013-12-31 (observation data set ends in 2013)
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate geoenv

CELLDIR="${DATA_DIR}/truth_chain/station_cells"

for REG in R4_longzhong R5_sichuan; do
  NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
  ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
  echo "########## $REG ##########"
  if [ -f "$NPZ" ]; then echo "skip 300 (exists)"; else
    python 11_dl_prep.py --region $REG --ncells 256 \
      --t0 2002-01-01 --t1 2013-12-31 --seed 42 --tag _sta \
      --station_cells "$CELLDIR/cells_${REG}.csv" > "_300_sta_${REG}.txt" 2>&1
    echo "300 $REG EXIT=$?"
  fi
  if [ -f "$ET" ]; then echo "skip 310 (exists)"; else
    python 14_prep_et.py --npz "$NPZ" --region $REG \
      --vars ET,PET,Tr,Eb,Ei --tag _sta > "_310_sta_${REG}.txt" 2>&1
    echo "310 $REG EXIT=$?"
  fi
done
echo "========== NEWPREP ALLDONE =========="
