# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Night batches 2 + 3: six-region 20-year full-grid training (PERS/HYBNS/LSTM share a tag) -> 6x6 transfer matrix
# Automatic wait: (1) batch 1 (R3 supplement) finishes  (2) 20-year data for each region are ready
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

BENCH="${BENCH_DIR}"
COMMON="--pet_src psp --lamet 5 --device cuda --epochs 40 --stride 10 \
 --bwin 16 --hidden 128 --sched cosine \
 --ttrain 2000-01-01,2016-12-31 --tval 2017-01-01,2018-12-31 \
 --ttest 2019-01-01,2019-12-31"

# ---- wait for batch 1 to finish (up to 4 hours) ----
W=0
while ! grep -q "NIGHT_R3_DONE" _night_r3.log 2>/dev/null; do
 sleep 120; W=$((W+1))
 if [ $W -gt 120 ]; then echo "!! wait for batch 1 timed out, giving up"; break; fi
done
echo "batch 1 finished (waited $((W*2)) min), starting six-region training $(date +%H:%M:%S)"

for R in R1_yangtze R0_northchina R3_northeast R2_loess; do
 NPZ="$BENCH/dl_${R}_nc1369_20000101_20191231_full20.npz"
 ET="$BENCH/et_${R}_2000_2019_full20.npz"
 W=0
 while [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; do
 sleep 120; W=$((W+1))
 if [ $W -gt 150 ]; then echo "!! data wait timed out $R"; break; fi
 done
 echo "########## train $R $(date +%H:%M:%S) ##########"
 python 360_dl_st.py $COMMON --npz "$NPZ" --etnpz "$ET" \
 --cfgs PERS,HYBNS,LSTM --tag "_20y_$R" \
 2>&1 | grep -E "^ \[test\]|drift ratio|early|Error|Traceback|window count|WROTE|model "
done
echo "SIX_REGIONS_TRAINED $(date +%H:%M:%S)"

# ---- transfer matrix: zero-shot evaluation of each source-region model on 6 target regions ----
for R in R1_yangtze R0_northchina R3_northeast R2_loess; do
 echo "########## transfer row $R $(date +%H:%M:%S) ##########"
 python 362_transfer_eval.py --src $R --src_tag "_20y_$R" --bench_tag _20y \
 --cfgs PERS,HYBNS,LSTM --device cuda --bwin 16 \
 2>&1 | grep -E "self|transfer|WROTE|Error|Traceback|skip"
done
echo "NIGHT_REGIONS_DONE $(date +%H:%M:%S)"
