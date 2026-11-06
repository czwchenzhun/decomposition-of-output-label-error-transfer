# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Night batch 1: R3 full-grid 20-year run -- robustness supplement
#   (1) across random seeds (identifiability / conclusion robustness)  (2) ET auxiliary-loss weight lambda scan
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

NPZ="${BENCH_DIR}/dl_R0_northchina_nc1369_20000101_20191231_full20.npz"
ET="${BENCH_DIR}/et_R0_northchina_2000_2019_full20.npz"
COMMON="--npz $NPZ --etnpz $ET --pet_src psp --device cuda \
        --epochs 40 --stride 10 --bwin 16 --sched cosine \
        --ttrain 2000-01-01,2016-12-31 --tval 2017-01-01,2018-12-31 \
        --ttest 2019-01-01,2019-12-31"

go () { # $1=tag  $2..=extra (incl. --cfgs/--seed/--lamet)
  local tag=$1; shift
  echo "########## R3 $tag  $(date +%H:%M:%S) ##########"
  python 360_dl_st.py $COMMON --tag "_20y_$tag" "$@" \
      2>&1 | grep -E "^  \[test\]|drift ratio|early|Error|Traceback|window count|WROTE"
}

# (1) across seeds: two extra seeds each for HYBNS and LSTM (seed42 already exists)
go s7_hybns  --cfgs HYBNS --seed 7
go s21_hybns --cfgs HYBNS --seed 21
go s7_lstm   --cfgs LSTM  --seed 7
go s21_lstm  --cfgs LSTM  --seed 21
# (2) ET auxiliary-loss weight scan (0 = pure SM supervision; 20 = strong ET constraint)
go lam0  --cfgs HYBNS --lamet 0
go lam20 --cfgs HYBNS --lamet 20
echo "NIGHT_R3_DONE $(date +%H:%M:%S)"
