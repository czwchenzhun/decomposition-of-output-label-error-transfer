# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# R3 full grid x 20 years (2000-2019): train 2000-2016 / val 2017-2018 / test 2019
# Purpose: test whether a longer training sequence (cross-year weather diversity) can further reduce absolute error
# Note: stride is set to 10 to limit GPU/CPU memory (about 610 training windows x 1369 grid cells)
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

NPZ="${BENCH_DIR}/dl_R0_northchina_nc1369_20000101_20191231_full20.npz"
ET="${BENCH_DIR}/et_R0_northchina_2000_2019_full20.npz"
COMMON="--npz $NPZ --etnpz $ET --pet_src psp --lamet 5 --device cuda \
        --epochs 40 --stride 10 --bwin 16 --sched cosine \
        --ttrain 2000-01-01,2016-12-31 --tval 2017-01-01,2018-12-31 \
        --ttest 2019-01-01,2019-12-31"

go () { # $1=tag $2=cfgs $3..=extra
  local tag=$1; local cfgs=$2; shift 2
  echo "########## 20y $tag ($cfgs) ##########"
  python 360_dl_st.py $COMMON --tag "_20y_$tag" --cfgs "$cfgs" "$@" \
      2>&1 | grep -E "model |^  \[test\]|early|Error|Traceback|window count"
}

# $EXTRA passes cfgs and extra arguments from the caller (e.g. --hidden 256)
go "${TAG:-lstm}" "${CFGS:-PERS,LSTM}" ${EXTRA:-}
echo "RUN20Y_DONE"
