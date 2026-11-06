# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Full-grid (1369 grid cells / 4 years) second batch: spatial models + fairness / expressiveness diagnostics
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

NPZ="${BENCH_DIR}/dl_R0_northchina_nc1369_20160101_20191231_full.npz"
ET="${BENCH_DIR}/et_R0_northchina_2016_2019_full.npz"
COMMON="--npz $NPZ --etnpz $ET --pet_src psp --lamet 5 --device cuda \
        --epochs 40 --stride 2 --bwin 16 --hidden 128 --ctx 32 --sched cosine"

go () { # $1=tag  $2=cfgs  $3..=extra
  local tag=$1; local cfgs=$2; shift 2
  echo "########## $tag ($cfgs) ##########"
  python 360_dl_st.py $COMMON --tag "_$tag" --cfgs "$cfgs" "$@" \
      2>&1 | grep -E "model |^  \[test\]|early|Error|Traceback"
}

go stconv   STCONV
go hybnss   HYBNSS
go lstmns   LSTMNS
go hybnsdyn HYBNS   --dynw 1
go hybnssdyn HYBNSS --dynw 1
echo "BATCH_DONE"
