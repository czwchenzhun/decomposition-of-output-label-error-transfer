# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Fast GPU hyperparameter scan: probe the effect of capacity / epochs / schedule on absolute accuracy on a 256-grid-cell subset
# Usage: bash _sweep_gpu.sh
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20160101_20191231_lag.npz"
ET="${BENCH_DIR}/et_R0_northchina_2016_2019.npz"
COMMON="--npz $NPZ --device cuda --stride 5 --batch 4096 --threads 4 \
        --lamet 5 --pet_src psp --etnpz $ET"

run () {  # $1=label $2..=remaining arguments
  local tag=$1; shift
  echo "########## $tag ##########"
  python 13_dl_hybrid.py $COMMON --tag "_sw_$tag" "$@" 2>&1 | grep -E "^  ep|^(HYBNS|LSTM|PERS) |model |early"
}

run h064e40 --cfgs HYBNS,LSTM --hidden 64  --epochs 40 --sched cosine
run h128e40 --cfgs HYBNS,LSTM --hidden 128 --epochs 40 --sched cosine
run h256e40 --cfgs HYBNS,LSTM --hidden 256 --epochs 40 --sched cosine
echo "SWEEP_DONE"
