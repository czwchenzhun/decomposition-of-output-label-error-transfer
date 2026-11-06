# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Diagnostic set: answers three attribution questions
#   A. LSTMNS -- LSTM without scheduled sampling, removing the confound of "training tricks"
#   B. HYBNS +dynw -- inter-layer evapotranspiration allocation changed from a global scalar to state-dependent layer-wise root water uptake
#   C. HYBNSS +dynw -- spatial context + dynamic allocation (candidate best combination)
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

NPZ="${BENCH_DIR}/dl_R0_northchina_nc1369_20160101_20191231_full.npz"
ET="${BENCH_DIR}/et_R0_northchina_2016_2019_full.npz"
COMMON="--npz $NPZ --etnpz $ET --pet_src psp --lamet 5 --device cuda \
        --epochs 40 --stride 2 --bwin 16 --hidden 128 --ctx 32 --sched cosine"

echo "########## A: LSTMNS (LSTM without SS) ##########"
python 360_dl_st.py $COMMON --tag _diag_a --cfgs LSTMNS 2>&1 | grep -E "model |^  \[test\]|^  ep(10|20|30|40) "

echo "########## B: HYBNS + dynw ##########"
python 360_dl_st.py $COMMON --tag _diag_b --cfgs HYBNS --dynw 1 2>&1 | grep -E "model |^  \[test\]|^  ep(10|20|30|40) "

echo "########## C: HYBNSS + dynw ##########"
python 360_dl_st.py $COMMON --tag _diag_c --cfgs HYBNSS --dynw 1 2>&1 | grep -E "model |^  \[test\]|^  ep(10|20|30|40) "

echo "DIAG_DONE"
