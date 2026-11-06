# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Six-region cross-region transfer matrix: train each source region once, zero-shot evaluate the other five regions
# Usage: bash run_xfer.sh
B="${BENCH_DIR}"
REGS="R1_yangtze R0_northchina R3_northeast R2_loess"
src=$1
short=$(echo $src | cut -d_ -f1)

xnpz=""; xet=""; xnm=""
for r in $REGS; do
 [ "$r" = "$src" ] && continue
 [ -z "$xnpz" ] || { xnpz="$xnpz,"; xet="$xet,"; xnm="$xnm,"; }
 xnpz="${xnpz}${B}/dl_${r}_nc256_20160101_20191231_lag.npz"
 if [ "$r" = "R0_northchina" ]; then
 xet="${xet}${B}/et_${r}_2016_2019.npz"
 else
 xet="${xet}${B}/et_${r}_2016_2019_lag.npz"
 fi
 xnm="${xnm}${r}"
done

if [ "$src" = "R0_northchina" ]; then
 setnpz="${B}/et_${src}_2016_2019.npz"
else
 setnpz="${B}/et_${src}_2016_2019_lag.npz"
fi

source /d/Software/Anaconda/etc/profile.d/conda.sh && conda activate torch
cd ${PKG_CODE}
python 13_dl_hybrid.py \
 --npz "${B}/dl_${src}_nc256_20160101_20191231_lag.npz" \
 --tag "_xfer${short}" --srcname "$src" \
 --xnpz "$xnpz" --xet "$xet" --xname "$xnm" \
 --ttrain 2016-01-01,2017-12-31 --tval 2018-01-01,2018-12-31 --ttest 2019-01-01,2019-12-31 \
 --cfgs HYBNS,LSTM,PERS --stride 30 --threads 4 --epochs 15 --K 30 \
 --lamet 5 --pet_src psp --etnpz "$setnpz" --seed 42
echo "EXIT=$? src=$src"
