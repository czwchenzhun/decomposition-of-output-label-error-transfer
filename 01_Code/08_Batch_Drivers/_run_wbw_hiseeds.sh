# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Add seeds 7/21 for w=10/30 -- confirm that the "high-strength soft-constraint plateau does not fall back"
set -u
cd "$PKG_CODE"
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred --pet_src psp --threads 8"

for W in 10 30; do
  for SD in 7 21; do
    TAG="_wbw${W}s${SD}"
    echo "--- train LPMIX w=$W seed=$SD ---"
    if [ ! -f "${BENCH_DIR}/pred_LPMIX${TAG}.npz" ]; then
      python 13_dl_hybrid.py $COMMON --cfgs LPMIX --wbw $W --seed $SD --tag $TAG \
        > _302${TAG}.txt 2>&1 || echo "302 $TAG FAILED"
      grep -E "^LPMIX " _302${TAG}.txt | head -1
    else
      echo "skip (pred exists)"
    fi
    echo "--- 321+370 $TAG ---"
    python 16_model_vs_truth.py --pred_tag "$TAG" --cfgs LPMIX \
      --out_suffix "gpu${TAG}" > /dev/null 2>&1 || echo "321 $TAG FAILED"
    python 19_err_transfer.py --pair_tag "gpu${TAG}" --nboot 1000 \
      --out_suffix "gpu${TAG}" > /dev/null 2>&1 || echo "370 $TAG FAILED"
  done
done
echo "WBW HI-SEEDS DONE"
