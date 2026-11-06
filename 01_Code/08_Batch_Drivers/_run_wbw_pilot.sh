# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# LPMIX soft water-balance penalty: dose-response pilot for constraint strength w (R3 station period, seed 42)
#
# Design points:
#   LPMIX and LSTM share the same architecture and capacity (cells/head initialisation value-by-value identical, verified),
#   the only difference is the extra w * (water-balance residual)^2 in the loss. w=0 is LSTM.
#   Hence w is a **pure constraint-strength axis**, not confounded with architecture / state form / parameter count as in HYBNS vs LSTM.
#   λ (ET auxiliary loss) was ruled out by the 3-seed experiment (effect < seed sd); this script is a replacement causal knob.
#
# Scale reference for w: the residual is a mean square in units of Δθ/day, typical |Δθ|~0.02, the same order as the main loss MSE(4e-4),
# so w=1 is already a "barely comparable" strength; from w=3 onward the free increment should be strongly suppressed.
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8 --cfgs LPMIX"

echo "########## A) training: w scan (seed 42) ##########"
conda activate rocm
for W in 0.03 0.1 0.3 1 3; do
  TAG="_wbw${W}"
  if [ -f "../out/_bench/pred_LPMIX${TAG}.npz" ]; then echo "skip $TAG"; continue; fi
  echo "--- train LPMIX w=$W ---"
  python 13_dl_hybrid.py $COMMON --wbw $W --seed 42 --tag "$TAG" \
      > "_302${TAG}.txt" 2>&1
  echo "302 $TAG EXIT=$?"
  grep -E "^LPMIX " "_302${TAG}.txt" | head -1
done

echo "########## B) 321 + 370 ##########"
conda activate geoenv
for W in 0.03 0.1 0.3 1 3; do
  TAG="_wbw${W}"
  SUF="gpu_wbw${W}"
  echo "---------- 321 $TAG ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$TAG" --region R0_northchina \
      --cfgs LPMIX --out_suffix "$SUF" > "_321_$SUF.txt" 2>&1
  if [ $? -ne 0 ]; then echo "321 $SUF FAIL"; continue; fi
  python 19_err_transfer.py \
      --pair "${DATA_DIR}/truth_chain/truth_chain_L2_monthly_$SUF.csv" \
      --out_suffix "$SUF" --nboot 1000 > "_370_$SUF.txt" 2>&1
  echo "370 $SUF EXIT=$?"
  sed -n '/error budget/,/^$/p' "_370_$SUF.txt" | head -8
done

echo "========== WBW PILOT DONE =========="
