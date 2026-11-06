# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# LPMIX dose response: seed confirmation + whether the high-w segment falls back (an "inverted U" or not)
#
# Pilot (seed 42) results: on 10cm, alpha rises monotonically with w from 1.91% -> 3.59%, rho -0.103 -> -0.158,
# i.e. **stronger constraints instead enhance bias correction**, opposite to the original hypothesis. 30cm is flat with no trend.
# But the pilot has only one seed -- and a single seed misled the previous lambda conclusion, so extra seeds are required.
#
# Design:
#   A) w ∈ {0.3, 3} × seed ∈ {7,21}, together with the existing seed42 forming a 3-point × 3-seed trend test
#      (the w=0 baseline already has three LSTM seeds: 42/7/21)
#   B) w ∈ {10, 30} seed 42: test whether the curve falls back at higher w
#      -- the 10cm alpha of HYBNS (hard skeleton) is only 2.03%; if LPMIX converges to it as w increases,
#         then the true relation is an "inverted U": moderate soft constraint is optimal, while the hard skeleton instead suppresses headroom for bias correction.
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8 --cfgs LPMIX"

echo "########## A) training ##########"
conda activate rocm
for W in 0.3 3; do
  for SD in 7 21; do
    TAG="_wbw${W}s${SD}"
    if [ -f "../out/_bench/pred_LPMIX${TAG}.npz" ]; then echo "skip $TAG"; continue; fi
    echo "--- train LPMIX w=$W seed=$SD ---"
    python 13_dl_hybrid.py $COMMON --wbw $W --seed $SD --tag "$TAG" \
        > "_302${TAG}.txt" 2>&1
    echo "302 $TAG EXIT=$?"; grep -E "^LPMIX " "_302${TAG}.txt" | head -1
  done
done
for W in 10 30; do
  TAG="_wbw${W}"
  if [ -f "../out/_bench/pred_LPMIX${TAG}.npz" ]; then echo "skip $TAG"; continue; fi
  echo "--- train LPMIX w=$W seed=42 ---"
  python 13_dl_hybrid.py $COMMON --wbw $W --seed 42 --tag "$TAG" \
      > "_302${TAG}.txt" 2>&1
  echo "302 $TAG EXIT=$?"; grep -E "^LPMIX " "_302${TAG}.txt" | head -1
done

echo "########## B) 321 + 370 ##########"
conda activate geoenv
for spec in "0.3:7" "0.3:21" "3:7" "3:21" "10:42" "30:42"; do
  W="${spec%%:*}"; SD="${spec##*:}"
  if [ "$SD" = "42" ]; then TAG="_wbw${W}"; else TAG="_wbw${W}s${SD}"; fi
  SUF="gpu_wbw${W}s${SD}"
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

echo "========== WBW SEEDS DONE =========="
