# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# R3 station period: seed replication + scheduled-sampling control
#
# Background: the λ=0/5/20 "fidelity--bias-correction" frontier has only one seed, and α is non-monotonic
#       (10cm: λ0 2.3% > λ20 1.2% > λ5 0.5%), so it is unclear whether the λ effect exceeds seed noise.
#       Also, LSTM uses scheduled sampling by default while HYBNS does not, so the α gap may include an SS contribution.
#
# This script:
#   A) HYBNS λ ∈ {0,5,20} × seed ∈ {7,21}   (6 run)
#   B) LSTM  seed ∈ {7,21}                  (2 run)
#   C) LSTMNS seed 42                       (1 run, LSTM control without scheduled sampling)
# Together with the existing seed 42 (HYBNS and LSTM at λ=0/5/20), this forms a full 3 seed × 3 λ grid.
#
# Data and time split are identical to the CPU version (train 2002-2009 / val 2010 / test 2011-2013).
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8"

echo "########## A) training: 3λ × 2seed HYBNS + LSTM 2seed + LSTMNS ##########"
conda activate rocm

for lam in 0 5 20; do
  for sd in 7 21; do
    TAG="_staG_lam${lam}_s${sd}"
    if [ -f "../out/_bench/pred_HYBNS${TAG}.npz" ]; then echo "skip train $TAG"; continue; fi
    echo "--- train HYBNS lam=$lam seed=$sd ---"
    python 13_dl_hybrid.py $COMMON --lamet $lam --seed $sd --tag "$TAG" --cfgs HYBNS \
        > "_302${TAG}.txt" 2>&1
    echo "302 $TAG EXIT=$?"
  done
done

for sd in 7 21; do
  TAG="_staGs${sd}"
  if [ -f "../out/_bench/pred_LSTM${TAG}.npz" ]; then echo "skip train $TAG"; continue; fi
  echo "--- train LSTM seed=$sd ---"
  python 13_dl_hybrid.py $COMMON --lamet 5 --seed $sd --tag "$TAG" --cfgs LSTM \
      > "_302${TAG}.txt" 2>&1
  echo "302 $TAG EXIT=$?"
done

TAG="_staG_lstmns"
if [ -f "../out/_bench/pred_LSTMNS${TAG}.npz" ]; then echo "skip train $TAG"; else
  echo "--- train LSTMNS seed=42 ---"
  python 13_dl_hybrid.py $COMMON --lamet 0 --seed 42 --tag "$TAG" --cfgs LSTMNS \
      > "_302${TAG}.txt" 2>&1
  echo "302 $TAG EXIT=$?"
fi

echo "########## B) run truth chain 321 + error decomposition 370 for each tag ##########"
conda activate geoenv

run_pair () {   # $1=tag  $2=cfgs  $3=suffix
  echo "---------- 321 $1 ($2) ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$1" --region R0_northchina \
      --cfgs "$2" --out_suffix "$3" > "_321_gpu_$3.txt" 2>&1
  if [ $? -ne 0 ]; then echo "321 $3 FAIL"; return; fi
  python 19_err_transfer.py \
      --pair "${DATA_DIR}/truth_chain/truth_chain_L2_monthly_$3.csv" \
      --out_suffix "$3" --nboot 1000 > "_370_gpu_$3.txt" 2>&1
  echo "370 $3 EXIT=$?"
}

for lam in 0 5 20; do
  for sd in 7 21; do
    run_pair "_staG_lam${lam}_s${sd}" "HYBNS" "gpu_lam${lam}_s${sd}"
  done
done
for sd in 7 21; do
  run_pair "_staGs${sd}" "LSTM" "gpu_LSTMs${sd}"
done
run_pair "_staG_lstmns" "LSTMNS" "gpu_lstmns"

echo "========== SEEDS ALLDONE =========="
