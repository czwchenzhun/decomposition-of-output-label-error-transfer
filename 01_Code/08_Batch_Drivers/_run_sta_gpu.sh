# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# R3 station-period GPU retraining -- replace the under-configured CPU numbers in 321/370 with fully trained versions
#
# Original CPU configuration (_302_dl_hybrid_sta.txt):
#   hidden=64  epochs=15 (early stop)  stride=30  K=30  lamet=5  pet_src=psp
#   train 2002-01-01..2009-12-31 / val 2010 / test 2011-01-01..2013-12-31
# This script keeps the same data and time split, upgrading only the training side to the 20y main-experiment convention:
#   hidden=128  epochs=40  sched=cosine  device=cuda
# Two additional HYBNS variants (lamet=0 / 20) for the fidelity--bias-correction Pareto frontier (E2)
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
# batch=64 measured 100 s/epoch (GPU idle); batch=256 measured 25 s/epoch (4x).
# After a 4x batch increase, lr is raised from 1e-3 to 2e-3 by sqrt scaling to avoid diluting the effective learning rate.
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8"

echo "########## STEP 1: HYBNS(lamet=5) + LSTM ##########"
conda activate rocm
python 13_dl_hybrid.py $COMMON --lamet 5 --tag _staG --cfgs HYBNS,LSTM \
    > _302_staG.txt 2>&1
echo "302 _staG EXIT=$?"; grep -E "^model |h1|all|drift ratio" _302_staG.txt | tail -20

conda activate geoenv
echo "---------- 321 _staG ----------"
python 16_model_vs_truth.py --npz "$NPZ" --tag _staG --region R0_northchina \
    --cfgs HYBNS,LSTM --out_suffix gpu > _321_gpu.txt 2>&1
echo "321 gpu EXIT=$?"; grep -E "MODEL|SMCI|ubRMSE|hit" _321_gpu.txt | head -20

echo "---------- 370 gpu ----------"
python 19_err_transfer.py \
    --pair "${DATA_DIR}/truth_chain/truth_chain_L2_monthly_gpu.csv" \
    --out_suffix gpu --nboot 1000 > _370_gpu.txt 2>&1
echo "370 gpu EXIT=$?"; sed -n '/error budget/,/TOTAL PASS/p' _370_gpu.txt

echo "########## STEP 2: fidelity--bias-correction frontier lamet=0 / 20 ##########"
conda activate rocm
python 13_dl_hybrid.py $COMMON --lamet 0 --tag _staG_lam0 --cfgs HYBNS \
    > _302_staG_lam0.txt 2>&1
echo "302 lam0 EXIT=$?"
python 13_dl_hybrid.py $COMMON --lamet 20 --tag _staG_lam20 --cfgs HYBNS \
    > _302_staG_lam20.txt 2>&1
echo "302 lam20 EXIT=$?"

conda activate geoenv
for T in lam0 lam20; do
  echo "---------- 321 _staG_$T ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "_staG_$T" --region R0_northchina \
      --cfgs HYBNS --out_suffix "gpu_$T" > "_321_gpu_$T.txt" 2>&1
  echo "321 $T EXIT=$?"
  python 19_err_transfer.py \
      --pair "${DATA_DIR}/truth_chain/truth_chain_L2_monthly_gpu_$T.csv" \
      --out_suffix "gpu_$T" --nboot 1000 > "_370_gpu_$T.txt" 2>&1
  echo "370 $T EXIT=$?"; sed -n '/error budget/,/^$/p' "_370_gpu_$T.txt" | head -12
done

echo "========== ALLDONE =========="
