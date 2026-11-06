# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# M2 response (fourth review round): rerun the no-SS variant LPMIXNS of LPMIX
#
# Background: the headline conclusion "the soft-penalty family outperforms the hard-skeleton family" was, in the original text, a SOFT(SS) vs HARD(NS)
# **cross-training-regime** comparison, while the regime effect (Δρ≈+0.093) is no smaller than the effect to be shown (≈0.084).
# LPMIXNS = component-wise identical to LPMIX except that scheduled sampling is removed (free-running rollout training),
# thereby providing a within-regime comparison of SOFT-NS against HARD-NS.
#
# Hyperparameters, period, data split and evaluation convention are identical to _run_sta_gpu.sh / _run_expand_train.sh:
#   hidden=128 epochs=40 sched=cosine batch=256 lr=2e-3 K=30 stride=30
#   train 2002-2009 / val 2010 / test 2011-2013, --wbw 1 (aligned with replication-region LPMIX w=1)
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"

# ============================ primary region R3: 3 seeds ============================
echo "##################### R0_northchina (3 seeds) #####################"
NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8 --wbw 1"

conda activate rocm
for SD in 42 7 21; do
  TAG="_staG_lpmixns_s${SD}"
  if [ -f "../out/_bench/pred_LPMIXNS${TAG}.npz" ]; then
    echo "  skip train $TAG (exists)"; continue
  fi
  echo "  --- train LPMIXNS seed=$SD ---"
  python 13_dl_hybrid.py $COMMON --seed "$SD" --tag "$TAG" --cfgs LPMIXNS \
      > "_302_lpmixns_s${SD}.txt" 2>&1
  echo "  302 s${SD} EXIT=$?"
  grep -E "parameter count|scheduled_sampling" "_302_lpmixns_s${SD}.txt" | head -2
done

conda activate geoenv
for SD in 42 7 21; do
  echo "  ---------- 321 lpmixns_s${SD} ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "_staG_lpmixns_s${SD}" \
      --region R0_northchina --cfgs LPMIXNS --out_suffix "lpmixns_s${SD}" \
      > "_321_lpmixns_s${SD}.txt" 2>&1
  if [ $? -ne 0 ]; then echo "  321 s${SD} FAIL"; tail -5 "_321_lpmixns_s${SD}.txt"; else
    grep -E "stations in window|monthly pairing" "_321_lpmixns_s${SD}.txt" | head -2
  fi
done
echo "  ---------- 395 R3 (full recomputation, incl. lpmixns_*) ----------"
python 46_region_analysis.py --region R0_northchina --pattern 'gpu*' --nboot 2000 \
    > "_395_R0_northchina.txt" 2>&1
echo "  395 R3 EXIT=$?"
sed -n '/Family summary/,/^$/p' "_395_R0_northchina.txt" | head -16

# ====================== three replication regions: 1 seed each (seed 42) ======================
for REG in R1_yangtze R3_northeast R2_loess; do
  case $REG in
    R1_yangtze)   PX="xr2";;
    R3_northeast) PX="xr4";;
    R2_loess)     PX="xr5";;
  esac
  NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
  ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
  if [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; then echo "!! $REG data missing, skipping"; continue; fi
  COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
          --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
          --pet_src psp --threads 8 --wbw 1"

  echo "##################### $REG #####################"
  conda activate rocm
  TAG="_staG_${REG}_lpmixns_s42"
  if [ -f "../out/_bench/pred_LPMIXNS${TAG}.npz" ]; then
    echo "  skip train $TAG (exists)"
  else
    echo "  --- train LPMIXNS seed=42 ---"
    python 13_dl_hybrid.py $COMMON --seed 42 --tag "$TAG" --cfgs LPMIXNS \
        > "_302_${REG}_lpmixns.txt" 2>&1
    echo "  302 $REG EXIT=$?"
    grep -E "parameter count|scheduled_sampling" "_302_${REG}_lpmixns.txt" | head -2
  fi

  conda activate geoenv
  echo "  ---------- 321 ${PX}_lpmixns_s42 ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$TAG" --region $REG \
      --cfgs LPMIXNS --out_suffix "${PX}_lpmixns_s42" \
      > "_321_${PX}_lpmixns.txt" 2>&1
  if [ $? -ne 0 ]; then echo "  321 $REG FAIL"; tail -5 "_321_${PX}_lpmixns.txt"; else
    grep -E "stations in window|monthly pairing" "_321_${PX}_lpmixns.txt" | head -2
  fi
  echo "  ---------- 395 ${REG} ----------"
  python 46_region_analysis.py --region $REG --pattern "${PX}_*" --nboot 2000 \
      > "_395_${REG}.txt" 2>&1
  echo "  395 $REG EXIT=$?"
  sed -n '/Family summary/,/^$/p' "_395_${REG}.txt" | head -14
done

echo "========== LPMIXNS ALL DONE =========="
