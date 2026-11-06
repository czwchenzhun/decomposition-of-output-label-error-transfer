# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Rerun LPMIXNS for R4 / R5 (seed 42). R3 and R2 are already done and are not repeated here.
# Identical parameters to _run_lpmixns.sh: hidden=128 epochs=40 sched=cosine batch=256
# lr=2e-3 K=30 stride=30, train 2002-2009 / val 2010 / test 2011-2013, --wbw 1
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"

for REG in R3_northeast R2_loess; do
  case $REG in
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

echo "========== R4/R5 LPMIXNS DONE =========="
