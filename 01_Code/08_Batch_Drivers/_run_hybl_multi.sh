# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Reinforcement: intermediate HYBLST -- expansion regions R2 (21 stations) / R4 (14 stations) / R5 (19 stations)
# Same hyperparameters, period, data split and evaluation convention as the primary-region _run_hybl_R3.sh.
# R5 takes about 2.5-3 h per configuration (data-side cost is high for this region), so only the HYBLSTNS regime is run.
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"

conda activate rocm
for REG in R2_loess R1_yangtze R3_northeast; do
  case $REG in
    R1_yangtze)   PX="xr2"; CFGLIST="HYBLSTNS,HYBLST";;
    R3_northeast) PX="xr4"; CFGLIST="HYBLSTNS,HYBLST";;
    R2_loess)     PX="xr5"; CFGLIST="HYBLSTNS";;
  esac
  NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
  ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
  if [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; then echo "!! $REG data missing, skipping"; continue; fi
  COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
          --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
          --pet_src psp --threads 8"

  echo "##################### $REG #####################"
  TAG="_staG_${REG}_hbl_s42"
  if [ -f "../out/_bench/pred_${CFGLIST%%,*}${TAG}.npz" ]; then
    echo "  skip train $TAG (exists)"
  else
    echo "  --- train $CFGLIST $TAG seed=42 ---"
    python 13_dl_hybrid.py $COMMON --seed 42 --tag "$TAG" \
        --cfgs "$CFGLIST" > "_302_${REG}_hbl_s42.txt" 2>&1
    echo "  302 EXIT=$?"; tail -3 "_302_${REG}_hbl_s42.txt" | head -2
  fi

  conda activate geoenv
  echo "  ---------- 321 ${PX}_hbl_s42 ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$TAG" --region $REG \
      --cfgs "$CFGLIST" --out_suffix "${PX}_hbl_s42" \
      > "_321_${PX}_hbl_s42.txt" 2>&1
  if [ $? -ne 0 ]; then echo "  321 FAIL"; tail -5 "_321_${PX}_hbl_s42.txt"; else
    grep -E "stations in window|matched model grid-cell subset|monthly pairing" "_321_${PX}_hbl_s42.txt" | head -3
  fi

  echo "  ---------- 395 ${REG} (incl. existing ${PX}_*) ----------"
  python 46_region_analysis.py --region $REG --pattern "${PX}_*" --nboot 2000 \
      > "_395_${REG}.txt" 2>&1
  echo "  395 EXIT=$?"
  sed -n '/Family summary/,/^$/p' "_395_${REG}.txt" | head -14
  conda activate rocm
done
echo "========== HYBL MULTI DONE =========="
