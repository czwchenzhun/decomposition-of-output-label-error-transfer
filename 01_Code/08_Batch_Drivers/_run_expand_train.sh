# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Expansion-region training + truth chain: R1 middle--lower Yangtze (21 stations) / R2 Loess Plateau (19 stations)
# Hyperparameters, period and data split strictly match the R3 station-period GPU version; only region and data differ.
#
# 7 training runs per region, producing 10 configurations:
#   HYBNS  λ=5 × seed{42,7,21}   -> HARD   (3)
#   LSTM        × seed{42,7,21}   -> FREE   (3)
#   LSTMNS      × seed 42         -> FREE   (1)
#   LPMIX  w∈{1,3,10} × seed 42   -> SOFT   (3)
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"

for REG in R1_yangtze R2_loess R3_northeast; do
  case $REG in
    R1_yangtze)   PX="xr2";;
    R2_loess)     PX="xr5";;
    R3_northeast) PX="xr4";;
  esac
  NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
  ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
  if [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; then echo "!! $REG data missing, skipping"; continue; fi
  COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
          --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
          --pet_src psp --threads 8"

  echo "##################### $REG #####################"
  conda activate rocm

  run_train () {   # $1=tag suffix $2=cfg $3=extra arguments
    local SUF="$1" CFG="$2"; shift 2
    local TAG="_staG_${REG}_${SUF}"
    if [ -f "../out/_bench/pred_${CFG}${TAG}.npz" ]; then
      echo "  skip train ${CFG}${TAG} (exists)"; return 0
    fi
    echo "  --- train ${CFG} ${TAG} $* ---"
    python 13_dl_hybrid.py $COMMON --seed "${SEED:-42}" --tag "$TAG" \
        --cfgs "$CFG" "$@" > "_302_${REG}_${SUF}.txt" 2>&1
    echo "  302 ${SUF} EXIT=$?"
  }

  SEED=42; run_train a        HYBNS,LSTM --lamet 5
  SEED=7;  run_train s7       HYBNS,LSTM --lamet 5
  SEED=21; run_train s21      HYBNS,LSTM --lamet 5
  SEED=42; run_train lstmns   LSTMNS     --lamet 0
  SEED=42; run_train w1       LPMIX      --wbw 1
  SEED=42; run_train w3       LPMIX      --wbw 3
  SEED=42; run_train w10      LPMIX      --wbw 10

  conda activate geoenv
  run_pair () {   # $1=suffix tail $2=cfgs $3=tag tail
    local SUF="$1" CFG="$2" TG="$3"
    echo "  ---------- 321 ${PX}_${SUF} ($CFG) ----------"
    python 16_model_vs_truth.py --npz "$NPZ" --tag "_staG_${REG}_${TG}" \
        --region $REG --cfgs "$CFG" --out_suffix "${PX}_${SUF}" \
        > "_321_${PX}_${SUF}.txt" 2>&1
    if [ $? -ne 0 ]; then echo "  321 ${SUF} FAIL"; tail -5 "_321_${PX}_${SUF}.txt"; return; fi
    grep -E "stations in window|matched model grid-cell subset|monthly pairing" "_321_${PX}_${SUF}.txt" | head -3
  }
  run_pair a      HYBNS,LSTM a
  run_pair s7     HYBNS,LSTM s7
  run_pair s21    HYBNS,LSTM s21
  run_pair lstmns LSTMNS     lstmns
  run_pair w1     LPMIX      w1
  run_pair w3     LPMIX      w3
  run_pair w10    LPMIX      w10

  echo "  ---------- 395 ${REG} ----------"
  python 46_region_analysis.py --region $REG --pattern "${PX}_*" --nboot 2000 \
      > "_395_${REG}.txt" 2>&1
  echo "  395 $REG EXIT=$?"
  tail -25 "_395_${REG}.txt"
done
echo "========== EXPAND ALLDONE =========="
