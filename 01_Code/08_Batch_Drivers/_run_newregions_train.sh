# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# New-region training + truth chain + region analysis (v20 expansion)
# Usage: the REGIONS environment variable overrides the default, which runs only R4_longzhong
#   12 configurations per region: FREE 4 (LSTM x3 seeds + LSTMNS) / SOFT 4 (LPMIX w=1,3,10 + LPMIXNS)
#                   HARD 3 (HYBNS x3 seeds) / HYBL 1 (HYBLSTNS)
#   Value-by-value identical to _run_expand_train.sh / _run_lpmixns.sh / _run_hybl_multi.sh.
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
REGS="${REGIONS:-R4_longzhong}"

for REG in $REGS; do
  case $REG in
    R4_longzhong)  PX="xr7";;
    R5_sichuan)    PX="xr8";;
    *) echo "!! unknown region $REG"; continue;;
  esac
  NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
  ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
  if [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; then echo "!! $REG data missing, skipping"; continue; fi
  COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
          --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
          --pet_src psp --threads 8"
  COMMON_W1="$COMMON --wbw 1"

  echo "##################### $REG #####################"
  conda activate rocm

  run_train () {   # $1=suffix $2=cfgs $3=COMMON [other args...]
    local SUF="$1" CFG="$2" CM="$3"; shift 3
    local TAG="_staG_${REG}_${SUF}"
    if [ -f "../out/_bench/pred_${CFG%%,*}${TAG}.npz" ]; then
      echo "  skip train ${CFG}${TAG} (exists)"; return 0
    fi
    echo "  --- train ${CFG} ${TAG} seed=${SEED:-42} $* ---"
    python 13_dl_hybrid.py $CM --seed "${SEED:-42}" --tag "$TAG" \
        --cfgs "$CFG" "$@" > "_302_${REG}_${SUF}.txt" 2>&1
    echo "  302 ${SUF} EXIT=$?"
    grep -E "parameter count" "_302_${REG}_${SUF}.txt" | head -2
  }

  SEED=42; run_train a        HYBNS,LSTM "$COMMON" --lamet 5
  SEED=7;  run_train s7       HYBNS,LSTM "$COMMON" --lamet 5
  SEED=21; run_train s21      HYBNS,LSTM "$COMMON" --lamet 5
  SEED=42; run_train lstmns   LSTMNS     "$COMMON" --lamet 0
  SEED=42; run_train w1       LPMIX      "$COMMON" --wbw 1
  SEED=42; run_train w3       LPMIX      "$COMMON" --wbw 3
  SEED=42; run_train w10      LPMIX      "$COMMON" --wbw 10
  SEED=42; run_train lpmixns  LPMIXNS    "$COMMON_W1"
  SEED=42; run_train hbl_s42  HYBLSTNS   "$COMMON"

  conda activate geoenv
  run_pair () {   # $1=suffix $2=cfgs
    echo "  ---------- 321 ${PX}_$1 ($2) ----------"
    python 16_model_vs_truth.py --npz "$NPZ" --tag "_staG_${REG}_$1" \
        --region $REG --cfgs "$2" --out_suffix "${PX}_$1" \
        > "_321_${PX}_$1.txt" 2>&1
    if [ $? -ne 0 ]; then echo "  321 $1 FAIL"; tail -5 "_321_${PX}_$1.txt"; return; fi
    grep -E "stations in window|matched model grid-cell subset|monthly pairing" "_321_${PX}_$1.txt" | head -3
  }
  run_pair a        HYBNS,LSTM
  run_pair s7       HYBNS,LSTM
  run_pair s21      HYBNS,LSTM
  run_pair lstmns   LSTMNS
  run_pair w1       LPMIX
  run_pair w3       LPMIX
  run_pair w10      LPMIX
  run_pair lpmixns  LPMIXNS
  run_pair hbl_s42  HYBLSTNS

  echo "  ---------- 395 ${REG} ----------"
  python 46_region_analysis.py --region $REG --pattern "${PX}_*" --nboot 2000 \
      > "_395_${REG}.txt" 2>&1
  echo "  395 $REG EXIT=$?"
  sed -n '/Family summary/,/^$/p' "_395_${REG}.txt" | head -16
done
echo "========== NEWREGION TRAIN ALLDONE =========="
