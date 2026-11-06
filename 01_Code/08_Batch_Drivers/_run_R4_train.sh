# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# R3 Northeast China black soil (14 stations) parallel training driver -- runs alongside the main expansion script (R5 finalisation)
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

REG=R3_northeast
PX=xr4
NPZ="${BENCH_DIR}/dl_${REG}_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_${REG}_2002_2013_sta.npz"
if [ ! -f "$NPZ" ] || [ ! -f "$ET" ]; then echo "!! $REG data missing"; exit 1; fi
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 4"

run_train () {
  local SUF="$1" CFG="$2"; shift 2
  local TAG="_staG_${REG}_${SUF}"
  if [ -f "../out/_bench/pred_${CFG}${TAG}.npz" ]; then echo "skip ${CFG}${TAG}"; return 0; fi
  echo "--- train ${CFG} ${TAG} ---"
  python 13_dl_hybrid.py $COMMON --seed "${SEED:-42}" --tag "$TAG" \
      --cfgs "$CFG" "$@" > "_302_${REG}_${SUF}.txt" 2>&1
  echo "302 ${SUF} EXIT=$?"
}

conda activate rocm
SEED=42; run_train a        HYBNS,LSTM --lamet 5
SEED=7;  run_train s7       HYBNS,LSTM --lamet 5
SEED=21; run_train s21      HYBNS,LSTM --lamet 5
SEED=42; run_train lstmns   LSTMNS     --lamet 0
SEED=42; run_train w1       LPMIX      --wbw 1
SEED=42; run_train w3       LPMIX      --wbw 3
SEED=42; run_train w10      LPMIX      --wbw 10

conda activate geoenv
run_pair () {
  local SUF="$1" CFG="$2" TG="$3"
  echo "---------- 321 ${PX}_${SUF} ($CFG) ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "_staG_${REG}_${TG}" \
      --region $REG --cfgs "$CFG" --out_suffix "${PX}_${SUF}" \
      > "_321_${PX}_${SUF}.txt" 2>&1
  echo "321 ${SUF} EXIT=$?"
}
run_pair a      HYBNS,LSTM a
run_pair s7     HYBNS,LSTM s7
run_pair s21    HYBNS,LSTM s21
run_pair lstmns LSTMNS     lstmns
run_pair w1     LPMIX      w1
run_pair w3     LPMIX      w3
run_pair w10    LPMIX      w10

python 46_region_analysis.py --region $REG --pattern "${PX}_*" --nboot 2000 \
    > "_395_${REG}.txt" 2>&1
echo "395 $REG EXIT=$?"
tail -30 "_395_${REG}.txt"
echo "========== R4 ALLDONE =========="
