# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Non-physical degradation control: no physical mechanisms; raise E3 only via under-training / capacity reduction / weight decay,
# and check whether alpha rises with it. If it rises -> "physical constraints" are not the active ingredient; if not -> physics is essential.
set -u
cd "$PKG_CODE"
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --seed 42 --batch 256 --lr 2e-3 \
        --sched cosine --device cuda --savepred --pet_src psp --threads 8 --cfgs LSTM"

run () {   # $1=tag  rest=extra arguments
  TAG="$1"; shift
  SUF="gpu${TAG}"
  echo "---------- $TAG : $* ----------"
  python 13_dl_hybrid.py $COMMON "$@" --tag "$TAG" > "_302${TAG}.txt" 2>&1
  if [ $? -ne 0 ]; then echo "302 FAIL"; tail -3 "_302${TAG}.txt"; return; fi
  grep -E "^LSTM " "_302${TAG}.txt" | head -1
}

run _deg5   --epochs 5  --hidden 128
run _deg10  --epochs 10 --hidden 128
run _deg32  --epochs 40 --hidden 32
run _degwd  --epochs 40 --hidden 128 --wd 0.1

echo "========== DEG DONE =========="

# Analysis (using geoenv)
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate geoenv
for TAG in _deg5 _deg10 _deg32 _degwd; do
  SUF="gpu${TAG}"
  echo "---------- 321+370 $TAG ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$TAG" --region R0_northchina \
      --cfgs LSTM --out_suffix "$SUF" > "_321_$SUF.txt" 2>&1
  if [ $? -ne 0 ]; then echo "321 FAIL"; tail -3 "_321_$SUF.txt"; continue; fi
  python 19_err_transfer.py \
      --pair "${DATA_DIR}/truth_chain/truth_chain_L2_monthly_$SUF.csv" \
      --out_suffix "$SUF" --nboot 1000 > "_370_$SUF.txt" 2>&1
  echo "370 EXIT=$?"
  sed -n '/error budget/,/^$/p' "_370_$SUF.txt" | head -8
done
echo "========== DEG ANALYSIS DONE =========="
