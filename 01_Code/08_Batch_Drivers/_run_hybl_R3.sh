# Upstream chain: set the following variables as described in Data_Acquisition_and_Reproduction_Guide.md
PKG_CODE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RAW_DIR:=${RAW_ROOT:?please set RAW_ROOT}}"
: "${WORK_DIR:=${PKG_CODE}/06_Reproduction_Outputs/_Upstream_Intermediate_Products}"
: "${BENCH_DIR:=${WORK_DIR}/_bench}"
: "${DATA_DIR:=${PKG_CODE}/02_Results_Data}"

#!/bin/bash
# Reinforcement: intermediate HYBLST (hard update + LSTM backbone) -- R3 primary region
#
# Purpose: decompose the overall gap of the original "HYBNS (hard) vs LSTM (soft)" into
#     LSTMNS  -> HYBLSTNS  : pure form effect (same LSTM backbone, only the update rule changes)
#     HYBLSTNS -> HYBNS    : pure backbone/design effect (both hard update, only backbone and residual implementation change)
# plus two residual-difference probes:
#     --cell gru  : backbone replaced by GRUCell (quantify how much the "memory mechanism" itself contributes)
#     --hxd free  : percolation parameterisation replaced by the HYBNS-style softplus (quantify how much flux parameterisation contributes)
# Hyperparameters, period and data split are identical to _run_sta_gpu.sh.
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh

NPZ="${BENCH_DIR}/dl_R0_northchina_nc256_20020101_20131231_sta.npz"
ET="${BENCH_DIR}/et_R0_northchina_2002_2013_sta.npz"
TT="--ttrain 2002-01-01,2009-12-31 --tval 2010-01-01,2010-12-31 --ttest 2011-01-01,2013-12-31"
COMMON="--npz $NPZ --etnpz $ET $TT --K 30 --stride 30 --hidden 128 --epochs 40 \
        --batch 256 --lr 2e-3 --sched cosine --device cuda --savepred \
        --pet_src psp --threads 8"

conda activate rocm
run_train () {   # $1=tag $2=cfg $3=seed  $4..=extra arguments
  local TAG="$1" CFG="$2" SEED="$3"; shift 3
  if [ -f "../out/_bench/pred_${CFG%%,*}${TAG}.npz" ]; then
    echo "  skip train ${CFG} ${TAG} (exists)"; return 0
  fi
  echo "  --- train ${CFG} ${TAG} seed=$SEED $* ---"
  python 13_dl_hybrid.py $COMMON --seed "$SEED" --tag "$TAG" \
      --cfgs "$CFG" "$@" > "_302${TAG}.txt" 2>&1
  echo "  302${TAG} EXIT=$?"
  tail -3 "_302${TAG}.txt" | head -2
}

echo "########## R3: HYBLSTNS + HYBLST (3 seeds) ##########"
run_train _staG_hbl_s42  HYBLSTNS,HYBLST 42
run_train _staG_hbl_s7   HYBLSTNS,HYBLST 7
run_train _staG_hbl_s21  HYBLSTNS,HYBLST 21
echo "########## R3: residual-difference probes ##########"
run_train _staG_hblg_s42 HYBLSTNS      42 --cell gru
run_train _staG_hblf_s42 HYBLSTNS      42 --hxd free

conda activate geoenv
pair () {   # $1=tag $2=cfgs $3=suffix
  echo "  ---------- 321 $3 ($2) ----------"
  python 16_model_vs_truth.py --npz "$NPZ" --tag "$1" --region R0_northchina \
      --cfgs "$2" --out_suffix "$3" > "_321_$3.txt" 2>&1
  if [ $? -ne 0 ]; then echo "  321 $3 FAIL"; tail -5 "_321_$3.txt"; return 1; fi
  grep -E "stations in window|matched model grid-cell subset|monthly pairing" "_321_$3.txt" | head -3
}
pair _staG_hbl_s42  HYBLSTNS,HYBLST gpu_hbl_s42
pair _staG_hbl_s7   HYBLSTNS,HYBLST gpu_hbl_s7
pair _staG_hbl_s21  HYBLSTNS,HYBLST gpu_hbl_s21
pair _staG_hblg_s42 HYBLSTNS        gpu_hblg_s42
pair _staG_hblf_s42 HYBLSTNS        gpu_hblf_s42

echo "########## 395 summary (incl. existing gpu* configurations) ##########"
python 46_region_analysis.py --region R0_northchina --pattern 'gpu*' --nboot 2000 \
    > _395_R0_northchina.txt 2>&1
echo "395 EXIT=$?"
sed -n '/Family summary/,/^$/p;/Station-clustered bootstrap/,/^$/p' _395_R0_northchina.txt | head -40

echo "========== R3 HYBL DONE =========="
