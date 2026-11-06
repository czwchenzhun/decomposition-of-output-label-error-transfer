#!/bin/bash
# Reinforcement finalisation: uniformly rerun 395 (stratified) / 398 (confound decomposition) / 399 (figures) / 396 (cross-region) / 397 (figures) for four regions
set -u
cd "$(dirname "$0")"
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate geoenv

for spec in "R0_northchina|gpu*" "R1_yangtze|xr2_*" "R2_loess|xr5_*" "R3_northeast|xr4_*"; do
  REG="${spec%%|*}"; PT="${spec##*|}"
  echo "===== 395/398 $REG ====="
  python 46_region_analysis.py --region "$REG" --pattern "$PT" --nboot 2000 \
      > "_395_${REG}.txt" 2>&1; echo "  395 EXIT=$?"
  python 37_confound.py --region "$REG" --pattern "$PT" --nboot 2000 \
      > "_398_${REG}.txt" 2>&1; echo "  398 EXIT=$?"
done

echo "===== 399 figures ====="
python 38_confound_figs.py 2>&1 | tail -4
echo "===== 396 cross-region summary ====="
python 47_crossregion.py > _396_crossregion.txt 2>&1; echo "396 EXIT=$?"
echo "===== 397 cross-region figures ====="
python 397_multiregion_figs.py 2>&1 | tail -4
echo "===== cross-region confound concatenation ====="
python - <<'EOF'
import pandas as pd, glob, os
OD='${DATA_DIR}/truth_chain/multi'
fs=sorted(glob.glob(OD+'/confound_*.csv'))
if fs:
    pd.concat([pd.read_csv(f,encoding='utf-8-sig') for f in fs],ignore_index=True)\
     .to_csv(OD+'/confound_ALL.csv',index=False,encoding='utf-8-sig')
    print('confound_ALL.csv:', len(fs), 'regions')
# Quick summary: FORM / BACKBONE / TOTAL for the NS regime (10/30 cm combined)
import numpy as np
d=pd.read_csv(OD+'/confound_ALL.csv',encoding='utf-8-sig')
c=d[(d.kind.isin(['FORM','BACKBONE','TOTAL']))]
for q in ['rho','ratio']:
    s=c[c.label==q]
    g=s.groupby('kind')[[q,q+'_lo',q+'_hi']].mean()
    print('--- mean Δ%s ---'%q); print(g.round(4).to_string())
EOF
echo "========== FINALIZE DONE =========="
