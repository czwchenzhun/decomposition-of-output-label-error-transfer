#!/bin/bash
# Night batch 4: automatic summary after six-region training + transfer evaluation finishes (NIGHT_REGIONS_DONE)
source /d/Software/Anaconda/etc/profile.d/conda.sh
conda activate rocm
cd ${PKG_CODE}

W=0
while ! grep -q "NIGHT_REGIONS_DONE" _night_regions.log 2>/dev/null; do
  sleep 300; W=$((W+1))
  if [ $W -gt 120 ]; then echo "!! six-region wait timed out"; break; fi
done
echo "six regions finished, starting summary $(date +%H:%M:%S)"

echo "########## 361 summary ##########"
python 361_collect.py 2>&1 | tail -40

echo "########## 340 transfer matrix ##########"
python 340_transfer_matrix.py --tag _20y --pattern "transfer_row_20y_*.json" \
    --cfgs HYBNS,LSTM,PERS 2>&1 | tail -60
echo "NIGHT_SUMMARY_DONE $(date +%H:%M:%S)"
