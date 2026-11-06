# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""59: Unify the North China family-level bootstrap confidence intervals to the "family-level aggregation excludes LPMIXNS" convention

Problem (self-review m2): the SOFT row point estimate and win count in main-text Table tab:fam follow "family-level aggregation excludes
LPMIXNS" (n=15), whereas the SOFT row of bootstrap_R0_northchina.csv is ncfg=18
(including the three within-regime controls LPMIXNS). The point estimate and the confidence interval come from different configuration sets,
which is a mixing of conventions.

This script imports the loading and bootstrap functions of 46 and replaces only the family assignment (SOFT recognises only LPMIX),
recomputes under the same random seed and the same number of bootstrap draws, and outputs two sets of numbers for side-by-side comparison.
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: comparison of the two configuration conventions n=15 / n=18 in main-text Tables 5 and 8
#

import os
import re
import importlib.util
import numpy as np
import pandas as pd

OD = MULTI
LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load395():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '46_region_analysis.py')
    spec = importlib.util.spec_from_file_location('m395', fp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fam_of_n15(run, cfg):
    """Family-level aggregation convention: SOFT contains only LPMIX, LPMIXNS is kept as a within-regime control"""
    if cfg == 'HYBNS':
        return 'HARD'
    if cfg in ('HYBLST', 'HYBLSTNS'):
        return 'HYBL'
    if cfg == 'LPMIX':
        return 'SOFT'
    if re.search(r'wd', run):
        return 'DAMAGED'
    if cfg in ('LSTM', 'LSTMNS'):
        return 'FREE'
    return None


def main():
    P('=== 59: North China family-level bootstrap unified to the convention excluding LPMIXNS ===')
    m = load395()

    old = pd.read_csv(os.path.join(OD, 'bootstrap_R0_northchina.csv'),
                      encoding='utf-8-sig')

    m.fam_of = fam_of_n15
    Pairs = m.load_pairs('gpu*')
    P('Loaded %d (run,cfg,layer) combinations' % len(Pairs))
    n15 = m.stage_bootstrap(Pairs, 'R0_northchina', nboot=2000)
    n15.to_csv(os.path.join(OD, 'bootstrap_R0_northchina_n15.csv'),
               index=False, encoding='utf-8-sig')
    P('Wrote bootstrap_R0_northchina_n15.csv')

    P('')
    P('%-8s %-6s | %-34s | %-34s' %
      ('family', 'layer', 'includes LPMIXNS (n=18, current main text)', 'excludes LPMIXNS (n=15, point-estimate convention)'))
    P('-' * 100)
    for lay in ['10cm', '30cm']:
        for fam in ['FREE', 'SOFT', 'HARD', 'DAMAGED']:
            a = old[(old.layer == lay) & (old.fam == fam)]
            b = n15[(n15.layer == lay) & (n15.fam == fam)]
            if not len(a) or not len(b):
                continue
            a, b = a.iloc[0], b.iloc[0]
            P('%-8s %-6s | n=%2d rho %+.3f [%+.3f,%+.3f] | n=%2d rho %+.3f [%+.3f,%+.3f]'
              % (fam, lay, a.ncfg, a.rho, a.rho_lo, a.rho_hi,
                 b.ncfg, b.rho, b.rho_lo, b.rho_hi))
            P('%-8s %-6s |      E2E1 %.4f [%.4f,%.4f] |      E2E1 %.4f [%.4f,%.4f]'
              % ('', '', a.ratio, a.ratio_lo, a.ratio_hi,
                 b.ratio, b.ratio_lo, b.ratio_hi))
    P('')
    P('Note: the FREE/HARD/DAMAGED families contain no NS control configurations, so the numbers under the two conventions should be exactly identical,')
    P('    if so, the recomputation affects only the SOFT row (a correctness check for this recomputation).')

    from common import report
    report('59_bootstrap_n15', LOG)


if __name__ == '__main__':
    main()
