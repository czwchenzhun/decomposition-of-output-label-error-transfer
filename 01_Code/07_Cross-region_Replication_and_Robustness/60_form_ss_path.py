# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""60: Estimate of the form effect FORM on the SS regime path (completing m10)

The chain decomposition in §confound of the main text uses the NS path
    LPMIXNS --FORM--> HYBLSTNS --BACKBONE--> HYBNS
of which the FORM step has another independent estimate on the SS regime
    LPMIX --FORM--> HYBLST
(both ends trained with scheduled sampling, both with an LSTM backbone, both with parameter count 2.12e5).
Appendix note 2 discloses this but gives no numbers, which is selective reporting; this script places the two side by side,
and gives the magnitude of the difference. BACKBONE can only be defined on the NS side (HYBNS is an NS regime),
so there is no corresponding SS estimate.
"""

#
# ---- Correspondence with the main text (script numbers no longer appear in the main text) ----
#   Main-text counterpart: §4.3 independent estimate of the chain decomposition under the SS training regime (the second path of FORM)
#

import os
import re
import importlib.util
import numpy as np
import pandas as pd

OD = MULTI
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load406():
    fp = os.path.join(HERE, '39_chain_soft.py')
    spec = importlib.util.spec_from_file_location('m406', fp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lab_of_ss(nm, cfg):
    """SS path: keep only the intermediate itself and the soft penalty itself, excluding the hblg(GRU)/hblf(softplus) probes"""
    if cfg == 'HYBLST' and re.search(r'hblg|hblf', nm):
        return None
    return cfg if cfg in ('LPMIX', 'HYBLST') else None


def main():
    P('=== 60: SS-path estimate of the FORM effect (side by side with the NS path) ===')
    m = load406()
    m.lab_of = lab_of_ss            # replace the family/label assignment inside 39
    LEVELS_SS = ['LPMIX', 'HYBLST']

    old = pd.read_csv(os.path.join(OD, 'chain_soft_ALL.csv'))
    rows = []
    for region, pat in m.REGIONS:
        Pairs = m.M395.load_pairs(pat)
        if not len(Pairs):
            P('!! no samples %s' % region)
            continue
        ncfg = {}
        for (nm, cfg, lay) in Pairs:
            if lay == '10cm' and lab_of_ss(nm, cfg) in LEVELS_SS:
                ncfg[lab_of_ss(nm, cfg)] = ncfg.get(lab_of_ss(nm, cfg), 0) + 1
        BS = m.boot(Pairs, LEVELS_SS, nboot=2000)
        for lay in m.LAYERS:
            if lay not in BS:
                continue
            d = BS[lay]
            if 'HYBLST' not in d or 'LPMIX' not in d:
                P('  %s %s: missing SS-path configuration (%s), skipped'
                  % (region, lay, 'no HYBLST' if 'HYBLST' not in d else 'no LPMIX'))
                continue
            nsta = int(d['_nsta'][0])
            for key in ['rho', 'x', 'ratio']:
                pass
            dr = d['HYBLST']['rho'] - d['LPMIX']['rho']
            dx = d['HYBLST']['x'] - d['LPMIX']['x']
            drt = d['HYBLST']['ratio'] - d['LPMIX']['ratio']

            def s_(a):
                return (float(np.mean(a)), float(np.percentile(a, 2.5)),
                        float(np.percentile(a, 97.5)))
            # The same effect on the NS path
            ns_ = old[(old.region == region) & (old.layer == lay) &
                      (old.kind == 'FORM(new chain)')]
            rns = (float(ns_.rho.iloc[0]), float(ns_.rho_lo.iloc[0]),
                   float(ns_.rho_hi.iloc[0])) if len(ns_) else (np.nan,) * 3
            nsr = (float(ns_.ratio.iloc[0]), float(ns_.ratio_lo.iloc[0]),
                   float(ns_.ratio_hi.iloc[0])) if len(ns_) else (np.nan,) * 3
            mr, lr, hr = s_(dr)
            mt, lt, ht = s_(drt)
            rows.append(dict(region=region, layer=lay, nsta=nsta,
                             ncfg_lpmix=ncfg.get('LPMIX', 0),
                             ncfg_hyblst=ncfg.get('HYBLST', 0),
                             rho_ss=mr, rho_ss_lo=lr, rho_ss_hi=hr,
                             rho_ns=rns[0], rho_ns_lo=rns[1], rho_ns_hi=rns[2],
                             ratio_ss=mt, ratio_ss_lo=lt, ratio_ss_hi=ht,
                             ratio_ns=nsr[0], ratio_ns_lo=nsr[1],
                             ratio_ns_hi=nsr[2],
                             d_rho=mr - rns[0], d_ratio=mt - nsr[0]))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OD, 'form_ss_path.csv'), index=False,
             encoding='utf-8-sig')

    P('')
    P('%-16s %-6s %4s | %-26s | %-26s | %s'
      % ('region', 'layer', 'station', 'FORM(SS)  HYBLST-LPMIX', 'FORM(NS) HYBLSTNS-LPMIXNS',
         'difference(SS-NS)'))
    P('-' * 120)
    for _, r in R.iterrows():
        P('%-16s %-6s %4d | rho %+.3f [%+.3f,%+.3f] | rho %+.3f [%+.3f,%+.3f] | %+.3f'
          % (r.region, r.layer, r.nsta,
             r.rho_ss, r.rho_ss_lo, r.rho_ss_hi,
             r.rho_ns, r.rho_ns_lo, r.rho_ns_hi, r.d_rho))
        P('%-16s %-6s %4s | E2E1 %+.4f [%+.4f,%+.4f] | E2E1 %+.4f [%+.4f,%+.4f] | %+.4f'
          % ('', '', '',
             r.ratio_ss, r.ratio_ss_lo, r.ratio_ss_hi,
             r.ratio_ns, r.ratio_ns_lo, r.ratio_ns_hi, r.d_ratio))
    P('')
    P('Summary: mean delta_rho of FORM on the SS path %+.4f (range %+.4f ~ %+.4f),'
      % (R.rho_ss.mean(), R.rho_ss.min(), R.rho_ss.max()))
    P('      mean delta_rho of FORM on the NS path %+.4f (range %+.4f ~ %+.4f);'
      % (R.rho_ns.mean(), R.rho_ns.min(), R.rho_ns.max()))
    P('      mean of the per-combination difference between the two paths %+.4f, maximum absolute value %.4f.'
      % (R.d_rho.mean(), R.d_rho.abs().max()))
    P('      Sign consistency: combinations with negative delta_rho (improvement) on the SS path %d/8, on the NS path %d/8.'
      % (int((R.rho_ss < 0).sum()), int((R.rho_ns < 0).sum())))
    P('')
    P('Note: the two paths are not repeated measurements of the same quantity -- both ends of the SS path are trained with scheduled sampling,')
    P('    both ends of the NS path use free-running-rollout training, so the magnitude of the difference reflects the training-regime effect itself.')

    from common import report
    report('60_form_ss_path', LOG)


if __name__ == '__main__':
    main()
