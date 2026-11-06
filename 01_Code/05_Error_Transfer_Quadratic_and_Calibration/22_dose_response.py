# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
22 Dose-response main figure: α / ρ / E3 response to soft-constraint strength w + a four-quadrant "form vs strength" decomposition

Reads err_transfer_wbw_full.csv (produced by 21).
Four panels:
  (a,b) α, ρ vs w (with seed error bars), hard skeleton HYBNS shown as a horizontal band
  (c)   E3 vs w -- proving that w is a genuine "constraint strength" axis (fidelity deteriorates monotonically)
  (d)   α = (−ρ) × (E3/E1) decomposition scatter: deviation magnitude × alignment efficiency, contours of α
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TDIR = TRUTH
FDIR = REPRO + '/process_figures/23_err_transfer'

for f in ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

CLR = {'10cm': '#1F77B4', '30cm': '#D62728'}
MARK = {'10cm': 'o', '30cm': 's'}


def main():
    os.makedirs(FDIR, exist_ok=True)
    A = pd.read_csv(os.path.join(TDIR, 'err_transfer_wbw_full.csv'))
    S = A[(~A.w.isna()) & (A.cfg != 'LSTMNS')]        # dose-response series
    H = A[A.w.isna()]                                  # hard skeleton HYBNS
    NS = A[A.cfg == 'LSTMNS']

    W = np.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    xs = np.arange(len(W))
    pos = {w: i for i, w in enumerate(W)}

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.0))

    # ---------- (a) alpha ----------
    a0 = ax[0, 0]
    for lay in ['10cm', '30cm']:
        s = S[S.layer == lay]
        m, e, lo, hi, n = [], [], [], [], []
        for w in W:
            v = s[s.w == w].alpha.values
            if len(v) == 0:
                m.append(np.nan); e.append(0); lo.append(np.nan)
                hi.append(np.nan); n.append(0)
                continue
            m.append(100 * v.mean())
            e.append(100 * v.std(ddof=1) if len(v) > 1 else 0.0)
            lo.append(100 * v.min()); hi.append(100 * v.max()); n.append(len(v))
        m = np.array(m); e = np.array(e)
        a0.errorbar(xs, m, yerr=e, marker=MARK[lay], color=CLR[lay],
                    capsize=4, linewidth=1.8, markersize=7, label=lay)
        a0.scatter(xs, lo, marker='_', s=100, color=CLR[lay], alpha=.7)
        a0.scatter(xs, hi, marker='_', s=100, color=CLR[lay], alpha=.7)
        for i, k in enumerate(n):
            if k == 1:
                a0.plot(xs[i], m[i], marker='s', mfc='none', color=CLR[lay],
                        markersize=9, alpha=.8)
    for lay in ['10cm', '30cm']:
        h = H[H.layer == lay].alpha
        if len(h):
            a0.axhline(100 * h.mean(), color=CLR[lay], linestyle=':', linewidth=1.5,
                       alpha=.75)
            a0.axhspan(100 * h.min(), 100 * h.max(), color=CLR[lay], alpha=.07)
    a0.axhline(0, color='#888', linewidth=.8, linestyle='--')
    a0.set_xticks(xs)
    a0.set_xticklabels(['0', '0.03', '0.1', '0.3', '1', '3', '10', '30'])
    a0.set_xlabel('soft water-balance constraint strength w (w=0 means unconstrained LSTM)')
    a0.set_ylabel('α  label-error elimination rate (%)')
    a0.set_title('(a) bias-correction rate rises with constraint strength (10cm monotonic, 30cm reversed)\n'
                 'dashed line + band = hard skeleton HYBNS mean and seed range; open squares = n=1',
                 fontsize=10.5)
    a0.grid(alpha=.25)
    a0.legend(title='soil layer', fontsize=9)

    # ---------- (b) rho ----------
    a1 = ax[0, 1]
    for lay in ['10cm', '30cm']:
        s = S[S.layer == lay]
        m, e = [], []
        for w in W:
            v = s[s.w == w].rho.values
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        a1.errorbar(xs, np.array(m), yerr=np.array(e), marker=MARK[lay],
                    color=CLR[lay], capsize=4, linewidth=1.8, markersize=7,
                    label=lay)
    for lay in ['10cm', '30cm']:
        h = H[H.layer == lay].rho
        if len(h):
            a1.axhline(h.mean(), color=CLR[lay], linestyle=':', linewidth=1.5, alpha=.75)
    a1.axhline(0, color='#888', linewidth=.8, linestyle='--')
    a1.set_xticks(xs)
    a1.set_xticklabels(['0', '0.03', '0.1', '0.3', '1', '3', '10', '30'])
    a1.set_xlabel('soft water-balance constraint strength w')
    a1.set_ylabel('ρ  correction efficiency (more negative means more bias correction)')
    a1.set_title('(b) alignment efficiency improves with constraint strength (10cm)', fontsize=10.5)
    a1.grid(alpha=.25)
    a1.legend(title='soil layer', fontsize=9)

    # ---------- (c) E3 ----------
    a2 = ax[1, 0]
    for lay in ['10cm', '30cm']:
        s = S[S.layer == lay]
        m, e = [], []
        for w in W:
            v = s[s.w == w].E3.values
            m.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        a2.errorbar(xs, np.array(m), yerr=np.array(e), marker=MARK[lay],
                    color=CLR[lay], capsize=4, linewidth=1.8, markersize=7,
                    label=lay)
    for lay in ['10cm', '30cm']:
        h = H[H.layer == lay].E3
        if len(h):
            a2.axhline(h.mean(), color=CLR[lay], linestyle=':', linewidth=1.5, alpha=.75)
    a2.set_xticks(xs)
    a2.set_xticklabels(['0', '0.03', '0.1', '0.3', '1', '3', '10', '30'])
    a2.set_xlabel('soft water-balance constraint strength w')
    a2.set_ylabel('E3 output vs label (fidelity error)')
    a2.set_title('(c) fidelity deteriorates monotonically with the constraint -> w is a genuine constraint-strength axis', fontsize=10.5)
    a2.grid(alpha=.25)
    a2.legend(title='soil layer', fontsize=9)

    # ---------- (d) decomposition ----------
    a3 = ax[1, 1]
    for lay in ['10cm', '30cm']:
        s = S[S.layer == lay]
        dev = 100 * s.E3 / s.E1          # relative deviation magnitude (%)
        al = -s.rho                      # alignment efficiency
        a3.scatter(dev, al, c=CLR[lay], marker=MARK[lay], s=48,
                   label=lay, alpha=.75, edgecolors='w', linewidths=.6)
        for _, r in s.iterrows():
            if r.w in (0.0, 3.0, 30.0):
                a3.annotate('w=%g' % r.w, (100 * r.E3 / r.E1, -r.rho),
                            fontsize=7.5, color=CLR[lay], alpha=.9,
                            xytext=(3, 3), textcoords='offset points')
        h = H[H.layer == lay]
        if len(h):
            a3.scatter(100 * h.E3 / h.E1, -h.rho, c=CLR[lay], marker='D', s=90,
                       edgecolors='k', linewidths=.9, zorder=5)
            a3.annotate('HARD', (100 * h.E3.mean() / h.E1.mean(), -h.rho.mean()),
                        fontsize=8.5, color=CLR[lay], fontweight='bold',
                        xytext=(4, -10), textcoords='offset points')
    for lev in [0.01, 0.02, 0.03, 0.04, 0.05]:
        xx = np.linspace(1, 30, 200)
        a3.plot(xx, lev * 100 / xx, color='#999', linewidth=.7, linestyle='--')
        a3.annotate('α=%.0f%%' % (100 * lev), (xx[-1], lev * 100 / xx[-1]),
                    fontsize=7, color='#777')
    a3.set_xlabel('E3/E1  relative deviation magnitude (%)')
    a3.set_ylabel('−ρ  alignment efficiency')
    a3.set_title('(d) α = (−ρ) × (E3/E1): the hard skeleton deviates most yet barely aligns', fontsize=10.5)
    a3.set_xlim(8, 32); a3.set_ylim(0, 0.36)
    a3.grid(alpha=.25)
    a3.legend(title='soil layer (◆=hard skeleton)', fontsize=9)

    fig.suptitle('Dose response of soft/hard physical constraints: the "form" of the constraint determines bias-correction ability'
                 ' (R0 North China 85 stations, monthly scale, 2011-2013)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fp = os.path.join(FDIR, 'dose_response_wbw.png')
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    print('saved', fp)


if __name__ == '__main__':
    main()
