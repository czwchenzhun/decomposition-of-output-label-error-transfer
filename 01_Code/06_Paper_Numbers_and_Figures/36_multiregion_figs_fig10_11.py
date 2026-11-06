# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""36 Cross-region replication figures

  fig_r1_families   four regions x four families of E2/E1 (with station-clustered bootstrap CI)
  fig_r2_rho        cross-region comparison of rho: whether "hard skeleton destroys alignment" holds consistently for FREE vs HARD
  fig_r3_calib      cross-region check of the calibration condition: theoretical lambda* vs empirical optimal lambda
  fig_r4_price      price of truth across regions: benefit after calibration on k=3 stations vs mismatch
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Fig. 10 (cross-region family outcome fig12a and alignment fig12b), Fig. 11 (cross-region price of truth fig14_price_cross.png); also produces fig13_calibres.png (not cited in the main text)
#   Data source: cross-region summary CSV files under 02_Results_Data/truth_chain/multi/. To regenerate: cd latex_paper && python _regen_figs.py
#

import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---- EN-edition layout adaptation ---------------------------------------
# English panel titles are longer than the Chinese originals. Wrap long titles
# onto two lines at a word boundary (never inside maths) and cap the size so
# neighbouring panels do not collide.
import matplotlib.axes as _en_axes
_EN_ORIG_SET_TITLE = _en_axes.Axes.set_title


def _en_set_title(self, label, *args, **kwargs):
    if isinstance(label, str) and '\n' not in label and len(label) > 30:
        _words = label.split(' ')
        _best, _gap = None, 10 ** 9
        for _i in range(1, len(_words)):
            _left, _right = ' '.join(_words[:_i]), ' '.join(_words[_i:])
            if _left.count('$') % 2 or _right.count('$') % 2:
                continue
            _d = abs(len(_left) - len(_right))
            if _d < _gap:
                _gap, _best = _d, (_left, _right)
        if _best:
            label = _best[0] + '\n' + _best[1]
    _fd = kwargs.get('fontdict')
    _fs = kwargs.get('fontsize', (_fd or {}).get('fontsize', 12.5))
    try:
        _fs = float(_fs)
    except (TypeError, ValueError):
        _fs = 12.5
    kwargs['fontsize'] = min(_fs, 12.5)
    if isinstance(_fd, dict) and 'fontsize' in _fd:
        kwargs['fontdict'] = {_k: _v for _k, _v in _fd.items() if _k != 'fontsize'}
    return _EN_ORIG_SET_TITLE(self, label, *args, **kwargs)


_en_axes.Axes.set_title = _en_set_title
# -------------------------------------------------------------------------


OD = MULTI
FD = REPRO + '/process figure/25_multiregion'
os.makedirs(FD, exist_ok=True)
LF = MANU_FIG
os.makedirs(LF, exist_ok=True)

REGCN = {'R0_northchina': 'R0 North China\n85 stations',
         'R1_yangtze': 'R1 middle--lower Yangtze\n21 stations',
         'R2_loess': 'R2 Loess Plateau\n19 stations',
         'R3_northeast': 'R3 Northeast China black soil\n14 stations',
         'R4_longzhong': 'R4 Longzhong\n32 stations',
         'R5_sichuan': 'R5 Sichuan Basin\n23 stations'}
ORDER = ['R0_northchina', 'R1_yangtze', 'R2_loess', 'R3_northeast',
         'R4_longzhong', 'R5_sichuan']
# With six regions the x-axis tick labels overlap, so use single-line short names (region ID + two-character abbreviation + station count)
REGCN_SHORT = {'R0_northchina': 'R0 N. China (85)', 'R1_yangtze': 'R1 Yangtze (21)',
               'R2_loess': 'R2 Loess (19)', 'R3_northeast': 'R3 NE China (14)',
               'R4_longzhong': 'R4 Longzhong (32)', 'R5_sichuan': 'R5 Sichuan (23)'}


def xtname(reg):
    """Use short names when there are more than four regions, otherwise keep the two-line long names"""
    return REGCN_SHORT[reg]
FAMS = ['FREE', 'SOFT', 'HARD', 'DAMAGED']
COL = {'FREE': '#0072B2', 'SOFT': '#009E73', 'HYBL': '#CC79A7',
       'HARD': '#D55E00', 'DAMAGED': '#000000'}
MRK = {'10cm': 'o', '30cm': 's'}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5


def load(name):
    d = {}
    for f in sorted(glob.glob(os.path.join(OD, '%s_*.csv' % name))):
        reg = os.path.basename(f).replace(name + '_', '').replace('.csv', '')
        d[reg] = pd.read_csv(f, encoding='utf-8-sig')
    return d


def main():
    W = load('withinstation')
    B = load('bootstrap')
    Q = load('quadratic')
    P = load('price')
    regs = [r for r in ORDER if r in W]
    print('region:', regs)

    # ---------- fig 1: four regions x families of E2/E1 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for li, lay in enumerate(['10cm', '30cm']):
        a = ax[li]
        xs, pos = [], 0
        xt, xp = [], []
        for reg in regs:
            b = B[reg]
            s = b[(b.layer == lay) & (b.fam.isin(FAMS))]
            if not len(s):
                continue
            for f in FAMS:
                g = s[s.fam == f]
                if not len(g):
                    continue
                r = g.iloc[0]
                a.errorbar(pos, r.ratio,
                           yerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]],
                           fmt=MRK[lay], color=COL[f], ms=7, capsize=3, lw=2.1)
                xs.append(pos)
                pos += 1
            xt.append(reg)
            xp.append(pos - 0.5)
            pos += 1.2
        a.axhline(1.0, color='#333333', lw=1.5, ls='--')
        # a.text(a.get_xlim()[1], 1.195, 'Level with the training label', ha='right',
        #        fontsize=10.8, color='#555555')
        a.set_ylim(0.885, 1.21)
        a.set_xticks(xp)
        a.set_xticklabels([xtname(r) if len(xt) > 4 else REGCN[r] for r in xt],
                          fontsize=10.8,
                          rotation=20 if len(xt) > 4 else 0,
                          ha='right' if len(xt) > 4 else 'center')
        a.set_xlim(-0.6, pos)
        # Panel titles omit the (a)(b) prefix: the outer LaTeX subcaption already numbers them (a)(b),
        # double numbering would confuse readers about whether "Fig. 11(a)" means the whole figure or the left panel
        a.set_title(lay.replace('cm', ' cm'), fontsize=15.9)
        if li == 0:
            # A single-line long label is clipped by the top of the canvas, so split it into two lines;
            # the full explanation ("output vs training label") is moved into the main-text Fig. 10 caption
            a.set_ylabel('$E_2/E_1$' + chr(10) + '($<1$ = better)')
        a.grid(axis='y', alpha=0.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[f], marker='o', ls='', ms=7,
                label={'FREE': 'unconstrained LSTM', 'SOFT': 'soft penalty LPMIX',
                       'HYBL': 'intermediate HYBLST', 'HARD': 'hard skeleton HYBNS',
                       'DAMAGED': 'training-damaged LSTM'}[f])
         for f in FAMS]
    # Legend placed horizontally at the bottom of the figure (upper left inside the panel competes for space above the R4 black points,
    # and panel (b) has only squares, so mixing in a 10 cm/30 cm marker legend would be misleading)
    # Legend placed at the top, outside the axes
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    fig.legend(handles=h, fontsize=12.0, ncol=5, loc='lower center',
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_r1_families.png'), dpi=300,
                    bbox_inches='tight')
    plt.close(fig)

    # ---------- fig 2: family ranking of rho -- SOFT is best in 6 of 8 combinations ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for li, lay in enumerate(['10cm', '30cm']):
        a = ax[li]
        for f in ['FREE', 'SOFT', 'HYBL', 'HARD']:
            xs, ys, lo, hi = [], [], [], []
            for i, reg in enumerate(regs):
                b = B[reg]
                g = b[(b.layer == lay) & (b.fam == f)]
                if not len(g):
                    continue
                r = g.iloc[0]
                xs.append(i)
                ys.append(r.rho)
                lo.append(r.rho - r.rho_lo)
                hi.append(r.rho_hi - r.rho)
            if not xs:
                continue
            off = {'FREE': -0.26, 'SOFT': -0.09, 'HYBL': 0.09, 'HARD': 0.26}[f]
            a.errorbar(np.array(xs) + off, ys, yerr=[lo, hi],
                       fmt=MRK[lay], color=COL[f], ms=7, capsize=3, lw=2.1,
                       alpha=0.9, label=f if li == 0 else None)
        a.set_xticks(range(len(regs)))
        a.set_xticklabels(
            [xtname(r) if len(regs) > 4 else REGCN[r] for r in regs],
            fontsize=10.8,
            rotation=20 if len(regs) > 4 else 0,
            ha='right' if len(regs) > 4 else 'center')
        # Panel titles omit the (a)(b) prefix (to avoid conflict with the outer subcaption numbering)
        a.set_title(lay.replace('cm', ' cm'), fontsize=15.9)
        a.grid(axis='y', alpha=0.25)
        if li == 0:
            a.set_ylabel(r'$\rho=\mathrm{Cov}(e_1,e_3)/(E_1E_3)$' + '\n(more negative = better)',
                         fontsize=13.5)
            # upper limit set to 0.16: the top of the error bar for R3 Northeast China 10 cm HARD is about +0.095
            a.set_ylim(-0.44, 0.16)
            hr, lr = a.get_legend_handles_labels()
        else:
            # sharey hides the right-panel ticks; readers must be able to read values independently
            a.tick_params(axis='y', labelleft=True)
            a.set_ylabel(r'$\rho$ (more negative = better)', fontsize=13.5)
    # Legend placed at the top, outside the axes (the in-figure suptitle duplicated the LaTeX subcaption and was removed)
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    fig.legend(hr, lr, fontsize=12.0, ncol=4, loc='lower center',
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_r2_rho.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------- fig 3: cross-region numerical residuals of the calibration condition (per configuration--layer) ----------
    # The y-axis shows two residuals: P2 |Delta lambda| and P3 |(E2/E1)_opt - lower bound|.
    # Configurations with rho>=0 have lambda*=-rho/x <=0 clipped by the feasible lower bound lambda=0 and are marked separately.
    groups = []
    RSHORT = {'R0_northchina': 'N. China', 'R1_yangtze': 'Yangtze',
              'R2_loess': 'Loess', 'R3_northeast': 'NE China',
              'R4_longzhong': 'Longzhong', 'R5_sichuan': 'Sichuan'}
    for reg in regs:
        D = Q[reg]
        for lay in ['10cm', '30cm']:
            s = D[D.layer == lay] if 'layer' in D.columns else D
            if not len(s):
                continue
            groups.append(('%s\n%s' % (RSHORT[reg], lay), s))

    PANELS = [
        # A single-line long label is clipped by the top of the canvas, so split it into two lines
        ('lam', r'$|\Delta\lambda|$'
         + chr(10) + r'(analytic $-$ grid)',
         '(a) P2: analytic vs grid-search $\\lambda^{\\ast}$'),
        ('opt', r'$\left|(E_2/E_1)_{\mathrm{opt}}-\sqrt{1-\rho^{2}}\right|$',
         '(b) P3: optimum vs theoretical lower bound'),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2))
    rng = np.random.default_rng(7)
    bnd_lab_done = False
    for pi, (key, ylab, ttl) in enumerate(PANELS):
        a = ax[pi]
        for gi, (lab, s) in enumerate(groups):
            if key == 'lam':
                y = (s.lam_th - s.lam_em).abs().values.astype(float)
            else:
                y = (s.E2E1_opt - s.floor).abs().values.astype(float)
            bnd = (s.rho.values >= 0)
            jit = rng.normal(0, 0.16, len(y))
            m = np.isfinite(y)
            # non-boundary configurations
            if (~bnd & m).any():
                a.scatter(gi + jit[~bnd & m], y[~bnd & m], s=17, c='#0072B2',
                          alpha=0.75, zorder=3)
            # boundary configurations (λ* clipped by λ=0)
            if (bnd & m).any():
                a.scatter(gi + jit[bnd & m], y[bnd & m], s=52, c='#D55E00',
                          marker='^', edgecolor='black', lw=0.9, zorder=5,
                          label='boundary configs ($\\rho\\geq0$)'
                                if (pi == 0 and not bnd_lab_done) else None)
                if pi == 0:
                    bnd_lab_done = True
        for ref in (1e-8, 1e-4):
            a.axhline(ref, color='#999999', lw=1, ls=':')
            a.text(-0.55, ref * 1.6, '$10^{%d}$' % int(np.log10(ref)),
                   fontsize=11.1, color='#777777')
        a.set_yscale('log')
        a.set_xticks(range(len(groups)))
        # Six regions x two layers give 12 groups; a horizontal layout would crowd them, so rotate all by 45 degrees
        a.set_xticklabels([g[0].replace('\n', ' ') for g in groups],
                          fontsize=11.6, rotation=45, ha='right')
        a.set_ylabel(ylab, fontsize=13.8)
        a.set_title(ttl, fontsize=15.2)
        a.grid(axis='y', alpha=0.22)
        if pi == 0:
            pass
    # Legend decoupled from plotting: non-boundary configurations (blue points) dominate, but the original figure gave only a boundary entry
    # so readers had to infer it from the caption; here one entry is added and the order fixed
    from matplotlib.lines import Line2D as _L2b
    lb = ['remaining configs ($\\rho<0$)',
          'boundary configs ($\\rho\\geq0$)']
    hb = [_L2b([], [], marker='o', ls='', ms=6, c='#0072B2',
                label=lb[0]),
          _L2b([], [], marker='^', ls='', ms=8, c='#D55E00',
                markeredgecolor='black', markeredgewidth=0.9,
                label=lb[1])]
    # Legend placed at the top, outside the axes
    fig.tight_layout(rect=[0, 0.26, 1, 1.0])
    fig.legend(hb, lb, fontsize=11.5, ncol=1, loc='lower center',
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_r3_calib.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------- fig 4: price of truth ----------
    # capture fraction = (base - k3) / (base - oracle). When a family is already close to its own theoretical lower bound,
    # (base - oracle) -> 0 and the ratio loses meaning (for North China 10 cm the FREE/SOFT denominator is only
    # 0.0008/0.0034, so the ratio can reach -500x), which would blow up the y-axis and flatten all other points.
    # Therefore groups with a denominator below GAP_MIN are not plotted and are noted in the figure; the remaining points use y-limits [-180,130].
    GAP_MIN = 0.005
    FAMCN = {'FREE': 'unconstrained LSTM', 'SOFT': 'soft penalty LPMIX',
             'HYBL': 'intermediate HYBLST', 'HARD': 'hard skeleton HYBNS',
             'DAMAGED': 'training-damaged LSTM'}
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    skipped = []
    for li, lay in enumerate(['10cm', '30cm']):
        a = ax[li]
        for reg in regs:
            PP = P[reg]
            s = PP[PP.layer == lay]
            if 'k3' not in s.columns:
                continue
            for f in ['FREE', 'SOFT', 'HYBL', 'HARD']:
                g = s[s.fam == f]
                if not len(g):
                    continue
                gap = float((g.base - g.oracle).mean())
                cap = (g.base - g.k3) / (g.base - g.oracle)
                cap = np.where(np.isfinite(cap), cap, np.nan)
                v = 100 * float(np.nanmean(cap))
                if abs(gap) < GAP_MIN:
                    skipped.append('%s %s %s (denominator %.4f)'
                                   % (REGCN[reg].split('\n')[0], lay, f, gap))
                    continue
                a.scatter((g.base - 1.0).mean(), v, s=54, color=COL[f],
                          alpha=0.8, marker=MRK[lay], label=None)
        a.axhline(0, color='#333333', lw=1.5)
        a.set_xlabel('mismatch ($E_2/E_1-1$)')
        a.set_title('(%s) %s' % ('ab'[li], lay), fontsize=15.2)
        if li == 0:
            a.set_ylabel('Captured benefit / %' + chr(10) + '(3-station calibration)')
        a.set_ylim(-180, 130)
        # The leftmost/rightmost points tend to hug the axis, so leave a 0.010 margin on each side
        x0, x1 = a.get_xlim()
        a.set_xlim(x0 - 0.010, x1 + 0.010)
        a.grid(alpha=0.25)
    if skipped:
        head = ('Not plotted (headroom $<%.3f$, ratio undefined):' % GAP_MIN)
        parts = ['; '.join(skipped[i:i + 3]) for i in range(0, len(skipped), 3)]
        fig.text(0.5, 0.012, head + '\n' + '\n'.join(parts),
                 ha='center', va='bottom', fontsize=10.4, color='#777777')
    # Legend decoupled from plotting: with get_legend_handles_labels(), the first region (North China)
    # FREE/SOFT are skipped because headroom < GAP_MIN, so the legend would omit these two families.
    from matplotlib.lines import Line2D as _L2
    _ORD = ['FREE', 'SOFT', 'HYBL', 'HARD']
    hp = [_L2([], [], color=COL[f], marker=MRK['10cm'], ls='', ms=8,
                label=FAMCN[f]) for f in _ORD]
    lp = [FAMCN[f] for f in _ORD]
    # Legend placed at the top, outside the axes (the in-figure suptitle duplicating the main-text caption was removed)
    fig.tight_layout(rect=[0, 0.30, 1, 1.0])
    fig.legend(hp, lp, fontsize=11.5, ncol=4, loc='lower center',
               frameon=False, bbox_to_anchor=(0.5, 0.11))
    for d in (FD, LF):
        fig.savefig(os.path.join(d, 'fig_r4_price.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Groups not plotted in Fig. 14:', skipped)
    print('Figures ->', FD)


if __name__ == '__main__':
    main()
