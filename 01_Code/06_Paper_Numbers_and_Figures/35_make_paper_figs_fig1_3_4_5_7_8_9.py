# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""
35 Paper figures (new main line: surpassing the training label)

Output directory: 06_Reproduction_Outputs/24_label_bound/
and copied to 00_Manuscript/figures/
"""

#
# ---- Correspondence with the main text (script numbers no longer appear) ----
#   Main text: Fig. 1 (three-layer error transfer and the quadratic-form framework), Fig. 3 (decomposition of label error E1), Fig. 4 (quadratic-form decomposition), Fig. 5 (separation of the roles of form and strength), Fig. 7 (calibration trajectory and optimal calibration), Fig. 8 (price of truth), Fig. 9 (one-sided metric trap)
#   Output (00_Manuscript/figures/, renamed by the _regen_figs.py mapping after regeneration): fig03_framework.png, fig04_e1_decomposition.png, fig05_quadratic.png, fig06_form_vs_strength.png, fig09_calibration.png, fig10_price.png, fig11_trap.png
#   Data source: CSV files under 02_Results_Data/truth_chain/ (multi/ = six-region merged subdirectory). To regenerate: cd latex_paper && python _regen_figs.py
#

import os
import re
import shutil
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

from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14.5
plt.rcParams['mathtext.fontset'] = 'dejavusans'

B = TRUTH
OUT = REPRO + '/process figure/24_label_bound'
DEST = MANU_FIG
os.makedirs(OUT, exist_ok=True)

FREE_RUNCFG = [('gpu', 'LSTM'), ('gpu_LSTMs7', 'LSTM'), ('gpu_LSTMs21', 'LSTM'),
               ('gpu_lstmns', 'LSTMNS'), ('gpu_deg5', 'LSTM'),
               ('gpu_deg10', 'LSTM'), ('gpu_deg32', 'LSTM')]
DAMAGED = [('gpu_degwd', 'LSTM'), ('gpu_wd0.01', 'LSTM'), ('gpu_wd0.03', 'LSTM')]
FAMS = ['FREE', 'SOFT', 'HARD', 'DAMAGED']
FAMCN = {'FREE': 'unconstrained LSTM', 'SOFT': 'soft penalty LPMIX',
         'HARD': 'hard skeleton HYBNS', 'DAMAGED': 'training-damaged LSTM'}
COL = {'FREE': '#0072B2', 'SOFT': '#009E73', 'HARD': '#D55E00', 'DAMAGED': '#000000'}
MK = {'FREE': 'o', 'SOFT': '^', 'HARD': 's', 'DAMAGED': 'v'}


def assign(row):
    key = (row['run'], row['cfg'])
    if key in DAMAGED:
        return 'DAMAGED'
    if key in FREE_RUNCFG:
        return 'FREE'
    if row['cfg'] == 'LPMIX':
        return 'SOFT'
    if row['cfg'] == 'HYBNS':
        return 'HARD'
    return None


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    shutil.copy2(p, os.path.join(DEST, name))
    plt.close(fig)
    print('  ->', name)


# =========================== data ===========================
W = pd.read_csv(os.path.join(B, 'err_transfer_withinstation.csv'), encoding='utf-8-sig')
W['fam'] = W.apply(assign, axis=1)
W['x'] = W.E3w / W.E1w
W['ratio'] = W.E2w / W.E1w
W['floor'] = np.sqrt(1 - W.rho_w ** 2)
W['exc'] = W.x + W.rho_w
W = W.dropna(subset=['fam'])
W['w'] = [float(m.group(1)) if (m := re.match(r'gpu_wbw([0-9.]+)(?:s\d+)?$', r)) else np.nan
          for r in W.run]

PR = pd.read_csv(os.path.join(B, 'shrink_price.csv'), encoding='utf-8-sig')
PR['fam'] = PR.apply(assign, axis=1)
OP = pd.read_csv(os.path.join(B, 'shrink_optimality.csv'), encoding='utf-8-sig')
OP['fam'] = OP.apply(assign, axis=1)
BS = pd.read_csv(os.path.join(B, 'paper_bootstrap.csv'), encoding='utf-8-sig')


# =========================== Fig 1 framework ===========================
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.0))
    a = ax[0]
    a.set_xlim(-0.1, 3.6); a.set_ylim(-0.42, 2.9); a.axis('off')
    P = dict(s=340, zorder=3)
    pts = {'O': (0.15, 0.25), 'R': (2.6, 0.55), 'M': (1.5, 2.0)}
    lab = {'O': 'ground truth\n$o$ (station observation)', 'R': 'training label\n$r$ (gridded product)',
           'M': 'model output\n$m$'}
    cl = {'O': '#333333', 'R': '#0072B2', 'M': '#D55E00'}
    for k, v in pts.items():
        a.scatter(*v, color=cl[k], **P)
        a.text(v[0], v[1] - 0.28, lab[k], ha='center', va='top', fontsize=15.2)

    def edge(p, q, txt, off, col='#444444', ls='-'):
        xp, yp = pts[p]; xq, yq = pts[q]
        a.annotate('', xy=(xq, yq), xytext=(xp, yp),
                   arrowprops=dict(arrowstyle='-', color=col, lw=2.7, ls=ls))
        mx, my = (xp + xq) / 2 + off[0], (yp + yq) / 2 + off[1]
        a.text(mx, my, txt, fontsize=15.9, color=col, ha='center',
               bbox=dict(fc='white', ec='none', pad=1.2))

    edge('O', 'R', '$E_1=\\|r-o\\|$', (0.02, -0.22))
    edge('R', 'M', '$E_3=\\|m-r\\|$', (0.72, 0.10))
    edge('O', 'M', '$E_2=\\|m-o\\|$', (-0.42, 0.10))
    a.text(1.55, 2.62,
           r'$E_2^2=E_1^2+E_3^2+2\,\mathrm{Cov}(e_1,e_3)$',
           ha='center', fontsize=16.7)
    a.text(1.55, 2.34, r'$\rho=\mathrm{Cov}/(E_1E_3)$, $x=E_3/E_1$',
           ha='center', fontsize=16.7, color='#555555')
    a.set_title('(a) Three-layer error transfer', fontsize=17.4, pad=4)

    b = ax[1]
    rho = -0.45
    xstar = -rho
    xx = 0.66             # a deviation magnitude beyond the optimum
    fl = np.sqrt(1 - rho ** 2)
    exc = xx + rho
    b.set_xlim(-0.28, 1.30); b.set_ylim(-0.16, 1.34)
    tri = Polygon([(xstar, 0), (xx, 0), (xx, fl)], closed=True,
                  fc='#eef3fb', ec='#0072B2', lw=2.1)
    b.add_patch(tri)
    b.plot([xstar, xx], [0, 0], color='#D55E00', lw=3.9)
    b.plot([xx, xx], [0, fl], color='#0072B2', lw=3.9)
    b.plot([xstar, xx], [0, fl], color='#333333', lw=3)
    b.text((xstar + xx) / 2, -0.085, '$x+\\rho$ (excess)', ha='center',
           fontsize=15.2, color='#D55E00')
    b.text(xx + 0.045, fl / 2, '$\\sqrt{1-\\rho^2}$\nlower bound', va='center',
           fontsize=14.5, color='#0072B2')
    b.text((xstar + xx) / 2 - 0.05, fl / 2 + 0.03, '$E_2/E_1$', fontsize=15.2,
           rotation=55, ha='right')
    b.plot([xstar], [0], 'o', color='#D55E00', ms=8, zorder=5)
    b.annotate('$x^{\\ast}=-\\rho$\noptimum', xy=(xstar, 0), xytext=(xstar - 0.24, 0.30),
               fontsize=14.5, color='#D55E00', ha='center',
               arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.5))
    b.axhline(1.0, color='#999999', lw=1.4, ls='--')
    b.text(1.28, 0.995, '$E_2/E_1=1$: level with the label', ha='right', va='top',
           fontsize=13, color='#666666')
    b.text(-0.26, 1.30, r'$\left(\frac{E_2}{E_1}\right)^{2}=(1-\rho^{2})+(x+\rho)^{2}$',
           fontsize=18.1, va='top')
    b.text(-0.26, 1.12, 'Two legs: alignment $\\rho$ and deviation $x$',
           fontsize=13.8, color='#555555', va='top')
    b.set_xlabel('$x=E_3/E_1$ (relative deviation from label)', fontsize=15.2)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for sp in ['top', 'right']:
        b.spines[sp].set_visible(False)
    b.set_title('(b) Geometry of the identity ($\\rho=-0.45$)', fontsize=17.4, pad=4)
    fig.tight_layout()
    save(fig, 'fig1_framework.png')


# =========================== Fig 2 E1 decomposition ===========================
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))
    # (a) stacked bars
    a = ax[0]
    lay_lab = ['10cm', '30cm']
    e1f = [W[W.layer == l].E1f.iloc[0] ** 2 for l in lay_lab]
    e1w = [W[W.layer == l].E1w.iloc[0] ** 2 for l in lay_lab]
    inter = [f - w for f, w in zip(e1f, e1w)]
    idx = np.arange(2)
    a.bar(idx, inter, 0.5, color='#bdbdbd', label='Between-station static offset (irreducible)')
    a.bar(idx, e1w, 0.5, bottom=inter, color='#0072B2',
          label='Within-station temporal mismatch (correctable)')
    for i in range(2):
        a.text(i, e1f[i] * 1.02, '%.1f%%' % (100 * inter[i] / e1f[i]),
               ha='center', fontsize=15.9, fontweight='bold', color='#555555')
        a.text(i, inter[i] / 2, '%.1f%%' % (100 * inter[i] / e1f[i]),
               ha='center', color='white', fontsize=14.5)
        a.text(i, inter[i] + e1w[i] / 2, '%.1f%%' % (100 * e1w[i] / e1f[i]),
               ha='center', color='white', fontsize=14.5)
    a.set_xticks(idx); a.set_xticklabels(['10cm', '30cm'])
    a.set_ylabel(r'MSE$(E_1)$ / $(\mathrm{m^3\,m^{-3}})^2$')
    a.set_ylim(0, max(e1f) * 1.5)
    h2a, l2a = a.get_legend_handles_labels()
    a.set_title('(a) Two sources of label error $E_1$', fontsize=15.9)

    # (b) alpha convention dilution
    b = ax[1]
    cfgs = [('gpu', 'LSTM'), ('gpu', 'HYBNS')]
    names = ['LSTM', 'HYBNS']
    af, aw = [], []
    for c in cfgs:
        s = W[(W.run == c[0]) & (W.cfg == c[1]) & (W.layer == '10cm')]
        af.append(float(s.a_f.iloc[0]) * 100)
        aw.append(float(s.a_w.iloc[0]) * 100)
    xb = np.arange(2); w = 0.34
    b.bar(xb - w / 2, af, w, color='#bdbdbd', label='full convention $\\alpha$')
    b.bar(xb + w / 2, aw, w, color='#0072B2', label='within-station anomaly convention $\\alpha$')
    for i in range(2):
        b.text(i - w / 2, af[i] + 0.08, '%.2f%%' % af[i], ha='center', fontsize=13.8)
        b.text(i + w / 2, aw[i] + 0.08, '%.2f%%' % aw[i], ha='center', fontsize=13.8)
        b.annotate('', xy=(i + w / 2, aw[i] + 0.05), xytext=(i - w / 2, af[i] + 0.05),
                   arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.8))
        # label each bar with its actual dilution factor (the two bars differ by nearly 3x, so a single "about 3x" would be misleading)
        b.text(i + 0.42, (af[i] + aw[i]) / 2, '%.1f$\\times$' % (aw[i] / af[i]),
               ha='left', va='center', fontsize=13.8, color='#D55E00')
    b.set_xticks(xb); b.set_xticklabels(names)
    b.set_ylabel('$\\alpha=-\\mathrm{Cov}(e_1,e_3)/E_1^2$ / %')
    b.set_xlim(-0.55, 1.85)
    b.set_ylim(0, max(max(af), max(aw)) * 1.10)
    h2b, l2b = b.get_legend_handles_labels()
    b.set_title('(b) Irreducible denominator dilutes $\\alpha$', fontsize=15.9)

    # (c) E1 vs product claims
    c = ax[2]
    e1 = [W[W.layer == l].E1w.iloc[0] for l in lay_lab]
    e1_full = [W[W.layer == l].E1f.iloc[0] for l in lay_lab]
    e3 = [W[(W.layer == l) & (W.fam == 'FREE')].E3w.mean() for l in lay_lab]
    xb = np.arange(2); w = 0.26
    c.bar(xb - w, e1_full, w, color='#bdbdbd', label='label (full convention)')
    c.bar(xb, e1, w, color='#757575', label='label (within-station)')
    c.bar(xb + w, e3, w, color='#D55E00', label='model vs label')
    c.set_xticks(xb); c.set_xticklabels(['10cm', '30cm'])
    c.set_ylabel(r'RMSE / $\mathrm{m^3\,m^{-3}}$')
    c.set_ylim(0, max(e1_full) * 1.15)
    h2c, l2c = c.get_legend_handles_labels()
    c.set_title('(c) Within-station convention only', fontsize=15.9)
    # Legend placed at the top, outside the axes: the three panel legends are arranged horizontally above each panel
    fig.tight_layout(rect=[0, 0.22, 1, 1.0])
    fig.legend(h2a, l2a, fontsize=11.5, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.17, 0.0))
    fig.legend(h2b, l2b, fontsize=11.5, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.legend(h2c, l2c, fontsize=11.5, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.82, 0.0))
    save(fig, 'fig2_e1_decomposition.png')


# =========================== Fig 3 main figure: quadratic-form decomposition ===========================
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
    for j, lay in enumerate(['10cm', '30cm']):
        a = ax[j]
        s = W[W.layer == lay]
        bs = BS[BS.layer == lay].set_index('fam')
        for f in FAMS:
            v = s[s.fam == f]
            r0 = v.rho_w.mean()
            xs = np.linspace(0.001, max(0.75, v.x.max() * 1.15), 300)
            ys = np.sqrt(1 - r0 ** 2 + (xs + r0) ** 2)
            a.plot(xs, ys, color=COL[f], lw=2.2, alpha=0.55)
            a.scatter(v.x, v.ratio, s=52, color=COL[f], marker=MK[f],
                      zorder=4, label='%s (n=%d)' % (FAMCN[f], len(v)))
            a.axhline(np.sqrt(1 - r0 ** 2), color=COL[f], lw=1.2, ls=':', alpha=0.7)
            a.plot([-r0], [np.sqrt(1 - r0 ** 2)], '*', color=COL[f], ms=15, zorder=5)
            # CI cross
            if f in bs.index:
                b = bs.loc[f]
                a.errorbar(b.x, b.ratio,
                           xerr=[[b.x - b.x_lo], [b.x_hi - b.x]],
                           yerr=[[b.ratio - b.ratio_lo], [b.ratio_hi - b.ratio]],
                           fmt='none', ecolor=COL[f], alpha=0.55, capsize=3, lw=1.7)
        a.axhline(1.0, color='#333333', lw=1.5)
        # a.text(0.552, 1.004, 'Level with the training label', fontsize=13,
        #        color='#333333', ha='left')
        a.set_xlabel('$x=E_3/E_1$ (model deviation)', fontsize=15.2)
        a.set_title('%s' % lay, fontsize=16.7)
        if j == 0:
            a.set_ylabel('$E_2/E_1$ (vs truth)', fontsize=15.2)
            a.plot([], [], '*', color='#555555', ms=15,
                   label=r'analytic optimum $x^{\ast}=-\rho$ (lower bound for this family)')
            h3, l3 = a.get_legend_handles_labels()
    # Legend placed at the top, outside the axes (as in all other figures)
    fig.tight_layout(rect=[0, 0.14, 1, 1.0])
    fig.legend(h3, l3, fontsize=12.0, loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'fig3_quadratic.png')


# =========================== Fig 4 form vs strength ===========================
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 5.2))
    # (a) rho vs log w in SOFT
    a = ax[0]
    for lay, mk, cc in [('10cm', 'o', '#0072B2'), ('30cm', 's', '#CC79A7')]:
        s = W[(W.fam == 'SOFT') & (W.layer == lay)].dropna(subset=['w'])
        a.scatter(np.log10(s.w), s.rho_w, s=46, marker=mk, color=cc, label=lay, alpha=0.85)
        lw = np.log10(s.w.values)
        sl, ic, rv, pv, se = stats.linregress(lw, s.rho_w.values)
        xs = np.linspace(lw.min(), lw.max(), 50)
        a.plot(xs, ic + sl * xs, color=cc, lw=2.4,
               label='%s regression (p=%.2g)' % (lay, pv))
    a.axhline(W[(W.fam == 'HARD') & (W.layer == '10cm')].rho_w.mean(),
              color='#D55E00', ls='--', lw=1.8, label='HARD mean (10 cm)')
    a.axhline(W[(W.fam == 'FREE') & (W.layer == '10cm')].rho_w.mean(),
              color='#009E73', ls=':', lw=1.8, label='FREE mean (10 cm)')
    a.set_xlabel('$\\log_{10} w$ (penalty strength)')
    a.set_ylabel('$\\rho$')
    a.set_ylim(-0.30, -0.12)
    h4a, l4a = a.get_legend_handles_labels()
    a.set_title('(a) Strength barely moves $\\rho$', fontsize=15.9)

    # (b) x vs log w
    b = ax[1]
    for lay, mk, cc in [('10cm', 'o', '#0072B2'), ('30cm', 's', '#CC79A7')]:
        s = W[(W.fam == 'SOFT') & (W.layer == lay)].dropna(subset=['w'])
        b.scatter(np.log10(s.w), s.x, s=46, marker=mk, color=cc, label=lay, alpha=0.85)
        lw = np.log10(s.w.values)
        sl, ic, rv, pv, se = stats.linregress(lw, s.x.values)
        xs = np.linspace(lw.min(), lw.max(), 50)
        b.plot(xs, ic + sl * xs, color=cc, lw=2.4,
               label='%s regression (p=%.2g)' % (lay, pv))
    b.axhline(W[(W.fam == 'HARD') & (W.layer == '10cm')].x.mean(),
              color='#D55E00', ls='--', lw=1.8, label='HARD mean (10 cm)')
    b.set_xlabel('$\\log_{10} w$')
    b.set_ylabel('$x=E_3/E_1$')
    h4b, l4b = b.get_legend_handles_labels()
    b.set_title('(b) Strength raises deviation $x$', fontsize=15.9)

    # (c) between-family rho comparison
    c = ax[2]
    for j, f in enumerate(FAMS):
        for lay, off in [('10cm', -0.16), ('30cm', 0.16)]:
            v = W[(W.fam == f) & (W.layer == lay)].rho_w.values
            rng = np.random.default_rng(0)
            c.scatter(j + off + rng.normal(0, 0.035, len(v)), v, s=34,
                      color=COL[f], alpha=0.75, marker=MK[f],
                      label=FAMCN[f] if lay == '10cm' else None)
            c.plot([j + off - 0.11, j + off + 0.11], [v.mean(), v.mean()],
                   color=COL[f], lw=3.3)
    c.set_xticks(range(len(FAMS)))
    c.set_xticklabels(FAMS, fontsize=11)
    c.axhline(0, color='#999999', lw=1.2)
    c.set_ylim(-0.36, 0.02)
    c.set_xlabel('Left cluster: 10 cm; right: 30 cm', fontsize=12.5)
    c.set_ylabel('$\\rho$')
    c.set_title('(c) Form significantly changes $\\rho$', fontsize=15.9)
    # Keep the FREE vs HARD significance annotation (the caption cites p=4.5e-7)
    c.annotate('', xy=(2.0, -0.10), xytext=(0.16, -0.235),
               arrowprops=dict(arrowstyle='-', lw=1.4, color='#333'))
    c.text(0.02, -0.345, 'FREE vs HARD: $p=4.5\\times10^{-7}$',
           fontsize=12.8, color='#333333')
    # (c) coloured by family (COL) and shaped (MK), with 10/30 cm distinguished by left/right position within each group,
    # so the legend must list all four families; reusing the grey 10 cm/30 cm entries would clash with the figure colour scheme.
    from matplotlib.lines import Line2D as _L2
    h4c = [_L2([], [], color=COL[f], marker=MK[f], ls='', ms=7,
                 label=FAMCN[f]) for f in FAMS]
    l4c = [FAMCN[f] for f in FAMS]
    # Legend placed at the top, outside the axes: panel legends arranged horizontally above each panel
    fig.tight_layout(rect=[0, 0.31, 1, 1.0])
    fig.legend(h4a, l4a, fontsize=10.7, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.17, 0.0))
    fig.legend(h4b, l4b, fontsize=10.7, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.legend(h4c, l4c, fontsize=13, loc='lower center', ncol=1,
               frameon=False, bbox_to_anchor=(0.82, 0.0))
    save(fig, 'fig4_form_vs_strength.png')


# =========================== Fig 5 calibration condition ===========================
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4))
    a = ax[0]
    picks = [('gpu', 'HYBNS', 'HARD', 'HYBNS hard skeleton'),
             ('gpu', 'LSTM', 'FREE', 'LSTM unconstrained'),
             ('gpu_degwd', 'LSTM', 'DAMAGED', 'training-damaged LSTM')]
    for run, cfg, fam, nm in picks:
        s = W[(W.run == run) & (W.cfg == cfg) & (W.layer == '10cm')].iloc[0]
        rho, x = s.rho_w, s.x
        lam = np.linspace(0, 1.9, 400)
        r = np.sqrt(1 + (lam * x) ** 2 + 2 * lam * x * rho)
        a.plot(lam, r, color=COL[fam], lw=3, label=nm)
        a.plot([1], [s.ratio], 'o', color=COL[fam], ms=9)
        lamstar = -rho / x
        a.plot([lamstar], [np.sqrt(1 - rho ** 2)], '*', color=COL[fam], ms=16)
        a.axhline(np.sqrt(1 - rho ** 2), color=COL[fam], lw=1, ls=':', alpha=0.6)
    a.axhline(1.0, color='#333333', lw=1.5)
    a.text(1.88, 1.006, 'Tie', fontsize=13, ha='right')
    a.set_xlabel('Scaling factor $\\lambda$: $m_\\lambda=r+\\lambda(m-r)$')
    a.set_ylabel('$E_2/E_1$')
    h5a, l5a = a.get_legend_handles_labels()
    a.set_title('(a) Analytic calibration trajectory (circles = $\\lambda{=}1$; stars = optimum)', fontsize=15.2)

    b = ax[1]
    for lay in ['10cm', '30cm']:
        s = OP[OP.layer == lay]
        b.scatter(s.lam_th, s.lam_em, s=48, alpha=0.8,
                  label=lay, marker='o' if lay == '10cm' else 's')
    lim = [0, max(OP.lam_th.max(), OP.lam_em.max()) * 1.05]
    b.plot(lim, lim, 'k--', lw=1.5)
    b.set_xlabel('theoretical $\\lambda^{\\ast}=-\\rho/x$')
    b.set_ylabel('empirical, from grid search $\\lambda^{\\ast}$')
    h5b, l5b = b.get_legend_handles_labels()
    b.text(0.02, lim[1] * 0.86, 'max deviation $4.9\\times10^{-4}$ (n=34x2)', fontsize=13.8)
    b.set_title('(b) Analytic solution vs numerical search', fontsize=15.2)
    # Legend placed at the top, outside the axes: panel legends arranged horizontally above each panel
    fig.tight_layout(rect=[0, 0.18, 1, 1.0])
    fig.legend(h5a, l5a, fontsize=12.5, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.27, 0.0))
    fig.legend(h5b, l5b, fontsize=12.5, loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.75, 0.0))
    save(fig, 'fig5_calibration_law.png')


# =========================== Fig 6 price of truth ===========================
def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    ks = [('k1', 1), ('k2', 2), ('k3', 3), ('k5', 5), ('k10', 10), ('k20', 20), ('k42', 42)]
    for j, lay in enumerate(['10cm', '30cm']):
        a = ax[j]
        for f in FAMS:
            s = PR[(PR.fam == f) & (PR.layer == lay)]
            if not len(s):
                continue
            xs = [k for _, k in ks]
            ys = [s[k].mean() for k, _ in ks]
            a.plot(xs, ys, 'o-', color=COL[f], lw=2.7, ms=5, label=FAMCN[f])
            a.axhline(s.oracle.mean(), color=COL[f], lw=1, ls=':', alpha=0.55)
            a.plot([1], [s.base.mean()], marker='D', color=COL[f], ms=6, alpha=0.6)
        a.axhline(1.0, color='#333333', lw=1.5)
        a.set_xscale('log')
        a.set_xticks([1, 2, 3, 5, 10, 20, 42])
        a.set_xticklabels(['1', '2', '3', '5', '10', '20', '42'])
        a.set_xlabel('Calibration stations $k$')
        if j == 0:
            a.set_ylabel('$E_2/E_1$')
        a.set_title(lay, fontsize=16.7)
        if j == 0:
            # a.text(7, 1.085, 'Level with the training label', fontsize=13)
            h6, l6 = a.get_legend_handles_labels()
    # Legend placed at the top, outside the axes (the in-figure suptitle duplicating the main-text caption was removed)
    fig.tight_layout(rect=[0, 0.14, 1, 1.0])
    fig.legend(h6, l6, fontsize=12.0, loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'fig6_price_of_truth.png')


# =========================== Fig 7 alpha trap ===========================
def fig7():
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.2))
    lay = '10cm'
    s = W[(W.fam == 'SOFT') & (W.layer == lay)].dropna(subset=['w'])
    g = s.groupby('w').agg(alpha=('a_w', 'mean'), ratio=('ratio', 'mean'),
                           n=('w', 'size')).reset_index()
    sl1, ic1, r1, p1, _ = stats.linregress(np.log10(g.w), g.alpha * 100)
    sl2, ic2, r2, p2, _ = stats.linregress(np.log10(g.w), g.ratio)
    sl1, ic1, r1, p1, _ = stats.linregress(np.log10(s.w), s.a_w * 100)
    a = ax[0]
    a.scatter(np.log10(s.w), s.a_w * 100, s=44, color='#009E73', alpha=0.6, label='each configuration')
    a.plot(np.log10(g.w), g.alpha * 100, 'o-', color='#006400', lw=3, label='aggregated by $w$')
    xs = np.linspace(np.log10(s.w.min()), np.log10(s.w.max()), 50)
    a.plot(xs, ic1 + sl1 * xs, '--', color='#006400', lw=2.1,
           label='trend %+.2f/decade, p=%.1e' % (sl1, p1))
    a.set_xlabel('$\\log_{10} w$ (constraint strength)')
    a.set_ylabel('$\\alpha=-\\mathrm{Cov}(e_1,e_3)/E_1^2$ / %')
    a.set_ylim(6.4, 10.7)
    h7a, l7a = a.get_legend_handles_labels()
    a.set_title('(a) One-sided $\\alpha$ appears to improve with strength', fontsize=15.2)

    b = ax[1]
    b.scatter(np.log10(s.w), s.ratio, s=44, color='#009E73', alpha=0.6, label='each configuration')
    b.plot(np.log10(g.w), g.ratio, 'o-', color='#006400', lw=3, label='aggregated by $w$')
    sl3, ic3, r3, p3, _ = stats.linregress(np.log10(s.w), s.ratio)
    b.plot(xs, ic3 + sl3 * xs, '--', color='#006400', lw=2.1,
           label='trend %+.4f/decade, p=%.3f' % (sl3, p3))
    b.axhline(1.0, color='#333333', lw=1.4)
    h = W[(W.fam == 'HARD') & (W.layer == lay)].ratio.mean()
    b.axhline(h, color='#D55E00', ls=':', lw=1.8, label='hard-skeleton mean')
    b.set_xlabel('$\\log_{10} w$')
    b.set_ylabel('$E_2/E_1$ (net outcome)')
    b.set_ylim(0.955, 1.024)
    h7b, l7b = b.get_legend_handles_labels()
    b.set_title('(b) Net outcome: slight rise ($+0.0042$/decade)', fontsize=15.2)
    # Legend placed at the top, outside the axes: panel legends arranged horizontally above each panel
    fig.tight_layout(rect=[0, 0.24, 1, 1.0])
    fig.legend(h7a, l7a, fontsize=11.5, loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.27, 0.0))
    fig.legend(h7b, l7b, fontsize=11.5, loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.75, 0.0))
    save(fig, 'fig7_alpha_trap.png')


if __name__ == '__main__':
    print('generating figures ->', OUT)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print('done, copied to', DEST)
