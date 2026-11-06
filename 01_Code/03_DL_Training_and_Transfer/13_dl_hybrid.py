# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""13: Hybrid  --  physical skeleton + residual flux correction (evolution of the 12 smoke test)

12 conclusion: PHY best at a single step (h=1 0.00761) but severe multi-step drift (h=30 0.04205), 
LSTM stable at multiple steps(0.01920)but degenerates to"persistence"strategy and is not interpretable.
13 addresses"multi-step drift"introduces two improvements and separates the ablations:

Configurations (ablation chain):
  PHY   : pure physics, free-running rollout, no SS, no residual           --  original configuration of 12
  PHYSS : pure physics + scheduled sampling (feeding observed states with a probability during training)
  HYB   : physical skeleton + residual flux U1/U2 + SS
  HYBNS : physical skeleton + residual flux U1/U2, **no SS** (free-running rollout, same regime as PHY)  --  main-method candidate
  LSTM  : purely autoregressive LSTM + SS (same conditions as HYB, fair control)
  PERS  : persistence baseline (analytic, not trained)

Reading the ablations:
  PHY  -> PHYSS  = scheduled sampling net effect
  PHYSS-> HYB    = net effect of the residual (SS regime)
  PHY  -> HYBNS  = net effect of the residual (under the free-running regime, the cleanest chain)
  HYBNS vs LSTM  = physical skeleton vs black box

Why the residual is designed as"unknown flux"rather than"state correction": 
  Keeps the water-balance structure intact  --  U enters the equations as an extra source/sink term,
  Therefore ET/Q/D12/D2 remain diagnosable and can be compared with flux products.
  At the same time, lres penalty limits U magnitude, forcing the physical terms to carry the main explanation,
  Avoid letting the residual swallow the physics entirely (that would fall back into equifinality).

State (two layers):
  layer1 0-10cm  Zr1 = 100 mm     layer2 10-30cm  Zr2 = 01 mm
  S1 = th1*Zr1, S2 = th2*Zr2  (mm)
  S1' = S1 + P + Irr - Q - ET1 - D12 + U1
  S2' = S2 + D12 - ET2 - D2 + U2

Usage (torch environment,CPU): 
  conda activate torch
  python 13_dl_hybrid.py --npz <Training_Inputs npz> --epochs 40
"""

#
# ---- Correspondence with the main text (script numbers no longer appear there) ----
#   Main-text correspondence: main-text Table 3 (capacity and training fingerprints of each family); the 2 lower-panel structural schematic also follows the four classes in this script LSTM / LSTMPHY / LSTMHYB / PHY four classes in this script
#   Note: this script is the training entry point, called by 360_dl_st.py and others, so it is not renamed
#

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

ROOT = PKG
LOG = []
HORIZONS = [1, 3, 7, 15, 30]

# Global device: set by --device; GPU available -> default cuda
DEV = torch.device('cpu')
ZR1, ZR2 = 100.0, 200.0   # two-layer effective water depth mm


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_inputs(X, feats):
    """Assemble model inputs (T,N,Fm); return tensors and feature names"""
    F = list(feats)
    i = {k: F.index(k) for k in F}
    # Porosity converted from bulk density:por = 1 - BD/2.65 (must be computed before standardisation, otherwise identically equal to 1)
    por_top = 1.0 - X[:, :, i['BD_0_5cm']] / 2.65
    por_sub = 1.0 - X[:, :, i['BD_15_30cm']] / 2.65
    cols = [
        X[:, :, i['chm_precip_daily']],   # 0  P     (not standardised)
        X[:, :, i['pet_jh']],             # 1  PET   (not standardised)
        por_top,                          # 2  por1  (not standardised)
        por_sub,                          # 3  por2  (not standardised)
        X[:, :, i['cmfd_t2m']],           # 4  T
        X[:, :, i['cmfd_vpd']],           # 5  VPD
        X[:, :, i['cmfd_swrad']],         # 6  Rs
        X[:, :, i['doy_sin']],            # 7
        X[:, :, i['doy_cos']],            # 8
        X[:, :, i['clay_0_5cm']],         # 9
        X[:, :, i['sand_0_5cm']],         # 10
        X[:, :, i['clay_15_30cm']],       # 11
        X[:, :, i['sand_15_30cm']],       # 12
        X[:, :, i['slope']],              # 13
        # --- Antecedent meteorological history: key quantities for infiltration/key physical quantities for infiltration/runoff partitioning ---
        X[:, :, i['P_lag1']],             # 14
        X[:, :, i['P_lag2']],             # 15
        X[:, :, i['P_lag3']],             # 16
        X[:, :, i['P_sum3']],             # 17
        X[:, :, i['P_sum7']],             # 18
        X[:, :, i['P_sum15']],            # 19
        X[:, :, i['pet_m3']],             # 20
        X[:, :, i['pet_m7']],             # 21
        X[:, :, i['pet_m15']],            # 22
    ]
    A = np.stack(cols, axis=2).astype('float32')
    # Precipitation variables are right-skewed, take log1p flatten the long tail (lagged daily values and accumulation-window values)
    for c in (0, 14, 15, 16, 17, 18, 19):
        A[:, :, c] = np.log1p(np.clip(A[:, :, c], 0, None))
    names = ['P_log', 'PET', 'por1', 'por2', 't2m', 'vpd', 'swrad', 'doy_sin', 'doy_cos',
             'clay_top', 'sand_top', 'clay_sub', 'sand_sub', 'slope',
             'Plag1_log', 'Plag2_log', 'Plag3_log', 'Psum3_log', 'Psum7_log', 'Psum15_log',
             'pet_m3', 'pet_m7', 'pet_m15']
    return A, names


class SeqSet:
    """Split (T,N,F) cut into fixed-length sequences (nseq,K,F)"""

    def __init__(self, A, Y, sl, K, stride, nmax=None, E=None):
        T, N, Fm = A.shape
        starts = list(range(sl.start, sl.stop - K, stride))
        if nmax:
            starts = starts[:nmax]
        self.starts = np.asarray(starts, dtype=np.int64)
        self.A = np.empty((len(starts), N, K, Fm), 'float32')
        self.Y = np.empty((len(starts), N, K, 2), 'float32')
        self.S0 = np.empty((len(starts), N, 2), 'float32')
        self.E = np.empty((len(starts), N, K), 'float32') if E is not None else None
        for k, s in enumerate(starts):
            self.A[k] = np.transpose(A[s:s + K], (1, 0, 2))
            self.Y[k] = np.transpose(Y[s + 1:s + K + 1], (1, 0, 2))
            self.S0[k] = Y[s]
            if E is not None:
                # and Y at the same time as Y: the t step fluxes correspond to s+t -> s+t+1
                self.E[k] = np.transpose(E[s + 1:s + K + 1], (1, 0))
        shape = self.A.shape
        self.A = self.A.reshape(-1, K, Fm)
        self.Y = self.Y.reshape(-1, K, 2)
        self.S0 = self.S0.reshape(-1, 2)
        if self.E is not None:
            self.E = self.E.reshape(-1, K)
        self.n = self.A.shape[0]

    def tensors(self):
        E = torch.from_numpy(self.E) if self.E is not None else None
        return (torch.from_numpy(self.A), torch.from_numpy(self.Y),
                torch.from_numpy(self.S0), E)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def mlp(din, dhid, dout, nl=2):
    ls = []
    d = din
    for _ in range(nl):
        ls += [nn.Linear(d, dhid), nn.Tanh()]
        d = dhid
    ls += [nn.Linear(dhid, dout)]
    return nn.Sequential(*ls)


class ResFlux(nn.Module):
    """Residual-flux network:GRU passes the hidden state across steps, outputting two layers of"unmodelled flux" U (mm/day)

    Zero-init output layer → At the start of training, U≡0, the model fully degenerates to the pure physical skeleton,
    gradually introduce corrections during training, ensuring"can correct"but not"rely on the residual from the start". 
    """

    def __init__(self, din, dh=64, umax=2.0):
        super().__init__()
        self.cell = nn.GRUCell(din, dh)
        self.head = nn.Linear(dh, 2)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head.weight)
        self.umax = umax

    def forward(self, inp, h):
        h2 = self.cell(inp, h)
        u = self.umax * torch.tanh(self.head(h2))     # bounded ±umax mm/day
        return u, h2


class PHY(nn.Module):
    """Physically constrained two-layer water balance:NN only learns ETratio / runoff ratio / inter-layer percolation / deep drainage / irrigation
     (optional) residual flux U1/U2 as unmodelled source/sink terms, still within the water-balance framework
    """

    def __init__(self, din, dh=64, use_res=False, umax=2.0, irr_max=2.0, qmax=0.6,
                 rlo=0.2, rhi=1.2, fix_ratio=None, dynw=False):
        super().__init__()
        self.rlo, self.rhi = rlo, rhi
        self.dynw = dynw
        self.fix_ratio = fix_ratio          # non- None freeze the ET/PET ratio when ET/PET ratio, for an identifiability scan
        self.f_et = mlp(din, dh, 1)
        self.f_q = mlp(din, dh, 1)
        self.f_d12 = mlp(din, dh, 1)
        self.f_d2 = mlp(din, dh, 1)
        self.f_irr = mlp(din, dh, 1)                # irrigation (a key extra water source on the North China Plain)
        self.w1 = nn.Parameter(torch.tensor(1.0))   # tendency to allocate ET to the surface layer (dynw=0 when dynw=0)
        if dynw:
            # State-dependent layer-wise root uptake:g1/g2 are given by the network; surface-layer fraction = g1/(g1+g2)
            self.f_w1 = mlp(din, dh, 1)
            self.f_w2 = mlp(din, dh, 1)
        self.use_res = use_res
        self.irr_max = irr_max
        self.qmax = qmax                            # infiltration excess/impervious runoff ratio upper bound
        if use_res:
            # The residual network also sees the fluxes computed by the physical terms, so it can judge"where the physics is unspecified"
            self.res = ResFlux(din + 5, dh, umax)

    def forward(self, A, S0, K, Yobs=None, tf_prob=0.0):
        # A: (B,K,Fm)  S0: (B,2) units m3/m3
        B = A.shape[0]
        th1 = S0[:, 0].clone()
        th2 = S0[:, 1].clone()
        h = None
        out = []
        ets = []                                              # per-step total evapotranspiration mm/day (for comparison with ET product comparison)
        diag = {'ratio': [], 'q': [], 'd12': [], 'd2': [], 'P': [], 'pet': [],
                'irr': [], 'inf': [], 'u1': [], 'u2': [], 'upen': []}
        nb = 0
        for t in range(K):
            a = A[:, t]
            Praw = torch.expm1(a[:, 0])                       # Recover precipitation mm/day
            pet = a[:, 1]
            por1, por2 = a[:, 2], a[:, 3]                     # porosity (original)
            thmin = 0.02
            # Relative saturation as network input: scale 0-1 and has a clear physical meaning
            sat1 = ((th1 - thmin) / (por1 - thmin)).clamp(0, 1)
            sat2 = ((th2 - thmin) / (por2 - thmin)).clamp(0, 1)
            inp = torch.cat([sat1[:, None], sat2[:, None], a], dim=1)

            # Physical prior range:ET/PET falls in [0.2, 1.2]; runoff coefficient upper bound 0.35 (observed on the North China Plain approx. 0.1-0.2)
            if self.fix_ratio is not None:
                ratio = pet.new_full(pet.shape, float(self.fix_ratio))
            else:
                ratio = self.rlo + (self.rhi - self.rlo) * torch.sigmoid(
                    self.f_et(inp).squeeze(-1))
            et = ratio * pet
            if self.dynw:
                g1 = nn.functional.softplus(self.f_w1(inp).squeeze(-1)) + 1e-3
                g2 = nn.functional.softplus(self.f_w2(inp).squeeze(-1)) + 1e-3
                ssum = g1 + g2
                et1, et2 = et * g1 / ssum, et * g2 / ssum
            else:
                w = torch.sigmoid(self.w1)                    # surface-layer fraction (0,1)
                et1, et2 = w * et, (1 - w) * et
            d12 = nn.functional.softplus(self.f_d12(inp).squeeze(-1))
            d2 = nn.functional.softplus(self.f_d2(inp).squeeze(-1))
            # Cap irrigation to prevent it being abused, as in 12 being abused into 837 mm/yr fake water source
            irr = self.irr_max * torch.sigmoid(self.f_irr(inp).squeeze(-1))

            # --- runoff generation changed to infiltration-excess + saturation-excess dual mechanism, replacing the earlier ad hoc 0.35 runoff coefficient upper bound ---
            # The old code treated the basin annual runoff coefficient as a grid-cell daily infiltration upper bound, which is conceptually wrong,
            # and in practice every training run Q always hits about 0.9×upper bound, showing this constraint kept suppressing the model.
            # Infiltration-excess part: direct runoff from infiltration-excess / impervious surfaces, ratio upper bound qmax
            q_horton = Praw * (self.qmax * torch.sigmoid(self.f_q(inp).squeeze(-1)))
            avail = Praw - q_horton
            # Saturation-excess part: water the surface layer can accept = remaining pore space + same-day drainable amount (percolation+evapotranspiration)
            deficit = torch.clamp((por1 - th1) * ZR1 + et1 + d12, min=0.0)
            inf = torch.minimum(avail, deficit)
            q = Praw - inf

            if self.use_res:
                inp_r = torch.cat([inp, et[:, None], q[:, None], d12[:, None],
                                   d2[:, None], irr[:, None]], dim=1)
                u, h = self.res(inp_r, h)
                u1, u2 = u[:, 0], u[:, 1]
            else:
                u1 = torch.zeros_like(et)
                u2 = torch.zeros_like(et)

            s1 = th1 * ZR1 + inf + irr - et1 - d12 + u1
            s2 = th2 * ZR2 + d12 - et2 - d2 + u2
            n1 = s1 / ZR1
            n2 = s2 / ZR2
            nb += int(((n1 <= 0.0201) | (n1 >= por1 - 1e-4) |
                       (n2 <= 0.0201) | (n2 >= por2 - 1e-4)).sum().item())
            # clamp does not support (Tensor, float, Tensor) mixing: use a scalar for the lower bound clamp, upper bound uses minimum
            th1 = torch.minimum(torch.clamp(n1, min=thmin), por1)
            th2 = torch.minimum(torch.clamp(n2, min=thmin), por2)

            # Key: record"true prediction", then do scheduled sampling replacement.
            # If the replacement is appended afterwards, at tf=1 the output is identically equal to the observation → loss=0 → the gradient vanishes.
            out.append(torch.stack([th1, th2], dim=1))
            ets.append(et)
            if Yobs is not None and tf_prob > 0:
                msk = (torch.rand(B, device=th1.device) < tf_prob).float()
                th1 = msk * Yobs[:, t, 0] + (1 - msk) * th1
                th2 = msk * Yobs[:, t, 1] + (1 - msk) * th2
            diag['ratio'].append(ratio.detach())
            diag['P'].append(Praw.detach())
            diag['pet'].append(pet.detach())
            diag['irr'].append(irr.detach())
            diag['q'].append(q.detach())
            diag['inf'].append(inf.detach())
            diag['d12'].append(d12.detach())
            diag['d2'].append(d2.detach())
            diag['u1'].append(u1.detach())
            diag['u2'].append(u2.detach())
            # Convert the penalty to SM same units as (u mm/day ÷ Zr mm = Δθ), 
            # so that lres magnitude comparable to MSE comparable, avoiding the need for a 1e-5 -level fudge factor
            diag['upen'].append(((u1.detach() / ZR1) ** 2 + (u2.detach() / ZR2) ** 2))
        out = torch.stack(out, dim=1)                         # (B,K,2)
        ets = torch.stack(ets, dim=1)                         # (B,K) mm/day
        for k in diag:
            diag[k] = torch.stack(diag[k], dim=1).mean().item()
        diag['bind'] = nb / max(1, K * B)
        diag['et'] = float(ets.mean().item())
        return out, diag, ets


class LSTM(nn.Module):
    """purely autoregressive LSTM control: predict the two layers directly SM

    Use LSTMCell unrolled manually rather than nn.LSTM  --  the two are mathematically equivalent when num_layers=1, 
    but bypasses ROCm/Windows under MIOpen RNN backward kernel compilation failure
     (HIPRTC and MSVC header conflict → miopenStatusUnknownError).
    """

    def __init__(self, din, dh=64, nlayers=1):
        super().__init__()
        self.dh = dh
        self.nlayers = nlayers
        self.cells = nn.ModuleList([
            nn.LSTMCell(din + 2 if i == 0 else dh, dh) for i in range(nlayers)])
        self.head = nn.Linear(dh, 2)
        # Zero-init output layer: initial behaviour degenerates to persistence, avoiding early autoregressive drift
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, A, S0, K, Yobs=None, tf_prob=0.0):
        B = A.shape[0]
        hs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        cs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        th = S0.clone()
        outs = []
        nb = 0
        for t in range(K):
            x = torch.cat([A[:, t], th], dim=1)
            for i, cl in enumerate(self.cells):
                hs[i], cs[i] = cl(x, (hs[i], cs[i]))
                x = hs[i]
            th = th + self.head(x)                            # predicted increment
            nb += int(((th <= 0.0201) | (th >= 0.5999)).sum().item())
            th = torch.clamp(th, 0.02, 0.60)
            outs.append(th)                                   # record the true prediction first
            if Yobs is not None and tf_prob > 0:
                msk = (torch.rand(B, device=th.device) < tf_prob).float()[:, None]
                th = msk * Yobs[:, t] + (1 - msk) * th
        return torch.stack(outs, dim=1), {'bind': nb / max(1, K * B)}, None


class LSTMPHY(nn.Module):
    """LSTM + learnable flux head + **soft**water-balance penalty (weight w continuously tunable)

    Why it is needed:
        LSTM and HYBNS comparison, physical constraints and architecture (LSTMCell vs GRU residual), state form
         (free increment vs hard skeleton), parameter count,E3 are all entangled, so no dose response can be obtained.
        λ (ET auxiliary loss) proves to be too weak a knob (shifting E3 shift 1.7%), 3 seed the effect vanishes.
        This model parameterises constraint strength alone as a continuous scalar w: 
            w = 0     -> and LSTM exactly equivalent to LSTM (the flux head does not take part in the forward pass; and cells/head before the flux head
                         created; under the same seed, the LSTM initialisation is value-by-value identical, which verifies correctness)
            w -> large   -> the increment is forced onto the water-balance skeleton, approaching "no residual, no GRU of HYBNS skeleton"
        Thus the constraint strength can be scanned under one architecture, one capacity and one training recipe, giving a genuine dose response.

    The flux head outputs only bounded ratios; the physical ranges are guaranteed by code:
        ET = PET * ratio,   ratio ∈ [rlo, rhi]
        D12 = dmax1 * sig * sat1,  D2 = dmax2 * sig * sat2
        Q  = infiltration excess(≤qmax*P) + saturation excess(the part exceeding pore space)
    """

    def __init__(self, din, dh=64, nlayers=1, w=0.0, rlo=0.2, rhi=1.2,
                 qmax=0.6, dmax1=8.0, dmax2=3.0):
        super().__init__()
        self.dh = dh
        self.nlayers = nlayers
        self.w = float(w)
        self.rlo, self.rhi = float(rlo), float(rhi)
        self.qmax, self.dmax1, self.dmax2 = float(qmax), float(dmax1), float(dmax2)
        self.cells = nn.ModuleList([
            nn.LSTMCell(din + 2 if i == 0 else dh, dh) for i in range(nlayers)])
        self.head = nn.Linear(dh, 2)
        # Zero-init output layer: initial behaviour degenerates to persistence, avoiding early autoregressive drift
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)
        # --- the flux head must be cells/head created after ---
        # otherwise the RNG consumption order would change, and w=0 would no longer be value-by-value identical to LSTM
        self.f_et = mlp(dh, dh, 1)
        self.f_d12 = mlp(dh, dh, 1)
        self.f_d2 = mlp(dh, dh, 1)
        self.f_q = mlp(dh, dh, 1)
        self.w_et = nn.Parameter(torch.tensor(0.0))     # surface-layer ET fraction = sigmoid(0) = 0.5

    def _wb_res(self, a, th_prev, th, h):
        """Water-balance residual (Δθ/day units, same dimension as the main loss)"""
        thmin = 0.02
        Praw = torch.expm1(a[:, 0]).clamp(min=0.0)      # Recover precipitation mm/day
        pet = a[:, 1]
        por1, por2 = a[:, 2], a[:, 3]
        s1 = ((th_prev[:, 0] - thmin) / (por1 - thmin)).clamp(0, 1)
        s2 = ((th_prev[:, 1] - thmin) / (por2 - thmin)).clamp(0, 1)
        ratio = self.rlo + (self.rhi - self.rlo) * torch.sigmoid(
            self.f_et(h).squeeze(-1))
        et = ratio * pet
        wet = torch.sigmoid(self.w_et)
        et1, et2 = wet * et, (1.0 - wet) * et
        d12 = self.dmax1 * torch.sigmoid(self.f_d12(h).squeeze(-1)) * s1
        d2 = self.dmax2 * torch.sigmoid(self.f_d2(h).squeeze(-1)) * s2
        # infiltration excess + saturation-excess runoff (same mechanism as PHY)
        q_horton = Praw * (self.qmax * torch.sigmoid(self.f_q(h).squeeze(-1)))
        avail = Praw - q_horton
        deficit = torch.clamp((por1 - th_prev[:, 0]) * ZR1 + et1 + d12, min=0.0)
        inf = torch.minimum(avail, deficit)
        q = Praw - inf
        d1_wb = (Praw - q - et1 - d12) / ZR1
        d2_wb = (d12 - et2 - d2) / ZR2
        r1 = (th[:, 0] - th_prev[:, 0]) - d1_wb
        r2 = (th[:, 1] - th_prev[:, 1]) - d2_wb
        return torch.mean(r1 ** 2 + r2 ** 2)

    def forward(self, A, S0, K, Yobs=None, tf_prob=0.0):
        B = A.shape[0]
        hs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        cs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        th = S0.clone()
        outs = []
        nb = 0
        wbs = 0.0
        for t in range(K):
            a = A[:, t]
            x = torch.cat([a, th], dim=1)
            for i, cl in enumerate(self.cells):
                hs[i], cs[i] = cl(x, (hs[i], cs[i]))
                x = hs[i]
            th_prev = th
            th = th + self.head(x)                            # predicted increment
            nb += int(((th <= 0.0201) | (th >= 0.5999)).sum().item())
            th = torch.clamp(th, 0.02, 0.60)
            outs.append(th)                                   # record the true prediction first
            if self.w > 0:
                wbs = wbs + self._wb_res(a, th_prev, th, x)
            if Yobs is not None and tf_prob > 0:
                msk = (torch.rand(B, device=th.device) < tf_prob).float()[:, None]
                th = msk * Yobs[:, t] + (1 - msk) * th
        pen = wbs / K if self.w > 0 else A.new_zeros(())
        return torch.stack(outs, dim=1), {'bind': nb / max(1, K * B), 'wbp': pen}, None


class LSTMHYB(nn.Module):
    """**hard** water-balance state update + LSTM backbone  --  key intermediate for ablating architecture confounding

    Originally HYBNS (hard) and LSTM (soft) at least three differences were entangled:
      (1) whether the state update is given rigidly by the water-balance equation (the form of interest here);
      (2) memory comes from LSTM cell, or "memoryless feed-forward flux head + GRU residual";
      (3)  Irrigation and other extra source/sink terms, and the percolation-flux parameterisation.
    Therefore "HYBNS worse than LSTM" cannot be attributed to(1). This class, relative to LPMIX **change only(1)**, everything else component-wise identical:
        backbone (cells), head, flux-head structure and inputs (hidden state h), SS regime and parameter count.
        Difference: the ET/Q/D given by the flux head ET/Q/D are no longer just a penalty term but**directly determine**state update;
        head output is changed to a bounded residual U entering the same equation (in the same position as HYBNS ResFlux).
    This yields a decomposable triplet (under the same training regime NS regime):
        LSTMNS   -- change only the state update -->  HYBLSTNS   : pure form effect
        HYBLSTNS -- change backbone/residual implementation --> HYBNS     : pure backbone-and-design effect
        The sum of the two is the total gap that was previously conflated.
    """

    def __init__(self, din, dh=64, nlayers=1, cell='lstm', umax=2.0,
                 rlo=0.2, rhi=1.2, qmax=0.6, dmax1=8.0, dmax2=3.0,
                 dform='sat', use_irr=False, irr_max=2.0):
        super().__init__()
        self.dh = dh
        self.nlayers = nlayers
        self.cell = cell
        self.umax = float(umax)
        self.rlo, self.rhi = float(rlo), float(rhi)
        self.qmax = float(qmax)
        self.dmax1, self.dmax2 = float(dmax1), float(dmax2)
        self.dform = dform
        self.use_irr = bool(use_irr)
        self.irr_max = float(irr_max)
        self.thmin = 0.02
        self.cells = nn.ModuleList([
            (nn.LSTMCell(din + 2 if i == 0 else dh, dh)
             if cell == 'lstm' else nn.GRUCell(din + 2 if i == 0 else dh, dh))
            for i in range(nlayers)])
        # head is here changed to the "unmodelled residual flux U" (mm/day), zero-initialised →
        # The training starts from **pure water balance**, gradually relaxed during training, in exactly the same position as HYBNS ResFlux
        self.head = nn.Linear(dh, 2)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head.weight)
        # --- the flux head must be created after cells/head (in the same order as LPMIX)---
        self.f_et = mlp(dh, dh, 1)
        self.f_d12 = mlp(dh, dh, 1)
        self.f_d2 = mlp(dh, dh, 1)
        self.f_q = mlp(dh, dh, 1)
        if self.use_irr:
            self.f_irr = mlp(dh, dh, 1)
        self.w_et = nn.Parameter(torch.tensor(0.0))     # surface-layer ET fraction = sigmoid(0) = 0.5

    def forward(self, A, S0, K, Yobs=None, tf_prob=0.0):
        B = A.shape[0]
        dev = A.device
        hs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        cs = [A.new_zeros(B, self.dh) for _ in range(self.nlayers)]
        th = S0.clone()
        outs, ets = [], []
        nb = 0
        dg = {'P': 0.0, 'q': 0.0, 'd12': 0.0, 'd2': 0.0, 'inf': 0.0,
              'irr': 0.0, 'u1': 0.0, 'u2': 0.0, 'pet': 0.0}
        for t in range(K):
            a = A[:, t]
            x = torch.cat([a, th], dim=1)
            for i, cl in enumerate(self.cells):
                if self.cell == 'lstm':
                    hs[i], cs[i] = cl(x, (hs[i], cs[i]))
                    x = hs[i]
                else:
                    hs[i] = cl(x, hs[i])
                    x = hs[i]
            inp = x                                     # flux-head input = hidden state (same as LPMIX)
            th_prev = th
            thmin = self.thmin
            Praw = torch.expm1(a[:, 0]).clamp(min=0.0)
            pet = a[:, 1]
            por1, por2 = a[:, 2], a[:, 3]
            s1 = ((th_prev[:, 0] - thmin) / (por1 - thmin)).clamp(0, 1)
            s2 = ((th_prev[:, 1] - thmin) / (por2 - thmin)).clamp(0, 1)
            ratio = self.rlo + (self.rhi - self.rlo) * torch.sigmoid(
                self.f_et(inp).squeeze(-1))
            et = ratio * pet
            wet = torch.sigmoid(self.w_et)
            et1, et2 = wet * et, (1.0 - wet) * et
            if self.dform == 'sat':                     # LPMIX-style: bounded and switched by saturation
                d12 = self.dmax1 * torch.sigmoid(self.f_d12(inp).squeeze(-1)) * s1
                d2 = self.dmax2 * torch.sigmoid(self.f_d2(inp).squeeze(-1)) * s2
            else:                                       # HYBNS -style:softplus unbounded above
                d12 = nn.functional.softplus(self.f_d12(inp).squeeze(-1))
                d2 = nn.functional.softplus(self.f_d2(inp).squeeze(-1))
            q_horton = Praw * (self.qmax * torch.sigmoid(self.f_q(inp).squeeze(-1)))
            avail = Praw - q_horton
            deficit = torch.clamp((por1 - th_prev[:, 0]) * ZR1 + et1 + d12, min=0.0)
            inf = torch.minimum(avail, deficit)
            q = Praw - inf
            irr = (self.irr_max * torch.sigmoid(self.f_irr(inp).squeeze(-1))
                   if self.use_irr else torch.zeros_like(et))
            u = self.umax * torch.tanh(self.head(inp))
            u1, u2 = u[:, 0], u[:, 1]

            # --- Hard update: the state is determined solely by the fluxes,head can only enter as the residual U intervene ---
            s1n = th_prev[:, 0] * ZR1 + inf + irr - et1 - d12 + u1
            s2n = th_prev[:, 1] * ZR2 + d12 - et2 - d2 + u2
            n1, n2 = s1n / ZR1, s2n / ZR2
            nb += int(((n1 <= thmin + 1e-4) | (n1 >= por1 - 1e-4) |
                       (n2 <= thmin + 1e-4) | (n2 >= por2 - 1e-4)).sum().item())
            th = torch.stack([
                torch.minimum(torch.clamp(n1, min=thmin), por1),
                torch.minimum(torch.clamp(n2, min=thmin), por2)], dim=1)
            outs.append(th)
            ets.append(et)
            if Yobs is not None and tf_prob > 0:
                msk = (torch.rand(B, device=dev) < tf_prob).float()[:, None]
                th = msk * Yobs[:, t] + (1 - msk) * th
            dg['P'] += float(Praw.detach().mean().item())
            dg['q'] += float(q.detach().mean().item())
            dg['d12'] += float(d12.detach().mean().item())
            dg['d2'] += float(d2.detach().mean().item())
            dg['inf'] += float(inf.detach().mean().item())
            dg['irr'] += float(irr.detach().mean().item())
            dg['u1'] += float(u1.detach().mean().item())
            dg['u2'] += float(u2.detach().mean().item())
            dg['pet'] += float(pet.detach().mean().item())
        out = torch.stack(outs, dim=1)
        ets_t = torch.stack(ets, dim=1)
        for k in dg:
            dg[k] = dg[k] / K
        dg['bind'] = nb / max(1, K * B)
        dg['ratio'] = float((ets_t.mean() / max(1e-6, pet.mean().item())))
        return out, dg, ets_t


# --------------------------------------------------------------------------
# train / evaluation
# --------------------------------------------------------------------------
def rmse_at(pred, obs):
    """pred/obs: (B,K,2) -> horizon-wise RMSE (this step and earlier? this step here)"""
    e = pred - obs
    r = {}
    for h in HORIZONS:
        j = min(h, e.shape[1]) - 1
        r[h] = float(torch.sqrt((e[:, j] ** 2).mean()).item())
    r['all'] = float(torch.sqrt((e ** 2).mean()).item())
    return r


def pers_rmse(S0, Y):
    """Persistence baseline: use the initial observation throughout"""
    e = S0.unsqueeze(1) - Y
    r = {}
    for h in HORIZONS:
        j = min(h, e.shape[1]) - 1
        r[h] = float(torch.sqrt((e[:, j] ** 2).mean()).item())
    r['all'] = float(torch.sqrt((e ** 2).mean()).item())
    return r


def fill_nan_cols(a):
    """Fill each column (grid cell) with its temporal mean; use the regional mean when a whole column is missing"""
    a = np.asarray(a, dtype='float64')
    m = np.isnan(a)
    if not m.any():
        return a.astype('float32')
    cm = np.nanmean(a, axis=0)
    g = float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else 0.0
    cm = np.where(np.isfinite(cm), cm, g)
    idx = np.where(m)
    a[idx] = cm[idx[1]]
    return a.astype('float32')


def build_xregion(paths, etpaths, names, mu, sd, args, src_names):
    """Load cross-region target data, reusing the source-region mu/sd standardisation, returning [(name, te_seqset), ...]"""
    out = []
    for i, p in enumerate(paths):
        p = p.strip()
        if not p:
            continue
        pp = p if os.path.isabs(p) else os.path.join(ROOT, p)
        dd = np.load(pp, allow_pickle=True)
        A2, _ = build_inputs(dd['X'], list(dd['feats']))
        Y2 = dd['Y']
        nm = names[i].strip() if i < len(names) and names[i].strip() else \
            os.path.basename(p).replace('.npz', '')
        if args.pet_src == 'psp':
            if i >= len(etpaths) or not etpaths[i].strip():
                raise SystemExit('!! pet_src=psp every target region must be given --xet (%s missing)' % nm)
            ep = etpaths[i].strip()
            ep = ep if os.path.isabs(ep) else os.path.join(ROOT, ep)
            de = np.load(ep, allow_pickle=True)
            etm = de['time'].astype('datetime64[D]')
            tv2 = dd['time'].astype('datetime64[D]')
            if len(etm) != len(tv2) or not np.array_equal(etm, tv2):
                raise SystemExit('!! %s of ET ET time axis does not match the target region' % nm)
            T0, H, W = de['ET'].shape
            cells = np.asarray(dd['cells']) if 'cells' in dd.files else np.arange(H * W)
            A2[:, :, src_names.index('PET')] = fill_nan_cols(
                de['PET'].reshape(T0, -1)[:, cells])
        A2 = (A2 - mu) / sd
        tv2 = dd['time'].astype('datetime64[D]')
        i0 = int(np.searchsorted(tv2, np.datetime64(args.ttest.split(',')[0])))
        i1 = int(np.searchsorted(tv2, np.datetime64(args.ttest.split(',')[1]), side='right'))
        sl = slice(max(0, i0), min(A2.shape[0], i1))
        te2 = SeqSet(A2, Y2, sl, args.K, args.K)
        out.append((nm, te2, float(np.nanmean(Y2))))
        P('  cross-region target %-16s test %d days  sequences %d  target-field mean %.4f'
          % (nm, sl.stop - sl.start, te2.n, float(np.nanmean(Y2))))
    return out


def xeval(net, xr, cfg):
    """Evaluate on cross-region targets with metric conventions identical to the source-region test"""
    res = {}
    for nm, te2, _ in xr:
        A2e, Y2e, S02e, _ = te2.tensors()
        A2e, Y2e, S02e = A2e.to(DEV), Y2e.to(DEV), S02e.to(DEV)
        with torch.no_grad():
            if cfg == 'PERS':
                r = pers_rmse(S02e, Y2e)
                res[nm] = {'all': r['all'], 'h1': r[1], 'h30': r[30],
                           'l1': float('nan'), 'l2': float('nan')}
            else:
                pt2, _, _ = net(A2e, S02e[:te2.n], te2.K if hasattr(te2, 'K') else A2e.shape[1])
                r = rmse_at(pt2, Y2e)
                r1 = rmse_at(pt2[:, :, 0:1], Y2e[:, :, 0:1])
                r2 = rmse_at(pt2[:, :, 1:2], Y2e[:, :, 1:2])
                res[nm] = {'all': r['all'], 'h1': r[1], 'h30': r[30],
                           'l1': r1['all'], 'l2': r2['all']}
    return res


def run(cfg, A, Y, mu, sd, train_sl, val_sl, test_sl, args, Eall=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    K = args.K
    Fm = A.shape[2]

    tr = SeqSet(A, Y, train_sl, K, args.stride, E=Eall)
    va = SeqSet(A, Y, val_sl, K, args.stride, E=Eall)
    te = SeqSet(A, Y, test_sl, K, K, E=Eall)   # test windows do not overlap
    P('Sequence count: train=%d val=%d test=%d  (K=%d)' % (tr.n, va.n, te.n, K))
    if Eall is not None:
        P('ET auxiliary loss: lamet=%g  observed ET mean %.3f mm/day' %
          (args.lamet, float(np.nanmean(Eall))))

    Atr, Ytr, S0tr, Etr = tr.tensors()
    Ava, Yva, S0va, Eva = va.tensors()
    Ate, Yte, S0te, Ete = te.tensors()
    # ---- GPUGPU: move to device once; all later index slicing happens on the device ----
    def _mv(t):
        return None if t is None else t.to(DEV, non_blocking=True)
    Atr, Ytr, S0tr, Etr = _mv(Atr), _mv(Ytr), _mv(S0tr), _mv(Etr)
    Ava, Yva, S0va = _mv(Ava), _mv(Yva), _mv(S0va)
    Ate, Yte, S0te, Ete = _mv(Ate), _mv(Yte), _mv(S0te), _mv(Ete)

    if cfg == 'PERS':
        r = pers_rmse(S0te, Yte)
        return {'cfg': cfg, 'test': r, 'nparam': 0}, None

    if cfg in ('PHY', 'PHYSS', 'HYB', 'HYBNS'):
        net = PHY(Fm + 2, args.hidden, use_res=(cfg in ('HYB', 'HYBNS')),
                  umax=args.umax, irr_max=args.irrmax, qmax=args.qmax,
                  rlo=args.rlo, rhi=args.rhi, fix_ratio=args.fix_ratio,
                  dynw=bool(getattr(args, 'dynw', 0)))
    elif cfg in ('LSTM', 'LSTMNS'):
        net = LSTM(Fm, args.hidden, nlayers=getattr(args, 'lstm_layers', 1))
    elif cfg in ('LPMIX', 'LPMIXNS'):
        # Continuous constraint-strength axis:w=0 ≡ LSTM (verified value-by-value identical),w increases → approaches the hard water-balance skeleton
        net = LSTMPHY(Fm, args.hidden, nlayers=getattr(args, 'lstm_layers', 1),
                      w=args.wbw, rlo=args.rlo, rhi=args.rhi, qmax=args.qmax)
    elif cfg in ('HYBLST', 'HYBLSTNS'):
        # **intermediate**: hard state update + LSTM backbone. Relative to LPMIX, change only the "update rule", 
        # relative to HYBNS change only the backbone/residual implementation", to separate the form effect from the architecture effect.
        net = LSTMHYB(Fm, args.hidden, nlayers=getattr(args, 'lstm_layers', 1),
                      cell=getattr(args, 'cell', 'lstm'), umax=args.umax,
                      rlo=args.rlo, rhi=args.rhi, qmax=args.qmax,
                      dform=getattr(args, 'hxd', 'sat'),
                      use_irr=bool(getattr(args, 'hxirr', 0)),
                      irr_max=args.irrmax)
    # PHY Keep 12 unchanged (free-running rollout) as the ablation starting point;
    # HYBNS = residual + free-running rollout, cleanly separating the net effect of the residual itself (free from SS interference)
    # LSTMNS: exactly the same structure as LSTM, but without feeding observations (free-running training), 
    # to separate "scheduled sampling the scheduled-sampling training trick" out of the structural comparison
    use_ss = cfg in ('PHYSS', 'HYB', 'LSTM', 'LPMIX', 'HYBLST')
    npar = sum(p.numel() for p in net.parameters())
    P('Model %s  parameter count %d  scheduled_sampling=%s  lres=%g' %
      (cfg, npar, use_ss, args.lres if cfg == 'HYB' else 0.0))

    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = None
    if getattr(args, 'sched', 'none') == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    warm = int(getattr(args, 'warmup', 0))
    nbatch = int(np.ceil(tr.n / args.batch))
    best = 1e9
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(tr.n, device=DEV)
        tot = 0.0
        # scheduled sampling scheduled-sampling probability decays linearly: more observations early (easier optimisation), approaching free-running rollout later (matching the test)
        tf = 0.0
        if use_ss and args.ss_frac > 0:
            tf = args.ss_min + (1.0 - args.ss_min) * max(
                0.0, 1.0 - (ep - 1) / max(1.0, args.ss_frac * args.epochs))
        dg = {'ratio': 0.0, 'q': 0.0, 'd12': 0.0, 'd2': 0.0, 'bind': 0.0,
              'P': 0.0, 'pet': 0.0, 'irr': 0.0, 'inf': 0.0,
              'u1': 0.0, 'u2': 0.0, 'upen': 0.0, 'wbp': 0.0}
        for b in range(nbatch):
            ix = perm[b * args.batch:(b + 1) * args.batch]
            pred, d, ets = net(Atr[ix], S0tr[ix], K,
                               Yobs=(Ytr[ix] if tf > 0 else None), tf_prob=tf)
            mse = torch.mean((pred - Ytr[ix]) ** 2)
            loss = mse
            if cfg == 'HYB':
                loss = loss + args.lres * d['upen']
            if cfg in ('LPMIX', 'LPMIXNS') and args.wbw > 0:
                # Soft water-balance penalty:d['wbp'] is already Δθ/day dimensionofmean squaredresidual, and mse same dimension
                loss = loss + args.wbw * d['wbp']
            if ets is not None and Etr is not None and args.lamet > 0:
                # ET error converted to θ units (mm/day ÷ Zr mm = Δθ), same dimension as the main loss,
                # so that lamet is then O(1) order of magnitude, avoiding a 1e-5 -level fudge factor
                eo = Etr[ix]
                mk = ~torch.isnan(eo)
                if bool(mk.any()):
                    loss = loss + args.lamet * torch.mean(((ets - eo)[mk] / ZR1) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(mse.item()) * len(ix)
            for k in dg:
                if k in d:
                    dg[k] += d[k]
        net.eval()
        with torch.no_grad():
            pv, _, _ = net(Ava, S0va, K)       # validation/validation/test always free-running (tf=0)
            vloss = float(torch.mean((pv - Yva) ** 2).item())
        msg = '  ep%3d  train MSE %.6f (RMSE %.5f)  val RMSE %.5f  %.1fs' % (
            ep, tot / tr.n, np.sqrt(tot / tr.n), np.sqrt(vloss), time.time() - t0)
        msg += '  | tf %.2f bind %.3f' % (tf, dg['bind'] / nbatch)
        if DEV.type == 'cuda':
            msg += '  | mem %.2fGB' % (torch.cuda.max_memory_allocated() / 1e9)
        if cfg in ('PHY', 'PHYSS', 'HYB', 'HYBNS'):
            msg += '  P %.2f INF %.2f Q %.2f PET %.2f I %.3f ratio %.3f D12 %.3f D2 %.3f' % (
                dg['P'] / nbatch, dg['inf'] / nbatch, dg['q'] / nbatch,
                dg['pet'] / nbatch, dg['irr'] / nbatch,
                dg['ratio'] / nbatch, dg['d12'] / nbatch, dg['d2'] / nbatch)
        if cfg in ('HYB', 'HYBNS'):
            msg += '  | U1 %+.3f U2 %+.3f pen %.2e' % (
                dg['u1'] / nbatch, dg['u2'] / nbatch, dg['upen'] / nbatch)
        if cfg in ('LPMIX', 'LPMIXNS') and args.wbw > 0:
            _w = dg['wbp']
            _w = float(_w.item()) if hasattr(_w, 'item') else float(_w)
            msg += '  | w=%g wbpen %.3e (RMSE %.5f)' % (
                args.wbw, _w / nbatch, float(np.sqrt(max(_w / nbatch, 0.0))))
        P(msg)
        if warm and ep <= warm:
            for g in opt.param_groups:
                g['lr'] = args.lr * ep / warm
        elif sched is not None:
            sched.step()
        hist.append((ep, tot / tr.n, vloss))
        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), os.path.join(BENCH, 'best_%s%s.pt' % (cfg, args.tag)))
        if ep - np.argmin([h[2] for h in hist]) > args.patience and ep > args.patience:
            P('  early stopping @ ep%d' % ep)
            break

    net.load_state_dict(torch.load(os.path.join(BENCH, 'best_%s%s.pt' % (cfg, args.tag)),
                                  map_location=DEV))
    net.eval()
    with torch.no_grad():
        pt, dg, pt_et = net(Ate, S0te[:te.n], K)
    res = {'cfg': cfg, 'nparam': int(npar),
           'test': rmse_at(pt, Yte),
           'test_l1': rmse_at(pt[:, :, 0:1], Yte[:, :, 0:1]),
           'test_l2': rmse_at(pt[:, :, 1:2], Yte[:, :, 1:2])}
    # ET with the independent product (PSP) comparison: this is"whether identifiability is resolved"direct evidence
    if pt_et is not None and Ete is not None:
        mk = ~torch.isnan(Ete)
        if bool(mk.any()):
            e = (pt_et - Ete)[mk]
            res['et_rmse'] = float(torch.sqrt((e ** 2).mean()).item())
            res['et_bias'] = float(e.mean().item())
            res['et_pred'] = float(pt_et[mk].mean().item())
            res['et_obs'] = float(Ete[mk].mean().item())
            P('  [ET vs PSP] predicted %.3f  observed %.3f  RMSE %.3f  bias %+.3f mm/day'
              % (res['et_pred'], res['et_obs'], res['et_rmse'], res['et_bias']))
    # Record test-period physical-flux diagnostics for cross-seed identifiability checks (whether the same metric is stable under different initialisations)
    if dg:
        for k in ('P', 'inf', 'q', 'pet', 'irr', 'ratio', 'd12', 'd2', 'u1', 'u2'):
            if k in dg:
                res['flux_' + k] = float(dg[k])
        if 'pet' in dg and 'ratio' in dg:
            res['flux_et'] = float(dg['ratio'] * dg['pet'])
        if 'u1' in dg and 'u2' in dg:
            res['flux_usum'] = float(dg['u1'] + dg['u2'])
    P('  [test-period fluxes mm/day] ' + '  '.join(
        '%s=%.3f' % (k[5:], v) for k, v in res.items() if k.startswith('flux_')))
    # write test-period daily predictions: flattened sequence order is k(start)*N+c(grid cells), the t step corresponds to time starts[k]+t+1
    if getattr(args, 'savepred', False):
        import numpy as _np
        _op = os.path.join(BENCH, 'pred_%s%s.npz' % (cfg, args.tag))
        _np.savez_compressed(_op,
                             pred=pt.detach().cpu().numpy().astype('float32'),
                             obs=Yte.detach().cpu().numpy().astype('float32'),
                             s0=S0te.detach().cpu().numpy().astype('float32'),
                             starts=te.starts, N=_np.int64(A.shape[1]),
                             K=_np.int64(K), cfg=cfg)
        P('  WROTE daily predictions %s' % _op)
    return res, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--K', type=int, default=30)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cfgs', default='PHY,PHYSS,HYB,HYBNS,LSTM,PERS')
    ap.add_argument('--umax', type=float, default=2.0)      # residual flux upper bound mm/day
    ap.add_argument('--irrmax', type=float, default=2.0)    # irrigation upper bound mm/day
    ap.add_argument('--qmax', type=float, default=0.6)      # infiltration-excess runoff ratio upper bound (saturation excess is bounded by porosity)
    ap.add_argument('--lres', type=float, default=0.5)      # residual penalty coefficient (θ units)
    ap.add_argument('--wbw', type=float, default=0.0,
                    help='LPMIX soft water-balance penalty weight (continuous constraint-strength axis;0=pure LSTM)')
    ap.add_argument('--rlo', type=float, default=0.2)       # ET/PET ratio lower bound
    ap.add_argument('--rhi', type=float, default=1.2)       # ET/PET ratio upper bound
    ap.add_argument('--fix_ratio', type=float, default=None)  # freeze ET/PET freeze the ET/PET ratio for an identifiability scan
    ap.add_argument('--ss_frac', type=float, default=0.7)   # SS within the first 70% epoch decays to ss_min
    ap.add_argument('--ss_min', type=float, default=0.0)
    ap.add_argument('--tag', default='')   # result filename suffix, to distinguish feature/data version
    ap.add_argument('--threads', type=int, default=4,
                    help='torch thread count (this machine has 8 threads, default 4 uses only physical cores)')
    ap.add_argument('--savepred', action='store_true',
                    help='write test-period daily predictions for station-truth comparison (16 for use)')
    ap.add_argument('--ttrain', default='2016-01-01,2017-12-31')
    ap.add_argument('--tval', default='2018-01-01,2018-12-31')
    ap.add_argument('--ttest', default='2019-01-01,2019-12-31')
    ap.add_argument('--etnpz', default='')     # PSP evapotranspiration product (14 produced by), used for ET auxiliary loss
    ap.add_argument('--etvar', default='ETsoil')  # ETsoil = ET - Ei (interception removed, truly taken from soil)
    ap.add_argument('--lamet', type=float, default=0.0)   # ET auxiliary loss weight (θ units,O(1))
    ap.add_argument('--pet_src', default='jh', choices=['jh', 'psp'])  # PET baseline
    ap.add_argument('--xnpz', default='',
                    help='comma-separated cross-region target npz (zero-shot evaluation, not used for training)')
    ap.add_argument('--xet', default='',
                    help='and xnpz one-to-one target region PSP evapotranspiration npz (pet_src=psp is required)')
    ap.add_argument('--xname', default='', help='and xnpz one-to-one target-region names')
    ap.add_argument('--srcname', default='src', help='source-region name (row label of the transfer matrix)')
    ap.add_argument('--dynw', type=int, default=0,
                    help='1=inter-layer ET partitioning changed to state-dependent layer-wise root uptake (default 0, comparable with old results)')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                    help='auto=has GPU use GPU')
    ap.add_argument('--lstm_layers', type=int, default=1, help='pure LSTM number of layers of the control')
    ap.add_argument('--cell', default='lstm', choices=['lstm', 'gru'],
                    help='HYBLST backbone type of the intermediate')
    ap.add_argument('--hxd', default='sat', choices=['sat', 'free'],
                    help='HYBLST percolation flux parameterisation:sat=LPMIX -style (bounded and modulated by saturation);free=HYBNS -style softplus unbounded above')
    ap.add_argument('--hxirr', type=int, default=0,
                    help='1=HYBLST add the irrigation term (as in HYBNS same as HYBNS), default 0 to keep the single-variable comparison with LPMIX single-variable comparison')
    ap.add_argument('--sched', default='none', choices=['none', 'cosine'],
                    help='learning-rate schedule:cosine=cosine annealing to lr of 5%')
    ap.add_argument('--warmup', type=int, default=0, help='first n entries epoch linear warm-up')
    a = ap.parse_args()

    global DEV
    if a.device == 'auto':
        DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEV = torch.device(a.device)
    torch.set_num_threads(a.threads)
    P('torch %s  device=%s  threads=%d' % (
        torch.__version__, DEV, torch.get_num_threads()))
    if DEV.type == 'cuda':
        P('GPU: %s  GPU memory %.1f GB' % (torch.cuda.get_device_name(0),
                                    torch.cuda.get_device_properties(0).total_memory / 1e9))
    d = np.load(a.npz, allow_pickle=True)
    X, Y, tm, feats = d['X'], d['Y'], d['time'], list(d['feats'])
    P('Loaded %s  X%s Y%s' % (os.path.basename(a.npz), X.shape, Y.shape))

    A, names = build_inputs(X, feats)
    tv = tm.astype('datetime64[D]')
    T = A.shape[0]

    def sl_of(s0, s1):
        i0 = int(np.searchsorted(tv, np.datetime64(s0)))
        i1 = int(np.searchsorted(tv, np.datetime64(s1), side='right'))
        return slice(max(0, i0), min(T, i1))

    tr_sl = sl_of(*a.ttrain.split(','))
    va_sl = sl_of(*a.tval.split(','))
    te_sl = sl_of(*a.ttest.split(','))
    P('train %s(%ddays) validation %s(%ddays) test %s(%ddays)'
      % (a.ttrain, tr_sl.stop - tr_sl.start, a.tval, va_sl.stop - va_sl.start,
         a.ttest, te_sl.stop - te_sl.start))

    # ---- PSP Evapotranspiration product: for ET auxiliary loss (the remedy for the identifiability problem)----
    Eall = None
    Epet = None
    if a.etnpz:
        p = a.etnpz if os.path.isabs(a.etnpz) else os.path.join(ROOT, a.etnpz)
        de = np.load(p, allow_pickle=True)
        etm = de['time'].astype('datetime64[D]')
        if len(etm) != len(tv) or not np.array_equal(etm, tv):
            raise SystemExit('!! ET ET time axis does not match the training data: %s vs %s' % (etm[0], tv[0]))
        T0, H, W = de['ET'].shape
        cells = np.asarray(d['cells']) if 'cells' in d.files else np.arange(H * W)
        Eall = fill_nan_cols(de[a.etvar].reshape(T0, -1)[:, cells])
        if 'PET' in de.files:
            Epet = fill_nan_cols(de['PET'].reshape(T0, -1)[:, cells])
        if np.isnan(np.asarray(de[a.etvar]).reshape(T0, -1)[:, cells]).any():
            P('  [NaN filled] PSP product has missing grid cells, filled with the per-cell temporal mean')
        P('ET product %s  variables %s  ->  %s  annual total %.1f mm/yr' %
          (os.path.basename(p), a.etvar, Eall.shape, np.nanmean(Eall) * 365.25))
        if Eall.shape != Y.shape[:2]:
            raise SystemExit('!! ET shape %s and Y %s does not match' % (Eall.shape, Y.shape[:2]))

    if a.pet_src == 'psp':
        if Epet is None:
            raise SystemExit('!! pet_src=psp requires --etnpz provide PET')
        i_pet = names.index('PET')
        old = float(np.nanmean(A[:, :, i_pet]))
        A[:, :, i_pet] = Epet
        P('PET baseline switch: self-computed Jensen-Haise %.3f -> PSP %.3f mm/day' %
          (old, float(np.nanmean(Epet))))

    # Standardisation (training period only)
    # Important: the first two columns (log1p(precipitation) and PET) must keep original units,
    #       the model must use expm1 recover the true mm/day; after standardisation, applying expm1 would give a wrong order of magnitude.
    m = A[tr_sl].reshape(-1, A.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    mu[:4] = 0.0        # P_log / PET / por1 / por2 keep original units
    sd[:4] = 1.0
    A = (A - mu) / sd
    P('Standardisation complete (training-period statistics);P_log/PET/por1/por2 four columns keep original units')

    # ---- cross-region zero-shot target regions (one row of the transfer matrix)----
    XREG = []
    if a.xnpz:
        P('')
        P('--- cross-region zero-shot target regions (reusing source-region standardisation statistics)---')
        XREG = build_xregion(a.xnpz.split(','), a.xet.split(','), a.xname.split(','),
                             mu, sd, a, names)

    allres = []
    for cfg in a.cfgs.split(','):
        P('')
        P('================ %s ================' % cfg)
        r, net = run(cfg, A, Y, mu, sd, tr_sl, va_sl, te_sl, a, Eall=Eall)
        if XREG:
            xr = xeval(net, XREG, cfg)
            r['xreg'] = xr
            P('  [cross-region zero-shot] ' + '  '.join(
                '%s %.5f' % (k, v['all']) for k, v in xr.items()))
        allres.append(r)

    P('')
    P('================ Summary (test period %s, RMSE m3/m3)==============' % a.ttest)
    P('%-6s %8s %8s %8s %8s %8s %8s   %s' %
      ('cfg', 'h=1', 'h=3', 'h=7', 'h=15', 'h=30', 'all', 'params'))
    for r in allres:
        t = r['test']
        P('%-6s %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f   %d' %
          (r['cfg'], t[1], t[3], t[7], t[15], t[30], t['all'], r['nparam']))
    P('')
    P('--- multi-step drift ratio h30/h1 (smaller is more stable)---')
    for r in allres:
        t = r['test']
        P('  %-6s  %.2f' % (r['cfg'], t[30] / max(1e-9, t[1])))
    P('')
    P('--- by layer (10cm / 30cm)all-step RMSE ---')
    for r in allres:
        if 'test_l1' in r:
            P('  %-6s 10cm %.5f   30cm %.5f' % (r['cfg'], r['test_l1']['all'], r['test_l2']['all']))

    if XREG:
        P('')
        P('================ cross-region zero-shot transfer matrix (all-step RMSE)==============')
        tg = [nm for nm, _, _ in XREG]
        P('%-6s %10s %10s   %s' % ('cfg', 'source region(self)', 'drift ratio',
                                   ' '.join('%10s' % t[:10] for t in tg)))
        for r in allres:
            xr = r.get('xreg', {})
            P('%-6s %10.5f %10.2f   %s' % (
                r['cfg'], r['test']['all'], r['test'][30] / max(1e-9, r['test'][1]),
                ' '.join('%10.5f' % xr[t]['all'] for t in tg)))
        # write the transfer-matrix row separately, for 340 assembled into the full matrix
        tm_op = os.path.join(BENCH, 'transfer_row%s_%s.json'
                             % (a.tag, a.srcname))
        json.dump({'src': a.srcname, 'tag': a.tag, 'cfgs': {r['cfg']: r.get('xreg', {})
                                                            for r in allres},
                   'self': {r['cfg']: r['test'] for r in allres},
                   'self_l': {r['cfg']: {'l1': r.get('test_l1', {}).get('all'),
                                         'l2': r.get('test_l2', {}).get('all')}
                              for r in allres}},
                  open(tm_op, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        P('WROTE transfer-matrix row %s' % tm_op)

    # by cfg merge into existing results so that separate runs of different configurations do not overwrite each other
    op = os.path.join(BENCH, 'dl_hybrid_result%s.json' % a.tag)
    prev = []
    if os.path.exists(op):
        try:
            prev = json.load(open(op, encoding='utf-8'))
        except Exception:
            prev = []
    byc = {r['cfg']: r for r in prev}
    for r in allres:
        byc[r['cfg']] = r
    json.dump(list(byc.values()), open(op, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    P('WROTE %s  (cumulative %d  configurations)' % (op, len(byc)))
    lp = os.path.join(LOG, '_302_dl_hybrid%s.txt' % a.tag)
    old = ''
    if os.path.exists(lp):
        try:
            old = open(lp, encoding='utf-8').read() + '\n\n'
        except Exception:
            old = ''
    open(lp, 'w', encoding='utf-8').write(old + '\n'.join(LOG))
    P('WROTE log %s' % lp)


if __name__ == '__main__':
    main()
