# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), '../..', '01_Code', '00_Common_Modules')))
from paths import (PKG, CODE, DATA, TRUTH, MULTI, STATION, BENCH_JSON,
                   TRAIN_SUM, FIG, MANU, MANU_FIG, LOG, REPRO, RAW, WORK, BENCH)
"""12: deep learning smoke test  --  physically constrained water-balance state model vs pure LSTM

Three configurations:
  PHY   : two-layer water-balance hard constraints,NN only parameterises 4  processes
            ET = ratio*PET (ratio in (0,2])  surface runoff Q = P*sigmoid  inter-layer percolation D12  deep drainage D2
  LSTM  : purely autoregressive LSTM (parameter count matched), no physical constraints
  PERS  : persistence baseline (analytic, not trained)

State (two layers):
  layer1 0-10cm  Zr1 = 100 mm     layer2 10-30cm  Zr2 = 01 mm
  S1 = th1*Zr1, S2 = th2*Zr2  (mm)
  S1' = S1 + P - Q - ET1 - D12
  S2' = S2 + D12 - ET2 - D2

Usage (torch environment,CPU): 
  conda activate torch
  python 12_dl_smoke.py --npz out/_bench/dl_R0_northchina_nc64_20160101_20191231.npz --epochs 20
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn

ROOT = PKG
LOG = []
HORIZONS = [1, 3, 7, 15, 30]
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
    ]
    A = np.stack(cols, axis=2).astype('float32')
    A[:, :, 0] = np.log1p(np.clip(A[:, :, 0], 0, None))   # precipitation is right-skewed, take log1p
    names = ['P_log', 'PET', 'por1', 'por2', 't2m', 'vpd', 'swrad', 'doy_sin', 'doy_cos',
             'clay_top', 'sand_top', 'clay_sub', 'sand_sub', 'slope']
    return A, names


class SeqSet:
    """Split (T,N,F) cut into fixed-length sequences (nseq,K,F)"""

    def __init__(self, A, Y, sl, K, stride, nmax=None):
        T, N, Fm = A.shape
        starts = list(range(sl.start, sl.stop - K, stride))
        if nmax:
            starts = starts[:nmax]
        self.A = np.empty((len(starts), N, K, Fm), 'float32')
        self.Y = np.empty((len(starts), N, K, 2), 'float32')
        self.S0 = np.empty((len(starts), N, 2), 'float32')
        for k, s in enumerate(starts):
            self.A[k] = np.transpose(A[s:s + K], (1, 0, 2))
            self.Y[k] = np.transpose(Y[s + 1:s + K + 1], (1, 0, 2))
            self.S0[k] = Y[s]
        shape = self.A.shape
        self.A = self.A.reshape(-1, K, Fm)
        self.Y = self.Y.reshape(-1, K, 2)
        self.S0 = self.S0.reshape(-1, 2)
        self.n = self.A.shape[0]

    def tensors(self):
        return (torch.from_numpy(self.A), torch.from_numpy(self.Y),
                torch.from_numpy(self.S0))


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


class PHY(nn.Module):
    """Physically constrained two-layer water balance:NN only learns ETratio / runoff ratio / inter-layer percolation / deep drainage"""

    def __init__(self, din, dh=64):
        super().__init__()
        self.f_et = mlp(din, dh, 1)
        self.f_q = mlp(din, dh, 1)
        self.f_d12 = mlp(din, dh, 1)
        self.f_d2 = mlp(din, dh, 1)
        self.f_irr = mlp(din, dh, 1)                # irrigation (a key extra water source on the North China Plain)
        self.w1 = nn.Parameter(torch.tensor(1.0))   # tendency to allocate ET to the surface layer

    def forward(self, A, S0, K):
        # A: (B,K,Fm)  S0: (B,2) units m3/m3
        B = A.shape[0]
        th1 = S0[:, 0].clone()
        th2 = S0[:, 1].clone()
        out = []
        diag = {'ratio': [], 'q': [], 'd12': [], 'd2': [], 'P': [], 'pet': [], 'irr': []}
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
            ratio = 0.2 + 1.0 * torch.sigmoid(self.f_et(inp).squeeze(-1))
            et = ratio * pet
            w = torch.sigmoid(self.w1)                        # surface-layer fraction (0,1)
            et1, et2 = w * et, (1 - w) * et
            q = Praw * (0.35 * torch.sigmoid(self.f_q(inp).squeeze(-1)))
            d12 = nn.functional.softplus(self.f_d12(inp).squeeze(-1))
            d2 = nn.functional.softplus(self.f_d2(inp).squeeze(-1))
            irr = nn.functional.softplus(self.f_irr(inp).squeeze(-1))

            s1 = th1 * ZR1 + Praw + irr - q - et1 - d12
            s2 = th2 * ZR2 + d12 - et2 - d2
            n1 = s1 / ZR1
            n2 = s2 / ZR2
            nb += int(((n1 <= 0.0201) | (n1 >= por1 - 1e-4) |
                       (n2 <= 0.0201) | (n2 >= por2 - 1e-4)).sum().item())
            # clamp does not support (Tensor, float, Tensor) mixing: use a scalar for the lower bound clamp, upper bound uses minimum
            th1 = torch.minimum(torch.clamp(n1, min=thmin), por1)
            th2 = torch.minimum(torch.clamp(n2, min=thmin), por2)

            out.append(torch.stack([th1, th2], dim=1))
            diag['ratio'].append(ratio.detach())
            diag['P'].append(Praw.detach())
            diag['pet'].append(pet.detach())
            diag['irr'].append(irr.detach())
            diag['q'].append(q.detach())
            diag['d12'].append(d12.detach())
            diag['d2'].append(d2.detach())
        out = torch.stack(out, dim=1)                         # (B,K,2)
        for k in diag:
            diag[k] = torch.stack(diag[k], dim=1).mean().item()
        diag['bind'] = nb / max(1, K * B)
        return out, diag


class LSTM(nn.Module):
    """purely autoregressive LSTM control: predict the two layers directly SM"""

    def __init__(self, din, dh=64):
        super().__init__()
        self.dh = dh
        self.lstm = nn.LSTM(din + 2, dh, batch_first=True)
        self.head = nn.Linear(dh, 2)
        # Zero-init output layer: initial behaviour degenerates to persistence, avoiding early autoregressive drift
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, A, S0, K):
        B = A.shape[0]
        h = None
        th = S0.clone()
        outs = []
        nb = 0
        for t in range(K):
            inp = torch.cat([A[:, t], th], dim=1).unsqueeze(1)
            o, h = self.lstm(inp, h)
            th = th + self.head(o.squeeze(1))                 # predicted increment
            nb += int(((th <= 0.0201) | (th >= 0.5999)).sum().item())
            th = torch.clamp(th, 0.02, 0.60)
            outs.append(th)
        return torch.stack(outs, dim=1), {'bind': nb / max(1, K * B)}


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


def run(cfg, A, Y, mu, sd, train_sl, val_sl, test_sl, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    K = args.K
    Fm = A.shape[2]

    tr = SeqSet(A, Y, train_sl, K, args.stride)
    va = SeqSet(A, Y, val_sl, K, args.stride)
    te = SeqSet(A, Y, test_sl, K, K)          # test windows do not overlap
    P('Sequence count: train=%d val=%d test=%d  (K=%d)' % (tr.n, va.n, te.n, K))

    Atr, Ytr, S0tr = tr.tensors()
    Ava, Yva, S0va = va.tensors()
    Ate, Yte, S0te = te.tensors()

    if cfg == 'PERS':
        r = pers_rmse(S0te, Yte)
        return {'cfg': cfg, 'test': r, 'nparam': 0}, None

    if cfg == 'PHY':
        net = PHY(Fm + 2, args.hidden)
    else:
        net = LSTM(Fm, args.hidden)
    npar = sum(p.numel() for p in net.parameters())
    P('Model %s  parameter count %d' % (cfg, npar))

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    nbatch = int(np.ceil(tr.n / args.batch))
    best = 1e9
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(tr.n)
        tot = 0.0
        dg = {'ratio': 0.0, 'q': 0.0, 'd12': 0.0, 'd2': 0.0, 'bind': 0.0,
              'P': 0.0, 'pet': 0.0, 'irr': 0.0}
        for b in range(nbatch):
            ix = perm[b * args.batch:(b + 1) * args.batch]
            pred, d = net(Atr[ix], S0tr[ix], K)
            loss = torch.mean((pred - Ytr[ix]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * len(ix)
            for k in dg:
                if k in d:
                    dg[k] += d[k]
        net.eval()
        with torch.no_grad():
            pv, _ = net(Ava, S0va, K)
            vloss = float(torch.mean((pv - Yva) ** 2).item())
        msg = '  ep%3d  train MSE %.6f (RMSE %.5f)  val RMSE %.5f  %.1fs' % (
            ep, tot / tr.n, np.sqrt(tot / tr.n), np.sqrt(vloss), time.time() - t0)
        msg += '  | bind %.3f' % (dg['bind'] / nbatch)
        if cfg == 'PHY':
            msg += '  P %.2f PET %.2f I %.3f ratio %.3f Q %.3f D12 %.3f D2 %.3f' % (
                dg['P'] / nbatch, dg['pet'] / nbatch, dg['irr'] / nbatch,
                dg['ratio'] / nbatch, dg['q'] / nbatch, dg['d12'] / nbatch, dg['d2'] / nbatch)
        P(msg)
        hist.append((ep, tot / tr.n, vloss))
        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), os.path.join(BENCH, 'best_%s.pt' % cfg))
        if ep - np.argmin([h[2] for h in hist]) > args.patience and ep > args.patience:
            P('  early stopping @ ep%d' % ep)
            break

    net.load_state_dict(torch.load(os.path.join(BENCH, 'best_%s.pt' % cfg)))
    net.eval()
    with torch.no_grad():
        pt, _ = net(Ate, S0te[:te.n], K)
    res = {'cfg': cfg, 'nparam': int(npar),
           'test': rmse_at(pt, Yte),
           'test_l1': rmse_at(pt[:, :, 0:1], Yte[:, :, 0:1]),
           'test_l2': rmse_at(pt[:, :, 1:2], Yte[:, :, 1:2])}
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
    ap.add_argument('--cfgs', default='PHY,LSTM,PERS')
    ap.add_argument('--ttrain', default='2016-01-01,2017-12-31')
    ap.add_argument('--tval', default='2018-01-01,2018-12-31')
    ap.add_argument('--ttest', default='2019-01-01,2019-12-31')
    a = ap.parse_args()

    P('torch %s  threads=%d' % (torch.__version__, torch.get_num_threads()))
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

    # Standardisation (training period only)
    # Important: the first two columns (log1p(precipitation) and PET) must keep original units,
    #       the model must use expm1 recover the true mm/day; after standardisation, applying expm1 would give a wrong order of magnitude.
    m = A[tr_sl].reshape(-1, A.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    mu[:4] = 0.0        # P_log / PET / por1 / por2 keep original units
    sd[:4] = 1.0
    A = (A - mu) / sd
    P('Standardisation complete (training-period statistics);P_log/PET/por1/por2 four columns keep original units')

    allres = []
    for cfg in a.cfgs.split(','):
        P('')
        P('================ %s ================' % cfg)
        r, _ = run(cfg, A, Y, mu, sd, tr_sl, va_sl, te_sl, a)
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
    P('--- by layer (10cm / 30cm)all-step RMSE ---')
    for r in allres:
        if 'test_l1' in r:
            P('  %-6s 10cm %.5f   30cm %.5f' % (r['cfg'], r['test_l1']['all'], r['test_l2']['all']))

    op = os.path.join(BENCH, 'dl_smoke_result.json')
    json.dump(allres, open(op, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    P('WROTE %s' % op)
    lp = os.path.join(LOG, '_12_dl_smoke.txt')
    open(lp, 'w', encoding='utf-8').write('\n'.join(LOG))
    P('WROTE log %s' % lp)


if __name__ == '__main__':
    main()
