# train_real.py — the v3 verdict machine: can a demapper learn the REAL camera channel?
# Real data = harvested 1024x1024 warps (harvest.py) with seeded truth labels; the
# whole dataset lives on the GPU (uint8) and crops are vectorized gathers.
# Synthetic data = web-exact rendered fields through a channel sim (blur/noise/
# gradient/crosstalk at logical-px scale) for pretraining / domain-gap measurement.
# Results -> results_real.txt (appended).      Run:  python train_real.py
import os, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import harvest as HV

DEV = "cuda"
DIR = os.path.dirname(os.path.abspath(__file__))
NP_, NC, M = len(HV.PAY), HV.NC, len(HV.LEVELS)   # M: PAM order (env V3LV)
torch.backends.cudnn.benchmark = True

def grids(y0, x0):                           # (K,24,24) gather index grids
    ar = torch.arange(24)
    YY = torch.as_tensor(y0)[:, None, None] + ar[None, :, None]
    XX = torch.as_tensor(x0)[:, None, None] + ar[None, None, :]
    return YY.to(DEV), XX.to(DEV)

RYY, RXX = grids(16 + HV.RC[:, 0] * 16 - 4, 16 + HV.RC[:, 1] * 16 - 4)

class Demapper(nn.Module):                   # pictrain/test-v3 arch, k=20 input
    def __init__(self, w=1):
        super().__init__()
        c1, c2, fc = 32 * w, 64 * w, 512 * w
        self.f = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(c2 * 25, fc), nn.ReLU(),
            nn.Linear(fc, 3 * NC * M))
    def forward(self, x):
        return self.f(x).view(-1, 3 * NC, M)

# ---- data on the GPU ----------------------------------------------------------------
def load_real(tag):
    rows = [l.split(",") for l in open(f"{DIR}/data/{tag}/meta.csv").read().splitlines()[1:]]
    fields = torch.empty(len(rows), 1024, 1024, 3, dtype=torch.uint8, device=DEV)
    labels = torch.empty(len(rows), NP_, 3 * NC, dtype=torch.uint8, device=DEV)
    idxs = np.zeros(len(rows), int)
    for i, r in enumerate(rows):
        seg, idx = int(r[0]), int(r[1])
        f = np.load(f"{DIR}/data/{tag}/s{seg}f{idx:05d}.npy")
        fields[i] = torch.from_numpy(f[:, :, ::-1].copy()).to(DEV)      # BGR -> RGB
        labels[i] = torch.from_numpy(HV.pays(idx).reshape(NP_, -1)).to(DEV)
        idxs[i] = idx
    print(f"{tag}: {len(rows)} frames on GPU "
          f"({fields.numel() / 1e9:.1f} GB)", flush=True)
    return fields, labels, idxs

def crops_of(fields, f, c, YY=RYY, XX=RXX):  # (B,24,24,3) uint8 gather on GPU
    return fields[f[:, None, None], YY[c], XX[c]]

CS = None                                    # (NP,2) per-cell shift correction (the
                                             # measured lens-distortion field), or None
def sample20(x8, off):                       # crop24 -> 20x20 sampled at +off px
    x = x8.float().permute(0, 3, 1, 2)
    B = x.shape[0]
    base = torch.linspace(-9.5, 9.5, 20, device=DEV)
    gy = (base[None, :, None] + off[:, 1, None, None]) / 12
    gx = (base[None, None, :] + off[:, 0, None, None]) / 12
    grid = torch.stack([gx.expand(B, 20, 20), gy.expand(B, 20, 20)], -1)
    return F.grid_sample(x, grid, align_corners=False)

def aug(x8, c=None, sub=0.4, bright=0.3, noise=4.0, jitter=None):
    B = x8.shape[0]
    if jitter is not None:                   # glyph-style INTEGER jitter (ablation)
        ox, oy = np.random.randint(0, 2 * jitter + 1, 2)
        x = x8.float().permute(0, 3, 1, 2)[:, :, oy:oy + 20, ox:ox + 20]
    else:                                    # sub-pixel shift +-sub px, center 20x20
        off = (torch.rand(B, 2, device=DEV) * 2 - 1) * sub
        if CS is not None and c is not None: off = off + CS[c]
        x = sample20(x8, off)
    if bright: x = x * (1 + (torch.rand(B, 3, 1, 1, device=DEV) * 2 - 1) * bright)
    if noise: x = x + torch.randn_like(x) * noise
    return x.div(255).sub(0.5)

def center(x8, c=None):
    if CS is not None and c is not None:
        return sample20(x8, CS[c]).div(255).sub(0.5)
    return x8.float().permute(0, 3, 1, 2)[:, :, 2:22, 2:22].div(255).sub(0.5)

@torch.no_grad()
def eval_real(net, fields, labels, which):
    net.eval()
    err = torch.zeros(3 * NC, device=DEV)
    n = 0
    fidx = torch.arange(NP_, device=DEV)
    for f in which:
        x = center(crops_of(fields, torch.full((NP_,), int(f), device=DEV), fidx), fidx)
        y = labels[f].long()
        for k0 in range(0, NP_, 8192):
            pred = net(x[k0:k0 + 8192]).argmax(-1)
            err += (pred != y[k0:k0 + 8192]).float().sum(0)
        n += NP_
    net.train()
    return (err / n).view(3, NC).cpu().numpy()

def real_batcher(fields, labels, frames):
    fr = torch.as_tensor(np.asarray(frames), device=DEV)
    def get(bs):
        f = fr[torch.randint(0, len(fr), (bs,), device=DEV)]
        c = torch.randint(0, NP_, (bs,), device=DEV)
        return crops_of(fields, f, c), labels[f, c].long(), c
    return get

# ---- synthetic (web-exact render + channel sim at logical scale) --------------------
def synth_batcher(nfields=600, cells=24, seed=0):
    rng = np.random.default_rng(seed)
    W = cells * 16
    fs = torch.empty(nfields, W, W, 3, dtype=torch.uint8, device=DEV)
    ys = torch.empty(nfields, cells * cells, 3 * NC, dtype=torch.uint8, device=DEV)
    for i in range(nfields):
        sym = rng.integers(0, M, (cells * cells, 3, NC))
        blocks = 128 + np.einsum("kcj,jyx->kyxc", HV.LEVELS[sym], HV.BAS)
        f = np.zeros((W, W, 3), np.float32)
        for k in range(cells * cells):
            r, c = k // cells, k % cells
            f[r * 16:r * 16 + 16, c * 16:c * 16 + 16] = blocks[k]
        f = cv2.GaussianBlur(np.clip(f, 0, 255), (0, 0), rng.uniform(0.4, 1.1))
        gy, gx = np.linspace(-1, 1, W)[:, None], np.linspace(-1, 1, W)[None, :]
        ang = rng.uniform(0, 2 * np.pi)
        f *= (1 + 0.08 * (np.cos(ang) * gx + np.sin(ang) * gy))[..., None]
        xt = np.eye(3) * rng.uniform(0.86, 0.94) + rng.uniform(0.02, 0.07)
        f = f @ (xt / xt.sum(1, keepdims=True)).T
        f += rng.normal(0, rng.uniform(2, 5), f.shape)
        fs[i] = torch.from_numpy(np.clip(f, 0, 255).astype(np.uint8)).to(DEV)
        ys[i] = torch.from_numpy(sym.reshape(cells * cells, -1).astype(np.uint8)).to(DEV)
    inner = np.array([r * cells + c for r in range(1, cells - 1)
                      for c in range(1, cells - 1)])
    YY, XX = grids((inner // cells) * 16 - 4, (inner % cells) * 16 - 4)
    ki = torch.arange(len(inner), device=DEV)
    innert = torch.as_tensor(inner, device=DEV)
    def get(bs):
        f = torch.randint(0, nfields, (bs,), device=DEV)
        k = ki[torch.randint(0, len(inner), (bs,), device=DEV)]
        return fs[f[:, None, None], YY[k], XX[k]], ys[f, innert[k]].long(), None
    return get

# ---- shared loop --------------------------------------------------------------------
def train(net, get_batch, steps, lr_fn, log, tag, evalf=None, every=2000,
          bs=512, augkw=dict()):
    opt = torch.optim.Adam(net.parameters(), lr_fn(0))
    t0 = time.time()
    for s in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = lr_fn(s)
        x8, y, c = get_batch(bs)
        loss = F.cross_entropy(net(aug(x8, c, **augkw)).reshape(-1, M), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if evalf and s % every == 0:
            log(f"    {tag} step {s:6d}: val SER {evalf(net).mean() * 100:6.3f}%  "
                f"({time.time() - t0:.0f}s)")

def fmt_coeffs(ser):
    return " ".join(f"{s * 100:.1f}" for s in ser.mean(0))

def main():
    def log(s):
        print(s, flush=True)
        open(f"{DIR}/results_real.txt", "a", encoding="utf-8").write(s + "\n")
    log(f"\n==== run {time.strftime('%Y-%m-%d %H:%M')} ====")
    fields, labels, idxs = load_real("4k")
    val = np.where(idxs % 7 == 0)[0]
    trn = np.where(idxs % 7 != 0)[0]
    log(f"4k: {len(trn)} train / {len(val)} val frames, {NP_} cells each")
    ev = lambda net: eval_real(net, fields, labels, val[::4])
    ev_full = lambda net: eval_real(net, fields, labels, val)

    # 1. synthetic pretrain (channel-sim) + zero-shot on real
    torch.manual_seed(0); np.random.seed(0)
    net = Demapper().to(DEV)
    train(net, synth_batcher(), 15000, lambda s: 1e-3, log, "pretrain")
    ser = ev_full(net)
    log(f"[1] synth-pretrained zero-shot on real: SER {ser.mean() * 100:.2f}%  "
        f"coeffs {fmt_coeffs(ser)}")
    torch.save(net.state_dict(), f"{DIR}/pre.pt")

    # 2. finetune on ALL real frames
    lr = lambda s: 1e-3 if s < 12000 else 1e-4
    rb = real_batcher(fields, labels, trn)
    train(net, rb, 16000, lr, log, "ft-all", ev)
    ser = ev_full(net)
    log(f"[2] pretrain + finetune ALL ({len(trn)}f): SER {ser.mean() * 100:.3f}%  "
        f"ch R/G/B {' '.join(f'{s * 100:.2f}' for s in ser.mean(1))}  "
        f"coeffs {fmt_coeffs(ser)}")
    torch.save(net.state_dict(), f"{DIR}/ft_all.pt")

    # 3. real-only from scratch
    torch.manual_seed(0)
    net2 = Demapper().to(DEV)
    train(net2, rb, 16000, lr, log, "scratch", ev)
    log(f"[3] real-only from scratch: SER {ev_full(net2).mean() * 100:.3f}%")

    # 4. data-efficiency curve (pretrained init)
    for nfr in (5, 20, 50, 100, 300):
        torch.manual_seed(0)
        net3 = Demapper().to(DEV)
        net3.load_state_dict(torch.load(f"{DIR}/pre.pt"))
        sub = trn[np.linspace(0, len(trn) - 1, nfr).astype(int)]
        train(net3, real_batcher(fields, labels, sub), 8000,
              lambda s: 1e-3 if s < 6000 else 1e-4, log, f"ft-{nfr}")
        log(f"[4] finetune {nfr:4d} frames: SER {ev_full(net3).mean() * 100:.3f}%")

    # 5. lr variants (pretrained, 100 frames — the phone-session-like regime)
    sub = trn[np.linspace(0, len(trn) - 1, 100).astype(int)]
    for name, lrf in [("const 1e-3", lambda s: 1e-3),
                      ("cosine 1e-3", lambda s: 1e-3 * 0.5 * (1 + np.cos(np.pi * s / 8000))),
                      ("phone tier 1e-4", lambda s: 1e-4),
                      ("hot 3e-4", lambda s: 3e-4)]:
        torch.manual_seed(0)
        net4 = Demapper().to(DEV)
        net4.load_state_dict(torch.load(f"{DIR}/pre.pt"))
        train(net4, real_batcher(fields, labels, sub), 8000, lrf, log, name)
        log(f"[5] lr {name}: SER {ev_full(net4).mean() * 100:.3f}%")

    # 6. augmentation ablation (pretrained, all frames)
    for name, augkw in [("none", dict(sub=0, bright=0, noise=0)),
                        ("subpix only", dict(sub=0.4, bright=0, noise=0)),
                        ("all (default)", dict()),
                        ("INTEGER jitter 2px", dict(jitter=2))]:
        torch.manual_seed(0)
        net5 = Demapper().to(DEV)
        net5.load_state_dict(torch.load(f"{DIR}/pre.pt"))
        train(net5, rb, 10000, lr, log, f"aug-{name}", augkw=augkw)
        log(f"[6] aug {name}: SER {ev_full(net5).mean() * 100:.3f}%")

    # 7. width x2
    torch.manual_seed(0)
    net6 = Demapper(w=2).to(DEV)
    train(net6, rb, 16000, lr, log, "w2-scratch", ev)
    log(f"[7] width x2 (scratch, all): SER {ev_full(net6).mean() * 100:.3f}%")

    # 8. 1080p: 4k-model zero-shot, then finetune
    if os.path.exists(f"{DIR}/data/1080p/meta.csv"):
        f2, l2, i2 = load_real("1080p")
        v2 = np.where(i2 % 7 == 0)[0]; t2 = np.where(i2 % 7 != 0)[0]
        net = Demapper().to(DEV)
        net.load_state_dict(torch.load(f"{DIR}/ft_all.pt"))
        log(f"[8a] 4k-model on 1080p zero-shot: "
            f"SER {eval_real(net, f2, l2, v2).mean() * 100:.2f}%")
        train(net, real_batcher(f2, l2, t2), 10000, lr, log, "ft-1080p",
              lambda n: eval_real(n, f2, l2, v2[::4]))
        ser = eval_real(net, f2, l2, v2)
        log(f"[8b] finetuned on 1080p: SER {ser.mean() * 100:.3f}%  "
            f"coeffs {fmt_coeffs(ser)}")

if __name__ == "__main__":
    main()
