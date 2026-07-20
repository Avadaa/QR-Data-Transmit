# run_v31.py — the v3.1 validation round: PAM-2, NC=8, A=40 (the redesign the round-1
# channel measurements pointed at). Bootstrap a PC-trained demapper on web-exact
# synthetic fields (channel sim; sub-pixel shifts, NO integer jitter), then fine-tune
# on the phone footage; measure the distortion field and A/B the piecewise-geometry
# correction; linear baseline; wire throughput verdict.
# Run:  V3A=40 V3LV=2 V3NC=8 python run_v31.py     -> results_v31.txt
import os, time
import numpy as np
import torch
import harvest as HV
import train_real as TR

assert TR.M == 2 and TR.NC == 8 and HV.A == 40.0, "set V3A=40 V3LV=2 V3NC=8"
DEV, DIR = "cuda", TR.DIR
NP_ = TR.NP_

def log(s):
    print(s, flush=True)
    open(f"{DIR}/results_v31.txt", "a", encoding="utf-8").write(s + "\n")
log(f"\n==== v3.1 run {time.strftime('%Y-%m-%d %H:%M')} (PAM-2 NC8 A40) ====")

SKIP = os.path.exists(f"{DIR}/ft_v31.pt")   # resume after the [1]-[5] steps completed

# ---- 1. PC bootstrap: synthetic pretrain, convergence to sub-1% -----------------------
torch.manual_seed(0); np.random.seed(0)
net = TR.Demapper().to(DEV)
if SKIP: net.load_state_dict(torch.load(f"{DIR}/pre_v31.pt"))
sb = TR.synth_batcher(600)
sve = TR.synth_batcher(80, seed=123)
@torch.no_grad()
def synth_eval(n):
    n.eval(); wrong = tot = 0
    for _ in range(12):
        x8, y, _ = sve(4096)
        wrong += (n(TR.center(x8)).argmax(-1) != y).float().sum().item()
        tot += y.numel()
    n.train(); return wrong / tot
opt = torch.optim.Adam(net.parameters(), 1e-3)
t0, hit1, hit02 = time.time(), None, None
for s in ([] if SKIP else range(1, 15001)):
    x8, y, c = sb(512)
    loss = torch.nn.functional.cross_entropy(
        net(TR.aug(x8, c)).reshape(-1, TR.M), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 1000 == 0:
        e = synth_eval(net)
        if e < 0.01 and hit1 is None: hit1 = s
        if e < 0.002 and hit02 is None: hit02 = s
        log(f"    pretrain step {s:6d}: synth SER {e * 100:6.3f}%  ({time.time() - t0:.0f}s)")
if not SKIP:
    log(f"[1] PC pretrain: sub-1% at {hit1} steps, sub-0.2% at {hit02} steps")
    torch.save(net.state_dict(), f"{DIR}/pre_v31.pt")

# ---- 2. real data: zero-shot, fine-tune curve, floor ----------------------------------
fields, labels, idxs = TR.load_real("v31")
val = np.where(idxs % 7 == 0)[0]
trn = np.where(idxs % 7 != 0)[0]
log(f"v31: {len(trn)} train / {len(val)} val frames")
ev_full = lambda n: TR.eval_real(n, fields, labels, val)
if not SKIP:
    ser = ev_full(net)
    log(f"[2] zero-shot on real: SER {ser.mean() * 100:.3f}%  coeffs {TR.fmt_coeffs(ser)}")

AUG = dict(sub=0.4, bright=0.15, noise=0)    # round-1 lesson: no synthetic noise on real
lr = lambda s: 1e-3 if s < 8000 else 1e-4
for nfr in ([] if SKIP else (5, 20, 50, 100, 300)):
    torch.manual_seed(0)
    n3 = TR.Demapper().to(DEV)
    n3.load_state_dict(torch.load(f"{DIR}/pre_v31.pt"))
    sub = trn[np.linspace(0, len(trn) - 1, min(nfr, len(trn))).astype(int)]
    TR.train(n3, TR.real_batcher(fields, labels, sub), 8000, lr, log, f"ft-{nfr}",
             augkw=AUG)
    log(f"[3] finetune {nfr:4d} frames: SER {ev_full(n3).mean() * 100:.3f}%")

torch.manual_seed(0)
netf = TR.Demapper().to(DEV)
netf.load_state_dict(torch.load(f"{DIR}/ft_v31.pt" if SKIP else f"{DIR}/pre_v31.pt"))
rb = TR.real_batcher(fields, labels, trn)
if not SKIP: TR.train(netf, rb, 16000, lambda s: 1e-3 if s < 12000 else 1e-4, log, "ft-all",
         lambda n: TR.eval_real(n, fields, labels, val[::4]), augkw=AUG)
if not SKIP:
    ser = ev_full(netf)
    log(f"[4] finetune ALL: SER {ser.mean() * 100:.3f}%  "
        f"ch R/G/B {' '.join(f'{s * 100:.2f}' for s in ser.mean(1))}  "
        f"coeffs {TR.fmt_coeffs(ser)}")
    torch.save(netf.state_dict(), f"{DIR}/ft_v31.pt")
    torch.manual_seed(0)
    nsc = TR.Demapper().to(DEV)
    TR.train(nsc, rb, 16000, lambda s: 1e-3 if s < 12000 else 1e-4, log, "scratch",
             augkw=AUG)
    log(f"[5] real-only scratch: SER {ev_full(nsc).mean() * 100:.3f}%")

# ---- 3. distortion field: per-cell shift by truth-NCC, then A/B ----------------------
log("estimating per-cell distortion field...")
sample = val[np.linspace(0, len(val) - 1, 24).astype(int)]
offs = [(dx, dy) for dx in np.arange(-1.5, 1.51, 0.25) for dy in np.arange(-1.5, 1.51, 0.25)]
fidx = torch.arange(NP_, device=DEV)
votes = np.zeros((len(sample), NP_, 2))
for si, f in enumerate(sample):
    tf = HV.truth_field(idxs[f])
    tc = np.stack([np.pad(tf, ((8, 8), (8, 8), (0, 0)), "edge")
                   [16 + r * 16 - 2 + 8:16 + r * 16 + 18 + 8,
                    16 + c * 16 - 2 + 8:16 + c * 16 + 18 + 8] for r, c in HV.RC])
    tct = torch.as_tensor(tc, device=DEV).float().permute(0, 3, 1, 2)
    tct = tct - tct.mean((2, 3), keepdim=True)
    tn = tct / (tct.norm(dim=(2, 3), keepdim=True) + 1e-6)
    x8 = TR.crops_of(fields, torch.full((NP_,), int(f), device=DEV), fidx)
    best = torch.full((NP_,), -9.0, device=DEV)
    barg = torch.zeros(NP_, 2, device=DEV)
    for dx, dy in offs:
        off = torch.tensor([dx, dy], device=DEV).float().expand(NP_, 2)
        w = TR.sample20(x8, off)
        w = w - w.mean((2, 3), keepdim=True)
        w = w / (w.norm(dim=(2, 3), keepdim=True) + 1e-6)
        ncc = (w * tn).sum((1, 2, 3)) / 3
        upd = ncc > best
        best = torch.where(upd, ncc, best)
        barg[upd] = torch.tensor([dx, dy], device=DEV).float()
    votes[si] = barg.cpu().numpy()
cs = np.median(votes, 0)                     # (NP,2) robust per-cell shift
G = np.zeros((62, 62, 2))
for k, (r, c) in enumerate(HV.RC): G[r, c] = cs[k]
import cv2
G = cv2.blur(G, (3, 3))                      # smooth the field (single-cell NCC noise)
cs = np.stack([G[r, c] for r, c in HV.RC])
np.save(f"{DIR}/data/v31/cellshift.npy", cs)
mag = np.hypot(cs[:, 0], cs[:, 1])
log(f"[6] distortion field: |shift| mean {mag.mean():.2f} px  p90 {np.percentile(mag, 90):.2f}  max {mag.max():.2f}")

TR.CS = torch.as_tensor(cs, device=DEV).float()
ser = ev_full(netf)
log(f"[7] ft-all + shift-corrected EVAL: SER {ser.mean() * 100:.3f}%")
torch.manual_seed(0)
ncs = TR.Demapper().to(DEV)
ncs.load_state_dict(torch.load(f"{DIR}/pre_v31.pt"))
TR.train(ncs, rb, 16000, lambda s: 1e-3 if s < 12000 else 1e-4, log, "ft-CS",
         lambda n: TR.eval_real(n, fields, labels, val[::4]), augkw=AUG)
ser_cs = ev_full(ncs)
log(f"[8] finetune WITH shift correction: SER {ser_cs.mean() * 100:.3f}%  "
    f"coeffs {TR.fmt_coeffs(ser_cs)}")
torch.save(ncs.state_dict(), f"{DIR}/ft_v31_cs.pt")

# spatial map of the corrected model
err = torch.zeros(NP_, device=DEV); n3c = torch.zeros(NP_, device=DEV)
ncs.eval()
with torch.no_grad():
    for f in val[::2]:
        x = TR.center(TR.crops_of(fields, torch.full((NP_,), int(f), device=DEV), fidx), fidx)
        wrong = (ncs(x).argmax(-1) != labels[f].long()).float().mean(1)
        err += wrong; n3c += 1
rate = (err / n3c).cpu().numpy()
Gm = np.full((62, 62), np.nan)
for k, (r, c) in enumerate(HV.RC): Gm[r, c] = rate[k]
m8 = np.array([[np.nanmean(Gm[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]) for j in range(8)]
               for i in range(8)])
log("[9] spatial SER map (8x8 blocks, %):")
for row in m8: log("    " + " ".join(f"{v * 100:5.2f}" for v in row))

# ---- 4. wire verdict ------------------------------------------------------------------
per = ser_cs.mean(0)                         # per-coeff BER (PAM-2: SER == BER)
log(f"[10] per-coeff BER (best model): {TR.fmt_coeffs(ser_cs)}")
for target, code in ((0.02, 0.83), (0.05, 0.71)):
    use = per < target
    raw = int(use.sum()) * 3
    net_b = raw * code
    for fps in (15, 30):
        log(f"     BER<{target:.0%} coeffs: {int(use.sum())}/8 -> {raw} raw b/cell "
            f"x r{code} -> {NP_ * net_b / 8 * fps / 1024:6.1f} KB/s net @{fps}fps")
log("     reference: glyph record 126.6 KB/s; v2 web grid @15fps nsym32 ~ 41 KB/s")
