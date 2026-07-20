# analyze.py — measure the REAL channel per coefficient and compute what wire design
# would survive it. The PAM-4 data is a built-in amplitude sweep: error rates
# conditioned on |level| 1 vs 3 give the noise CDF at two amplitudes; a linear
# matched-filter estimate per coefficient gives gain g and noise sigma (in LEVEL
# units), from which achievable constellations (PAM-2/4/8 at fixed peak +-3A') and
# throughput follow. Also runs the trained NN per-|level| for the learned bound.
# Run:  python analyze.py [4k|1080p]
import math, os, sys
import numpy as np
import torch
import harvest as HV
import train_real as TR

DIR = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "4k"
fields, labels, idxs = TR.load_real(tag)
val = np.where(idxs % 7 == 0)[0]
NPc, NC = TR.NP_, TR.NC
Q = lambda x: 0.5 * math.erfc(x / math.sqrt(2))

# ---- linear matched filter on exact 16x16 cells -------------------------------------
bas = torch.as_tensor(HV.BAS, device="cuda")            # (NC,16,16) includes A
bnorm = (bas * bas).sum((1, 2))                          # per-pattern energy
meas, true = [], []
for f in val:
    fr = fields[f].float()                               # (1024,1024,3) RGB
    cells = torch.stack([fr[16 + r * 16:32 + r * 16, 16 + c * 16:32 + c * 16]
                         for r, c in HV.RC])             # (NP,16,16,3)
    cells = cells - cells.mean((1, 2), keepdim=True)     # kill DC/illumination
    m = torch.einsum("kyxc,jyx->kcj", cells, bas) / bnorm[None, None, :]
    meas.append(m.cpu().numpy())
    true.append(HV.LEVELS[labels[f].cpu().numpy().reshape(NPc, 3, NC)])
meas = np.concatenate(meas); true = np.concatenate(true)  # (N,3,NC) in level units

print(f"\n==== {tag}: linear matched-filter channel measurement "
      f"({len(val)} val frames) ====")
print("coeff |  gain  sigma(lvl)  SER_lin  |  per-|level| err:  |1|      |3|")
rows = []
for j in range(NC):
    g = (meas[:, :, j] * true[:, :, j]).sum() / (true[:, :, j] ** 2).sum()
    n = meas[:, :, j] - g * true[:, :, j]
    sig = n.std() / abs(g)                               # noise in level units
    eq = meas[:, :, j] / g
    pred = np.abs(eq[..., None] - np.array([-3., -1., 1., 3.])).argmin(-1)
    tl = ((true[:, :, j] + 3) / 2).astype(int)
    ser = (pred != tl).mean()
    inner = np.abs(true[:, :, j]) == 1
    e1, e3 = (pred != tl)[inner].mean(), (pred != tl)[~inner].mean()
    rows.append((g, sig, ser, e1, e3))
    print(f"  {j:2d}  | {g:6.3f}  {sig:8.3f}  {ser * 100:6.2f}%  |"
          f"          {e1 * 100:6.2f}%  {e3 * 100:6.2f}%")

# ---- NN per-|level| (the learned bound) --------------------------------------------
if os.path.exists(f"{DIR}/ft_all.pt"):
    net = TR.Demapper().to("cuda")
    net.load_state_dict(torch.load(f"{DIR}/ft_all.pt"))
    net.eval()
    errs = np.zeros((2, NC)); cnts = np.zeros((2, NC))
    with torch.no_grad():
        for f in val:
            fidx = torch.arange(NPc, device="cuda")
            x = TR.center(TR.crops_of(fields, torch.full((NPc,), int(f), device="cuda"),
                                      fidx))
            y = labels[f].long()
            pred = net(x).argmax(-1)
            wrong = (pred != y).cpu().numpy().reshape(NPc, 3, NC)
            lv = np.abs(HV.LEVELS[y.cpu().numpy()]).reshape(NPc, 3, NC)
            for a, l in ((0, 1.0), (1, 3.0)):
                sel = lv == l
                errs[a] += (wrong & sel).sum((0, 1)); cnts[a] += sel.sum((0, 1))
    print("\nNN per-|level| SER by coeff:")
    print("  coeff  |1|-err   |3|-err")
    for j in range(NC):
        print(f"   {j:2d}   {errs[0, j] / cnts[0, j] * 100:7.2f}%  "
              f"{errs[1, j] / cnts[1, j] * 100:7.2f}%")

# ---- EMPIRICAL PAM-2: sign-error rates, no noise model needed -----------------------
# P(sign flip | |level|=3) IS a PAM-2 wire at today's peak amplitude, measured on the
# real channel; | |level|=1 ) is PAM-2 at a third of it. Linear and (below) NN.
print("\n==== empirical PAM-2 (sign) error rates, linear matched filter ====")
print("coeff   sign-err@|3|   sign-err@|1|")
for j in range(NC):
    sgn = np.sign(meas[:, :, j])
    ok = sgn == np.sign(true[:, :, j])
    i3 = np.abs(true[:, :, j]) == 3
    print(f"  {j:2d}    {(~ok)[i3].mean() * 100:7.2f}%     {(~ok)[~i3].mean() * 100:7.2f}%")

# ---- wire redesign calculator -------------------------------------------------------
# fixed PEAK amplitude 3*A' (A' = amplitude scale vs today's A=20). PAM-M spacing
# 6A'/(M-1) in today's level units = 2A'... noise sigma measured in level units at
# A=20; scaling amplitude by s scales sigma_rel by 1/s. SER_M ~ 2(1-1/M) Q(d/2sigma).
print("\n==== achievable bits/cell vs amplitude scale (target SER 2%/coeff) ====")
sigs = np.array([r[1] for r in rows])
for s in (1.0, 1.5, 2.0):
    tot = 0
    detail = []
    for j in range(NC):
        sig = sigs[j] / s
        bits = 0
        for M, b in ((8, 3), (4, 2), (2, 1)):
            d = 6.0 / (M - 1)                            # spacing, level units (peak 3)
            if 2 * (1 - 1 / M) * Q(d / 2 / sig) < 0.02: bits = b; break
        detail.append(bits)
        tot += bits * 3                                  # 3 color channels
    kbs30 = TR.NP_ * tot / 8 * 30 / 1024
    print(f"  A x{s:.1f}: bits/coeff {detail}  -> {tot} bits/cell "
          f"-> {kbs30:7.1f} KB/s @30fps raw (pre-RS)")
print("reference: glyph wire record 126.6 KB/s goodput; glyph cell = 6 raw bits")
