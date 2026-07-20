# integrate_test.py — is the channel noise TEMPORAL? Compare linear matched-filter
# stats on the same idx harvested two ways: best single shot (data/4k) vs the mean of
# the dwell's clean shots (data/4kavg). If averaging helps, the live receiver can
# integrate its ~6 shots per dwell — a free SNR lever.
import numpy as np
import torch
import harvest as HV

def stats(dirname, idxs):
    bas = torch.as_tensor(HV.BAS, device="cuda")
    bn = (bas * bas).sum((1, 2))
    meas, true = [], []
    for seg, idx in idxs:
        f = np.load(f"data/{dirname}/s{seg}f{idx:05d}.npy")[:, :, ::-1].copy()
        fr = torch.as_tensor(f, device="cuda").float()
        cells = torch.stack([fr[16 + r * 16:32 + r * 16, 16 + c * 16:32 + c * 16]
                             for r, c in HV.RC])
        cells = cells - cells.mean((1, 2), keepdim=True)
        meas.append((torch.einsum("kyxc,jyx->kcj", cells, bas) / bn[None, None, :])
                    .cpu().numpy())
        true.append(HV.LEVELS[HV.pays(idx)])
    meas, true = np.concatenate(meas), np.concatenate(true)
    out = []
    for j in range(HV.NC):
        g = (meas[:, :, j] * true[:, :, j]).sum() / (true[:, :, j] ** 2).sum()
        sig = (meas[:, :, j] - g * true[:, :, j]).std() / abs(g)
        i3 = np.abs(true[:, :, j]) == 3
        se3 = (np.sign(meas[:, :, j]) != np.sign(true[:, :, j]))[i3].mean()
        out.append((sig, se3))
    return out

rows = [l.split(",") for l in open("data/4kavg/meta.csv").read().splitlines()[1:]]
shared = [(int(r[0]), int(r[1])) for r in rows]
shared = [(0, i) for _, i in shared]          # best-shot set is all seg 0
a = stats("4k", shared)
b = stats("4kavg", shared)
print(f"{len(shared)} shared idx   BEST-SHOT          DWELL-MEAN")
print("coeff        sigma  signerr@3      sigma  signerr@3")
for j in range(HV.NC):
    print(f"  {j:2d}      {a[j][0]:7.3f}  {a[j][1] * 100:7.2f}%   {b[j][0]:7.3f}  "
          f"{b[j][1] * 100:7.2f}%")
