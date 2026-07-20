# export_demap.py — phone factory demappers. NC8 = the real-finetuned ft_v31; NC12 =
# fresh synthetic pretrain (no real NC12 footage yet). Raw fp32 in state_dict order
# (conv1 w/b, conv2 w/b, fc1 w/b, fc2 w/b) — the Swift Demap reader mirrors it.
# Run:  V3A=40 V3LV=2 V3NC=12 python export_demap.py
import os, time
import numpy as np
import torch
import train_real as TR

DIR = TR.DIR
assert TR.M == 2 and TR.NC == 12, "run with V3A=40 V3LV=2 V3NC=12"

torch.manual_seed(0); np.random.seed(0)
net12 = TR.Demapper().to("cuda")
sb = TR.synth_batcher(600)
t0 = time.time()
opt = torch.optim.Adam(net12.parameters(), 1e-3)
for s in range(1, 15001):
    x8, y, c = sb(512)
    loss = torch.nn.functional.cross_entropy(
        net12(TR.aug(x8, c)).reshape(-1, 2), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
sve = TR.synth_batcher(60, seed=123)
with torch.no_grad():
    net12.eval()
    x8, y, _ = sve(8192)
    ser = (net12(TR.center(x8)).argmax(-1) != y).float().mean().item()
print(f"nc12 pretrain: {time.time() - t0:.0f}s, synth SER {ser * 100:.3f}%")
torch.save(net12.state_dict(), f"{DIR}/pre_v31_nc12.pt")

def export(sd, path):
    with open(path, "wb") as f:
        for k in ["f.0.weight", "f.0.bias", "f.3.weight", "f.3.bias",
                  "f.7.weight", "f.7.bias", "f.9.weight", "f.9.bias"]:
            f.write(sd[k].cpu().float().numpy().tobytes())
    print(path, os.path.getsize(path), "bytes")

export(net12.state_dict(), f"{DIR}/demap16_nc12.bin")
export(torch.load(f"{DIR}/ft_v31.pt"), f"{DIR}/demap16_nc8.bin")
