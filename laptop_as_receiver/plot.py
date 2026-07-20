# plot.py <model_name> — per-rung validation error over time from the training log.
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glyphs import LADDER

name = sys.argv[1]
import os
import config
p = f"models/{config.MVER}/{name}_log.csv"
if not os.path.exists(p): p = f"models/{config.MPREV}/{name}_log.csv"
d = np.genfromtxt(p, delimiter=",", names=True)
t = (d["t"] - d["t"][0]) / 60
plt.figure(figsize=(9, 5))
for px in LADDER:
    plt.plot(t, np.maximum(1 - d[f"acc{px}"], 1e-4), label=f"{px}px")
plt.yscale("log"); plt.xlabel("minutes"); plt.ylabel("val error (1 - joint acc)")
plt.legend(); plt.grid(True, alpha=0.3); plt.title(f"{name}: error vs time per rung")
plt.savefig(p.replace("_log.csv", "_err.png"), dpi=110, bbox_inches="tight")
print(f"models/{name}_err.png")
