# pictrain.py — picture-trained bootstraps for all four wires, phone-pipeline-exact:
# frames arrive in BATCHES OF 20, each frame harvests 1200 random tile crops into a
# 20k reservoir (10% to val), every batch triggers one chunk = 120 steps x bs 256
# with the receiver's augmentation (0..4 px jitter crop, per-channel brightness
# 0.7-1.3x, sigma-4 noise) and the receiver's lr tiers (fresh Adam per chunk:
# err>=3% 1e-4, >=1% 5e-5, >=0.5% 2e-5, else 1e-5). Val = clean center crop, no aug
# (the phone's terr). Direct image inputs — rendered frames, no camera.
# Models: v2 glyph nets (k=16 native, k=12 native) and v3 DMT demappers (px 16, 12).
# Run:  python pictrain.py            -> results.txt
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transmitter"))
from glyphs import PALETTE, shape_mask

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_STEPS, TRAIN_BS, BATCH_FRAMES = 120, 256, 20
HARVEST, RES_N, VAL_N = 1200, 20000, 3000
TARGET, MAX_CHUNKS = 0.002, 60
NC, A, LEVELS = 20, 20.0, np.array([-3.0, -1.0, 1.0, 3.0])

def lr_tier(err):             # the receiver's runChunk tiers, verbatim
    return 1e-4 if err is None or err >= 0.03 else 5e-5 if err >= 0.01 \
        else 2e-5 if err >= 0.005 else 1e-5

# ---- frame renderers (direct images) ------------------------------------------------
def render_v2(rng, px, cols, rows):   # glyph field + labels (shape*4+color per tile)
    shp = rng.integers(0, 16, rows * cols)
    col = rng.integers(0, 4, rows * cols)
    img = np.zeros((rows * px + 16, cols * px + 16, 3), np.float32)
    for t in range(rows * cols):
        m = shape_mask(int(shp[t]), px)
        r, c = t // cols, t % cols
        img[8 + r * px:8 + (r + 1) * px, 8 + c * px:8 + (c + 1) * px] = \
            m[..., None] * PALETTE[col[t]]
    return img, (shp * 4 + col).astype(np.int64)

_DCT = {}
def zz_basis(px):             # NC per-coefficient IDCT patterns (like the web page)
    if px not in _DCT:
        D = np.zeros((px, px))
        for k in range(px):
            for i in range(px):
                D[k, i] = np.sqrt((1 if k == 0 else 2) / px) \
                    * np.cos(np.pi * (2 * i + 1) * k / (2 * px))
        zz = sorted(((u, v) for u in range(px) for v in range(px) if u + v > 0),
                    key=lambda t: (t[0] + t[1], max(t)))[:NC]
        _DCT[px] = np.stack([np.outer(D[u], D[v]) for u, v in zz]) * A
    return _DCT[px]

def render_v3(rng, px, cols, rows):   # DMT field + labels (3*NC symbols per cell)
    pat = zz_basis(px)
    sym = rng.integers(0, 4, (rows * cols, 3, NC))
    blocks = 128 + np.einsum("ncj,jkl->nklc", LEVELS[sym], pat)
    img = np.full((rows * px + 16, cols * px + 16, 3), 128, np.float32)
    for t in range(rows * cols):
        r, c = t // cols, t % cols
        img[8 + r * px:8 + (r + 1) * px, 8 + c * px:8 + (c + 1) * px] = blocks[t]
    return np.clip(img, 0, 255), sym.reshape(rows * cols, -1).astype(np.int64)

def harvest(rng, img, labels, px, C, cols, rows):   # crops C x C around random tiles
    take = rng.permutation(rows * cols)[:HARVEST]
    m = (C - px) // 2
    crops = np.zeros((len(take), C, C, 3), np.uint8)
    for i, t in enumerate(take):
        r, c = t // cols, t % cols
        y, x = 8 + r * px - m, 8 + c * px - m
        crops[i] = img[y:y + C, x:x + C]
    return crops, labels[take]

# ---- nets ---------------------------------------------------------------------------
class GlyphNet(nn.Module):            # the receiver's net, torch twin (heads 16+4)
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * (k // 4) ** 2, 64), nn.ReLU())
        self.s, self.c = nn.Linear(64, 16), nn.Linear(64, 4)
    def forward(self, x):
        h = self.f(x)
        return self.s(h), self.c(h)

class Demapper(nn.Module):            # test/v3 arch: 3*NC PAM-4 heads
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * (k // 4) ** 2, 512), nn.ReLU(),
            nn.Linear(512, 3 * NC * 4))
    def forward(self, x):
        return self.f(x).view(-1, 3 * NC, 4)

# ---- the phone-exact chunk loop -----------------------------------------------------
def err_of(net, kind, VX, Vy, K):
    with torch.no_grad():
        x = VX[:, :, 2:2 + K, 2:2 + K].float().div(255).sub(0.5)
        wrong = 0
        for i in range(0, len(x), 4096):
            out = net(x[i:i + 4096].to(DEV))
            if kind == "v2":
                s, c = out
                wrong += ((s.argmax(1) != Vy[i:i + 4096, 0].to(DEV))
                          | (c.argmax(1) != Vy[i:i + 4096, 1].to(DEV))).sum().item()
            else:
                wrong += (out.argmax(-1) != Vy[i:i + 4096].to(DEV)) \
                    .float().sum().item() / (3 * NC)   # per-SYMBOL rate (tile-err analog)
        return wrong / len(x)

def run(kind, px, log):
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    cols = rows = (1024 - 32) // px            # the web field's grid
    if kind == "v2":
        K, C = px, px + 4
        net = GlyphNet(K).to(DEV)
        render = lambda: render_v2(rng, px, cols, rows)
    else:
        K, C = 20, 24                          # demapper input 20, crop 24 for jitter
        net = Demapper(K).to(DEV)
        render = lambda: render_v3(rng, px, cols, rows)
    RX = torch.zeros(RES_N, 3, C, C, dtype=torch.uint8)
    VX = torch.zeros(VAL_N, 3, C, C, dtype=torch.uint8)
    ndim = 2 if kind == "v2" else 3 * NC
    Ry = torch.zeros(RES_N, ndim, dtype=torch.long)
    Vy = torch.zeros(VAL_N, ndim, dtype=torch.long)
    rtn = rvn = 0
    err, images, t_all = None, 0, time.time()
    for chunk in range(1, MAX_CHUNKS + 1):
        t_h = time.time()
        for _ in range(BATCH_FRAMES):          # gather the 20-image batch
            img, labels = render()
            cr, lb = harvest(rng, img, labels, px, C, cols, rows)
            if kind == "v2":
                lb = np.stack([lb // 4, lb % 4], 1)
            for i in range(len(cr)):
                if i % 10 == 0:
                    j = rvn if rvn < VAL_N else rng.integers(0, VAL_N)
                    VX[j] = torch.from_numpy(cr[i].transpose(2, 0, 1))
                    Vy[j] = torch.from_numpy(lb[i])
                    rvn = min(rvn + 1, VAL_N)
                else:
                    j = rtn if rtn < RES_N else rng.integers(0, RES_N)
                    RX[j] = torch.from_numpy(cr[i].transpose(2, 0, 1))
                    Ry[j] = torch.from_numpy(lb[i])
                    rtn = min(rtn + 1, RES_N)
        images += BATCH_FRAMES
        t_c = time.time()
        opt = torch.optim.Adam(net.parameters(), lr_tier(err))   # fresh Adam per chunk
        for _ in range(TRAIN_STEPS):
            i = torch.randint(0, rtn, (TRAIN_BS,))
            ox, oy = np.random.randint(0, 5, 2)
            x = RX[i, :, oy:oy + K, ox:ox + K].float()
            x = x * torch.empty(TRAIN_BS, 3, 1, 1).uniform_(0.7, 1.3)
            x = (x + torch.randn_like(x) * 4).div(255).sub(0.5).to(DEV)
            y = Ry[i].to(DEV)
            if kind == "v2":
                s, c = net(x)
                loss = nn.functional.cross_entropy(s, y[:, 0]) \
                     + nn.functional.cross_entropy(c, y[:, 1])
            else:
                loss = nn.functional.cross_entropy(net(x).reshape(-1, 4), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        err = err_of(net, kind, VX[:rvn], Vy[:rvn], K)
        log(f"  chunk {chunk:2d} ({images:4d} imgs): err {err*100:6.3f}%  "
            f"(harvest {t_c-t_h:4.1f}s, train {time.time()-t_c:4.1f}s)")
        if err < TARGET:
            log(f"  -> SUB-0.2%: {images} images, {chunk} chunks, "
                f"{time.time()-t_all:.0f}s total")
            return
    log(f"  -> NOT reached in {MAX_CHUNKS} chunks ({images} imgs), final {err*100:.3f}%")

def main():
    lines = [f"pictrain — phone-exact chunks (20 imgs -> 120x256 steps, lr tiers), {DEV}"]
    def log(s):
        print(s, flush=True)
        lines.append(s)
    for kind, px in [("v2", 16), ("v2", 12), ("v3", 16), ("v3", 12)]:
        log(f"{kind}-{px}px:")
        run(kind, px, log)
    open(os.path.join(os.path.dirname(__file__), "results.txt"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
