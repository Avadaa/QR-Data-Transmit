# train.py — watches screen.png, syncs via QR counter, harvests labeled tile crops,
# trains the two-headed CNN. Ctrl+C-safe: saves on exit and every 10 frames.
# Usage: train.py <name> [--init <parent>]          interleaved harvest+train (default)
#        train.py <name> --harvest                  capture-only, full speed (weak CPUs)
#        train.py <name> --offline <steps>          epochs over stored shards, no capture
import json, os, sys, time
import numpy as np, cv2, torch, torch.nn as nn
from pyzbar.pyzbar import ZBarSymbol, decode
import config
from glyphs import LADDER, PALETTE, RING, GAP, spec, grid, inks, set_ladder

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(HERE, "screen.png")
MDIR = os.path.join(HERE, "models", config.MVER); os.makedirs(MDIR, exist_ok=True)
OLD = os.path.join(HERE, "models", config.MPREV)
C = 28                  # stored crop px (24 used for training after jitter crop)
CAP, VCAP = 20000, 3000 # reservoir caps per rung (train / val)
PER_FRAME = 4000        # max crops harvested per frame
STEPS = 30              # train minibatches per harvested frame
DEV = "cuda" if torch.cuda.is_available() else "cpu"

name = sys.argv[1]
parent = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else None
if "--ladder" in sys.argv: set_ladder(sys.argv[sys.argv.index("--ladder") + 1])
if "--harvest" in sys.argv: STEPS = 0
OFFLINE = int(sys.argv[sys.argv.index("--offline") + 1]) if "--offline" in sys.argv else 0
if OFFLINE: torch.set_num_threads(max(1, os.cpu_count() - 1))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Flatten(), nn.Linear(32 * 6 * 6, 64), nn.ReLU())
        s.shape, s.color = nn.Linear(64, 16), nn.Linear(64, len(PALETTE))
    def forward(s, x):
        h = s.f(x); return s.shape(h), s.color(h)

net = Net().to(DEV)
src = parent or (name if os.path.exists(os.path.join(MDIR, name + ".npz")) else None)
if src:
    p = os.path.join(MDIR, src + ".npz")
    if not os.path.exists(p): p = os.path.join(OLD, src + ".npz")
    w = np.load(p)
    net.load_state_dict({k: torch.tensor(w[k]) for k in w.files})
    print(f"init from {src}", flush=True)
opt = torch.optim.Adam(net.parameters(), 1e-3)

R = {px: {"X": np.zeros((CAP, C, C, 3), np.uint8), "y": np.zeros((CAP, 2), np.int64), "n": 0,
          "VX": np.zeros((VCAP, C, C, 3), np.uint8), "Vy": np.zeros((VCAP, 2), np.int64),
          "vn": 0, "acc": 0.0} for px in LADDER}
for px in LADDER:       # resume this setup's shards if present
    p = os.path.join(MDIR, f"data_{name}_r{px}.npz")
    if os.path.exists(p):
        d = np.load(p); k, vk = min(len(d["X"]), CAP), min(len(d["VX"]), VCAP)
        R[px]["X"][:k], R[px]["y"][:k], R[px]["n"] = d["X"][:k], d["y"][:k], k
        R[px]["VX"][:vk], R[px]["Vy"][:vk], R[px]["vn"] = d["VX"][:vk], d["Vy"][:vk], vk

def refine_quad(hull, quad):      # hull corners get chamfered by lit corner tiles ->
    pts = hull.reshape(-1, 2).astype(np.float64)      # robust-fit the 4 side lines,
    lines = []                                        # intersect for the true corners
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a; L = np.hypot(*ab)
        t = ((pts - a) @ ab) / (L * L)
        d = np.abs((pts - a)[:, 0] * ab[1] - (pts - a)[:, 1] * ab[0]) / L
        sel = pts[(d < 0.05 * L) & (t > 0.1) & (t < 0.9)]
        if len(sel) < 2: sel = pts[d < 0.05 * L]
        lines.append(cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = []
    for i in range(4):
        vx1, vy1, x1, y1 = lines[i - 1]; vx2, vy2, x2, y2 = lines[i]
        t1 = np.linalg.solve(np.array([[vx1, -vx2], [vy1, -vy2]]),
                             np.array([x2 - x1, y2 - y1]))[0]
        out.append([x1 + t1 * vx1, y1 + t1 * vy1])
    return np.float32(out)

def pick_H(img, qr_rect, n):   # ring INNER edge = a threshold hole: immune to daylight
    px, cols, rows = spec(n)   # window blobs merging with the ring from outside
    _, th = cv2.threshold(img.max(2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ctr = (img.shape[1] / 2, img.shape[0] / 2)
    holes = [c for c, h in zip(cnts, hier[0]) if h[3] >= 0]
    s, o = C / px, GAP * C / px
    W, H = cols * px + 2 * GAP, rows * px + 2 * GAP
    dst = np.float32([[-o, -o], [W * s - o, -o], [W * s - o, H * s - o], [-o, H * s - o]])
    qc = np.float32([[[qr_rect.left + qr_rect.width / 2, qr_rect.top + qr_rect.height / 2]]])
    shp, col = grid(n)
    exp = PALETTE[col] * inks(px)[shp][..., None]
    down = np.diag([1 / C, 1 / C, 1.0])            # cheap pass: 1 px per tile
    for hole in sorted(holes, key=cv2.contourArea, reverse=True)[:8]:
        hull = cv2.convexHull(hole)
        if cv2.pointPolygonTest(hull, ctr, False) < 0: continue  # tile field holds center
        for eps in (0.02, 0.04, 0.08):
            quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(quad) == 4: break
        if len(quad) != 4: continue
        quad = refine_quad(hull, quad.astype(np.float64))
        cands = []
        for f in (1, -1):
            for r in range(4):
                q = np.float32(np.roll(quad[::f], r, 0))
                Hm = cv2.getPerspectiveTransform(q, dst)
                x, y = cv2.perspectiveTransform(qc, Hm)[0, 0]
                if x < cols * C: continue          # QR must land right of the tile area
                small = cv2.warpPerspective(img, (down @ Hm).astype(np.float32), (cols, rows))
                cc0 = np.corrcoef(small.ravel(), exp.ravel())[0, 1]
                cands.append((0 if np.isnan(cc0) else cc0, Hm))
        for cc0, Hm in sorted(cands, key=lambda c: -c[0]):   # full verify, best first
            w = cv2.warpPerspective(img, Hm, (cols * C, rows * C))
            meas = w.reshape(rows, C, cols, C, 3).mean((1, 3))
            cc = np.corrcoef(meas.ravel(), exp.ravel())[0, 1]
            if cc >= 0.5: return Hm, cc
    return None, 0

def batch(px, k=256, val=False):
    r = R[px]
    N = r["vn"] if val else r["n"]
    if N < 64: return None
    i = np.random.randint(0, N, k)
    X = torch.tensor((r["VX"] if val else r["X"])[i]).to(DEV).float()
    y = torch.tensor((r["Vy"] if val else r["y"])[i]).to(DEV)
    if val:
        X = X[:, 2:26, 2:26]
    else:                                          # jitter crop + photometric augmentation
        ox, oy = np.random.randint(0, 5, 2)
        X = X[:, oy:oy + 24, ox:ox + 24]
        X = X * torch.empty((k, 1, 1, 3), device=DEV).uniform_(0.7, 1.3)
        X = X + torch.randn_like(X) * 4
    return (X / 255 - 0.5).permute(0, 3, 1, 2), y

def save():
    np.savez(os.path.join(MDIR, name + ".npz"),
             **{k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    meta = {"parent": parent, "ladder": list(LADDER), "frames": frames, "updated": time.strftime("%F %T"),
            "rungs": {px: {"train": R[px]["n"], "val": R[px]["vn"],
                           "val_acc": round(R[px]["acc"], 4)} for px in LADDER}}
    json.dump(meta, open(os.path.join(MDIR, name + ".json"), "w"), indent=1)

def save_shards():
    for px in LADDER:
        r = R[px]
        if r["n"]:
            np.savez(os.path.join(MDIR, f"data_{name}_r{px}.npz"),
                     X=r["X"][:r["n"]], y=r["y"][:r["n"]],
                     VX=r["VX"][:r["vn"]], Vy=r["Vy"][:r["vn"]])

frames, last_n, last_mt = 0, -1, 0
if OFFLINE:
    print(f"offline: {OFFLINE} steps on {DEV}, batch 1024", flush=True)
    try:
        for i in range(OFFLINE):
            trpx = LADDER[np.random.randint(len(LADDER))]
            b = batch(trpx, 1024)
            if b is None: continue
            X, yy = b
            ls, lc = net(X)
            loss = nn.functional.cross_entropy(ls, yy[:, 0]) + \
                   nn.functional.cross_entropy(lc, yy[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
            if (i + 1) % 250 == 0:
                net.eval()
                with torch.no_grad():
                    for px in LADDER:
                        b = batch(px, 1024, val=True)
                        if b:
                            vs, vc = net(b[0])
                            R[px]["acc"] = ((vs.argmax(1) == b[1][:, 0]) &
                                            (vc.argmax(1) == b[1][:, 1])).float().mean().item()
                net.train()
                print(f"step {i+1}: " + " ".join(f"r{p}:{R[p]['acc']:.3f}" for p in LADDER), flush=True)
                save()
    except KeyboardInterrupt:
        pass
    save(); print("saved", flush=True)
    sys.exit()

print(f"{'harvesting' if not STEPS else 'training'} '{name}' on {DEV}, watching {SHOT}", flush=True)
try:
    while True:
        try: mt = os.path.getmtime(SHOT)
        except OSError: time.sleep(0.5); continue
        if mt == last_mt: time.sleep(0.5); continue
        last_mt = mt
        img = cv2.imread(SHOT)
        if img is None: time.sleep(0.2); continue
        img = img[:, :, ::-1].copy()               # BGR -> RGB, match generator
        qr = [d for d in decode(img, symbols=[ZBarSymbol.QRCODE]) if d.data.startswith(b"SNC")]
        if not qr: continue
        n = int(qr[0].data[3:])
        if n == last_n: continue
        last_n = n
        px, cols, rows = spec(n)
        Hm, cc = pick_H(img, qr[0].rect, n)
        if Hm is None or cc < 0.5:
            print(f"n={n} rejected (corr {cc:.2f})", flush=True); continue
        w = cv2.warpPerspective(img, Hm, (cols * C, rows * C))
        crops = w.reshape(rows, C, cols, C, 3).transpose(0, 2, 1, 3, 4).reshape(-1, C, C, 3)
        shp, col = grid(n)
        y = np.stack([shp.ravel(), col.ravel()], 1)
        idx = np.random.permutation(len(crops))[:PER_FRAME]
        r = R[px]
        for i in idx:
            if i % 10 == 0:                        # tile-index split: stable val set
                j = r["vn"] if r["vn"] < VCAP else np.random.randint(VCAP)
                r["VX"][j], r["Vy"][j] = crops[i], y[i]; r["vn"] = min(r["vn"] + 1, VCAP)
            else:
                j = r["n"] if r["n"] < CAP else np.random.randint(CAP)
                r["X"][j], r["y"][j] = crops[i], y[i]; r["n"] = min(r["n"] + 1, CAP)
        net.train()
        for _ in range(STEPS):
            trpx = LADDER[np.random.randint(len(LADDER))]
            b = batch(trpx)
            if b is None: continue
            X, yy = b
            ls, lc = net(X)
            loss = nn.functional.cross_entropy(ls, yy[:, 0]) + \
                   nn.functional.cross_entropy(lc, yy[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            b = batch(px, 512, val=True)
            if b:
                X, yy = b
                ls, lc = net(X)
                r["acc"] = ((ls.argmax(1) == yy[:, 0]) & (lc.argmax(1) == yy[:, 1])).float().mean().item()
        frames += 1
        print(f"n={n} r{px} corr={cc:.2f} +{len(idx)} | " +
              " ".join(f"r{p}:{R[p]['n']}/{R[p]['acc']:.3f}" for p in LADDER), flush=True)
        logp = os.path.join(MDIR, name + "_log.csv")
        if not os.path.exists(logp):
            open(logp, "w").write("t,n,px,corr," + ",".join(f"acc{p}" for p in LADDER) + "\n")
        open(logp, "a").write(f"{time.time():.1f},{n},{px},{cc:.3f}," +
                              ",".join(f"{R[p]['acc']:.4f}" for p in LADDER) + "\n")
        if frames % 10 == 0: save()
        if frames % 50 == 0: save_shards()
except KeyboardInterrupt:
    pass
finally:
    save(); save_shards()
    print("saved model + shards", flush=True)
