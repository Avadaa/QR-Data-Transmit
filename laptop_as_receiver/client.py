# client.py — the air-gapped client (this is the "input computer"). Local camera ->
# QR sync -> harvest -> batch train, everything on this machine, nothing sent anywhere.
# Cycles: RECORD (fill BATCH frames per rung) -> TRAIN (offline steps) -> repeat.
# Live table: err%% per rung (start + a column each 5 min), batch fill, state.
# Usage: client.py <model_name> [--init <parent>]
import json, os, shutil, sys, time
import numpy as np, cv2, torch, torch.nn as nn
from pyzbar.pyzbar import ZBarSymbol, decode
import config
from glyphs import (LADDER, PALETTE, RING, GAP, SHAPE_NAMES, COLOR_NAMES,
                    set_canvas, set_grid, spec, grid, inks, set_ladder)

sys.argv[1:] = config.override(sys.argv[1:])   # bare key=value tokens -> config
set_canvas(config.CANVAS_W, config.CANVAS_H)   # fallback only: the SNC QR announces
                                               # the grid (set_grid) on new servers
HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, "models", config.MVER); os.makedirs(MDIR, exist_ok=True)
OLD = os.path.join(HERE, "models", config.MPREV)
CAP, VCAP = 20000, 3000
BATCH = 100            # frames per rung per recording phase
TRAIN_STEPS, BS = 1500, 512
name = sys.argv[1]
parent = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else None
if "--ladder" in sys.argv: set_ladder(sys.argv[sys.argv.index("--ladder") + 1])
K = LADDER[0] if config.NATIVE_TILES else 24   # NN input side (config.NATIVE_TILES)
if config.NATIVE_TILES and len(LADDER) > 1:
    sys.exit("NATIVE_TILES: single-rung ladders only (one net, one input size)")
C = K + 4              # stored crop side (4px jitter margin)
SAVEF = int(sys.argv[sys.argv.index("--save-frames") + 1]) if "--save-frames" in sys.argv else 0
FDIR = os.path.join(HERE, "frames", "L" + "-".join(map(str, LADDER)))  # archive is
SAVEF and os.makedirs(FDIR, exist_ok=True)   # ladder-bound: n only decodes with it
FCAP = 10 * 2**30                      # stop hoarding frames past 10 GB
fbytes = sum(e.stat().st_size for e in os.scandir(FDIR)) if SAVEF else 0
torch.set_num_threads(os.cpu_count())   # benchmarked: 4 threads > 3 (HT contention hole)
LR_TIERS = [(.03, 1e-3), (.02, 2e-4), (.01, 1e-4), (.005, 3e-5), (0, 1e-5)]

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Flatten(), nn.Linear(32 * (K // 4) ** 2, 64), nn.ReLU())
        s.shape, s.color = nn.Linear(64, 16), nn.Linear(64, len(PALETTE))
    def forward(s, x):
        h = s.f(x); return s.shape(h), s.color(h)

net = Net()
src = parent or (name if os.path.exists(os.path.join(MDIR, name + ".npz")) else None)
if src:   # parent never rewritten; resolved from MVER then MPREV
    p = os.path.join(MDIR, src + ".npz")
    if not os.path.exists(p): p = os.path.join(OLD, src + ".npz")
    w = np.load(p)
    net.load_state_dict({k: torch.tensor(w[k]) for k in w.files})
net = net.to(memory_format=torch.channels_last)   # benchmarked: +62% train, 2.8x infer
opt = torch.optim.Adam(net.parameters(), 1e-3)

R = {px: {"X": np.zeros((CAP, C, C, 3), np.uint8), "y": np.zeros((CAP, 2), np.int64), "n": 0,
          "VX": np.zeros((VCAP, C, C, 3), np.uint8), "Vy": np.zeros((VCAP, 2), np.int64),
          "vn": 0, "acc": None} for px in LADDER}

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
                if x < cols * C: continue
                small = cv2.warpPerspective(img, (down @ Hm).astype(np.float32), (cols, rows))
                cc0 = np.corrcoef(small.ravel(), exp.ravel())[0, 1]
                cands.append((0 if np.isnan(cc0) else cc0, Hm))
        for cc0, Hm in sorted(cands, key=lambda c: -c[0]):   # full verify, best first
            w = cv2.warpPerspective(img, Hm, (cols * C, rows * C))
            meas = w.reshape(rows, C, cols, C, 3).mean((1, 3))
            cc = np.corrcoef(meas.ravel(), exp.ravel())[0, 1]
            if cc >= 0.5: return Hm, cc
    return None, 0

def batch(px, k, val=False):
    r = R[px]
    N = r["vn"] if val else r["n"]
    if N < 64: return None
    i = np.random.randint(0, N, k)
    X = torch.tensor((r["VX"] if val else r["X"])[i]).float()
    y = torch.tensor((r["Vy"] if val else r["y"])[i])
    if val:
        X = X[:, 2:2 + K, 2:2 + K]
    else:
        ox, oy = np.random.randint(0, 5, 2)
        X = X[:, oy:oy + K, ox:ox + K]
        X = X * torch.empty((k, 1, 1, 3)).uniform_(0.7, 1.3) + torch.randn(k, K, K, 3) * 4
    return (X / 255 - 0.5).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last), y

F = {px: np.zeros((16, len(PALETTE), 2)) for px in LADDER}   # per (shape,color): seen, wrong

def evaluate():
    net.eval()
    with torch.no_grad():
        for px in LADDER:
            b = batch(px, 1024, val=True)
            if b:
                vs, vc = net(b[0])
                wrong = ((vs.argmax(1) != b[1][:, 0]) | (vc.argmax(1) != b[1][:, 1])).numpy()
                R[px]["acc"] = 1 - wrong.mean()
                s, c = b[1][:, 0].numpy(), b[1][:, 1].numpy()
                F[px] *= 0.98                             # decay: stats track recent model
                np.add.at(F[px], (s, c, 0), 1)
                np.add.at(F[px], (s, c, 1), wrong)
    net.train()
    csv_row()

def fail_report():          # ALL (shape,color) combos per rung, worst first, min 30 seen
    rep = {}
    for px in LADDER:
        seen, bad = F[px][:, :, 0], F[px][:, :, 1]
        rate = np.where(seen >= 30, bad / np.maximum(seen, 1), -1)
        order = np.dstack(np.unravel_index(np.argsort(rate, None)[::-1], rate.shape))[0]
        rep[f"r{px}"] = [{"shape": SHAPE_NAMES[s], "color": COLOR_NAMES[c],
                          "err": round(float(rate[s, c]), 4), "seen": int(seen[s, c])}
                         for s, c in order if rate[s, c] >= 0]
    return rep

def expected(px):           # frames of this rung the server has shown since we first synced
    if n0 is None: return 0
    total, i = nmax - n0 + 1, LADDER.index(px)
    return total // len(LADDER) + sum(1 for j in range(total % len(LADDER))
                                      if (n0 + j) % len(LADDER) == i)

def csv_row():              # accuracies + capture stats, one timestamped row
    p = os.path.join(MDIR, name + "_log.csv")
    hdr = ("t," + ",".join(f"acc{x}" for x in LADDER) + ","
           + ",".join(f"exp{x},cap{x}" for x in LADDER))
    if os.path.exists(p) and open(p).readline().strip() != hdr:
        os.replace(p, p + ".old")   # rotate logs with stale headers
    fresh = not os.path.exists(p)
    with open(p, "a") as f:
        if fresh: f.write(hdr + "\n")
        f.write(time.strftime("%F %T") + ","
                + ",".join("" if R[x]["acc"] is None else f"{R[x]['acc']:.4f}" for x in LADDER)
                + "," + ",".join(f"{expected(x)},{cap[x]}" for x in LADDER) + "\n")

def save():
    np.savez(os.path.join(MDIR, name + ".npz"),
             **{k: v.detach().numpy() for k, v in net.state_dict().items()})

def set_lr():          # tiered LR: mature model -> gentler steps (stays plastic, never 0)
    errs = [1 - R[p]["acc"] for p in LADDER if R[p]["acc"] is not None]
    if not errs: return opt.param_groups[0]["lr"]
    e = sum(errs) / len(errs)
    lr = next(v for lim, v in LR_TIERS if e >= lim)
    for g in opt.param_groups: g["lr"] = lr
    return lr

def err(px, accs=None):
    a = (accs or {p: R[p]["acc"] for p in LADDER})[px]
    return "  -- " if a is None else f"{(1 - a) * 100:4.1f}%"

start_err, hist, drawn = None, [], 0   # hist: (minute_mark, {px: acc}) every 5 min
def table(state, cnt, images):
    global drawn
    os.system("")                       # enable ANSI on Windows consoles
    cols = ["start"] + [f"{m}m" for m, _ in hist[-4:]]
    out = ["err% | " + " | ".join(f"{c:>5}" for c in cols) + " | batch   | captured",
           "-----+" + "-" * (8 * len(cols) + 30)]
    for px in LADDER:
        cells = [err(px, start_err)] + [err(px, h) for _, h in hist[-4:]]
        e = expected(px)
        out.append(f"r{px:>2}  | " + " | ".join(cells) + f" | {min(cnt[px], BATCH):>3}/{BATCH}"
                   + f" | {cap[px]}/{e} {100 * cap[px] // max(e, 1):>2}%")
    out.append(f"[{state}] images={images} elapsed={int((time.time()-t0)/60)}m  (ctrl+c saves)")
    if drawn: sys.stdout.write(f"\x1b[{drawn}F")
    sys.stdout.write("\n".join(l + "\x1b[K" for l in out) + "\n"); sys.stdout.flush()
    drawn = len(out)

cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920), cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
for _ in range(120): cam.read()
if config.CAM_FOCUS: cam.set(cv2.CAP_PROP_FOCUS, config.CAM_FOCUS)
if config.CAM_EXPOSURE: cam.set(cv2.CAP_PROP_EXPOSURE, config.CAM_EXPOSURE)
for _ in range(30): cam.read()
t0, last_n, images, next_hist = time.time(), -1, 0, 300
n0, nmax, cap = None, 0, {px: 0 for px in LADDER}
try:
    while True:
        cnt = {px: 0 for px in LADDER}
        # RECORD until ~85% total fill — don't let the slowest rung hold the phase hostage
        while sum(min(cnt[px], BATCH) for px in LADDER) < 0.85 * BATCH * len(LADDER):
            ok, f = cam.read()
            if not ok: continue
            img = f[:, :, ::-1].copy()
            qr = [d for d in decode(img, symbols=[ZBarSymbol.QRCODE]) if d.data.startswith(b"SNC")]
            if not qr: continue
            txt = qr[0].data[3:].decode()
            if "," in txt:               # server announces its grid (auto-scaled canvas)
                txt, dims = txt.split(",")
                set_grid(tuple(int(x) for x in dims.split("x")))
            n = int(txt)
            if n == last_n: continue
            last_n = n
            if n0 is None: n0 = n
            nmax = max(nmax, n)
            px, colsn, rowsn = spec(n)
            Hm, cc = pick_H(img, qr[0].rect, n)
            if Hm is None or cc < 0.5: continue
            w = cv2.warpPerspective(img, Hm, (colsn * C, rowsn * C))
            crops = w.reshape(rowsn, C, colsn, C, 3).transpose(0, 2, 1, 3, 4).reshape(-1, C, C, 3)
            shp, col = grid(n)
            y = np.stack([shp.ravel(), col.ravel()], 1)
            r = R[px]
            for i in np.random.permutation(len(crops))[:4000]:
                if i % 10 == 0:
                    j = r["vn"] if r["vn"] < VCAP else np.random.randint(VCAP)
                    r["VX"][j], r["Vy"][j] = crops[i], y[i]; r["vn"] = min(r["vn"] + 1, VCAP)
                else:
                    j = r["n"] if r["n"] < CAP else np.random.randint(CAP)
                    r["X"][j], r["y"][j] = crops[i], y[i]; r["n"] = min(r["n"] + 1, CAP)
            cnt[px] += 1; cap[px] += 1; images += 1
            if SAVEF and images % SAVEF == 0 and fbytes < FCAP:  # frame hoard, capped
                p = os.path.join(FDIR, f"{int(time.time())}_{n}.jpg")
                cv2.imwrite(p, img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                fbytes += os.path.getsize(p)
            if images % 100 == 0:      # numbered snapshot + its worst-combo report
                save()
                sd = os.path.join(MDIR, "snaps"); os.makedirs(sd, exist_ok=True)
                shutil.copy(os.path.join(MDIR, name + ".npz"),
                            os.path.join(sd, f"{name}_{images:06d}.npz"))
                json.dump(fail_report(), open(os.path.join(
                    sd, f"{name}_{images:06d}_fails.json"), "w"), indent=1)
            if time.time() - t0 > next_hist:
                hist.append((int(next_hist / 60), {p: R[p]["acc"] for p in LADDER})); next_hist += 300
                csv_row()      # capture-rate progress also logged during recording
            table("recording", cnt, images)
        if start_err is None:
            evaluate(); start_err = {p: R[p]["acc"] for p in LADDER}
        for i in range(TRAIN_STEPS):                        # TRAIN
            trpx = LADDER[np.random.randint(len(LADDER))]
            b = batch(trpx, BS)
            if b is None: continue
            X, yy = b
            ls, lc = net(X)
            loss = nn.functional.cross_entropy(ls, yy[:, 0]) + \
                   nn.functional.cross_entropy(lc, yy[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
            if (i + 1) % 100 == 0:
                evaluate(); save()
                lr = set_lr()
                if time.time() - t0 > next_hist:
                    hist.append((int(next_hist / 60), {p: R[p]["acc"] for p in LADDER})); next_hist += 300
                table(f"training {i+1}/{TRAIN_STEPS} lr={lr:g}", cnt, images)
except KeyboardInterrupt:
    pass
finally:
    save(); evaluate()
    print("\nsaved", name, flush=True)
    cam.release()
