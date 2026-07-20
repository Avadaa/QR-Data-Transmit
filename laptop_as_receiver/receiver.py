# receiver.py — data-mode receiver: capture thread (dedup -> queue) + NN worker.
# START QR gives FRAMES/SIZE/PX/FPS -> loads models/<MVER>/pc_L<PX>. Worker decodes queued
# frames (hole+refine geometry, orientation from sync corners, accept = header CRC),
# frees each processed frame from RAM, reassembles payload by true header index at the
# end -> received_payload.bin (+ MD5 to compare with the transmit console).
# Usage: receiver.py [--save <dir>] [--model <name>] [--tune-name <name>] [key=value ...]
#        (--save dir is WIPED each run; key=value overrides config.py, e.g. train_every=20)
import hashlib, json, os, queue, shutil, sys, threading, time, zlib
from collections import deque
import numpy as np, cv2, torch, torch.nn as nn
import config, rs
from pyzbar.pyzbar import ZBarSymbol, decode
from glyphs import CANVAS_H, GAP, PALETTE, RING, shape_mask

argv = config.override(sys.argv[1:])
SAVE = argv[argv.index("--save") + 1] if "--save" in argv else None
MODEL = argv[argv.index("--model") + 1] if "--model" in argv else None
TUNE = argv[argv.index("--tune-name") + 1] if "--tune-name" in argv else None
LRT = [(.03, 1e-4), (.01, 5e-5), (.005, 2e-5), (0, 1e-5)]
if SAVE:
    shutil.rmtree(SAVE, ignore_errors=True); os.makedirs(SAVE)
HERE = os.path.dirname(os.path.abspath(__file__))
W_USE = config.W_USE or 1800   # fallback grid for pre-v1.3 captures without COLS/ROWS
NSYM = config.NSYM             # wire format (docs/decode_practical.md)
TH_CHANGE, TH_STABLE = config.TH_CHANGE, config.TH_STABLE
torch.set_num_threads(max(1, os.cpu_count() - config.CAM_THREADS))   # capture headroom

class Net(nn.Module):
    def __init__(s, k=24):              # k = NN input side: 24 canonical, rung px native
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Flatten(), nn.Linear(32 * (k // 4) ** 2, 64), nn.ReLU())
        s.shape, s.color = nn.Linear(64, 16), nn.Linear(64, len(PALETTE))
    def forward(s, x):
        h = s.f(x); return s.shape(h), s.color(h)

def refine_quad(hull, quad):            # verbatim from client.py
    pts = hull.reshape(-1, 2).astype(np.float64)
    lines = []
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

def small(f): return cv2.cvtColor(f[::16, ::16], cv2.COLOR_BGR2GRAY).astype(np.int16)

def colorful(f):
    s = f[::16, ::16].astype(np.int16)
    return float((s.max(2) - s.min(2) > 60).mean()) > 0.15

def ranges(idxs, cap=10):   # cap: a wrapped line breaks the ANSI in-place redraw
    out = []
    for m in sorted(idxs):
        if out and m == out[-1][1] + 1: out[-1][1] = m
        else: out.append([m, m])
    parts = [f"{a}-{b}" if a != b else f"{a}" for a, b in out]
    if cap and len(parts) > cap:
        parts = parts[:cap] + [f"(+{len(parts) - cap} more)"]
    return ", ".join(parts)

# ---- worker (starts once START gives PX) --------------------------------------
G = {}               # grid constants, filled by setup()
D = {}               # true idx -> payload bit array
DSEEN = set()        # true idx of every decoded header (stats: kills phantom MISSING)
failed = 0
import threading, time  # (re)import INSIDE the exec slice: label_rx/bench exec this
                        # region into a bare env, and decode_at/nn_sym use these
NLOCK = threading.Lock()   # forward pass vs runtime weight swap (torn weights = junk)
TM = {k: 0.0 for k in ("load", "warp", "prep", "nn", "hdr", "rs", "proc", "train")}
MN = {"d": 0, "t": 0}   # per-stage wall time + frame counts -> table at the end
MEAS = [False]          # accumulate stage times only around DATA-frame processing
                        # (nn_sym also runs for training frames and search retries)

# ---- OpenVINO engine (config.OV_THREADS, docs/iGPU_CPU_performance.md): fp32 IR at
# setup, upgraded to int8 once real crops exist, rebuilt after every tune. torch stays
# the TRAINING net; nn_sym(live=True) reads it directly (train phase must track the
# live weights). All builds run in background threads; decode falls back to torch
# until an engine is ready.
OVC = {"req": None, "cm": None, "gen": 0, "prec": None, "busy": False}

def ov_prep(crops):
    k = G["K"]
    X = (crops[:, 2:2 + k, 2:2 + k].astype(np.float32) / 255 - 0.5).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(X)

def ov_swap(cal=None):     # torch weights -> ONNX -> OV IR (int8 if calibration crops
    try:                   # given, else fp32) -> compile -> hot-swap under NLOCK
        import tempfile
        import openvino as ov
        clone = Net(G["K"])
        with NLOCK:
            clone.load_state_dict(G["net"].state_dict())
        clone.eval()
        path = os.path.join(tempfile.gettempdir(), f"rx_{os.getpid()}.onnx")
        torch.onnx.export(clone, torch.zeros(1, 3, G["K"], G["K"]), path,
                          input_names=["x"], output_names=["shape", "color"],
                          dynamic_axes={"x": {0: "n"}}, dynamo=False)
        m = ov.convert_model(path)
        if cal is not None:
            import contextlib, io, logging
            import nncf
            logging.getLogger("nncf").setLevel(logging.ERROR)
            X = ov_prep(cal[np.random.permutation(len(cal))[:2048]])
            ds = nncf.Dataset([X[i:i + 256] for i in range(0, len(X), 256)])
            with contextlib.redirect_stdout(io.StringIO()):   # progress bars shred the
                m = nncf.quantize(m, ds, subset_size=8)       # stats line's ANSI redraw
        cm = ov.Core().compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": config.OV_THREADS,
                                                "NUM_STREAMS": 1})
        req = cm.create_infer_request()
        with NLOCK:
            OVC.update(req=req, cm=cm, gen=OVC["gen"] + 1,
                       prec="fp32" if cal is None else "int8")
    finally:
        OVC["busy"] = False

def setup(px):
    global NSYM
    k = px if config.NATIVE_TILES else 24   # NN input side (see config.NATIVE_TILES)
    c = k + 4                               # stored crop side (4px jitter margin)
    p = globals().get("plan", {})           # grid comes from the START/TRAIN QR (the
    NSYM = int(p.get("NSYM", NSYM))         # ...and so does the RS overhead (v1.4)
    cols = int(p.get("COLS", 0)) or (W_USE - 2 * (RING + GAP)) // px      # field scales
    rows = int(p.get("ROWS", 0)) or (CANVAS_H - 2 * (RING + GAP)) // px   # to the screen)
    T = rows * cols
    corners = [0, cols - 1, (rows - 1) * cols, T - 1]
    free = np.setdiff1d(np.arange(T), corners)
    sync_s, sync_c = [0, 1, 8, 9], [0, 1, 2, 3]     # solid/white X/green ring/red dot/yellow
    exp = np.float32([shape_mask(s, px)[px // 2, px // 2] for s in sync_s]  # cheap warp
                     )[:, None] * PALETTE[sync_c]   # POINT-samples tile centers, so the
                                                    # prior is the glyph center pixel
    net = Net(k)           # wrong-regime weights die here with a shape mismatch
    p = os.path.join(HERE, "models", config.MVER,
                     (MODEL or f"{config.MODEL_PREFIX}{px}") + ".npz")
    if not os.path.exists(p):
        p = os.path.join(HERE, "models", config.MPREV, os.path.basename(p))
    w = np.load(p)
    net.load_state_dict({kk: torch.tensor(w[kk]) for kk in w.files})
    net = net.to(memory_format=torch.channels_last).eval()
    CAP = len(free[49:]) * 6
    nblk = (CAP // 8) // 255
    G.update(px=px, K=k, C=c, cols=cols, rows=rows, T=T, corners=corners, HDR=free[:49],
             PAY=free[49:], CAP=CAP, NBLK=nblk, KPF=nblk * (255 - NSYM), exp=exp, net=net)
    RT["X"] = np.zeros((20000, c, c, 3), np.uint8)  # crop reservoirs follow the crop side
    RT["VX"] = np.zeros((3000, c, c, 3), np.uint8)
    if config.OV_THREADS:
        OVC["busy"] = True
        threading.Thread(target=ov_swap, daemon=True).start()   # fp32 while we wait
                                                                # for calibration crops

def decode_at(img, Hm):    # warp -> NN -> majority header; None unless the CRC passes
    t = time.perf_counter()
    w = cv2.warpPerspective(img, Hm, (G["cols"] * G["C"], G["rows"] * G["C"]))
    if MEAS[0]: TM["warp"] += time.perf_counter() - t
    return decode_warped(w, Hm)

def decode_warped(w, cache_hm=None):   # (pre-)warped field -> NN -> header -> RS
    rows, cols, c = G["rows"], G["cols"], G["C"]
    t = time.perf_counter()
    crops = w.reshape(rows, c, cols, c, 3).transpose(0, 2, 1, 3, 4).reshape(-1, c, c, 3)
    if MEAS[0]: TM["prep"] += time.perf_counter() - t
    sym = nn_sym(crops)
    t = time.perf_counter()
    hb = np.unpackbits(sym[G["HDR"]][:, None], axis=1)[:, 2:].reshape(7, 42)
    hb = (hb.sum(0) > 3).astype(np.uint8)              # majority over the 7 copies
    idx = int(hb[:30] @ (1 << np.arange(29, -1, -1, dtype=np.int64)))
    crc = int(hb[30:] @ (1 << np.arange(11, -1, -1, dtype=np.int64)))
    if MEAS[0]: TM["hdr"] += time.perf_counter() - t
    if crc != (zlib.crc32(idx.to_bytes(4, "big")) & 0xFFF): return None
    if cache_hm is not None:
        G["Hm"] = cache_hm     # header decoded -> this geometry is right; cache it
    if config.OV_THREADS and OVC["prec"] != "int8" and not OVC["busy"]:
        OVC["busy"] = True     # a CRC-proven field = real calibration data -> int8
        threading.Thread(target=ov_swap, args=(crops.copy(),), daemon=True).start()
    t = time.perf_counter()
    pb = np.unpackbits(sym[G["PAY"]][:, None], axis=1)[:, 2:].ravel()
    by = np.packbits(pb)[:G["NBLK"] * 255].reshape(255, G["NBLK"]).T  # deinterleave
    dec, ok = rs.decode(by, NSYM)
    if MEAS[0]: TM["rs"] += time.perf_counter() - t
    return (idx, dec.ravel() if ok.all() else None)    # None bytes = RS overrun

THS = ((config.ADAPT_BLOCK, config.ADAPT_C), (101, -5), (101, -10))  # threshold sweep:
# block 51 is too local under harsh sun (dim ring next to blinding glare); 101 spans
# enough context. Extra variants only cost time on frames that would fail anyway.

FAILS = [0]            # consecutive fast-path CRC failures (see config.RESEARCH_FAILS)

def process(img):                       # RGB frame -> (true_idx, payload bytes) or None
    if "Hm" in G:          # fast path: the rig is static, so reuse the last proven
        r = decode_at(img, G["Hm"])     # homography. Skips contour+verify (~240 ms) and
        if r:                                 # survives glare that breaks the ring hole.
            FAILS[0] = 0
            return r if r[1] is not None else None   # CRC passed -> geometry is RIGHT;
        FAILS[0] += 1                   # an RS overrun is channel noise, not a lost rig
        if not config.RESEARCH_FAILS or FAILS[0] < config.RESEARCH_FAILS:
            return None    # junk frame >> moved rig: skip the ~1.2s full re-search
    px, cols, rows, c = G["px"], G["cols"], G["rows"], G["C"]
    s, o = c / px, GAP * c / px
    W, H = cols * px + 2 * GAP, rows * px + 2 * GAP
    dst = np.float32([[-o, -o], [W * s - o, -o], [W * s - o, H * s - o], [-o, H * s - o]])
    down = np.array([[1 / c, 0, -0.5], [0, 1 / c, -0.5], [0, 0, 1.0]])  # -0.5: warp
    # point-samples at output-pixel coords, so shift half a tile to hit tile CENTERS
    ctr = (img.shape[1] / 2, img.shape[0] / 2)
    for blk, cc in THS:
        th = cv2.adaptiveThreshold(img.max(2), 255, cv2.ADAPTIVE_THRESH_MEAN_C,  # local
                                   cv2.THRESH_BINARY, blk, cc)
        # contrast: glare-proof; global Otsu put screen-black in noise, shredded the gap
        cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, h in zip(cnts, hier[0]) if h[3] >= 0]
        for hole in sorted(holes, key=cv2.contourArea, reverse=True)[:8]:
            hull = cv2.convexHull(hole)
            if cv2.pointPolygonTest(hull, ctr, False) < 0: continue
            for eps in (0.02, 0.04, 0.08):
                quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
                if len(quad) == 4: break
            if len(quad) != 4: continue
            quad = refine_quad(hull, quad.astype(np.float64))
            scores = []                 # orientation: sync corner colors, cheap 1px/tile
            for fl in (1, -1):
                for r in range(4):
                    q = np.float32(np.roll(quad[::fl], r, 0))
                    Hm = cv2.getPerspectiveTransform(q, dst)
                    sm = cv2.warpPerspective(img, (down @ Hm).astype(np.float32), (cols, rows))
                    m = sm[[0, 0, rows - 1, rows - 1], [0, cols - 1, 0, cols - 1]].astype(float)
                    cx = np.corrcoef(m.ravel(), G["exp"].ravel())[0, 1]
                    scores.append((0 if np.isnan(cx) else cx, Hm))
            scores.sort(key=lambda x: -x[0])
            for sc, Hm in scores[:2]:   # top-2: header CRC arbitrates
                r = decode_at(img, Hm)
                if r:
                    FAILS[0] = 0
                    return r if r[1] is not None else None  # all-or-nothing per frame
    return None

def locate(img):     # geometry only -> all tile crops for the best orientation, or None
    px, cols, rows, c = G["px"], G["cols"], G["rows"], G["C"]
    s, o = c / px, GAP * c / px
    W, H = cols * px + 2 * GAP, rows * px + 2 * GAP
    dst = np.float32([[-o, -o], [W * s - o, -o], [W * s - o, H * s - o], [-o, H * s - o]])
    down = np.array([[1 / c, 0, -0.5], [0, 1 / c, -0.5], [0, 0, 1.0]])
    ctr = (img.shape[1] / 2, img.shape[0] / 2)
    for blk, cc in THS:
        th = cv2.adaptiveThreshold(img.max(2), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blk, cc)
        cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, h in zip(cnts, hier[0]) if h[3] >= 0]
        for hole in sorted(holes, key=cv2.contourArea, reverse=True)[:8]:
            hull = cv2.convexHull(hole)
            if cv2.pointPolygonTest(hull, ctr, False) < 0: continue
            for eps in (0.02, 0.04, 0.08):
                quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
                if len(quad) == 4: break
            if len(quad) != 4: continue
            quad = refine_quad(hull, quad.astype(np.float64))
            best = (-2, None)
            for fl in (1, -1):
                for r in range(4):
                    q = np.float32(np.roll(quad[::fl], r, 0))
                    Hm = cv2.getPerspectiveTransform(q, dst)
                    sm = cv2.warpPerspective(img, (down @ Hm).astype(np.float32), (cols, rows))
                    m = sm[[0, 0, rows - 1, rows - 1], [0, cols - 1, 0, cols - 1]].astype(float)
                    cx = np.corrcoef(m.ravel(), G["exp"].ravel())[0, 1]
                    if not np.isnan(cx) and cx > best[0]: best = (cx, Hm)
            if best[1] is None: continue
            w = cv2.warpPerspective(img, best[1], (cols * c, rows * c))
            return (w.reshape(rows, c, cols, c, 3).transpose(0, 2, 1, 3, 4)
                     .reshape(-1, c, c, 3), best[1])
    return None

def truth_sym(idx):  # training frames: every tile derives from the frame index
    hdr = np.concatenate([np.unpackbits(np.frombuffer(idx.to_bytes(4, "big"), np.uint8))[2:],
                          np.unpackbits(np.frombuffer((zlib.crc32(idx.to_bytes(4, "big"))
                                                       & 0xFFF).to_bytes(2, "big"), np.uint8))[4:]])
    sym = np.zeros(G["T"], np.uint8)
    sym[G["HDR"]] = np.tile(hdr, 7).reshape(49, 6) @ (1 << np.arange(5, -1, -1))
    sym[G["PAY"]] = np.random.default_rng(config.TRAIN_SEED + idx).integers(
        0, 64, len(G["PAY"])).astype(np.uint8)
    for ci, (s, c) in zip(G["corners"], [(0, 0), (1, 1), (8, 2), (9, 3)]):
        sym[ci] = (s << 2) | c
    return sym

def nn_sym(crops, live=False):   # classify crops -> symbols; live=True forces the torch
    k = G["K"]                   # net (train phase reads headers off the LIVE weights)
    if OVC["req"] is not None and not live:
        t = time.perf_counter()
        X = ov_prep(crops)
        if MEAS[0]: TM["prep"] += time.perf_counter() - t
        t = time.perf_counter()
        with NLOCK:
            r = OVC["req"].infer({0: X})
        s, c = list(r.values())
        if MEAS[0]: TM["nn"] += time.perf_counter() - t
        return (s.argmax(1).astype(np.uint8) << 2) | c.argmax(1).astype(np.uint8)
    t = time.perf_counter()
    X = torch.tensor(crops[:, 2:2 + k, 2:2 + k]).float()
    X = (X / 255 - 0.5).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    if MEAS[0]: TM["prep"] += time.perf_counter() - t
    t = time.perf_counter()
    ps, pc = [], []
    with torch.no_grad(), NLOCK:
        for i in range(0, len(X), 4096):
            ls, lc = G["net"](X[i:i + 4096])
            ps.append(ls.argmax(1)); pc.append(lc.argmax(1))
    if MEAS[0]: TM["nn"] += time.perf_counter() - t
    return (torch.cat(ps).numpy().astype(np.uint8) << 2) | torch.cat(pc).numpy().astype(np.uint8)

TR = {"n": 0, "err": None, "saved": False}
RT = {"X": None, "y": np.zeros((20000, 2), np.int64), "n": 0,   # X/VX allocated in
      "VX": None, "Vy": np.zeros((3000, 2), np.int64), "vn": 0} # setup() (crop side)

def train_frame(img, opt, fld=False):
    if fld:                    # pre-warped field from the capture thread
        c = G["C"]
        crops, Hm = img.reshape(G["rows"], c, G["cols"], c, 3).transpose(0, 2, 1, 3, 4)\
                       .reshape(-1, c, c, 3), None
    else:
        r = locate(img)
        if r is None: return
        crops, Hm = r
    hb = np.unpackbits(nn_sym(crops[G["HDR"]], live=True)[:, None], axis=1)[:, 2:].reshape(7, 42)
    hb = (hb.sum(0) > 3).astype(np.uint8)
    idx = int(hb[:30] @ (1 << np.arange(29, -1, -1, dtype=np.int64)))
    crc = int(hb[30:] @ (1 << np.arange(11, -1, -1, dtype=np.int64)))
    if crc != (zlib.crc32(idx.to_bytes(4, "big")) & 0xFFF): return
    if Hm is not None:
        G["Hm"] = Hm           # CRC-proven geometry: FA/data mode start pre-seeded
    t = truth_sym(idx)
    lab = np.stack([t >> 2, t & 3], 1)
    for i in np.random.permutation(G["T"])[:1200]:
        if i % 10 == 0:
            j = RT["vn"] if RT["vn"] < 3000 else np.random.randint(3000)
            RT["VX"][j], RT["Vy"][j] = crops[i], lab[i]; RT["vn"] = min(RT["vn"] + 1, 3000)
        else:
            j = RT["n"] if RT["n"] < 20000 else np.random.randint(20000)
            RT["X"][j], RT["y"][j] = crops[i], lab[i]; RT["n"] = min(RT["n"] + 1, 20000)
    TR["n"] += 1
    if TR["n"] % config.TRAIN_EVERY == 0 and RT["n"] > 2000:
        net = G["net"]; net.train()
        e = TR["err"]
        lr = 1e-4 if e is None else next(v for lim, v in LRT if e >= lim)
        for g in opt.param_groups: g["lr"] = lr
        bs, k = config.TRAIN_BS, G["K"]
        for _ in range(config.TRAIN_STEPS):
            i = np.random.randint(0, RT["n"], bs)
            X = torch.tensor(RT["X"][i]).float()
            ox, oy = np.random.randint(0, 5, 2)
            X = X[:, oy:oy + k, ox:ox + k]
            X = X * torch.empty((bs, 1, 1, 3)).uniform_(0.7, 1.3) + torch.randn(bs, k, k, 3) * 4
            X = (X / 255 - 0.5).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            yy = torch.tensor(RT["y"][i])
            ls, lc = net(X)
            loss = nn.functional.cross_entropy(ls, yy[:, 0]) + nn.functional.cross_entropy(lc, yy[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        i = np.random.randint(0, RT["vn"], min(2048, RT["vn"]))
        X = (torch.tensor(RT["VX"][i][:, 2:2 + k, 2:2 + k]).float() / 255 - 0.5)
        X = X.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            ls, lc = G["net"](X)
        yy = torch.tensor(RT["Vy"][i])
        TR["err"] = 1 - ((ls.argmax(1) == yy[:, 0]) & (lc.argmax(1) == yy[:, 1])).float().mean().item()

def solve_repair(D, N):    # rebuild missing sources from repair frames (idx >= N):
    miss = sorted(set(range(N)) - set(D))      # GF(2) elimination, masks from the seed
    pos = {k: j for j, k in enumerate(miss)}
    piv = {}                                   # pivot bit -> [row mask, row bytes]
    for ri in sorted(i for i in D if i >= N):
        m = np.random.default_rng(config.REPAIR_SEED + ri - N).integers(0, 2, N).astype(bool)
        if not m.any(): m[(ri - N) % N] = True
        bm, rhs = 0, D[ri].copy()
        for k in np.where(m)[0]:
            if k in pos: bm |= 1 << pos[k]     # unknown source -> equation bit
            else: rhs = rhs ^ D[k]             # known source -> fold into the rhs
        while bm:
            low = (bm & -bm).bit_length() - 1
            if low not in piv:
                piv[low] = [bm, rhs]; break
            bm ^= piv[low][0]; rhs = rhs ^ piv[low][1]
    if not miss or len(piv) < len(miss): return 0
    for b in sorted(piv, reverse=True):        # back-substitute, highest pivot first
        bm, rhs = piv[b]
        for b2 in range(b + 1, len(miss)):
            if bm >> b2 & 1: rhs = rhs ^ piv[b2][1]
        piv[b] = [1 << b, rhs]
        D[miss[b]] = rhs
    return len(miss)

def save_tuned():
    name = TUNE or f"{config.TUNE_PREFIX}{G['px']}"
    np.savez(os.path.join(HERE, "models", config.MVER, name + ".npz"),
             **{k: v.detach().cpu().numpy() for k, v in G["net"].state_dict().items()})
    TR["saved"] = name

Q = queue.Queue()
qbytes = 0

# ---- runtime fine-tuning (data mode): decoded payload = free labels -------------
RTC = {"n": 0, "cand": [], "busy": False}

def sent_sym(idx, dat):    # decoded frame bytes -> the exact displayed symbol grid
    hdr = np.concatenate([np.unpackbits(np.frombuffer(idx.to_bytes(4, "big"), np.uint8))[2:],
                          np.unpackbits(np.frombuffer((zlib.crc32(idx.to_bytes(4, "big"))
                                                       & 0xFFF).to_bytes(2, "big"), np.uint8))[4:]])
    sym = np.zeros(G["T"], np.uint8)
    sym[G["HDR"]] = np.tile(hdr, 7).reshape(49, 6) @ (1 << np.arange(5, -1, -1))
    code = rs.encode(dat.reshape(G["NBLK"], 255 - NSYM).astype(np.int32), NSYM)
    fb = np.unpackbits(code.T.ravel())
    fb = np.concatenate([fb, np.zeros(G["CAP"] - len(fb), np.uint8)])
    sym[G["PAY"]] = fb.reshape(-1, 6) @ (1 << np.arange(5, -1, -1))
    for ci, (s, c) in zip(G["corners"], [(0, 0), (1, 1), (8, 2), (9, 3)]):
        sym[ci] = (s << 2) | c
    return sym

def runtime_keep(img, res, fld=False):   # every ~N/F-th decoded frame = training food
    if not (config.RUNTIME_TRAIN_EVERY_N and config.RUNTIME_TRAIN_FRAMES): return
    RTC["n"] += 1
    if RTC["n"] % max(1, config.RUNTIME_TRAIN_EVERY_N // config.RUNTIME_TRAIN_FRAMES) == 0:
        RTC["cand"].append((img, res[0], res[1], fld))
        RTC["cand"] = RTC["cand"][-config.RUNTIME_TRAIN_FRAMES:]   # freshest F only
    if len(RTC["cand"]) >= config.RUNTIME_TRAIN_FRAMES and not RTC["busy"]:
        batch, RTC["cand"] = RTC["cand"], []
        RTC["busy"] = True
        threading.Thread(target=runtime_tune, args=(batch,), daemon=True).start()

def runtime_tune(batch):   # background: harvest -> train a CLONE -> swap -> save
    try:
        c, k = G["C"], G["K"]
        for img, idx, dat, fld in batch:
            w = img if fld else cv2.warpPerspective(img, G["Hm"], (G["cols"] * c, G["rows"] * c))
            crops = w.reshape(G["rows"], c, G["cols"], c, 3).transpose(0, 2, 1, 3, 4)\
                     .reshape(-1, c, c, 3)
            t = sent_sym(idx, dat)
            lab = np.stack([t >> 2, t & 3], 1)
            for i in np.random.permutation(G["T"])[:1200]:   # same reservoir as train mode
                if i % 10 == 0:
                    j = RT["vn"] if RT["vn"] < 3000 else np.random.randint(3000)
                    RT["VX"][j], RT["Vy"][j] = crops[i], lab[i]; RT["vn"] = min(RT["vn"] + 1, 3000)
                else:
                    j = RT["n"] if RT["n"] < 20000 else np.random.randint(20000)
                    RT["X"][j], RT["y"][j] = crops[i], lab[i]; RT["n"] = min(RT["n"] + 1, 20000)
        if RT["n"] < 2000: return
        clone = Net(k)
        clone.load_state_dict(G["net"].state_dict())
        clone = clone.to(memory_format=torch.channels_last).train()
        e = TR["err"]
        opt = torch.optim.Adam(clone.parameters(),
                               1e-4 if e is None else next(v for lim, v in LRT if e >= lim))
        bs = config.TRAIN_BS
        for _ in range(config.TRAIN_STEPS):
            i = np.random.randint(0, RT["n"], bs)
            X = torch.tensor(RT["X"][i]).float()
            ox, oy = np.random.randint(0, 5, 2)
            X = X[:, oy:oy + k, ox:ox + k]
            X = X * torch.empty((bs, 1, 1, 3)).uniform_(0.7, 1.3) + torch.randn(bs, k, k, 3) * 4
            X = (X / 255 - 0.5).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            yy = torch.tensor(RT["y"][i])
            ls, lc = clone(X)
            loss = nn.functional.cross_entropy(ls, yy[:, 0]) + nn.functional.cross_entropy(lc, yy[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        clone.eval()
        i = np.random.randint(0, RT["vn"], min(2048, RT["vn"]))
        X = (torch.tensor(RT["VX"][i][:, 2:2 + k, 2:2 + k]).float() / 255 - 0.5)
        X = X.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            ls, lc = clone(X)
        yy = torch.tensor(RT["Vy"][i])
        TR["err"] = 1 - ((ls.argmax(1) == yy[:, 0]) & (lc.argmax(1) == yy[:, 1])).float().mean().item()
        TR["n"] += len(batch)
        with NLOCK:            # atomic swap: the worker never sees half-new weights
            G["net"].load_state_dict(clone.state_dict())
        save_tuned()
        if config.OV_THREADS and not OVC["busy"]:   # re-quantize the tuned weights
            OVC["busy"] = True                      # (already on a background thread)
            ov_swap(RT["X"][:RT["n"]])
    finally:
        RTC["busy"] = False

POOL = {}            # pool size -> [decoded, failed]: attribute failures to dwell health

def worker():
    global failed, qbytes
    setup(int(plan["PX"]))
    opt = torch.optim.Adam(G["net"].parameters(), 1e-4)
    while True:
        item = Q.get()
        if item is None: break
        tag, buf, pool, fld = item      # BGR frame/field or jpeg bytes (see enqueue)
        if tag == "crop":               # in-order marker: every later frame is cropped
            x0, y0 = buf                # by (x0,y0) -> rebase the cached geometry once
            G["Hm"] = G["Hm"] @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
            continue
        if config.ISOLATE_CAM_NN_THREADS and tag == "d":
            while phase not in ("done", "timeout", "interrupted"):
                time.sleep(0.2)         # isolation: no NN while the camera records
        t = time.perf_counter()
        img = (buf if buf.ndim == 3 else
               cv2.imdecode(buf, cv2.IMREAD_COLOR))[:, :, ::-1].copy()
        tload = time.perf_counter() - t
        if tag == "t":
            t = time.perf_counter()
            train_frame(img, opt, fld)
            TM["train"] += time.perf_counter() - t; MN["t"] += 1
        else:
            if TR["n"] and not TR["saved"]:
                save_tuned()               # training ended -> keep it; the pre-train OV
                if config.OV_THREADS:      # engine is stale -> torch until the rebuild
                    with NLOCK: OVC.update(req=None, prec=None)
                    if not OVC["busy"]:
                        OVC["busy"] = True
                        threading.Thread(target=ov_swap, args=(RT["X"][:RT["n"]],),
                                         daemon=True).start()
            TM["load"] += tload; MN["d"] += 1
            MEAS[0] = True
            t = time.perf_counter()
            res = decode_warped(img) if fld else process(img)   # fields: fast-path only
            TM["proc"] += time.perf_counter() - t
            MEAS[0] = False
            if fld and res is not None and res[1] is None: res = None   # all-or-nothing
            p = POOL.setdefault(pool, [0, 0])
            if res is None: failed += 1; p[1] += 1
            else:
                D[res[0]] = res[1]; DSEEN.add(res[0]); p[0] += 1
                runtime_keep(img, res, fld)
        qbytes -= buf.nbytes            # processed frame leaves RAM (queue pop + this)

# ---- capture ------------------------------------------------------------------
drawn = 0
def stats(state):
    global drawn
    os.system("")
    N = plan.get("FRAMES", "?")
    tot = "?" if N == "?" else N + plan.get("REPAIR", 0)
    src = len(D) if N == "?" else sum(1 for i in D if i < N)
    rate = (f"   KB/s {src * G['KPF'] / 1024 / (time.time() - t0):.2f}"
            if t0 and src and G.get("KPF") else "")   # decoded payload bytes over the
    lines = [f"[{state}]  Capturing: {(max(idx_est) + 1) if idx_est else 0}/{tot}"   # data
             f"   Processed: {src}/{N}" + (f"   failed: {failed}" if failed else "") + rate,
             f"Backlog: {Q.qsize()} frames  {max(qbytes, 0) / 2**20:.1f} MB RAM"
             + (f"  -> {SAVE}/" if SAVE else "")]
    if DIST:
        lines.append("Dedup candidates: " + "  ".join(f"{k}:{DIST[k]}" for k in sorted(DIST))
                     + (f"   culled: {CULL[0]}" if CULL[0] else ""))
    if FAST["win"]:
        lines.append(f"FA: named {FAST['named']}/{FAST['frames']}  committed {FAST['commits']}"
                     f"  {(FAST['remap'] + FAST['nn'] + FAST['logic']) / FAST['win'] * 1000:.0f} ms/win"
                     + (f"  dropped {FAST['drop']}" if FAST["drop"] else ""))
    if TR["n"]:
        e = "--" if TR["err"] is None else f"{TR['err']*100:.2f}%"
        lines.append(f"TRAIN: frames={TR['n']} err={e}"
                     + (f"  saved as {TR['saved']}" if TR["saved"] else "  (stop on server when happy)"))
    exp = min(tot, int((time.time() - t0) * plan["FPS"]) + 1) if t0 and tot != "?" else 0
    miss = set(range(exp)) - idx_est - DSEEN.copy()   # DECODED headers beat the time
    lines.append(f"MISSING (est): {ranges(miss)}"     # estimate: no phantom gaps once
                 if miss else "All frames captured so far")   # the worker catches up
    if drawn: sys.stdout.write(f"\x1b[{drawn}F")
    sys.stdout.write("\n".join(l + "\x1b[K" for l in lines) + "\n"); sys.stdout.flush()
    drawn = len(lines)

phase, plan, t0 = "wait-start", {}, None
idx_est, wt = set(), None

CAPB = {"box": None}   # capture-side crop/warp state (INSTANT_CROP_PICS / MISC_...)

def capture_payload(f):    # BGR camera frame -> what actually enters the queue
    if "Hm" not in G or not (config.INSTANT_CROP_PICS or config.MISC_IN_CAPTURE_THREAD):
        return f, False
    if CAPB["box"] is None:            # geometry just became known: fix the field box
        W, H = G["cols"] * G["C"], G["rows"] * G["C"]
        pts = cv2.perspectiveTransform(np.float32([[[0, 0], [W, 0], [W, H], [0, H]]]),
                                       np.linalg.inv(G["Hm"]))[0]
        CAPB["box"] = (max(0, int(pts[:, 0].min()) - 32), max(0, int(pts[:, 1].min()) - 32),
                       min(f.shape[1], int(pts[:, 0].max()) + 32),
                       min(f.shape[0], int(pts[:, 1].max()) + 32))
        if config.INSTANT_CROP_PICS and not config.MISC_IN_CAPTURE_THREAD:
            Q.put(("crop", np.float64(CAPB["box"][:2]), -1, False))   # worker rebases
    if config.MISC_IN_CAPTURE_THREAD:  # pre-warp: the worker gets a ready field
        return cv2.warpPerspective(f, G["Hm"], (G["cols"] * G["C"], G["rows"] * G["C"])), True
    x0, y0, x1, y1 = CAPB["box"]
    return np.ascontiguousarray(f[y0:y1, x0:x1]), False

def save_view(f):          # --save archives shrink to the field box too
    if CAPB["box"] and config.INSTANT_CROP_PICS:
        x0, y0, x1, y1 = CAPB["box"]
        return f[y0:y1, x0:x1]
    return f

def enqueue(tag, f, pool=-1):   # short backlog -> queue the RAW frame: skips the jpeg
    global qbytes      # encode (capture thread) AND decode (worker), ~90ms combined.
    f, fld = capture_payload(f)
    buf = f if Q.qsize() < config.QRAW else \
        cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_Q])[1]
    qbytes += buf.nbytes; Q.put((tag, buf, pool, fld))   # deep backlog -> jpeg, RAM flat

# ---- frame analysis (config.FA_*, validated in test/fa_test.py): fps-independent
# commit picking. Every data-phase camera frame lands in a window buffer; a background
# thread names each one by FA_INDEX_COUNT header copies (majority + CRC-12) and scores
# sharpness by probe-glyph logit margins -> the best NEW frame per idx is committed.
# Replaces the stability-pool pick for data frames once geometry is seeded; the old
# dedup keeps serving QR/held-screen detection and the pre-seed bootstrap.
FAB, FALOCK = [], threading.Lock()
FAON, FATH = [False], [None]
FAST = {"frames": 0, "named": 0, "commits": 0, "win": 0, "drop": 0, "remap": 0.0,
        "nn": 0.0, "logic": 0.0, "seen": set(), "req": None, "gen": -1, "map": None}

def fa_axis(dim):
    c = dim // 2; step = max(1, c // 3)
    return sorted({min(dim - 2, max(1, c + k * step)) for k in range(-3, 4)})

def fa_maps():      # probe tiles + ONE remap LUT for all their crops (static rig)
    rows, cols, c = G["rows"], G["cols"], G["C"]
    vr, vc = fa_axis(rows), fa_axis(cols)
    if config.FA_GLYPH_PATTERN == "grid":
        pts = {(r, x) for r in vr for x in vc}
    else:           # cross: center row + center column + both rectangle diagonals
        pts = {(r, cols // 2) for r in vr} | {(rows // 2, x) for x in vc}
        n = min(len(vr), len(vc))
        pts |= {(vr[k], vc[k]) for k in range(n)} | {(vr[k], vc[n - 1 - k]) for k in range(n)}
    spots = np.array(sorted({r * cols + x for r, x in pts} - set(G["corners"])))
    probe = np.concatenate([G["HDR"][:config.FA_INDEX_COUNT * 7], spots])
    ys = (probe // cols)[:, None, None] * c + np.arange(c)[None, :, None]
    xs = (probe % cols)[:, None, None] * c + np.arange(c)[None, None, :]
    p = np.stack([np.broadcast_to(xs, (len(probe), c, c)),
                  np.broadcast_to(ys, (len(probe), c, c))], -1).reshape(-1, 1, 2)
    m = cv2.perspectiveTransform(p.astype(np.float32),
                                 np.linalg.inv(G["Hm"]).astype(np.float32))
    m = m.reshape(len(probe) * c, c, 2)
    return np.ascontiguousarray(m[..., 0]), np.ascontiguousarray(m[..., 1]), len(probe)

def fa_infer(X, live=False):   # live=True -> torch (training: OV engine is stale)
    if OVC["cm"] is not None and not live:
        with NLOCK:
            if FAST["gen"] != OVC["gen"]:
                FAST["req"], FAST["gen"] = OVC["cm"].create_infer_request(), OVC["gen"]
        r = FAST["req"].infer({0: X})
        return list(r.values())
    with torch.no_grad(), NLOCK:
        o = G["net"](torch.tensor(X))
    return o[0].numpy(), o[1].numpy()

def fa_idx(sym):    # index-tile symbols -> frame idx, or None on CRC fail (blend)
    hb = np.unpackbits(sym[:, None], axis=1)[:, 2:].reshape(config.FA_INDEX_COUNT, 42)
    hb = (hb.sum(0) * 2 > config.FA_INDEX_COUNT).astype(np.uint8)
    idx = int(hb[:30] @ (1 << np.arange(29, -1, -1, dtype=np.int64)))
    crc = int(hb[30:] @ (1 << np.arange(11, -1, -1, dtype=np.int64)))
    return idx if crc == (zlib.crc32(idx.to_bytes(4, "big")) & 0xFFF) else None

def fa_flush():     # background: one probe batch per FA_BATCH_MS window
    ni = config.FA_INDEX_COUNT * 7
    while True:
        time.sleep(config.FA_BATCH_MS / 1000)
        with FALOCK:
            batch, FAB[:] = FAB[:], []
        if not batch or "Hm" not in G: continue
        c = G["C"]
        t = time.perf_counter()
        if FAST["map"] is None: FAST["map"] = fa_maps()
        MX, MY, npr = FAST["map"]
        crops = np.concatenate([cv2.remap(fr, MX, MY, cv2.INTER_LINEAR)
                                [:, :, ::-1].reshape(npr, c, c, 3) for _, fr, _ in batch])
        FAST["remap"] += time.perf_counter() - t
        t = time.perf_counter()
        live = any(ph == "train" for _, _, ph in batch)
        ls, lc = fa_infer(ov_prep(crops), live)
        FAST["nn"] += time.perf_counter() - t
        t = time.perf_counter()
        sym = (ls.argmax(1).astype(np.uint8) << 2) | lc.argmax(1).astype(np.uint8)
        p = np.partition(ls, -2, 1); mg = p[:, -1] - p[:, -2]
        p = np.partition(lc, -2, 1); mg = np.minimum(mg, p[:, -1] - p[:, -2])
        fmg = mg.reshape(len(batch), npr)[:, ni:].mean(1)   # per-frame probe score
        cand = {}       # (tag, idx) -> [best score, ts, frame, shots seen]
        for j, (ts, fr, ph) in enumerate(batch):
            idx = fa_idx(sym[j * npr:j * npr + ni])
            FAST["frames"] += 1
            if idx is None: continue
            FAST["named"] += 1
            e = cand.setdefault(("t" if ph == "train" else "d", idx), [-1.0, 0, None, 0])
            e[3] += 1
            if fmg[j] > e[0]: e[0], e[1], e[2] = float(fmg[j]), ts, fr
        FAST["logic"] += time.perf_counter() - t
        FAST["win"] += 1
        for (tag, idx), (score, ts, fr, n) in cand.items():
            if (tag, idx) in FAST["seen"]: continue    # duplicates are free to skip
            FAST["seen"].add((tag, idx))
            if tag == "t":
                if Q.qsize() < config.QCAP_TRAIN: enqueue("t", fr)
            else:
                idx_est.add(idx)
                enqueue("d", fr, n)                # pool = shots this idx had in-window
                if SAVE: cv2.imwrite(os.path.join(SAVE, f"{idx:05d}_{int(ts)}.jpg"),
                                     save_view(fr), [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_Q])
            FAST["commits"] += 1
        if live and not cand and Q.qsize() < config.QCAP_TRAIN:   # cold model named
            enqueue("t", batch[int(fmg.argmax())][1])   # nothing: still feed the best-
                                                        # margin shot (CRC-gated later)

def keep(f, held=False, pool=-1):
    global phase, t0, qbytes, wt
    ts = time.time()
    if held or not colorful(f):     # control QRs ride ON a glyph field now (black-bg
        h, w = f.shape[:2]          # QRs overexposed) -> they are colorful but HELD;
        crop = np.ascontiguousarray(f[max(0, h // 2 - 500):h // 2 + 500,
                                      max(0, w // 2 - 500):w // 2 + 500])  # QR is centered
        qr = decode(crop, symbols=[ZBarSymbol.QRCODE]) or \
            ([] if colorful(f) else decode(f[:, :, ::-1].copy(), symbols=[ZBarSymbol.QRCODE]))
        txt = qr[0].data.decode() if qr else ""
        if txt.startswith("---START---") or txt.startswith("---TRAIN---"):
            for kv in txt.split(",")[1:]:
                k, v = kv.split("="); plan[k] = float(v) if k == "FPS" else int(v)
            phase = "train" if txt.startswith("---TRAIN---") else "armed"
            torch.set_num_threads(config.TRAIN_THREADS if phase == "train"
                                  else max(1, os.cpu_count() - config.CAM_THREADS))
                                                                     # t frames are free;
            if wt is None:                                           # data dwells are not
                wt = threading.Thread(target=worker, daemon=True); wt.start()
            return
        elif txt.startswith("---END---"):
            phase = "done"; return
        if not colorful(f): return
    if pool == 0 and config.AUTO_CULL_DEDUP_0_POOL:
        CULL[0] += 1; return    # never-stable dwell -> blend; cull before any use
    if phase == "train":
        if not held and not FAON[0] and Q.qsize() < config.QCAP_TRAIN:
            enqueue("t", f)         # held+no-QR = a control screen the model can't
        return                      # read -- never train on it; FA owns its commits
    if phase == "armed" and not held: phase, t0 = "data", ts
    if phase != "data": return
    if FAON[0]: return         # FA owns data commits (QR/held logic already ran above)
    idx = min(round((ts - t0) * plan["FPS"]),                      # last frame commits
              plan["FRAMES"] + plan.get("REPAIR", 0) - 1)
    # one dwell late (on END's arrival) and would otherwise round past the end
    idx_est.add(idx); enqueue("d", f, pool)
    if SAVE: cv2.imwrite(os.path.join(SAVE, f"{idx:05d}_{int(ts)}.jpg"), save_view(f),
                         [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_Q])

cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W), cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)
print("warming up camera...", flush=True)
for _ in range(config.CAM_WARMUP): cam.read()
if config.CAM_FOCUS: cam.set(cv2.CAP_PROP_FOCUS, config.CAM_FOCUS)
if config.CAM_EXPOSURE: cam.set(cv2.CAP_PROP_EXPOSURE, config.CAM_EXPOSURE)
for _ in range(30): cam.read()     # ISP settles on the locked lens/exposure

# dedup: hold the NEWEST stable frame of the current dwell, commit it when the next
# change arrives (or after 1.5s for held screens like START/END). The ISP temporally
# blends across transitions — the first stable-looking frame is a two-frame ghost —
# and the LAST stable frame can straddle the NEXT swap: a rolling-shutter boundary in
# the final rows changes too little of the bottom band to trip TH_STABLE (seen live:
# doubled glyphs at the frame bottom). Commit picks MID-DWELL, clear of both edges.
# A dwell that never settles (sun shimmer) still commits its least-unstable view,
# as long as it is calmer than TH_BLEND (above that = a true transition blur).
prev_s = kept_s = pend = pend_s = None
pend_q = np.inf
pend_t = last_stat = 0
CANDS = deque(maxlen=5)        # stable frames of the dwell, chronological
DIST = {}                      # pool size -> committed frames that had that many
                               # candidates (0 = never-stable TH_BLEND fallback)
CULL = [0]                     # 0-pool commits dropped by AUTO_CULL_DEDUP_0_POOL

def commit(held=False):
    global pend, pend_s, pend_q, kept_s
    if pend is not None:
        n = len(CANDS)         # <=3 -> last (small pools: middle sits in the leading
        f = pend if n <= 3 else CANDS[(n - 1) // 2 + 1]   # ghost); >=4 -> middle+1
        th = 2.0 if held or not colorful(pend) else TH_CHANGE   # QR->QR swaps barely
        if kept_s is None or np.abs(pend_s - kept_s).mean() > th:   # move the global mean
            keep(f, held, n); kept_s = pend_s
            DIST[n] = DIST.get(n, 0) + 1
    pend, pend_s, pend_q = None, None, np.inf
    CANDS.clear()

try:
    while phase not in ("done", "timeout"):   # timeout must EXIT (it used to loop
        ok, f = cam.read()                    # forever re-setting itself — stuck)
        if not ok: continue
        s = small(f)
        FAON[0] = (config.FA_ANALYZE_FRAMES and config.CAMERA_TYPE == "stable"
                   and "Hm" in G and (phase == "data" or
                   (phase == "train" and config.FA_ENABLED_IN_TRAINING)))
        if FAON[0] and not (phase == "train" and Q.qsize() >= config.QCAP_TRAIN):
            # full training backlog -> every probe is guaranteed-discard work; pause
            # FA and give the trainer the CPU (cam keeps READING — a stopped stream
            # would serve stale frames on resume; dedup stays on for QR detection)
            if FATH[0] is None:
                FATH[0] = threading.Thread(target=fa_flush, daemon=True); FATH[0].start()
            with FALOCK:
                if len(FAB) < 64: FAB.append((time.time(), f, phase))
                else: FAST["drop"] += 1       # flusher stalled; cap RAM, count it
        if prev_s is not None:
            if pend_s is not None and np.abs(s - pend_s).mean() > TH_CHANGE:
                commit()                              # dwell ended -> keep its best view
            q = np.abs(s - prev_s).reshape(6, -1).mean(1).max()  # per-band instability:
            # LCD/rolling-shutter/ISP settle bottom-last; a bottom blend hides in a mean
            if q < min(config.TH_BLEND, max(TH_STABLE, pend_q)):
                # stable -> candidate list (commit picks mid-dwell); never-stable
                # dwell -> its calmest frame (pend doubles as that fallback)
                if pend_s is None: pend_t = time.time()
                pend, pend_s, pend_q = f, s, q
                if q < TH_STABLE: CANDS.append(f)
            if pend_s is not None and time.time() - pend_t > 1.5:
                commit(held=True)                     # held screen (START/END/last frame)
        prev_s = s
        if time.time() - last_stat > 0.25:
            stats(phase); last_stat = time.time()
        if t0 and time.time() - t0 > (plan["FRAMES"] + plan.get("REPAIR", 0)) / plan["FPS"] + 10:
            phase = "timeout"
except KeyboardInterrupt:
    phase = "interrupted"
cam.release()
if FATH[0]: time.sleep(config.FA_BATCH_MS / 1000 + 0.3)   # FA flushes its last window
torch.set_num_threads(os.cpu_count())    # capture thread is idle now -> drain the
if wt:                                   # backlog on all cores (bench: 4 > 3 > 2 on
    Q.put(None)                          # the m3; 3 hits an HT contention hole)
    while wt.is_alive():
        stats(phase + " / crunching"); time.sleep(0.25)
    wt.join()
    if TR["n"] and not TR["saved"]: save_tuned()
t_done = time.time()          # backlog drained: the live KB/s has converged to this
stats(phase)
if TR["saved"]: print(f"\ntuned model saved: models/{config.MVER}/{TR['saved']}.npz "
                      f"(err {TR['err']*100:.2f}% on {TR['n']} frames)", flush=True)

if plan.get("FRAMES"):
    N, KPF = plan["FRAMES"], G["KPF"]
    direct = sum(1 for i in D if i < N)
    rebuilt = solve_repair(D, N)               # fountain: fill losses from repair frames
    miss = set(range(N)) - set(D)
    out = np.zeros(N * KPF, np.uint8)
    for i, d in D.items():
        if i < N: out[i * KPF:(i + 1) * KPF] = d
    out = out.tobytes()[:plan["SIZE"]]
    open(os.path.join(HERE, "received_payload.bin"), "wb").write(out)
    print(f"\ndecoded {direct}/{N} frames + {rebuilt} rebuilt from repair (failed: {failed})", flush=True)
    print("pool size -> ok/failed: " + "  ".join(
        f"{k}:{v[0]}/{v[1]}" for k, v in sorted(POOL.items())), flush=True)
    if FAST["win"]:
        w = FAST["win"]
        print(f"FA: {FAST['named']}/{FAST['frames']} frames named "
              f"({FAST['frames'] - FAST['named']} blends), {FAST['commits']} committed "
              f"over {w} windows of {config.FA_BATCH_MS}ms"
              + (f", {FAST['drop']} dropped at buffer cap" if FAST["drop"] else ""), flush=True)
        print(f"FA per-window: remap {FAST['remap'] / w * 1000:5.1f} ms   "
              f"nn {FAST['nn'] / w * 1000:5.1f} ms   logic {FAST['logic'] / w * 1000:5.1f} ms   "
              f"(engine {OVC['prec'] or 'torch'})", flush=True)
    print("MISSING (by header): " + (ranges(miss, 0) if miss else "none"), flush=True)
    print(f"received_payload.bin  {len(out)} bytes  md5 {hashlib.md5(out).hexdigest()}", flush=True)
    if t0:
        print(f"goodput {len(out) / 1024 / (t_done - t0):.2f} KB/s "
              f"({len(out)} B in {t_done - t0:.1f}s, START->backlog drained)", flush=True)
    if MN["d"]:
        n = MN["d"]
        stages = [("load", TM["load"])] + [(k, TM[k]) for k in ("warp", "prep", "nn", "hdr", "rs")]
        stages.append(("search", max(0.0, TM["proc"] - sum(TM[k] for k in
                                     ("warp", "prep", "nn", "hdr", "rs")))))
        tot = TM["load"] + TM["proc"]
        print(f"\nper-stage timing over {n} data frames:", flush=True)
        for k, v in stages:
            print(f"  {k:6s} {v/n*1000:6.1f} ms  {v/tot*100:5.1f}%", flush=True)
        print(f"  TOTAL  {tot/n*1000:6.1f} ms  -> {n/tot:.1f} fps"
              + (f"   (+ training: {TM['train']:.1f}s over {MN['t']} frames)" if MN["t"] else ""),
              flush=True)
    if SAVE: json.dump({"plan": plan, "decoded": sorted(D)}, open(os.path.join(SAVE, "manifest.json"), "w"), indent=1)
