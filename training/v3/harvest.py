# harvest.py — mine the phone videos (web transmitter, v3 wire, TRAINING frames) for
# labeled real cell data. No NN needed anywhere: training payloads are seeded truth
# (PCG64(TRAIN_SEED+idx)), so frames are IDENTIFIED by correlating the warped camera
# field against candidate truth fields (top-vs-second margin test), and the homography
# is REFINED sub-pixel by maximizing that same correlation (the jitter law makes
# sub-pixel geometry the make-or-break). Best-correlating camera frame per idx wins
# (best-shot doctrine; the ISP blends pattern transitions). Videos may contain SEVERAL
# training runs (the sequence restarts) — a rescan re-locks after unidentified streaks.
# Output: per selected frame a 1024x1024 warp of the logical field (uint8 npy) +
# meta.csv with seg/idx/NCC/H.   Run:  python harvest.py "../vids/x.MOV" data/tag
import os, subprocess, sys, time
import numpy as np
import cv2
cv2.setNumThreads(2)                         # parallel workers: don't thrash the cores

PX, COLS, FIELD = 16, 62, 1024
TRAIN_SEED = 7000
NC = int(os.environ.get("V3NC", "20"))       # v3.1 knobs: match what the web page
A = float(os.environ.get("V3A", "20"))       # announced for the filmed session
LEVELS = np.array([-3.0, 3.0] if os.environ.get("V3LV") == "2"
                  else [-3.0, -1.0, 1.0, 3.0], np.float32)
T = COLS * COLS
CORNERS = {0, COLS - 1, T - COLS, T - 1}
FREE = [t for t in range(T) if t not in CORNERS]
PAY = np.array(FREE[49:])                    # 3791 payload cells
LOG4 = np.array([[0, 0], [FIELD, 0], [FIELD, FIELD], [0, FIELD]], np.float32)

def zz_basis():                              # (NC,16,16) IDCT patterns x A — web-exact
    D = np.zeros((PX, PX))
    for k in range(PX):
        for i in range(PX):
            D[k, i] = np.sqrt((1 if k == 0 else 2) / PX) \
                * np.cos(np.pi * (2 * i + 1) * k / (2 * PX))
    zz = sorted(((u, v) for u in range(PX) for v in range(PX) if u + v > 0),
                key=lambda t: (t[0] + t[1], max(t)))[:NC]
    return (np.stack([np.outer(D[u], D[v]) for u, v in zz]) * A).astype(np.float32)

BAS = zz_basis()
BAS4 = BAS.reshape(NC, 4, 4, 4, 4).mean((2, 4))   # 4x box-avg basis (ID path)
RC = np.stack([PAY // COLS, PAY % COLS], 1)

def pays(idx):                               # labels: (3791, 3, NC)
    return np.random.default_rng(TRAIN_SEED + idx) \
        .integers(0, len(LEVELS), (len(PAY), 3, NC))

MASK = np.zeros((FIELD, FIELD), bool)
for r, c in RC:
    MASK[16 + r * 16:32 + r * 16, 16 + c * 16:32 + c * 16] = True
M256 = MASK.reshape(256, 4, 256, 4).all((1, 3))
M512 = MASK.reshape(512, 2, 512, 2).all((1, 3))

def truth_field(idx):                        # payload-only field, float32 1024^2x3
    f = np.full((FIELD, FIELD, 3), 128, np.float32)
    blocks = 128 + np.einsum("kcj,jyx->kyxc", LEVELS[pays(idx)], BAS)
    np.clip(blocks, 0, 255, blocks)
    for (r, c), b in zip(RC, blocks):
        f[16 + r * 16:32 + r * 16, 16 + c * 16:32 + c * 16] = b
    return f

_tc = {}                                     # idx -> (t256norm, t512norm lazy)
def truth256(idx):
    if idx not in _tc:
        blocks = np.einsum("kcj,jyx->kyxc", LEVELS[pays(idx)], BAS4)   # 4x4 per cell
        f = np.zeros((256, 256, 3), np.float32)
        for (r, c), b in zip(RC, blocks):
            f[4 + r * 4:8 + r * 4, 4 + c * 4:8 + c * 4] = b            # mid-gray = 0
        t = f[M256].ravel()
        _tc[idx] = ((t - t.mean()) / (t.std() + 1e-9)).astype(np.float32)
        if len(_tc) > 400:                   # never evict the entry just built
            for k in sorted(_tc)[:100]:
                if k != idx: del _tc[k]
    return _tc[idx]

def truth512(idx):
    t = truth_field(idx).reshape(512, 2, 512, 2, 3).mean((1, 3))[M512].ravel()
    return ((t - t.mean()) / (t.std() + 1e-9)).astype(np.float32)

def warp(frame, H, size):                    # camera -> logical field, centers sampled
    s = FIELD / size
    S = np.array([[s, 0, s / 2], [0, s, s / 2], [0, 0, 1]])
    return cv2.warpPerspective(frame, H @ S, (size, size),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

def normed(w, m):
    v = w[m].astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-9)

def find_quad(frame):
    g = frame.max(2)
    th = (g > 200).astype(np.uint8)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = g.shape
    best = None
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if x < w / 2 < x + bw and y < h / 2 < y + bh and bw * bh > w * h / 8:
            if best is None or cv2.contourArea(c) > cv2.contourArea(best): best = c
    if best is None: return None
    pts = best.reshape(-1, 2).astype(np.float64)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    quad = np.array([pts[s.argmin()], pts[d.argmax()],
                     pts[s.argmax()], pts[d.argmin()]], np.float32)
    return cv2.getPerspectiveTransform(LOG4, quad)

def refine(frame, H, idx, coarse=(2, 1, 0.5, 0.25), fine=(0.3, 0.15)):
    quad = cv2.perspectiveTransform(LOG4.reshape(-1, 1, 2), H).reshape(4, 2)
    def cd(quad, steps, size, m, tvec):      # coordinate descent on the 4 corners
        def score(q):
            Hq = cv2.getPerspectiveTransform(LOG4, q.astype(np.float32))
            return float(normed(warp(frame, Hq, size), m) @ tvec) / len(tvec), Hq
        best, Hb = score(quad)
        for st in steps:
            improved = True
            while improved:
                improved = False
                for i in range(4):
                    for ax in range(2):
                        for sg in (st, -st):
                            q = quad.copy(); q[i, ax] += sg
                            sc, Hq = score(q)
                            if sc > best: best, Hb, quad, improved = sc, Hq, q, True
        return best, Hb, quad
    _, Hb, quad = cd(quad, coarse, 256, M256, truth256(idx))
    best, Hb, _ = cd(quad, fine, 512, M512, truth512(idx))
    return best, Hb

def frames_of(vid, t0, t1):                  # ffmpeg raw pipe: frame-accurate slicing
    # dims/fps/rotation from ffprobe CONTAINER values (cv2 props auto-rotate on new
    # builds — mixing the two double-corrects; cv2 SEEKING on HEVC lies anyway)
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height,r_frame_rate",
                         "-show_entries", "stream_side_data=rotation",
                         "-of", "default=noprint_wrappers=1:nokey=1", vid],
                        capture_output=True, text=True).stdout.split()
    W, Hh = int(pr[0]), int(pr[1])
    num, den = pr[2].split("/")
    fps = float(num) / float(den)
    if len(pr) > 3 and abs(int(float(pr[3]))) % 180 == 90:
        W, Hh = Hh, W                        # ffmpeg auto-rotates portrait phone MOVs
    cmd = ["ffmpeg", "-v", "error", "-threads", "4", "-ss", str(t0), "-i", vid] \
        + (["-t", str(t1 - t0)] if t1 else []) \
        + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * Hh * 3 * 4)
    def gen():
        while True:
            buf = p.stdout.read(W * Hh * 3)
            if len(buf) < W * Hh * 3: break
            yield np.frombuffer(buf, np.uint8).reshape(Hh, W, 3)
        p.wait()
    return gen(), fps

def main(vid, outdir, t0v=0.0, t1v=0.0, hint_lo=0, hint_hi=80):
    os.makedirs(outdir, exist_ok=True)
    frames, fps = frames_of(vid, t0v, t1v)
    meta = open(os.path.join(outdir, "meta.csv"), "w")
    meta.write("seg,idx,ncc,time,vframe,H\n")
    H, seg, cur = None, 0, -1
    best = (0.0, None, -1)
    AVG = os.environ.get("AVG") == "1"       # save the DWELL MEAN of clean shots
    shots = []                               # instead of the single best (integration
    last_idx, last_t, slope = 0, 0.0, []     # experiment: is the noise temporal?)
    n, saved, unid, streak, t0 = 0, 0, 0, 0, time.time()
    stats = []

    def flush():
        nonlocal H, saved
        ncc, frame, vf = best
        if cur <= 0 or frame is None: return
        if AVG and shots:
            sel = [f for (nc, f) in shots if nc >= ncc - 0.015]
            frame = np.mean(sel, 0).round().astype(np.uint8)
        if saved < 2:                        # static rig: full refine early, light after
            sc, Hr = refine(frame, H, cur)
        else:
            sc, Hr = refine(frame, H, cur, coarse=(0.3, 0.15), fine=(0.15,))
        H = Hr
        np.save(os.path.join(outdir, f"s{seg}f{cur:05d}.npy"), warp(frame, Hr, FIELD))
        meta.write(f"{seg},{cur},{sc:.4f},{t0v + vf / fps:.2f},{vf}," +
                   " ".join(f"{v:.8g}" for v in Hr.ravel()) + "\n")
        stats.append(sc)
        saved += 1

    def ident(w, cands):                     # (ncc, idx) with top-vs-2nd margin test
        nccs = sorted((float(w @ truth256(i)) / len(w), i) for i in cands)
        (n1, i1), n2 = nccs[-1], nccs[-2][0] if len(nccs) > 1 else 0.0
        return (n1, i1) if n1 > 0.06 and n1 > 3 * abs(n2) else (n1, -1)

    booted = False
    for frame in frames:
        n += 1
        if H is None:
            H = find_quad(frame)
            if H is None: continue
            print(f"quad found at frame {n}", flush=True)
        if not booted:                       # worker bootstrap: wide hint-window scan
            if n % 8 not in (1,): continue
            w = normed(warp(frame, H, 256), M256)
            ncc, idx = ident(w, range(hint_lo, hint_hi))
            if idx < 0: continue
            _, H = refine(frame, H, idx)
            booted = True
            cur, best = idx, (0.0, None, -1)
            last_idx, last_t = idx, n / fps
            print(f"booted at idx {idx} (t={t0v + n / fps:.1f}s)", flush=True)
            continue
        w = normed(warp(frame, H, 256), M256)
        tnow = n / fps
        exp = last_idx + (tnow - last_t) * (np.median(slope) if slope else 12)
        ncc, idx = ident(w, range(max(0, last_idx), int(exp) + 8))
        if idx < 0:
            streak += 1; unid += 1
            if streak % 25 == 0:             # rescan: restart / big gap / geometry drift
                wide = range(0, max(last_idx + 300, hint_hi))
                ncc, idx = ident(w, wide)
                if idx < 0 and find_quad(frame) is not None:   # maybe H drifted: re-lock
                    Hq = find_quad(frame)
                    wq = normed(warp(frame, Hq, 256), M256)
                    ncc, idx = ident(wq, wide)
                    if idx >= 0:
                        _, H = refine(frame, Hq, idx)
                        w = normed(warp(frame, H, 256), M256)
                        ncc = float(w @ truth256(idx)) / len(w)
                if idx >= 0:
                    flush()
                    if idx < last_idx - 2:
                        seg += 1
                        print(f"  RESTART detected -> segment {seg} (idx {idx})", flush=True)
                    cur, best = idx, (0.0, None, -1)
                    last_idx, last_t, slope = idx, tnow, []
            if idx < 0: continue
        streak = 0
        if idx != cur:
            flush()
            if idx < last_idx - 2:
                seg += 1
                print(f"  RESTART detected -> segment {seg} (idx {idx})", flush=True)
                slope = []
            cur, best = idx, (0.0, None, -1)
            shots = []
            if idx > last_idx and last_t > 0 and idx - last_idx <= 3:
                slope.append((idx - last_idx) / (tnow - last_t))
                slope = slope[-50:]
            last_idx, last_t = idx, tnow
        if ncc > best[0]: best = (ncc, frame.copy(), n)
        if AVG: shots.append((ncc, frame.copy()))
        if n % 1000 == 0:
            print(f"  {n} frames t={tnow:.0f}s seg={seg} idx={idx} ncc={ncc:.3f} "
                  f"saved={saved} unid={unid} ({time.time() - t0:.0f}s)", flush=True)
    flush()
    meta.close()
    print(f"DONE {vid}: {n} frames -> {saved} saved, {unid} unidentified, "
          f"NCC mean {np.mean(stats):.3f} min {np.min(stats):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":                   # vid outdir [t0 t1 hint_lo hint_hi]
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 0.0,
         float(a[4]) if len(a) > 4 else 0.0,
         int(a[5]) if len(a) > 5 else 0, int(a[6]) if len(a) > 6 else 80)
