# record.py — dumb recorder: grab camera frames at sensor rate, save the UNIQUE ones
# (new-content dedup vs the last saved frame; writer pool keeps capture at full rate).
# Usage: record.py <folder> [seconds=30]   ->  frames/<folder>/<seq>_<t>.jpg
import os, queue, sys, threading, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

name = sys.argv[1]
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames", name)
os.makedirs(OUT, exist_ok=True)

Q = queue.Queue(maxsize=90)    # ~3s of encode headroom; full -> frame dropped, counted

def writer():
    while True:
        item = Q.get()
        if item is None: return
        i, t, f = item
        cv2.imwrite(os.path.join(OUT, f"{i:05d}_{t:.3f}.jpg"), f,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_Q])

W = [threading.Thread(target=writer, daemon=True) for _ in range(3)]
for w in W: w.start()
cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W), cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)
print("warming up...", flush=True)
for _ in range(config.CAM_WARMUP): cam.read()
if config.CAM_FOCUS: cam.set(cv2.CAP_PROP_FOCUS, config.CAM_FOCUS)
if config.CAM_EXPOSURE: cam.set(cv2.CAP_PROP_EXPOSURE, config.CAM_EXPOSURE)
for _ in range(30): cam.read()
print(f"recording {SEC:g}s -> {OUT}  (ctrl+c stops early)", flush=True)

def small(f): return cv2.cvtColor(f[::16, ::16], cv2.COLOR_BGR2GRAY).astype(np.int16)

i = uniq = dup = dropped = 0
last_s = None
t0 = last_stat = time.time()
try:
    while time.time() - t0 < SEC:
        ok, f = cam.read()
        if not ok: continue
        i += 1
        s = small(f)
        if last_s is None or np.abs(s - last_s).mean() > config.TH_CHANGE:
            last_s = s; uniq += 1
            try: Q.put_nowait((uniq, time.time(), f))
            except queue.Full: dropped += 1
        else:
            dup += 1
        if time.time() - last_stat > 0.5:
            print(f"\rcaptured {i}  unique {uniq}  dup {dup}  in-mem {Q.qsize()}  ",
                  end="", flush=True)
            last_stat = time.time()
except KeyboardInterrupt:
    pass
dt = time.time() - t0
cam.release()
for _ in W: Q.put(None)
for w in W: w.join()
print(f"\n{i} frames in {dt:.1f}s = {i/dt:.1f} fps | unique saved {uniq}, "
      f"dup skipped {dup}, dropped {dropped}", flush=True)
