# cam_daemon.py — runs on the Surface. Keeps cam0 open (no per-shot reopen/warmup),
# writes the latest frame to cam.jpg ~3x/s, atomically. The PC pulls it via scp.
import os, time, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cam.jpg")

c = cv2.VideoCapture(0, cv2.CAP_MSMF)
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920), c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
for _ in range(120): c.read()          # ISP settle
print("streaming", flush=True)
i = 0
while True:
    ok, f = c.read()
    i += 1
    if not ok or i % 10: continue      # ~3 fps to disk
    cv2.imwrite(OUT + ".w.jpg", f)
    try:
        os.replace(OUT + ".w.jpg", OUT)
    except OSError:
        pass                           # scp had it open; next frame wins
