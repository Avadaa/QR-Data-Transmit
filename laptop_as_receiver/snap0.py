# snap0.py — one snapshot from cam0, saved next to this script (runs on the Surface;
# the PC pulls snap0.jpg via scp — the Surface has no key toward the PC).
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap0.jpg")
c = cv2.VideoCapture(0, cv2.CAP_MSMF)
c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920), c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
for _ in range(120): c.read()      # ISP/AWB settle
if config.CAM_FOCUS: c.set(cv2.CAP_PROP_FOCUS, config.CAM_FOCUS)
if config.CAM_EXPOSURE: c.set(cv2.CAP_PROP_EXPOSURE, config.CAM_EXPOSURE)
for _ in range(30): c.read()       # snap0 must see what the receiver sees
ok, f = c.read()
c.release()
cv2.imwrite(OUT, f, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved", f.shape, f"mean {f.mean():.1f}", flush=True)
