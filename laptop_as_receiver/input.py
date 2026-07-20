# input.py — capture stage of the shapes/colors link.
# Grabs whatever the receiver would see and overwrites screen.png every PERIOD s.
# SRC "screen" = screen recording of monitor 1 (loopback dev mode, default).
# SRC "camera" = pull a fresh cam0 frame from the Surface (ssh snap + scp).
# Consumers (train.py) only ever see screen.png — they never know the source.
# Config: constants below, overridden by argv: input.py [period_s] [out.png] [src]
import os, subprocess, sys, time
from PIL import ImageGrab

SRC    = "screen"
REGION = (1920, 0, 3840, 1030)     # monitor 1 in virtual-screen coords, minus 50px taskbar
PERIOD = 5
OUT    = "screen.png"
CAM    = ("dev@<laptop-ip>", r"C:\Users\dev\Desktop\QR_data_transmit")

a = sys.argv[1:]
PERIOD = float(a[0]) if len(a) > 0 else PERIOD
OUT    = a[1] if len(a) > 1 else OUT
SRC    = a[2] if len(a) > 2 else SRC

def grab_camera():     # pull latest frame from the Surface's cam_daemon.py
    host, rdir = CAM
    tmp = OUT + ".tmp.jpg"
    r = subprocess.run(f'scp {host}:"{rdir.replace(chr(92), "/")}/shapes_n_colors/cam.jpg" "{tmp}"',
                       shell=True, capture_output=True, timeout=30)
    if r.returncode:
        raise RuntimeError("pull failed — is cam_daemon.py running on the Surface?")
    from PIL import Image
    im = Image.open(tmp); im.load()
    im.save(OUT + ".w.png")
    os.replace(OUT + ".w.png", OUT)    # atomic swap: readers never see a partial file
    os.remove(tmp)
    return im.size

while True:
    t0 = time.time()
    if SRC == "camera":
        try:
            size = grab_camera()
        except Exception as e:
            print("pull failed, retrying:", e, flush=True)
            time.sleep(2); continue
    else:
        img = ImageGrab.grab(bbox=REGION, all_screens=True)
        img.save(OUT + ".w.png")
        os.replace(OUT + ".w.png", OUT)
        size = img.size
    print(time.strftime("%H:%M:%S"), "saved", OUT, size, flush=True)
    time.sleep(max(0, PERIOD - (time.time() - t0)))
