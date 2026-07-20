# transmit.py — data-mode server. Mode select -> (optional training) -> START QR ->
# data frames at fps -> END QR. Wire format: docs/decode_practical.md. Settings live in
# config.py; override per run with bare key=value args (fps=3 px=20 nsym=32).
# Usage: transmit.py [file] [key=value ...]   (no file -> SYNTH_KB of seeded random
#                                              bytes, written to sent_payload.bin)
import hashlib, io, math, os, queue, sys, threading, time, zlib
import numpy as np, pygame, segno
import config, rs
from glyphs import GAP, PALETTE, RING, shape_mask

a = config.override(sys.argv[1:])
FILE = a[0] if a else None
PX, FPS, NSYM = config.PX, config.FPS, config.NSYM
TFPS = config.FPS_TRAINING or FPS      # gentle training rate; data mode keeps FPS

pygame.init()          # the field scales to the screen (capped by config W_USE/H_USE =
screen = pygame.display.set_mode(pygame.display.get_desktop_sizes()[config.MONITOR],
                                 pygame.SCALED | pygame.FULLSCREEN, display=config.MONITOR,
                                 vsync=1)   # pin the PANEL to 60Hz in System Settings!
# SCALED is NOT cosmetic: pygame only honors vsync with SCALED/OpenGL. Plain FULLSCREEN
# free-ran (measured on the mac: 4.6ms flips + clock.tick(15) pacing at 67.8ms) — the
# ~1.1ms/frame drift walked swap boundaries across refresh lines every ~15-25 frames =
# the periodic odd-index dwell losses and the dissolved-frame-at-95% glitch.
pygame.mouse.set_visible(False)   # a cursor parked ON the ring breaks the hole search
clock = pygame.time.Clock()        # camera-view limits); the QR carries COLS/ROWS

def present(img, fps):   # hold img for exactly 60/fps refreshes when fps divides 60
    t0 = time.time()
    n = int(60 // fps) if fps <= 60 and 60 % fps == 0 else 0
    frame = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
    for _ in range(max(n, 1)):
        screen.fill((0, 0, 0))
        screen.blit(frame, (0, config.TOP_PAD))
        pygame.display.flip()
    if n == 0 or time.time() - t0 < (n - 0.5) / 60:
        clock.tick(fps)   # flips returned too fast = vsync NOT engaged (pygame honors
                          # vsync only with SCALED/OpenGL; plain FULLSCREEN ignores it —
                          # this silently made the display free-run and ignore FPS)
W_USE = min(screen.get_width(), config.W_USE or 10 ** 6)
H_USE = min(screen.get_height(), config.H_USE or 10 ** 6) - config.TOP_PAD  # notch room
cols = (W_USE - 2 * (RING + GAP)) // PX
rows = (H_USE - 2 * (RING + GAP)) // PX
W0, H0 = cols * PX + 2 * (RING + GAP), rows * PX + 2 * (RING + GAP)
T = rows * cols
CORNERS = [0, cols - 1, (rows - 1) * cols, T - 1]          # sync tiles (raster idx)
SYNC = [(0, 0), (1, 1), (8, 2), (9, 3)]                    # solid/white X/green ring/red dot/yellow
free = np.setdiff1d(np.arange(T), CORNERS)
HDR, PAY = free[:49], free[49:]        # header = 42 bits x 7 copies (majority-decoded)
CAP = len(PAY) * 6                                         # payload bits per frame
NBLK = (CAP // 8) // 255               # RS codewords per frame (byte-interleaved)
KPF = NBLK * (255 - NSYM)              # data bytes per frame after coding

data = open(FILE, "rb").read() if FILE else \
    np.random.default_rng(config.PAYLOAD_SEED).integers(
        0, 256, config.SYNTH_KB * 1024, dtype=np.uint8).tobytes()
if not FILE: open("sent_payload.bin", "wb").write(data)
FRAMES = math.ceil(len(data) / KPF)
buf = np.frombuffer(data, np.uint8)
rows_ = np.concatenate([buf, np.zeros(FRAMES * KPF - len(buf), np.uint8)]).reshape(FRAMES, KPF)
REPAIR = config.REPAIR and max(8, int(np.ceil(FRAMES * config.REPAIR / 100)))
if REPAIR:             # config.REPAIR is a PERCENT of sources (floor 8 frames)
    rep = np.zeros((REPAIR, KPF), np.uint8)
    for j in range(REPAIR):
        m = np.random.default_rng(config.REPAIR_SEED + j).integers(0, 2, FRAMES).astype(bool)
        if not m.any(): m[j % FRAMES] = True
        rep[j] = np.bitwise_xor.reduce(rows_[m], 0)
    rows_ = np.concatenate([rows_, rep])
TOT = len(rows_)       # frame idx >= FRAMES -> repair (receiver knows from the header)
code = rs.encode(rows_.reshape(TOT * NBLK, 255 - NSYM).astype(np.int32), NSYM)
code = code.reshape(TOT, NBLK, 255)
print(f"grid {cols}x{rows} r{PX}, {NBLK} RS blocks = {KPF} B/frame, {FRAMES}+"
      f"{REPAIR} repair frames, {len(data)} bytes -> {FPS:g} fps = "
      f"{KPF * FPS / 1024:.1f} KB/s goodput ({NSYM / 255:.0%} RS overhead)", flush=True)
print(f"payload md5 {hashlib.md5(data).hexdigest()}  (compare with receiver)", flush=True)

M = np.stack([shape_mask(i, PX) for i in range(16)]).astype(np.float32)
# Build the 64 possible rendered tiles once.  Expanding M[shp] as float32 for every
# full-screen frame used several times the framebuffer size in temporaries; on a
# Retina Mac that is enough allocation churn to beachball as training starts.
TILE_RGB = (M[:, None, :, :, None] * PALETTE[None, :, None, None, :]).astype(np.uint8)

def base_sym(i):     # header + sync corners for frame index i
    hdr = np.concatenate([np.unpackbits(np.frombuffer(i.to_bytes(4, "big"), np.uint8))[2:],
                          np.unpackbits(np.frombuffer((zlib.crc32(i.to_bytes(4, "big"))
                                                       & 0xFFF).to_bytes(2, "big"), np.uint8))[4:]])
    sym = np.zeros(T, np.uint8)
    sym[HDR] = np.tile(hdr, 7).reshape(49, 6) @ (1 << np.arange(5, -1, -1))
    for ci, (s, c) in zip(CORNERS, SYNC):
        sym[ci] = (s << 2) | c
    return sym

def render_sym(sym):
    shp, col = (sym >> 2).reshape(rows, cols), (sym & 3).reshape(rows, cols)
    tiles = TILE_RGB[shp, col]
    field = tiles.transpose(0, 2, 1, 3, 4).reshape(rows * PX, cols * PX, 3)
    img = np.zeros((H0, W0, 3), np.uint8)
    img[:RING], img[-RING:], img[:, :RING], img[:, -RING:] = 255, 255, 255, 255
    img[RING + GAP:RING + GAP + rows * PX, RING + GAP:RING + GAP + cols * PX] = field
    return img

def frame_img(i):
    sym = base_sym(i)
    fb = np.unpackbits(code[i].T.ravel())   # .T interleaves: bursts spread over blocks
    fb = np.concatenate([fb, np.zeros(CAP - len(fb), np.uint8)])
    sym[PAY] = fb.reshape(-1, 6) @ (1 << np.arange(5, -1, -1))
    return render_sym(sym)

def train_img(i):    # payload derives from i alone -> receiver knows every tile's truth
    sym = base_sym(i)
    sym[PAY] = np.random.default_rng(config.TRAIN_SEED + i).integers(
        0, 64, len(PAY)).astype(np.uint8)
    return render_sym(sym)

def qr_surf(text):
    q = segno.make(text, error="m", micro=False)
    buf = io.BytesIO(); q.save(buf, kind="png", scale=10, border=4); buf.seek(0)
    return pygame.image.load(buf, "q.png").convert()

BG = None            # control screens ride ON a glyph field: a black-bg QR made the
def field_bg():      # camera AE overexpose (white QR bloomed out, unreadable) and
    global BG        # parked the ISP in a state unlike the data frames
    if BG is None:
        BG = pygame.surfarray.make_surface(train_img(0).transpose(1, 0, 2))
    screen.fill((0, 0, 0)); screen.blit(BG, (0, config.TOP_PAD))

_redraw = None       # closure that repaints the CURRENT screen; the blocking waits call
                     # it each loop so the fullscreen display link never idles (an idle
                     # link makes the next present()/flip stall = beachball freeze)
def show_qr(text):
    global _redraw
    qs = qr_surf(text)
    def draw():
        field_bg()
        screen.blit(qs, ((screen.get_width() - qs.get_width()) // 2,
                         (screen.get_height() - qs.get_height()) // 2))
        pygame.display.flip()
    _redraw = draw
    draw()

# ---- keyboard stage control -------------------------------------------------------
# Keys are read from the fullscreen window; the console (type + ENTER) is a focus-
# independent fallback that does the same thing. Stage flow:
#   idle:            [t] -> training QR      [Enter] -> arm QR (skip training)
#   training QR:     [Enter] -> training frames start
#   training frames: [Enter] -> arm QR
#   arm QR:          [Enter] -> transmit the payload
#   (Esc quits anywhere; Space is an alias for Enter)
sig = queue.SimpleQueue()
def _console_reader():
    # Exactly one thread owns stdin.  Starting a new input() thread at every stage left
    # the previous one blocked whenever a pygame key advanced the stage; on macOS those
    # competing reads could consume a later Enter or wedge the terminal input lock.
    while True:
        try:
            sig.put(input())
        except (EOFError, OSError):
            return

threading.Thread(target=_console_reader, daemon=True).start()

def console(msg):    # ENTER typed in the terminal works even if the window lacks focus
    print(msg, end="", flush=True)

def _quit(): pygame.quit(); sys.exit(0)

def _line():         # a typed terminal line (lowercased) if one is waiting, else None
    try: return str(sig.get_nowait()).strip().lower()
    except queue.Empty: return None

def wait_choice():   # idle: block until [t] / [Enter] / Esc -> "train" or "go"
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                _quit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_t: return "train"
                if e.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER): return "go"
        c = _line()
        if c is not None: return "train" if c.startswith("t") else "go"
        if _redraw: _redraw()        # keep the display link alive (see _redraw note)
        time.sleep(0.02)

def wait_go():       # block until [Enter]/[Space] / Esc (keeps pumping = stays responsive)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                _quit()
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_RETURN,
                                                       pygame.K_KP_ENTER):
                return
        if _line() is not None: return
        if _redraw: _redraw()        # keep the display link alive (see _redraw note)
        time.sleep(0.02)

def advance():       # non-blocking (used inside the training-frame loop): True on [Enter]
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            _quit()
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_RETURN,
                                                   pygame.K_KP_ENTER):
            return True
    return _line() is not None

START_QR = (f"---START---,FRAMES={FRAMES},REPAIR={REPAIR},SIZE={len(data)},"
            f"PX={PX},FPS={FPS:g},COLS={cols},ROWS={rows},NSYM={NSYM}")

# ---- stage machine ----------------------------------------------------------------
_redraw = lambda: (field_bg(), pygame.display.flip())   # idle = plain field: the camera
_redraw()                                               # warms on data-frame brightness
console("Receiver running?   [t] = training QR first   "
        "[Enter] = arm QR (skip to transmit)   (Esc = quit): ")
if wait_choice() == "train":
    show_qr(f"---TRAIN---,PX={PX},FPS={TFPS:g},COLS={cols},ROWS={rows},NSYM={NSYM}")
    console("TRAIN QR up (receiver should say [train]).   [Enter] = start training frames: ")
    wait_go()
    console("TRAINING — watch the receiver's err.   [Enter] = stop and show the arm QR: ")
    i = 0
    while not advance():             # one flip per frame (NOT present()'s multi-flip
        # TRAIN QR already built frame zero as BG.  Reusing it makes the first transition
        # immediate and avoids a second full-screen allocation at the sensitive flip.
        surf = BG if i == 0 else pygame.surfarray.make_surface(
            train_img(i).transpose(1, 0, 2))                                  # vsync
        screen.fill((0, 0, 0)); screen.blit(surf, (0, config.TOP_PAD))          # busy-
        pygame.display.flip()        # loop, which stalled the display here); [Enter]
        clock.tick(TFPS)             # is checked every frame so stopping is instant.
        i += 1
    _redraw = lambda: None           # training frames self-refresh; nothing to keep alive

show_qr(START_QR)   # arm; after training this also makes the receiver save its tuned model
console("ARM QR up (receiver saves/arms).   [Enter] = transmit the payload: ")
wait_go()

t0 = time.time()
step = max(1, TOT // 5)              # ~5 progress lines however long the payload
aborted = False
for i in range(TOT):
    for e in pygame.event.get():     # Esc aborts mid-stream; the receiver still
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            aborted = True            # assembles + fountain-rebuilds what it got
    if aborted: break
    present(frame_img(i), FPS)
    if i == 0: time.sleep(3 / FPS)   # dwell on frame 0: ISP/AE settle on real content,
    if (i + 1) % step == 0: print(f"frame {i + 1}/{TOT}", flush=True)   # no black swing
print(f"sent {i if aborted else TOT}/{TOT} frames in {time.time() - t0:.1f}s", flush=True)
print(f"payload md5 {hashlib.md5(data).hexdigest()}  (compare with receiver)", flush=True)
show_qr(f"---END---,FRAMES={FRAMES}")
time.sleep(3)   # field-backed QRs are read from HELD screens (>=1.5s stable): a fast
                # quit used to flash END too briefly to ever be seen
console("END QR up.   [Enter] = quit: ")
wait_go()
pygame.quit()
