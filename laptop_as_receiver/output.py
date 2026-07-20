# output.py — training server (transmitter). Shows frame n: tile grid from the shared
# generator + a small sync QR carrying only n. The client derives everything else.
# Config: constants below, argv: output.py [dwell_s] [start_n] [ladder e.g. 24,20,16,12,8]
import io, os, sys
import numpy as np, pygame, segno
import config, glyphs
from glyphs import PANEL_W, render, set_canvas, set_ladder

DWELL = 7                   # seconds per frame (> input capture period, so every
START = 0                   # capture window contains one full stable frame)

a = config.override(sys.argv[1:])
DWELL = float(a[0]) if len(a) > 0 else DWELL
START = int(a[1]) if len(a) > 1 else START
if len(a) > 2: set_ladder(a[2])

def qr_surf(n):      # QR announces the grid -> the client scales with the server
    px, cols, rows = glyphs.spec(n)
    q = segno.make(f"SNC{n},{cols}x{rows}", error="m", micro=False)
    buf = io.BytesIO(); q.save(buf, kind="png", scale=10, border=2); buf.seek(0)
    return pygame.image.load(buf, "q.png").convert()

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=config.MONITOR)
pygame.mouse.set_visible(False)   # a cursor parked ON the ring breaks the hole search
set_canvas(min(screen.get_width(), config.CANVAS_W or 10 ** 6),        # auto-scale,
           min(screen.get_height() - config.TOP_PAD, config.CANVAS_H or 10 ** 6))
print(f"canvas {glyphs.CANVAS_W}x{glyphs.CANVAS_H} (grid announced in the QR)", flush=True)
clock, n, running = pygame.time.Clock(), START, True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False
    frame = pygame.surfarray.make_surface(render(n).transpose(1, 0, 2))
    screen.fill((0, 0, 0))
    screen.blit(frame, (0, config.TOP_PAD))
    qs = qr_surf(n)   # left-aligned in panel: camera view ends ~x1820, right of that is lost
    screen.blit(qs, (glyphs.CANVAS_W - PANEL_W + 8,
                     config.TOP_PAD + (glyphs.CANVAS_H - qs.get_height()) // 2))
    pygame.display.flip()
    print(f"frame {n}", flush=True)
    n += 1
    clock.tick(1 / DWELL)
pygame.quit()
