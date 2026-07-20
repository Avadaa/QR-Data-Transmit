# make_alphabet.py — render the full V2.1 glyph alphabet (16 shapes x 4 colors = 64
# symbols, 6 bits/cell) into a labeled reference grid for the README.
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "transmitter"))
from glyphs import shape_mask, PALETTE, SHAPE_NAMES, COLOR_NAMES

TILE = 96          # glyph render size
PAD = 12           # gap between cells
LEFT = 92          # left margin (color names)
TOP = 108          # top margin (title + shape index)
BOT = 96           # bottom margin (shape-name legend)
RIGHT = 24
COLS, ROWS = 16, 4
CELL = TILE + PAD

W = LEFT + COLS * CELL + RIGHT
H = TOP + ROWS * CELL + BOT

def font(sz, bold=False):
    for p in [r"C:/Windows/Fonts/consola.ttf", r"C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

img = Image.new("RGB", (W, H), (12, 12, 14))
d = ImageDraw.Draw(img)
f_title = font(26); f_hdr = font(20); f_small = font(16); f_leg = font(15)

d.text((LEFT, 18), "V2.1 glyph alphabet", font=f_title, fill=(235, 235, 235))
d.text((LEFT, 52), "16 shapes x 4 colors = 64 symbols  (6 bits per cell)",
       font=f_small, fill=(150, 150, 155))

# shape index numbers across the top
for c in range(COLS):
    x = LEFT + c * CELL + TILE // 2
    d.text((x, TOP - 22), str(c), font=f_hdr, fill=(120, 120, 128), anchor="mm")

# color names down the left
for r in range(ROWS):
    y = TOP + r * CELL + TILE // 2
    swatch = tuple(int(v) for v in PALETTE[r])
    d.text((LEFT - 14, y), COLOR_NAMES[r], font=f_small, fill=swatch,
           anchor="rm")

# the 64 glyphs, each on a black cell (as they appear on the field)
for r in range(ROWS):          # r = color
    for c in range(COLS):      # c = shape
        m = shape_mask(c, TILE)                       # (TILE,TILE) 0..1
        rgb = (m[..., None] * PALETTE[r]).round().astype(np.uint8)
        x0 = LEFT + c * CELL
        y0 = TOP + r * CELL
        d.rectangle([x0 - PAD // 2, y0 - PAD // 2,
                     x0 + TILE + PAD // 2, y0 + TILE + PAD // 2], fill=(0, 0, 0))
        img.paste(Image.fromarray(rgb), (x0, y0))

# shape-name legend at the bottom (index -> name), two rows of 8
d.text((LEFT, TOP + ROWS * CELL + 14), "shapes:", font=f_small, fill=(150, 150, 155))
per = 8
colw = (COLS * CELL) / per
for i, name in enumerate(SHAPE_NAMES):
    row = i // per
    col = i % per
    x = LEFT + col * colw
    y = TOP + ROWS * CELL + 38 + row * 22
    d.text((x, y), f"{i:>2} {name}", font=f_leg, fill=(175, 175, 182))

out = os.path.join(os.path.dirname(__file__), "..", "docs", "glyph_alphabet.png")
img.save(out)
print("wrote", out, img.size)
