# glyphs.py — shared generator, alphabet V2.1. Both ends derive frames from counter n.
# V2.1: positional glyphs (bars/halves/checkers) replaced by topological ones after the
# V2.0 fail-matrix analysis (docs/glyph_errors_V2.0.png). Saints keep their class indices
# so warm-started models retain them. Line glyphs are drawn analytically (any tile px ok).
import numpy as np, cv2

LADDER = [24, 20, 16]                  # default; set_ladder() overrides on both ends

def set_ladder(spec):                  # "24,20,16,12,8" — must match on server & client
    LADDER[:] = [int(x) for x in str(spec).split(",")]

def set_canvas(w, h):                  # legacy canvas (server side: min(screen, config))
    global CANVAS_W, CANVAS_H
    CANVAS_W, CANVAS_H = w, h

GRID = None                            # (cols, rows) announced by the server's SNC QR;
def set_grid(cr):                      # None -> derive from the canvas (old archives)
    global GRID
    GRID = cr

PALETTE = np.array([[255, 255, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]], np.uint8)
CANVAS_W, CANVAS_H = 1920, 1030
PANEL_W = 360
RING, GAP = 8, 8

SHAPE_NAMES = ["solid", "X", "plus", "cutSlash", "cutBackslash", "cornerTL", "cornerTR",
               "cornerBL", "ring", "dot", "Z", "N", "diagA", "diagB", "Y", "Y90"]
COLOR_NAMES = ["white", "green", "red", "yellow"]

_BLOCKY = {0: ("####", "####", "####", "####"), 5: ("###.", "##..", "#...", "...."),
           6: (".###", "..##", "...#", "...."), 7: ("....", "#...", "##..", "###."),
           8: ("####", "#..#", "#..#", "####"), 9: ("....", ".##.", ".##.", "...."),
           12: ("##..", "##..", "..##", "..##"), 13: ("..##", "..##", "##..", "##..")}
_cache = {}

def shape_mask(i, S):                  # (S, S) float mask, 0..1 (anti-aliased lines)
    if (i, S) not in _cache:
        if i in _BLOCKY:
            m4 = np.array([[c == "#" for c in r] for r in _BLOCKY[i]], np.float32)
            m = cv2.resize(m4, (S, S), interpolation=cv2.INTER_NEAREST)
        else:
            D = S * 4
            im = np.zeros((D, D), np.float32)
            w = max(1, D // 5); mg = w // 2 + 1
            L = lambda a, b, v=1.0: cv2.line(im, a, b, v, w)
            if i == 1:  L((0, 0), (D - 1, D - 1)); L((D - 1, 0), (0, D - 1))       # X
            elif i == 2: L((D // 2, 0), (D // 2, D - 1)); L((0, D // 2), (D - 1, D // 2))
            elif i == 3: im[:] = 1; L((0, D - 1), (D - 1, 0), 0.0)                 # cutSlash
            elif i == 4: im[:] = 1; L((0, 0), (D - 1, D - 1), 0.0)                 # cutBackslash
            elif i in (10, 11):                                                    # Z / N
                L((mg, mg), (D - mg, mg)); L((D - mg, mg), (mg, D - mg))
                L((mg, D - mg), (D - mg, D - mg))
                if i == 11: im = np.rot90(im).copy()
            elif i in (14, 15):                                                    # Y / Y90
                L((mg, mg), (D // 2, D // 2)); L((D - mg, mg), (D // 2, D // 2))
                L((D // 2, D // 2), (D // 2, D - mg))
                if i == 15: im = np.rot90(im).copy()
            m = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
        _cache[(i, S)] = np.clip(m, 0, 1)
    return _cache[(i, S)]

def inks(S):                           # per-shape ink fraction at tile size S
    return np.array([shape_mask(i, S).mean() for i in range(16)], np.float32)

def spec(n):                           # n -> tile px, grid cols/rows
    px = LADDER[n % len(LADDER)]
    if GRID: return px, GRID[0], GRID[1]
    inner_w = CANVAS_W - PANEL_W - 2 * (RING + GAP)
    inner_h = CANVAS_H - 2 * (RING + GAP)
    return px, inner_w // px, inner_h // px

def grid(n):                           # n -> (rows, cols) shape ids + color ids
    px, cols, rows = spec(n)
    rng = np.random.default_rng(n)
    return rng.integers(0, 16, (rows, cols)), rng.integers(0, len(PALETTE), (rows, cols))

def render(n):                         # n -> (CANVAS_H, CANVAS_W, 3) RGB frame (no QR)
    px, cols, rows = spec(n)
    shp, col = grid(n)
    img = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    w, h = cols * px + 2 * (RING + GAP), rows * px + 2 * (RING + GAP)
    img[:h, :w][:RING], img[h - RING:h, :w] = 255, 255
    img[:h, :RING], img[:h, w - RING:w] = 255, 255
    masks = np.stack([shape_mask(i, px) for i in range(16)])
    tiles = np.round(masks[shp][..., None] * PALETTE[col][:, :, None, None, :]).astype(np.uint8)
    o = RING + GAP
    img[o:o + rows * px, o:o + cols * px] = tiles.transpose(0, 2, 1, 3, 4).reshape(
        rows * px, cols * px, 3)
    return img
