# WIRE.md — the on-screen format (v2 glyphs / v3.1 DMT)

Both transmitters (Python and web) are bit-exact implementations of this spec;
the iOS receiver implements the decode side. All PRNG streams are numpy's
`default_rng(seed)` (PCG64) — the Swift and JS ports are validated bit-exact.

## Field geometry

- White ring `RING=8` logical px, black gap `GAP=8`, then the cell grid.
- Web: a 1:1 square field (default 1024 logical px → `cols = rows =
  (1024-32)//px`). Python: fills the screen (`cols = (w-32)//px` etc).
- 4 corner tiles are fixed sync glyphs (TL solid/white, TR X/green, BL
  ring/red, BR dot/yellow) — orientation disambiguation.
- The next 49 free tiles (row-major) are the HEADER: a 42-bit word — 30-bit
  frame index + 12-bit `crc32(idx_be32) & 0xFFF` — repeated 7×, 6 bits per
  glyph tile, majority-voted per bit at the receiver.
- All remaining tiles are payload.

## Control plane (QR codes between phases)

`---TRAIN---,PX=,FPS=,COLS=,ROWS=,NSYM=[,V=3,NC=,A=,LV=]`
`---START---,FRAMES=,REPAIR=,SIZE=,PX=,FPS=,COLS=,ROWS=,NSYM=,SCR=1[,NAME=file.ext][,V=3,NC=,A=,LV=]`
`---END---,FRAMES=`

The receiver builds its grid at the FIRST control QR and rebuilds if a later QR
announces a different NSYM. `NAME=` makes a complete transfer save as a real
file. `SCR=1` = whitening on (always, in current transmitters).

## Payload pipeline (both wires)

1. Payload bytes are split into rows of `kpf` bytes per frame.
2. **Whitening**: row i is XORed with `PCG64(5000+i).integers(256, kpf)` —
   content-shaped bright floods are impossible; receiver unscrambles at
   assembly only.
3. **Fountain**: after the FRAMES source rows, REPAIR extra rows, each the XOR
   of a random half of the sources (`mask = PCG64(9000+j).integers(2, FRAMES)`,
   all-zero guard flips bit j%N). Any ~(lost+2) surviving repairs rebuild any
   lost sources by GF(2) elimination. No feedback channel needed.
4. **Reed-Solomon**: each row is `nblk` RS(255, 255−nsym) blocks,
   byte-interleaved across the frame (byte i of the stream = block `i % nblk`,
   position `i / nblk`) so error bursts spread over blocks.
5. A frame decodes all-or-nothing: header CRC + every RS block, else it counts
   failed and the fountain covers it.

## Wire v2 — glyphs (proven, record 126.6 KB/s)

- Cell = one of 16 shapes × 4 colors (white/green/red/yellow) = 6 bits.
- `nblk = (pay_tiles*6/8)/255`, `kpf = nblk*(255-nsym)`.
- Training frames: payload tiles = `PCG64(7000+idx).integers(64, n)` — the
  receiver regenerates the same truth and trains on its own camera frames.

## Wire v3.1 — DMT / "modem mode" (offline-proven, no live run yet)

- Cell = `NC` 2-D DCT coefficients (zigzag order, DC excluded) × 3 color
  channels, PAM-modulated around mid-gray 128 with amplitude `A`:
  pixel = `128 + Σ level_j · A · basis_j`. Validated operating point:
  **PAM-2 (levels ±3), NC=8, A=40** → 24 raw bits/cell (vs 6 for a glyph).
- Header + corners STAY v2 glyphs (the glyph net keeps doing frame naming and
  orientation — hybrid frame). QRs add `V=3,NC=,A=,LV=`.
- `nblk = (pay_cells*3*NC*log2(LV)/8)/255`; bit order: cell-major, then
  channel (R,G,B), then coefficient; bits MSB-first into bytes.
- Training frames: `PCG64(7000+idx).integers(LV, cells*3*NC)`.
- Design laws from the real-footage study (`training/v3/REPORT.md`):
  geometry budget is SUB-PIXEL (±0.4 px; never integer-jitter augment a
  demapper), 4K capture mandatory, per-cell distortion correction is a static
  receiver calibration worth ~2.5× in BER.

## Seeds (shared constants)

`TRAIN_SEED=7000  REPAIR_SEED=9000  SCRAMBLE_SEED=5000  PAYLOAD_SEED=42`
(payload seed only for synthetic benchmark payloads).
