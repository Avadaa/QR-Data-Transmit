# v3 on REAL camera footage — verdict report (2026-07-19)

Question asked: can a v3 (DMT) demapper learn the real monitor→phone-camera channel
well enough to be reliable, or do we abandon the theory?

**Verdict in one line: the v3 configuration we shipped (PAM-4, NC=20, A=20) is DEAD
on the real channel (41% SER) — but the measurements that killed it also show,
empirically on the same footage, that a redesigned v3 wire (PAM-2, ~6–8 loaded
coefficients, higher amplitude, piecewise geometry) runs at ~2% raw bit error and
projects to 175–500 KB/s — above the 126.6 KB/s glyph record. Don't abandon the
theory; abandon the constellation.**

## Data

Two handheld-tripod videos of the web transmitter in v3 mode (PC monitor, r16 cells,
62×62 grid, training frames = seeded truth):
- `4k 60fps.MOV`: 3840×2160 @29.9fps, 246 s, HEVC 48 Mbit/s → **1601 labeled frames**
  (idx 1–1601, zero gaps), ~1.78 camera px / logical px.
- `1080p 60fps.MOV`: 1920×1080, 78 s → **492 labeled frames**, ~0.89 cam px/logical px.

Pipeline (`harvest.py`): ring-quad detect → frame idx by NCC against reconstructed
truth fields (no NN needed — training payloads are PCG64-seeded) → sub-pixel
homography refinement by maximizing truth-NCC (coordinate descent on the 4 corners,
256- then 512-res objectives) → best-shot per idx (ISP blends transitions) → 1024²
warp saved. Alignment verified at the full-res NCC peak within ±0.25 px (global).
8 parallel workers, ffmpeg raw-pipe slices (cv2 HEVC seeking is unreliable — that
cost one false "restart" diagnosis).

## Experiment battery (`train_real.py` → `results_real.txt`)

Demapper = the PoC arch (conv32-conv64-fc512, ~1M params), input 20×20 crops
(cell+2px margin), labels 3ch × NC=20 PAM-4 symbols. Val = every 7th frame.

| experiment                                   | val SER |
|----------------------------------------------|---------|
| synth-pretrained (channel-sim), zero-shot    | 50.5%   |
| pretrain + finetune ALL 1373 frames          | 41.3%   |
| real-only from scratch                       | 39.7%   |
| finetune 5 / 20 / 50 / 100 / 300 frames      | 45.7 / 42.3 / 42.1 / 42.0 / 41.8% |
| lr: const 1e-3 / cosine / 1e-4 / 3e-4        | 42.9 / 42.0 / 42.8 / 42.4% |
| aug: none / subpix-only / all / INT jitter   | **40.2** / 40.6 / 42.7 / 49.8% |
| width ×2                                     | 35.2%   |
| 1080p (finetuned on 1080p)                   | 53.5%   |

Reads: (a) the plateau is CHANNEL-LIMITED — no lr/data/aug lever moves it more than
~2 points; (b) integer jitter costs 9 points → the jitter law holds on real data;
(c) adding noise/brightness aug HURTS (real data already carries the real noise);
(d) width ×2 buys 4.5 points → some learnable structure exists beyond ×1 capacity;
(e) 1080p at this distance (0.89 px/logical) is unusable for v3. 4K or closer.

## Channel measurement (`analyze.py`, `analyze_4k.txt`)

Linear matched filter per coefficient (in PAM level units, spacing 2):
noise σ = 1.47 (coeff 0) rising to 4.0 (coeff 19); gain rolls 1.0 → 0.39.
At A=20/PAM-4 even the best coefficient is ~1.5σ per half-spacing → the shipped
wire never had a chance on this channel. The NN beats linear by ~10 points of SER
(41 vs 51) — real but not decisive at this SNR.

**Empirical PAM-2** (sign decisions conditioned on |level|=3 = a binary wire at the
same peak amplitude, measured, not modeled):

| coeff | linear sign-err | NN sign-err |
|-------|-----------------|-------------|
| 0     | 2.73%           | **1.92%**   |
| 1     | 2.98%           | **2.64%**   |
| 2–4   | 5.6–6.3%        | 3.4–4.4%    |
| 5–6   | 9.0–9.6%        | 5.6–5.8%    |

## The two structural findings

**1. The noise is STATIC, not temporal.** Dwell-mean harvest (average of ~4 clean
shots per idx) changes σ and sign-error by <2% relative (`integrate_test.py`).
Multi-shot integration is a dead lever. The noise is codec quantization (this
footage: HEVC 0.19 bpp — the encoder quantizes DCT detail, i.e. exactly our
modulation) + demosaic/moiré/monitor structure. The live receiver decodes ISP
frames with NO video codec → these numbers are likely a LOWER bound on live quality.

**2. Global homography violates the jitter law locally.** Spatial map of sign-err@|3|
(coeffs 0–5): field center 1.3–3%, right corners 18–42%. Per-block re-alignment
(±1 px translation search per 8×8-cell block) recovers the mid-field to **1.4–3.0%
everywhere** and halves the edges; the needed shifts are 0.5–1.4+ px with a radial,
lens-distortion-like pattern. v3 geometry must be piecewise (per-region offsets on
top of Hinv — the handheld shift-ring idea applied spatially), and the receiver's
camera toggle #6 (geometric distortion correction OFF in production) should flip ON
for v3 sessions.

## What a viable v3.1 wire looks like (from these numbers)

- **PAM-2** (levels ±3, all energy at max margin), **coeffs 0–5 or 0–7**, 3 channels
  → 18–24 raw bits/cell at ~2–3% raw BER (empirical, this footage, incl. codec).
- With bit-level coding at rate ~0.7 → **~12–17 net bits/cell → 175–245 KB/s @30 fps**
  on the web 62×62 grid. (Glyph record: 126.6 KB/s; glyph cell = 6 raw bits.)
- **Amplitude ×1.5–2** (A=30–40; pixel std still only ~19–25, field stays pastel):
  σ scales down linearly → projected 18–36 bits/cell at 2%/coeff → **250–500 KB/s
  raw**. Needs validation footage (additive-noise assumption).
- Requirements: 4K capture, piecewise geometry, subpixel-only augmentation, no
  synthetic noise aug on real data, NC>8 only if footage proves it.

## What I need from you (next filming round)

1. **A ProRes clip** (Settings → Camera → Formats → Apple ProRes ON, 4K30, 60–90 s
   of the same v3 training run). This isolates codec damage — the single biggest
   unknown. If σ drops a lot, live projections rise accordingly; ~6 GB/min is fine.
2. **A v3.1 clip**: I add experimental knobs to the web page (PAM-2 mode, amplitude
   A=20/30/40, NC picker); you film the same setup once per setting (60–90 s each,
   4K, HEVC is OK for A/B since codec hits all settings equally).
3. Optional: same at ~25% closer distance (px density lever, r12-era lesson).

## Files

`harvest.py` (video → labeled warps), `merge.py`, `train_real.py` (battery),
`analyze.py` (channel + redesign calculator), `integrate_test.py` (temporal test),
`debug_align.py`; data in `data/4k` (1601), `data/1080p` (492), `data/4kavg` (204);
raw numbers in `results_real.txt`, `analyze_4k.txt`, `battery.log`.

---

# v3.1 VALIDATION ROUND (same day): the redesign works — **0.17% BER, record-beating**

New clip: `4k 60fps v3.1 amp40 pam-2 nc8 tile16px trainfps15.MOV` (127 s, 4K60
portrait, HEVC 48 Mbit/s, different/bigger monitor, ~1.75 cam px/logical px) →
**1561 labeled frames, zero gaps** (rate ~12.9 idx/s). Wire = exactly what round 1
prescribed: PAM-2 (±3), NC=8, A=40. Results (`run_v31.py` → `results_v31.txt`):

| step                                        | BER      |
|---------------------------------------------|----------|
| PC-only pretrain (synthetic + channel sim)  | sub-0.2% at 1000 steps |
| **zero-shot on real footage**               | **7.6%** (round 1: 50.5%) |
| finetune 5 / 20 / 50 / 100 frames           | 2.7 / 1.1 / 0.8 / 0.8% |
| finetune all 1338 frames                    | 0.40%    |
| + measured distortion-field correction      | **0.163%** (eval-time only!) |

- **Distortion field**: mean |shift| 1.0 px, p90 2.0 px (real lens distortion).
  Applying the per-cell correction to the ALREADY-TRAINED model at eval time drops
  0.40% → 0.163%; retraining with it adds nothing more. It's a static calibration —
  exactly a receiver-side CAL step.
- **Spatial map**: 0.02–0.2% across virtually the whole field; only the 4 extreme
  corner blocks reach 1.6–2.6%.
- **All 8 coefficients usable** (0.1–0.3% each), both PAM levels, all 3 channels
  (B slightly worse as always).
- **Throughput verdict (measured BER, conservative r0.83 coding): ~138 KB/s net
  @15 fps, ~277 KB/s @30 fps on the web 62×62 grid — beats the 126.6 KB/s all-time
  glyph record at HALF the frame rate**, through codec-compressed footage (live ISP
  frames should be cleaner still). At 0.17% BER even plain RS nsym32 byte-coding
  has 4× headroom, and NC=12 at A=40 is the obvious next capacity step.

Verdict upgrade: v3 is not merely "not dead" — **v3.1 as filmed already beats the
glyph wire's best-ever number with margin**. The remaining work is engineering, not
physics: phone-side demapper port (same TileNet-family arch → the existing ANE
minting path), bit-level wire coding, distortion CAL in the receiver, NC/A tuning.
