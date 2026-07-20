# STATUS — what is proven, what is not (2026-07-19)

Everything below was measured on ONE rig: PC (RTX workstation) or MacBook M1 as
transmitter, iPhone 14 Pro as receiver, tripod-mounted at 40–60 cm, indoor
daylight. "Byte-perfect" = md5 of the received payload equals the sent payload.

## Proven, byte-perfect, repeatedly

| what | number | conditions |
|---|---|---|
| v2 glyphs, 16 px native, ANE | 43.4 → **78.7 KB/s** | 15 → 30 fps, mac panel, 4K capture |
| v2 glyphs, 12 px read at 16×16 ("12c16") | 94.8 → **126.6 KB/s** (record) | 20 → 30 fps, nsym 32, 4K |
| Real-file delivery into the Files app | 576 KB PDF, 275/275 frames, 0 rebuilt | web transmitter, wire whitening |
| Long transfer + thermal drift | **6.4 MB mp3 @ 65.7 KB/s**, md5 exact | runtime retraining every 50 frames kept err ≤0.5 % |
| On-device session training | ~0–1 % tile err in 1–8 chunks (20–160 frames) | warm-started models, all px |
| On-device ANE model minting | any grid ≤16383 tiles, logit-identical | batch-dim rewrite of the compiled model |
| Chunked ANE minting (>16383 tiles) | mac-validated; deployed | not yet exercised live |
| v2 glyphs, 8 px read at 16×16 ("8c16") | 7.8 KB/s, md5 exact — **CPU only** (693 ms/frame; ANE bugs found + fixed after this run, no ANE rerun yet) | mac fullscreen grid (T=21090), 163-frame train to 3.8 % |
| 30 px big glyphs ("30c16") | works in training/CAL; grid 33×33 @ web-1024 | never a full timed transfer |

## Implemented but NEVER live-proven

| what | state |
|---|---|
| **v3.1 DMT wire end-to-end** | Phone decode implemented (CPU demapper, bit-exact vs torch) and deployed; web transmits it; **no live phone transfer has ever run**. All v3.1 numbers (0.16 % BER, 138–277 KB/s projections) come from OFFLINE decoding of phone-camera FOOTAGE (`training/v3/REPORT.md`) — strong evidence, not a live run. |
| **Handheld** | Two architectures built (trust-but-verify v1.2, per-frame-naming v2 with tracked geometry). Field runs: mechanics worked (64 % frames named, re-locks firing) but training stalled at 8 % err → **no handheld transfer ever succeeded**. Parked in favor of the web/big-cell path. |
| v3.1 on-device demapper training / runtime tuning | not implemented (pretrained model only) |
| demapper on the ANE | not implemented (CPU ~0.7–3 s/frame at 16 px grid → keep fps ≤5) |
| receiver distortion calibration | measured offline (fixes 0.40 %→0.16 % BER), not on the phone |
| Android | nothing exists |

## Known limits & traps (paid for; don't rediscover)

- **The camera must be ABSOLUTELY stable.** Even micro-shakes (bumped desk,
  hand on the phone, phone cable tug) wreck training labels, per-frame
  recognition and the cached homography. 
  The handheld work exists precisely because this constraint is
  so hard; it never fully succeeded.
- **4K capture GREATLY helps even 16 px glyphs** (settings toggle #9): more
  camera px per tile is nearly free accuracy (measured: first-ever 0.00 %
  training error came with 4K; it costs only ~+7 ms texture upload). 1080p at
  ~0.9 camera px per logical px is **unusable** for v3 and marginal for small
  glyphs — 4K is mandatory for px ≤12 and strongly recommended always.
- The nsym handshake: TRAIN QR must announce NSYM (an nsym-64 codeword passes an
  nsym-32 syndrome check "cleanly" → silent garbage). Both ends handle this now.
- Structured payloads need the whitening scrambler (SCR=1): a PDF's xref once
  rendered 86 % white tiles and bloomed the camera dead.
- Lens distortion breaks a single global homography at the field corners
  (±1–2 px). Glyphs tolerate it; v3 DMT does not (sub-pixel geometry law).
- Video-based training data is HEVC-compressed: codec noise inflates measured
  error floors; live ISP frames are cleaner.
- Camera 3A must settle on FIELD content, then lock; CAL scores decoded symbols
  vs seeded truth (lapvar alone picks clipping exposures).
- iPhone free-signing apps expire after 7 days — rebuild + reinstall.
