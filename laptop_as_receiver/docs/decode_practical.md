# decode_practical.md — data mode: settled architecture (transmit.py built, receiver.py next)

## Session shape
- START: transmit.py shows one QR centered, payload "---START---,FRAMES=<n>,SIZE=<bytes>,PX=<px>"
  and waits for a keypress. Receiver is started during this; it reads the QR, learns the plan.
- DATA: on keypress, all frames stream at fps (default 5), each shown exactly 1/fps s. No side
  QR on data frames — index lives in the payload header (see wire format).
- END: transmit.py shows "---END---,FRAMES=<n>" and holds. Receiver sees it, closes capture,
  reports missing frame indices; decode workers keep crunching the spool until done.
- v1 recovery for missing frames: rerun the pass (receiver fills gaps by header index).
  Fountain outer layer later makes any-K-frames sufficient and kills the straggler tail.

## Receiver architecture (Surface, 4 GB RAM — hard constraint; as BUILT)
- capture thread: cam.read at full rate, dedup in real time (~2 ms: subsample every 16th
  px, per-band abs-diff). Commit the NEWEST stable frame of each dwell when the next
  change arrives (ISP blends transitions; first-stable is a ghost), after 1.5s for held
  screens, and — if a dwell never settles (sun shimmer) — its calmest frame under
  TH_BLEND. Stability must hold in EVERY of 6 horizontal bands (bottom-band blends
  hide in a global mean).
- kept frames: JPEG q85 in a RAM queue (qbytes tracked in stats); the worker frees each
  after processing. --save also spools to disk. ~1 MB/frame transient, 4 GB never limits.
- one NN worker thread: geometry + NN + RS per frame; capture never blocks on decode,
  the backlog drains after END.
- geometry: cached homography FIRST — the rig is static, so once any frame's header CRC
  passes, its Hm is reused (skips contour+verify ~240 ms AND survives glare that breaks
  the ring). Full search (adaptive-threshold ring hole + refine_quad + sync-corner
  orientation, top-2 by corr, header CRC arbitrates) only for the first frame or when
  the cached path fails.
- NN: torch forward, channels_last, center-crop 24x24 (val path = reference).
  Softmax low-confidence tiles -> RS erasures (worth 2x errors in coding budget) — TODO.

## Wire format v1 (transmit.py is the reference)
- field: no QR panel — SCALES TO THE SCREEN: field = min(screen, config W_USE/H_USE
  caps; the caps are camera-view limits of the PC rig, ~x1820). ring RING=8 + GAP=8;
  grid cols=(w-2*(RING+GAP))//px etc. The START/TRAIN QR announces COLS/ROWS — the
  receiver takes the grid from there (config fallback for pre-v1.3 captures).
  PC monitor r16: 110x62 = 6820 tiles (17.7 KB/s); mac 1512x982: 92x59 (14.0 KB/s)
- symbol = 6 bits: s in 0..63, shape = s>>2 (16), color = s&3 (4); bits MSB-first via
  np.packbits/unpackbits convention, raster order (row-major), corners+header skipped
- sync corners (orientation anchors, fixed): TL=solid/white TR=X/green BL=ring/red BR=dot/yellow
  (180-degree rotation maps TL<->BR: distinguishable pair -> full disambiguation)
- header = 42 bits (30-bit frame index + 12-bit crc32(idx_be32)&0xFFF) x 7 COPIES =
  49 tiles, majority-voted per bit at the receiver (a single copy is statistically dead
  at even 5% bit err — measured lesson)
- payload (v1.2): NBLK=(payload_bytes//255) RS(255,191) codewords per frame (nsym=64,
  25% overhead, 32 byte-errors/block), BYTE-INTERLEAVED (.T.ravel(): bursts spread over
  blocks). r16: 19 blocks = 3629 data B/frame -> 17.7 KB/s goodput at 5 fps. Frame is
  all-or-nothing: header CRC + all blocks RS-ok, else counted failed (repass/fountain
  recovers). Measured channel (daylight, pc_L16d): mean 0.58% bit err, worst ~4% -> RS
  headroom ~2.5x mean. transmit dwells 3 extra ticks on frame 0 (ISP ghost of START QR);
  NEVER flush with black - AE swing kills the first ~5 frames (measured lesson).
- v1.4: the START QR also announces NSYM — RS overhead is set on the TRANSMIT side
  only (nsym=32 on its command line; receiver follows the QR, config is the fallback
  for pre-v1.4 captures). nsym=32: 12.5% overhead, 16-error wall (~4.5% tile err) —
  needs the tuned-brightness rig; nsym=64 for hostile light.
- payload should be pre-compressed (zstd/gzip) — the link moves opaque bytes; 7-bit/char
  tricks are source coding and always lose to real compression.
- fountain (v1.3): after the FRAMES sources, REPAIR extra frames (config = PERCENT of
  sources, default 3%, floor 8; the QR announces the resulting COUNT), each the
  XOR of a random half of the sources (mask = rng(REPAIR_SEED + j), all-zero guard
  flips bit j%N). Header idx >= FRAMES marks repair; START QR carries REPAIR=R.
  Receiver rebuilds any missing sources by GF(2) elimination (solve_repair) — ANY
  (lost + ~2) surviving repair frames suffice, dense random masks are near-full-rank.
  No feedback, no repass. At K=100 this beats real LT/Raptor (those pay their
  complexity to make decoding cheap at K ~ 10^4+).

## Speed ledger (Surface, r16 — test/bench_rx.py replays saved frames)
- pre-data-mode (2026-07-06 am): capture 48 / qr 135 / contour 128 / cands 7 /
  verify 107 / prep 33 / nn 260 = 718 ms, 1.4 fps
- wire v1.3 fast path (2026-07-06 pm): jpeg 34 / warp 16 / prep 31 / nn 235 / rs 32
  = 348 ms, 2.9 fps at 2 threads. qr+contour+verify are GONE (cached Hm); rs syndromes
  vectorized (rs.py _syndromes — Horner loop was ~40ms even on clean frames).
- live-only wins bench_rx can't see: raw-frame queue under QRAW backlog (skips jpeg
  encode+decode ~90ms combined) and 4-thread drain after END (~25% off nn).
- nn is now 2/3 of the frame — remaining levers: 16x16-input retrain (~2x conv cost,
  matches r16's native camera resolution; needs per-rung input sizes) or int8 via
  onnxruntime/OpenVINO AVX2 kernels (Kaby Lake has no VNNI; naive torch quant may LOSE).

## Model / drift
- night-trained pc_L16: 0.32% val err but 4.32% LIVE err in daylight (13x) — lighting drift
  is the dominant error source; soak/fine-tune per lighting epoch before sizing RS
- rung choice by worst-decile err over a drift-spanning window, not mean (the mean lies)
- r12/r8 are physics-walled (22-25% even for GPU specialists) — production ladder is 24,20,16

## Gotchas already paid for (do not rediscover)
- ISP shifts image ~8px after stream start: warm 120 frames before trusting geometry
- AWB oscillates (~6s): model must be trained across states; pilot-free by design
- daylight windows merge with the ring in Otsu -> ONLY the inner-edge hole is safe (pick_H)
- harsh direct sun (2026-07-06 pm): AE dims the ring to ~40-110 brightness while glare
  raises the local mean -> adaptive threshold BREAKS the ring, the field hole merges
  with the background or shatters -> per-frame geometry died on 27/97 frames whose
  PIXELS were fine (1.5% tile err). Fix = cached homography (header CRC proves a frame's
  Hm; reuse it, rig is static). A saturation-blob fallback was tried and rejected:
  JPEG chroma bleed + glare fringes cap it at 20-70px corner error, never sub-tile.
- worse sun + a nudged rig (next run): ring broken in ALL frames -> nothing ever seeded
  the Hm cache -> 0/100 live. Adaptive block 51 is too LOCAL then (window sits entirely
  in moat/glare); block 101 recovered the field. process()/locate() now sweep
  (51,-5) -> (101,-5) -> (101,-10); extra variants only cost time on doomed frames.
  Replay: 98/100 byte-exact on that capture set.
- the same glare run: 3/100 dwells never had a stable frame -> dedup dropped them
  entirely; TH_BLEND best-effort commit now keeps the calmest view instead
- ssh to Surface: camera scripts need ssh -tt, else MSMF cam.read hangs forever in warmup
- segno: micro=False always; pyzbar: symbols=[QRCODE]; torn frames: header CRC rejects
- train/infer crop mismatch: train jitters 24x24 from 28x28; inference center-crops 24x24
- PER-SESSION CAMERA STATE (mac rig, 2026-07-07): a fresh 0.9%-err model scored 30%
  tile err (3/100) on a transmit run 37 min later — ambient light was CONSTANT (soak
  frames pixel-stable for 100 min). The receiver had warmed its 120 frames on the
  black START-QR screen -> ISP locked ~20-25% brighter and warmer (B rose half as
  much as R/G -> white reads as yellow) than the soak session, and kept moving DURING
  the 22s run (err 9% -> 30%). Brightness augmentation (0.7-1.3x linear) cannot cover
  exposure changes: clipping + tone curve + WB are nonlinear. The frames themselves
  were fine — a model trained on them reads them at 0.87%. Fixes, in order: (1)
  training-mode calibration right before ENTER (camera settles ON field content and
  the model tunes to that exact state); (2) frozen-model runs: warm the camera while
  the FIELD is on screen, never the black/QR screen. "Lighting drift" and "session
  state" are two different channel-movers; this rig only has the second.
