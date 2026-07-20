# train.md — shapes/colors training system spec

## Principle
One-way channel. Both ends share `glyphs.py`; the whole frame derives from the counter
`n` (rung = LADDER[n % 5], seed = n). The sync QR carries only `SNC{n}` — it is the
clock, not the label. The client regenerates ground truth locally from n.

## Pipeline (train.py, runs on the input machine)
1. **Watch** `screen.png` (written by input.py — screen recorder now, camera later;
   train.py must not know or care which).
2. **Sync**: pyzbar the frame -> n. No QR / seen n before -> discard. This self-rejects
   torn or mid-transition captures.
3. **Geometry (per frame, every frame)**: homography from the white ring's 4 corners +
   QR finder patterns as anchor points. Per-frame solving is what makes handheld/moving
   cameras workable — never rely on a stored calibration. (Loopback: near-identity.)
4. **Harvest**: warp tile area to canonical grid, cut one crop per tile at C px
   (C = 24 canonical, tiles resampled to C regardless of rung), label from glyphs.grid(n).
   Sanity gate: if quick-check accuracy vs truth is ~chance, discard frame (bad sync).
5. **Dataset**: append (crop, shape_id, color_id, rung, timestamp) to an .npz shard per
   rung. Cap + reservoir-sample so drift states stay represented.
6. **Train** (torch, PC): small CNN, two softmax heads.
   crop 24x24x3 -> conv3x3(16) relu -> pool2 -> conv3x3(32) relu -> pool2 ->
   fc(64) -> heads: shape(16), color(4). ~30k params.
   **Augmentation carries the robustness budget**: random +-2px shift, +-3% scale,
   +-2 deg rotation, directional motion blur, brightness/WB jitter, mild noise.
   This is what absorbs camera angle error, hand shake, and drift residuals.
7. **Export**: weights -> model.npz. Receiver inference = numpy im2col forward pass
   (no torch on the Surface).
8. **Report**: per-rung held-out symbol error rate (shape, color, joint), rolling over
   time so drift phases are visible. This table IS the product of training.

## Q1 — choosing the ladder rung
Never guessed, always measured: train until per-rung accuracy plateaus (no improvement
over the last k batches), then pick the densest rung whose WORST-DECILE joint symbol
error (over a window spanning drift states, >=30 min of captures) stays under the outer
ECC budget (~1-2% SER). Transmitter keeps sweeping all rungs regardless (it can't hear
us); the receiver simply knows which rungs it can read. The future beep backchannel's
only job is to tell the transmitter where to lock.

## Q2 — model errors after deployment
Three layers, in order of cost:
- **ECC eats the routine errors** (see Q3): the model never needs to be perfect, only
  under-budget. A model at 0.5% SER + RS is a working link.
- **Online trickle-training**: data mode interleaves occasional training frames (they
  self-identify via sync QR) -> receiver keeps fine-tuning forever. Training is not a
  phase, it is a background process. This tracks slow drift (AWB wave, lighting, angle).
- **Confidence alarm**: rising mean softmax entropy = channel left the training
  distribution -> receiver falls back one rung (it has models/accuracy for all rungs)
  and trickle-trains until the denser rung is trustworthy again.

## Q3 — inference-time verification
Same armor as v1, the NN is never trusted bare:
- Payload frames: per-channel RS blocks + CRC32 (reuse optic.py framing over the tile
  symbol stream; 6 bits/tile -> bytes).
- **Soft outputs are the upgrade over v1**: tiles whose top softmax prob is low are
  passed to RS as ERASURES (known-bad positions) — RS corrects 2x more erasures than
  errors, so NN confidence directly doubles the error budget.
- CRC32 is the final arbiter; a failed frame is dropped and re-arrives next cycle
  (later: fountain outer code makes any-K-frames sufficient).

## Model registry — one model per setup (camera + lighting + geometry)
The channel is dominated by rig-specific statics (moire phase, PSF, gamma, AWB habits);
per-setup models memorize them, a general model would have to ignore them. Models are
tiny (~30k params, ~120KB) so many is free.
- `models/<name>.npz` (weights) + `models/<name>.json` (per-rung accuracy table, sample
  counts, parent model, created/updated stamps).
- train.py <name> [--init <parent>]: train new model, warm-started from parent if given
  (from-scratch only for the very first). Warm-start makes new setups minutes, not hours.
- Existing name -> resumes/trickle-trains that model. Inference loads models/<name>.npz.
- Dataset shards are tagged per setup too; a model only eats its own setup's shards.

## Camera-readiness checklist (nothing changes in train.py when camera arrives)
- input.py swaps SRC screen->camera; same screen.png contract.
- Per-frame homography already assumed (step 3).
- Augmentation already covers expected optics; real camera data simply joins the dataset.
- Rung table re-learns itself: expect the camera to kill rung 8, maybe 12 — the report
  says so, we don't.
