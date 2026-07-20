# train_practical.md — how the training system is actually built

## V2.2 (2026-07-07): model epoch for the mac-transmitter rig
- SAME alphabet and wire as V2.1 — only the model folder rolls: config.MVER ("V2.2")
  names models/<MVER>/ everywhere; loaders and --init parents fall back to config.MPREV
  ("V2.1"), so mac models warm-start from surf_L16n/pc_L16n without copying files.
- rationale: models are rig-bound (docs/train.md registry section); the mac rig is a
  new camera geometry + panel, so its models get a clean folder instead of piling into
  the monitor-rig collection.

## V2.1 (2026-07-05): alphabet swap + configurable ladder + versioned models
- alphabet: 8 positional glyphs (vbar/hbar/left/right/top/bottom/chkA/chkB) replaced by
  X, plus, cutSlash, cutBackslash, Z, N, Y, Y90 — after V2.0 fail-matrix analysis
  (bars 7.5-8.6% err from position-only ambiguity; see docs/glyph_errors_V2.0.png,
  new set: docs/alphabet_V2.1.png). Saints keep class indices -> warm-start friendly.
- line glyphs render analytically (4x supersample + INTER_AREA) -> any tile px legal now
- ladder configurable per run, MUST match on both ends:
  server: output.py 0.6 0 24,20,16,12,8 | trainer: --ladder 24,20,16,12,8
  planned models: a=24,20,16  b=24,20,16,12,8  c..g=single-rung 24/20/16/12/8
- models/ split: V2.0/ = old-alphabet models+data+logs (frozen), V2.1/ = new. train.py &
  client.py write V2.1, resolve --init parents from V2.1 then V2.0. Surface mirrored;
  its 7GB frames/ archive deleted (old glyphs, no longer re-harvestable value).
- naming convention: <where>_<ladder> e.g. loop_L24-16, surf_L24-8, surf_L16only

## Files & roles
- input.py — dumb frame source: overwrites screen.png every PERIOD s (screen now, camera later; consumers never know which)
- output.py — server: frame n = tile grid (from shared glyphs.py) + sync QR "SNC{n}"; dwell 6-7s > capture period; sweeps ladder via n
- glyphs.py — shared generator: n -> rung, grid dims, full shape/color grid (seed = n)
- train.py <name> [--init <parent>] — harvester + trainer; models/<name>.npz + .json

## Choices
- torch (cuda if present) for training, weights exported to plain npz — receiver will run a ~30-line numpy forward pass, zero torch on the Surface
- crops stored 28x28, trained 24x24 — the 4px margin feeds random jitter-crop augmentation
- augmentation = +-2px crop jitter, per-channel brightness 0.7-1.3x, gaussian noise; motion blur to be added when camera data arrives
- reservoir per rung: 20k train / 3k val, random replacement when full (biases toward recent frames — desirable under drift)
- val split by tile index % 10 — stable, never leaks into train
- net: conv16-pool-conv32-pool-fc64 -> heads shape(16) + color(4); ~90k params, 323KB npz
- one optimizer session across frames (Adam 1e-3), STEPS=30 minibatches of 256 per harvested frame, rungs sampled uniformly
- per-frame homography: ring contour quad -> 8 candidate orderings -> keep those putting the QR right of the tile area -> pick by correlation of measured tile colors vs expected (ink fraction is constant by design, so COLOR is the orientation signal)
- sanity gate: color correlation < 0.5 -> frame rejected (torn capture / bad sync)
- pyzbar restricted to QRCODE symbols (its PDF417/DataBar decoders spam warnings on the tile field)

## Saving / kill behavior
- model + json: every 10 frames AND on Ctrl+C (finally-block) — npz is 323KB, saving costs milliseconds; could be every frame if we cared
- dataset shards (data_<name>_r<px>.npz, ~50-250MB total): every 50 frames + on Ctrl+C — np.savez is ~0.5-2s, too heavy per-frame
- hard kill (SIGKILL/timeout): model survives to last 10-frame checkpoint; shards to last 50-frame checkpoint. Ctrl+C loses nothing.
- restart with same name resumes: weights + shards reload automatically

## Surface training-speed benchmarks (2026-07-05, test/bench_train.py; m3 2C/4T, torch 2.12 cpu)
- baseline (threads=3, bs=512, NCHW): ~4.7-5.7k samples/s — what the client shipped with
- threads: 4 > 2 > 3 (!) — 3 threads hits an HT contention hole; use os.cpu_count()
- batch size: samples/s flat 256->1024 (~5.8k), drops at 2048 — batch is not a lever here
- **channels_last: train 5.9k -> 9.5k samples/s (+62%); inference 11.7k -> 32.9k (2.8x)**
  -> decode at r16 density = ~5.6 fps on this tablet; the future infer.py must use it
- bf16 autocast: 10x SLOWER (no bf16 hardware on Kaby Lake) — dead
- torch.compile: fails, no MSVC on the Surface — dead unless a compiler is installed
- torch-directml (iGPU): NO wheels for python 3.13 — blocked; expected gain (<2x, small-conv
  launch overhead on HD615) not worth side-loading python 3.12. Dead end, documented.
- APPLIED to client.py: threads=cpu_count + channels_last net/batches (~1.7-2x net training)

## Expectations
- loopback (perfect channel): all rungs -> val_acc 1.000 within ~2 ladder passes (verified 2026-07-04; confirms pipeline, not difficulty)
- camera: rungs will separate — expect 24/20/16 trainable, 12 marginal, 8 dead; the json per-rung table is the ground truth, not this guess
- warm-start (--init) should cut camera convergence to minutes: conv features transfer, later layers absorb the rig

## Approaches (why like this)
- geometry classical, classification learned: homography per frame handles camera motion/angle; augmentation absorbs the residual +-2px — NN never learns geometry
- everything derives from n: sync QR is a clock, not a label; torn frames self-reject (no QR, or color-corr gate)
- training never ends: same loop later trickle-trains during data mode (training frames self-identify by their QR)

## Verified (loopback, 2026-07-04)
- output 6s dwell + input 5s period + train.py: corr=1.00 every frame, 2.5-4k crops/frame, 5 rungs at val_acc 1.000 after ~10 frames, kill-save intact
