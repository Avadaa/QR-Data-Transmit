# laptop_as_receiver/ — the first receiver generation (reference)

Before the iPhone app, the receiver was a Python/torch pipeline on a Windows
tablet (a Surface with a 2-core i5-7300U, 4 GB RAM, rear camera) filming the
transmitter screen. **Superseded by `../ios/` — kept because it is a complete,
working, CPU-only reference implementation** of the whole receive side
(geometry search, NN decode, RS + fountain, in-session training), and because
an Android/desktop port would start from exactly this shape.

Run it with the shared wire modules on the path (they live in
`../transmitter/`): copy `config.py`, `glyphs.py`, `rs.py` next to these files
or add that folder to `PYTHONPATH`. Dependencies: numpy, opencv-python, torch
(CPU is fine), pyzbar.

```bash
python receiver.py --save rx_frames                 # decode a transmission
python receiver.py --model rx_L16 --tune-name rx_L16  # use/tune a named model
```

## What's here

| file | role |
|---|---|
| `receiver.py` | THE receiver: camera capture + dedup, geometry (ring contour + homography, header-CRC-verified, cached), torch NN decode, RS + fountain assembly, in-session + runtime training, OpenVINO int8 engine swap |
| `client.py`, `train.py`, `output.py`, `input.py` | the legacy self-training loop (server sweeps glyph "rungs" with a sync-QR clock; client harvests + trains autonomously for hours) |
| `record.py`, `snap0.py`, `cam_daemon.py`, `plot.py` | frame recorder, camera snapshot with locked settings, capture daemon, training-curve plots |
| `deploy.py` | rig ship tool (code + models to the other machines over ssh) — study for the multi-machine workflow, edit hosts before use |
| `docs/` | the era's documentation — see below |

## Era docs (2026-07-04 → 07-16, sorted oldest-first)

- `train.md` — the training-system SPEC: seeded ground truth ("the QR is a
  clock, not a label"), augmentation as the robustness budget, per-rig model
  registry, RS-erasure design. Still the project's intellectual foundation.
- `train_practical.md` — how it was actually built: alphabet V2.1 redesign
  (positional glyphs → topological), reservoirs, channels_last 2.8×,
  warm-start doctrine.
- `machines.md` / `commands.md` — the 3-machine rig (PC trains, mac transmits,
  tablet receives) and every operational command. Hostnames/paths sanitized.
- `surface_camera.md` — the MSMF camera-driver war: readback lies, the
  autofocus lottery, why frames are judged by decoded truth and never by
  sharpness metrics. The iPhone camera doctrine descends directly from this.
- `decode_practical.md` — data-mode architecture + wire v1 history + the
  gotchas ledger (ISP warmup, AWB oscillation, sun-glare geometry failures,
  per-session camera state).
- `iGPU_CPU_performance.md` — CPU inference study: OpenVINO int8 = 2.1× over
  torch at +0.1 % err on AVX2 (no VNNI needed); iGPU dead (2020 driver). The
  quantization result matters for ANY future CPU receiver (Android!).

## Why it was retired

Compute: ~40–90 ms/frame NN on 2 cores vs the iPhone's 16 ms on the ANE, and
the camera driver could not be trusted (locked-setting readback lies, AF
lottery). The performance evolution that led to the phone: 718 ms/frame
(1.4 fps) at first light → 121 ms (8.2 fps) after the cached-homography +
native-tiles + vectorized-RS campaign → the remaining gap closed by silicon.
