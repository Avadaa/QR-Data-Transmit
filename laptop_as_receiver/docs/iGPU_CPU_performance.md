# iGPU_CPU_performance.md — Surface NN inference: OpenVINO vs torch, CPU vs iGPU
(2026-07-07 probe; NEXT PLANNED WORK — this doc is the resume point)

## Verdict in one line
The iGPU is blocked by a 2020 driver, but **OpenVINO CPU int8 is 2.1x faster than the
current torch path at +0.1% error worst case** — that gain is sitting on the shelf,
already exported and benchmarked, waiting to be wired into the receiver.

## Hardware truth (corrects older docs)
The Surface is an **Intel i5-7300U** (Kaby Lake, 2C/4T, base 2.6 GHz) + HD Graphics
620 iGPU — not the "m3" older notes claim. 4 GB RAM. No VNNI (int8 runs on AVX2 —
and still wins, see below). GPU driver: 27.20.100.8682, dated **2020-09-06**.

## Results (Surface, full clock*, one "frame" = the rung's real tile batch)

| engine              | L16 (5244 t)     | L12 (9348 t)   | L20 (3330 t)   | L16 w=0.5      |
|---------------------|------------------|----------------|----------------|----------------|
| torch fp32 (= receiver today) | 76.9 ms / 13.0 fps | 94.1 / 10.6 | 78.4 / 12.8 | 50.9 / 19.7 |
| OV CPU fp32         | 53.6 / 18.7      | 59.4 / 16.8    | 57.1 / 17.5    | 35.1 / 28.5    |
| OV CPU fp16         | 53.9 (= fp32)    | 57.2           | 56.3           | 35.3           |
| **OV CPU int8**     | **36.4 / 27.5**  | **37.4 / 26.7**| **34.8 / 28.8**| **24.1 / 41.5**|
| OV GPU (any prec)   | COMPILE FAILED   | same           | same           | same           |

Accuracy deltas (PC eval on shard val crops): fp16 = fp32 exactly; int8 costs
+0.02-0.12% absolute. L12/L20 (synthetic shards): 0.000% at every precision.

*full clock: this bench ran AFTER discovering Windows "Power saver" had been capping
the CPU at ~1.4 GHz. torch nn was 104 ms/frame in that state vs 76.9 here — check the
power slider before believing ANY Surface benchmark (thermal state already known to
swing 2.3x on top).

## Thread-count sweep (2026-07-16, test/igpu/bench_threads.py — warm chassis, ~25%
## slower than the cool-state table above; trust ratios)

| threads | torch fp32 L16/L12 | OV fp32 L16/L12 | OV int8 L16/L12 |
|---------|--------------------|-----------------|-----------------|
| 1 (= live while filming) | 132.8 / 149.8 ms | 97.0 / 114.8 | **65.8 / 88.3** |
| 2       | 79.7 / 89.8        | 53.3 / 73.2     | **35.4 / 47.9** |
| 4       | 75.7 / 87.9        | 53.4 / 78.3     | 35.4 / 53.0     |

- **2 threads = saturation**: OV gains NOTHING from 4 threads (L12 int8 is WORSE at 4t
  — HT contention). The 2 physical cores are the machine; HT is a lie here.
- Consequence for the receiver: OV int8 pinned to 2 threads gets FULL NN speed while
  leaving 2 logical processors to the capture thread — the CAM_THREADS tug-of-war
  mostly dissolves (today torch-1t runs 132-150ms while filming; OV int8-2t: 35-48ms).
- The live 4MB run that motivated this (ISOLATE off, torch 1t + capture + tuner
  contention, warm): nn averaged 167ms — consistent with the torch-1t row.

## Key findings
1. **int8 is basically free accuracy-wise** and ~1.5x over OV fp32 despite no VNNI —
   the old "naive quant may LOSE on Kaby Lake" fear is dead (that was about torch's
   own quant kernels; OpenVINO's AVX2 kernels are good).
2. **OV fp32 alone beats torch fp32 by 1.4x** on identical math — graph fusion,
   not precision. Swapping runtimes pays even without quantization.
3. **fp16 on CPU is storage-only** (decompressed to fp32 at load) — never a speed lever
   on this chip; it's the iGPU's native format instead.
4. **L12 int8 = 26.7 fps NN-only** -> the r12@15fps live-decode target has NN headroom
   on CPU alone (37 ms of a 67 ms/frame budget).
5. **iGPU: CL_BUILD_PROGRAM_FAILURE** at kernel build on every model/precision = the
   2020 driver's OpenCL compiler predates what OpenVINO 2026 emits. Not a model
   problem, not fixable from our side without a driver update.

## Paths & scripts (everything re-runnable)
- `test/igpu/export_models.py` (PC) — npz -> ONNX (dynamo=False; the new exporter
  crashes cp1252 consoles, run with PYTHONUTF8=1) -> OpenVINO IR fp32/fp16 +
  NNCF post-training int8 (calibration = 2048 real shard crops, receiver's exact
  preprocessing) + accuracy eval per precision. Outputs `test/igpu/out/`:
  `<name>_<prec>.xml/.bin`, source `.npz` (for the torch baseline), `meta.json`.
- `test/igpu/bench.py` (Surface) — torch baseline (4 threads, channels_last, 4096-crop
  chunks = the receiver's exact loop) + OV CPU/GPU x fp32/fp16/int8, median of 12,
  prints the sorted table. Usage: `python bench.py [torch|CPU|GPU ...]`.
- Models benched: L16 = **the Surface's live tuned loop_L16_best** (pulled back as
  `test/igpu/loop_L16_best_surf.npz` — the PC's copy was stale!), L12 = loop_L12n,
  L20 = loop_L20n, L16w05 = testing/model_size/net_w0.5_mac16 (width 0.5).
- Calibration/eval shards: pc_subtraining/data_mac16_r16.npz (real mac-rig),
  data_loopn_r12/r20.npz (synthetic). NOTE: L16 shows ~10% err on the mac16 shard —
  the shard is from an OLD camera session; the live model has been runtime-tuned away
  from it. Fine for speed + precision-DELTA comparisons, not an absolute quality read.
- Both machines have the toolchain: PC = openvino 2026.2.1 + nncf 3.2.0 + onnx;
  Surface = openvino (pip, py3.13).

## Next steps (agreed direction)
0. DONE 2026-07-16: OV engine wired into receiver.py (config.OV_THREADS=2, 0=torch).
   fp32 IR built at setup (1.4s), int8 quantized from the first CRC-proven field's
   crops, rebuilt after every tune (t+ENTER save and each runtime_tune), training
   phases read the live torch net (nn_sym(live=True)). Smoke test (no camera):
   test/igpu/test_swap.py — fp32 agree 1.0000, on-device int8 build 6.5s, int8 agree
   0.972 on random-NOISE crops (worst case; real crops were +0.1%), torch 102.7 ->
   OV int8 59.0 ms/frame warm. Awaiting first live transfer for the real numbers.
1. **Wire OV int8 into the receiver's nn_sym** — biggest single win available (2.1x on
   the 53-75% nn share of frame time). Keep torch for the training/tuning side; after
   each runtime tune, re-export + NNCF-quantize the clone (<1s for this toy net) and
   hot-swap the compiled model under NLOCK. Moving parts: torch->onnx->IR on the
   Surface per tune cycle (onnx 1.22 + nncf 3.2 installed there 2026-07-16 — full
   toolchain now version-matched with the PC; iGPU direction PARKED: nothing about
   the HD620 transfers to the phone goal, CPU int8 does).
2. Combine with **frame-parallel workers** (2-3 decode threads, torch/OV threads=1
   each) — multiplies with the runtime swap; ~4-6x total decode throughput expected.
3. **iGPU retry after driver update**: last Kaby Lake driver is the 31.0.101.2xxx
   legacy branch; Surface OEM lock usually forces a manual "have disk" INF install.
   Expectation: HD620 fp16 lands NEAR CPU-int8 speed — its real value is OFFLOAD
   (both cores freed for load/geometry/RS during capture), not raw speed. Decide
   with the user present; it's the tablet's display driver.
4. If more speed is ever needed cheap: width 0.5 + int8 = 24 ms (41 fps) at ~2x the
   model error — only viable where the error budget allows (see model_size study).

## Notes / gotchas for the resume
- PowerShell mangles `ssh ... "python -c \"...\""` quoting — use the Bash tool or ship
  a script file; this bit twice.
- OV GPU compiles are slow on first run; bench.py sets CACHE_DIR=out/ovcache.
- bench feeds random floats — speed only. Accuracy numbers come from export_models.py
  on the PC (real crops).
- The receiver currently holds torch weights in memory and fine-tunes a clone; an OV
  swap must keep sent_sym/truth_sym label plumbing untouched — only nn_sym's forward
  changes engine.
- Power slider "Best performance" + plugged in, and let it COOL: this machine's
  numbers move 2-3x with power/thermal state. Re-bench in-state before comparing.
