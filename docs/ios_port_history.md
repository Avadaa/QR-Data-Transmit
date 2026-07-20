# iPhone.md — porting the receiver to the iPhone (2026-07-17, iPhone 14 Pro, iOS 26.5)

The Surface was the dinosaur we tuned around; the phone is the future receiver. This doc
records the port's first two days: what was built, the architectural tricks that mattered,
the traps, and the measured numbers. Companion docs: `iGPU_CPU_performance.md` (the Surface
numbers we compare against), the mac's `~/Desktop/iOS Dev/README.md` (toolchain, signing,
ssh build workflow — read §11 before touching anything over ssh).

## Status in one line
`loop_L16_best` runs on-phone at **7.5 ms / full r16 field (134 fps, ANE)** and the whole
NN side of a frame (warp + classify) is **8.3 ms via Metal + ANE** — decode-identical to
the PC's torch pipeline (5244/5244 tiles) — and the model **trains on-device** (perturb
5.14% -> 0.19% err in 8.6 s). Not ported yet: camera capture, geometry search, RS/fountain.

## The app: "QR File Transmit"
Mac: `~/Desktop/iOS Dev/QR File Transmit`, XcodeGen project (`project.yml` is the truth,
`xcodegen generate` after adding files), bundle `com.example.QRFileTransmit`.
~600 lines of Swift, only Apple frameworks (Accelerate, CoreML, Metal, SwiftUI):
- `Net.swift` — the CNN hand-rolled: im2col + `cblas_sgemm` forward AND backward + Adam.
  Weights read raw from `weights.bin` (torch `state_dict` tensors concatenated fp32,
  fixed order). Training uses the receiver's exact augmentation (4px jitter crop,
  0.7-1.3x per-channel brightness, sigma-4 noise) and LR tiers.
- `CoreMLEngine.swift` — the same net as a Core ML model (see "single-tensor" below).
- `Warp.swift` — gather-LUT warp (CPU) + Metal compute-kernel warp (GPU).
- `Runner.swift` — decode/bench/train orchestration; writes its log to `Documents/log.txt`
  which the PC pulls headlessly via `devicectl device copy from` (no screenshots needed).
- Assets: one real Surface capture (`frame.jpg`), PC-computed `meta.json` (grid, inverse
  homography, header idx), `ref_sym.bin` (PC torch symbols), `truth_sym.bin` (RS-proven
  transmitted symbols), built by the prep script (exec's the real receiver.py slice; grid
  was unknown -> swept candidates, ONLY a full RS decode arbitrates — 42-bit CRC alone
  false-positives across a 64-candidate sweep, measured).

## Verification doctrine
Every engine/path must match the reference exactly, not approximately: symbols vs the
PC's torch output (5244/5244 required), header CRC as the independent proof-of-decode,
and for warp variants byte/float equality against the naive path. This caught nothing
being wrong — but it is the only reason we can claim the fp16 ANE path is safe (0 tile
flips on real crops; random-noise inputs DO flip 8/5244, so noise tests overstate fp16 risk).

## Trick 1 — Core ML wants ONE tensor, never a batch of tiny arrays
First Core ML attempt fed 5244 individual (3,16,16) arrays through `MLArrayBatchProvider`:
199-291 ms/frame — WORSE than CPU. Per-item dispatch overhead swamps a 90k-param net.
Fix: rebuild the model with input shape `(5244, 3, 16, 16)` — one predict call per frame.
Same silicon: 199 ms -> 45 ms (CPU) and unlocked GPU/ANE entirely. The model is built on
the mac with coremltools' MIL Builder **straight from weights.bin — no torch anywhere**
(`build_mlmodel2.py`), validated against a numpy forward before shipping. Xcode compiles
the `.mlpackage` resource to `.mlmodelc` automatically.

## Trick 2 — the Neural Engine
Core ML's `computeUnits` picks the silicon: `.cpuOnly` / `.cpuAndGPU` (GPU) / `.all` (ANE).
The ANE is fp16 — and free accuracy-wise here (confident logits; see doctrine above).
NN engines, same 5244-tile r16 field, median of 12, forward only:

| engine                              | ms/frame | fps   | vs Surface OV int8 |
|-------------------------------------|----------|-------|--------------------|
| Surface torch fp32 (old live path)  | 76.9     | 13.0  | 0.5x               |
| Surface OV CPU int8 (live engine)   | 36.4     | 27.5  | 1.0x               |
| iPhone CPU hand-rolled fp32 1-core  | 116-117  | 8.5   | 0.3x               |
| iPhone CPU hand-rolled fp32 all-core| 37-43    | ~25   | ~1x                |
| iPhone CPU batched-GEMM             | 47       | 21    | dead end           |
| iPhone Core ML cpuOnly              | 45.4     | 22.0  | ~1x                |
| iPhone Core ML GPU                  | 15.8     | 63.2  | 2.3x               |
| **iPhone Core ML ANE**              | **7.5**  | **134**| **4.9x**          |

Notes: 1->2 CPU chunks scales 1.98x (2 P-cores), 4 E-cores add ~+55%; batched-GEMM on CPU
gained nothing (per-tile BLAS was already at the CPU ceiling — don't revisit). ANE
load+warm ~0.3 s once per process.

## Trick 3 — the warp becomes a gather (and then a GPU kernel)
The warp (camera quad -> rectified field, = cv2.warpPerspective) is per-pixel projective
math + bilinear sampling. Naive readable Swift: ~13 ms warm (the "50 ms" first seen was
cold-start page faults — always re-measure warm). Two attacks, both exploiting the static
rig (cached Hm -> the per-pixel math never changes):

1. **Gather-LUT** (CPU, = notes2 "warp->gather" + the FA remap trick): precompute every output
   pixel's source coordinates ONCE (6 ms/session), then each frame is pure 4-neighbor
   gathering. Fused two ways: 20x20 uint8 crops (training path) and — decode path —
   **directly into the NN's (T,3,16,16) float tensor, center-16x16 pixels only**: the
   separate warp, crop and prep passes collapse into one 5.2 ms gather, bit-identical
   to the naive chain.
2. **Metal** (GPU): Apple's CUDA-equivalent. One ~40-line MSL kernel, one thread per NN
   input pixel: projective math + manual bilinear from the camera texture, writing the
   normalized float straight into a `storageModeShared` buffer. That buffer is wrapped as
   the Core ML input array (`MLMultiArray(dataPointer:)`), so warp+prep+classify involve
   ZERO CPU pixel work. The MSL compiles from a Swift string at runtime — no build fuss.
   (vImage — Apple's SIMD CPU image library — was considered and skipped: it has affine
   but NO perspective warp, so it can't take Hm whole.)

Warp paths, camera frame -> NN-ready input, median of 12 (A: of 5), warm:

| path                                          | ms/frame | exactness vs naive        |
|-----------------------------------------------|----------|---------------------------|
| A naive: warp 12.9 + crops 5.8 + prep 5.4     | 24.2     | (reference)               |
| B gather-LUT -> 20x20 crops (training path)   | 9.9      | bit-identical             |
| C gather-LUT -> NN tensor (fused decode path) | 5.2      | bit-identical             |
| D Metal -> NN tensor (GPU)                    | 1.3      | 0.12% floats off by 1 LSB |
| E = D + ANE classify (whole NN side of frame) | **8.3**  | 0/5244 symbols differ     |

## On-device training (the bonus that worked)
Hand-written backprop (im2col gemms mirrored, col2im, unpool via stored argmax) + Adam.
Demo: perturb every tensor with sigma = 0.15 x its std -> val err 5.14% -> 200 steps x
bs256 on the frame's own tiles (truth symbols = labels) -> 0.19% in 8.6 s, single core
(~6k samples/s ~= the Surface's 4-thread torch). This is the same mechanism as the
receiver's t+ENTER / runtime tuning, so per-session calibration on the phone is proven
feasible. Training stays on CPU (it IS the torch-equivalent); ANE is inference-only.

## Traps paid for (do not rediscover)
- **Keychain over ssh**: unlock state is PER LOGIN SESSION; codesign fails
  `errSecInternalComponent` in any fresh ssh session. Unlock INSIDE the build command
  (`security unlock-keychain -p "$(cat ~/.kcpw)" && xcodebuild ...`) + one-time
  `set-key-partition-list`. Full recipe: mac README §11. `rm ~/.kcpw` revokes.
- **Phone auto-lock suspends the app mid-bench** (first ANE run "hung" for minutes).
  `UIApplication.shared.isIdleTimerDisabled = true` while working.
- Windows scp cannot address mac paths with spaces — stage via `/tmp`, then `cp -R`.
- Free personal-team signing: apps expire after 7 days; rebuild+reinstall refreshes.
- Swift init rule: closures may not touch members before ALL are initialized (build in
  locals, assign last).
- First-frame timings lie (cold caches/page faults): the naive warp measured 50 ms once
  and 12.9 ms warm. Median-of-N warm, like every Surface bench.

## What this means for the live phone receiver
Whole NN side of a frame = 8.3 ms -> ~120 fps of full r16 fields; the 5 fps mac-panel
link uses ~4% of that. The phone's bottlenecks are now capture and geometry, not compute:
- **Capture**: AVFoundation delivers `CVPixelBuffer`s that map to Metal textures zero-copy
  (`CVMetalTextureCache`) — the bench's texture upload cost disappears in production.
  And unlike the Surface driver, iOS locks focus/exposure/WB honestly (readback works).
- **Geometry**: port process()/locate() (adaptive threshold, ring-hole contour,
  refine_quad) or use OpenCV-iOS; only runs at bootstrap/on CRC-fail thanks to the Hm cache.
- **RS + fountain**: rs.py's vectorized syndromes + solve_repair -> Swift, few hundred lines.
- Realistic frame budget after those ports: geometry amortized ~0, warp+NN 8, RS ~3-5,
  header ~0 -> **the phone decodes faster than any display we own can flash frames**; the
  transmitter becomes the link's limiting element (ProMotion displays go to 120 Hz...).

## Files & scripts
- PC prep (assets from a saved frame): session scratchpad `prep.py` (exec's receiver.py's
  slice like sim_rx.py does; requires only a frame jpg + model name).
- Mac model build: `build_mlmodel2.py` (MIL Builder, fp16 mlprogram, numpy-validated).
- Bench workflow: build+install+launch over ssh (README §6/§11), results pulled from the
  app sandbox: `xcrun devicectl device copy from --domain-type appDataContainer
  --domain-identifier com.example.QRFileTransmit --source Documents/log.txt ...`
