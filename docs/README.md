# docs/ — reading guide (for humans and their LLMs)

If you're a future developer (or feeding this repo to a large-context model),
read in this order — it reconstructs the project's entire state of knowledge:

1. **[STATUS.md](STATUS.md)** — what is proven vs what never ran. Trust this
   over any enthusiasm elsewhere.
2. **[WIRE.md](WIRE.md)** — the on-screen format spec (v2 glyphs + v3.1 DMT),
   seeds, QR control plane, coding pipeline. Both transmitters and the iOS
   receiver implement exactly this.
3. **[engineering_log.md](engineering_log.md)** — the FULL dated engineering
   ledger (newest first). Every architectural decision, every measured number,
   every trap and its fix, in the order they happened. This is the highest-
   density context in the repo; a future dev who reads it inherits the whole
   journey — including WHY things are the way they are.
4. **[../training/v3/REPORT.md](../training/v3/REPORT.md)** — the v3.1
   evidence base (real-footage study: what killed PAM-4, what validated PAM-2,
   the jitter law, the distortion field).

Reference material:

- **[iphone_camera.md](iphone_camera.md)** — how the app drives the iPhone
  camera: the 3A settle-then-lock doctrine, truth-scored calibration, why
  lapvar is never allowed to decide.
- **[iphone_camera_specs.txt](iphone_camera_specs.txt)** — sourced external
  research on the iPhone camera pipeline (formats, HDR/EIS/tone-mapping traps,
  shutter/readout physics). The basis for the camera settings toggles.
- **[iphone_camera_handheld.txt](iphone_camera_handheld.txt)** — camera
  physics of handheld operation (blur vs exposure, OIS/EIS bundling trap,
  rolling-shutter jello).
- **[handheld_analysis.md](handheld_analysis.md)** — the full handheld
  feasibility analysis + de-risk plan (architecture built; never succeeded in
  the field — see STATUS).
- **[ios_port_history.md](ios_port_history.md)** — how the receiver was ported
  to the phone: the CoreML single-tensor trick, ANE benchmarks, Metal warp,
  on-device training. Explains the performance architecture.
- **[fountain_design_notes.md](fountain_design_notes.md)** — why the dense-XOR
  fountain beats LT/RaptorQ at this scale, and the designed-but-unbuilt
  generation-interleaved rateless tail.
- **[../laptop_as_receiver/docs/](../laptop_as_receiver/docs/)** — the FIRST
  receiver generation (Python on a Windows tablet): wire/receiver architecture
  (`decode_practical.md`), training system design (`train.md`,
  `train_practical.md`), the camera-driver war stories (`surface_camera.md`),
  CPU/iGPU inference study (`iGPU_CPU_performance.md`), rig operations
  (`machines.md`, `commands.md`). Era: 2026-07-04 → 07-16, before the iPhone.
  Many doctrines that survive today (truth-seeded training, warm starts,
  RS-erasure ideas, per-rig models) were born here.

Timeline orientation: laptop-era docs are dated ≤2026-07-16; the iPhone era
starts 2026-07-17; the web transmitter, v3, and everything in `training/v3/`
is 2026-07-18/19. When docs disagree, the newer one and STATUS.md win.
