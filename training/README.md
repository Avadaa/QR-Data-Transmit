# training/ — pretraining the decoders on a PC

Requirements: Python 3.10+, numpy, opencv-python, torch (CUDA strongly
recommended — every run below is seconds-to-minutes on a modern GPU, CPU works
but is slow), ffmpeg on PATH (v3 pipeline only). Scripts expect to import
`glyphs.py` from the transmitter folder for v2 rendering (adjust `sys.path` or
copy it next to them).

## Glyph nets (wire v2) — `pictrain/pictrain.py`

Renders web-exact glyph fields and trains the receiver's exact architecture
with the receiver's exact recipe (20-image chunks, 120×256 steps, lr tiers,
±2 px jitter + brightness + noise augmentation). Measured convergence on
rendered images: 16 px sub-0.2 % in 100 images, 12 px in 160.

Picture-training is the BOOTSTRAP: a picture-trained net reads real camera
frames poorly until the phone's in-session training adapts it (that is by
design — the phone trains in seconds off seeded truth frames). Export for iOS:
concatenate the torch `state_dict` tensors as raw fp32 in declaration order —
that IS the `.bin` format everywhere in this project.

## Demapper (wire v3.1) — `v3/`

The full real-footage pipeline that validated v3.1 (read `v3/REPORT.md` first —
it is the evidence base):

1. Film the web transmitter's v3 TRAINING phase with the phone camera
   (4K mandatory), copy the .MOV to the PC.
2. `harvest.py <video> <outdir> [t0 t1 hint_lo hint_hi]` — NN-free labeling:
   frames identify by correlating against reconstructed seeded-truth fields,
   the homography refines sub-pixel against truth, best shot per frame index
   wins. Env `V3A/V3LV/V3NC` must match what the web page announced. Run 8
   slice-parallel workers + `merge.py` for speed (HEVC sequential reads only —
   cv2 frame seeking lies).
3. `run_v31.py` — synthetic pretrain (sub-0.2 % in ~1000 steps), zero-shot on
   real, fine-tune curves, distortion-field estimation + correction, wire
   throughput verdict. `train_real.py` holds the shared machinery,
   `analyze.py` the linear-channel measurement.
4. `export_demap.py` — writes `demap16_nc{8,12}.bin` for the iOS bundle.

Training laws (measured, do not fight them): demappers need SUB-PIXEL
augmentation (±0.4 px bilinear), never integer jitter; do not add synthetic
noise on real data; const 1e-3 Adam from scratch (the phone's lr tiers are
fine-tuning tiers); 1080p footage is unusable.

## Loading models onto the phone

- Bundle: replace the `.bin` in `ios/Assets/` and rebuild.
- Without rebuilding: copy a `.bin` into the app's Documents via the Files app
  as `active_model_<key>.bin` (glyphs; key 16, 12, 816, 1216, 3016) or
  `v3_active_model_<key>_nc<n>.bin` (demapper) — sessions warm-start from
  active files automatically. The in-app Save/Load/Factory buttons manage the
  same files.
