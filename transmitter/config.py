# config.py — one place for the data-mode link settings, shared verbatim by the PC
# (transmit.py) and the Surface (receiver.py). Any bare key=value token on a script's
# command line overrides a setting for that run (case-insensitive), e.g.:
#   python transmit.py fps=3 px=20
#   python receiver.py --save rx_frames train_every=20 nsym=32
# --flags and positional args are untouched and parsed by the scripts themselves.

MONITOR = 0           # server: display index for the fullscreen canvas (PC: monitor=1)
W_USE = 0             # optional field CAPS, 0 = none: the field auto-scales to the
H_USE = 0             # screen and the QR announces COLS/ROWS. Only the PC-window rig
                      # needs caps (camera view ends ~x1820): w_use=1800 h_use=1030.
TOP_PAD = 32          # blit the frame this far below the screen top — clears the mac
                      # camera notch (bump per run if the bite still touches the ring)
CANVAS_W = 1920       # LEGACY optional caps — output.py auto-scales to its screen and
CANVAS_H = 1030       # the SNC QR announces the grid. Only pre-announcement frame
                      # archives still decode against these defaults (reharvest).
PX = 16               # tile rung (production: 16; safe fallback: 20)
FPS = 5               # data/training frame rate
FPS_TRAINING = 5     # transmit.py training-loop fps (0 = FPS): stabilize the
                      # model at a gentle rate before a high-fps live session.
                      # Announced in the TRAIN QR; data mode keeps FPS.
NSYM = 64             # RS parity bytes per 255-block: corrects nsym/2 errors. 32 =
                      # 12.5% overhead, 16-error wall (~4.5% tile err); 64 = 25%, twice
                      # the wall. The START QR announces it — set on transmit side only.

PAYLOAD_SEED = 42     # synthetic payload rng (transmit.py without a file)
SYNTH_KB = 100        # synthetic payload size in KB (frames follow from KPF) #4096
TRAIN_SEED = 7000     # training-frame payload rng base (seed = TRAIN_SEED + frame idx)

REPAIR = 6            # fountain repair frames as PERCENT of sources (floor 8 frames,
                      # 0 disables): each is the XOR of a random half of the sources;
                      # ANY (lost + 2) of them rebuild all losses — no feedback, no
                      # repass. Size per regime: clean 5fps ~2-3, high-fps FA runs 5-8
                      # (the 15fps fix is nsym=64, not heroic repair). repair=6 per run.
REPAIR_SEED = 9000    # repair-mask rng base (seed = REPAIR_SEED + repair ordinal)

RUNTIME_TRAIN_EVERY_N = 50   # DATA mode: per N successfully decoded frames, keep
RUNTIME_TRAIN_FRAMES = 4     # F of them (decoded payload = free labels) and fine-tune
                      # in a background thread; the tuned net swaps in live AND saves
                      # to --tune-name. Long transfers track drift by themselves.
                      # 0 = off. A cycle may span >N frames if training is still busy.
ISOLATE_CAM_NN_THREADS = False   # hold DATA-frame decoding until recording ENDS: the
                      # camera phase gets the machine to itself (t+ENTER frames still
                      # process live for the err readout). Costs the capture/decode
                      # overlap — total time grows by roughly the decode time — buys
                      # zero NN interference while filming.
CAM_THREADS = 2       # logical cores RESERVED for the capture thread while filming:
                      # torch gets cpu_count - CAM_THREADS; after the camera releases,
                      # torch takes everything for the drain. (m3 note: torch at 3
                      # threads is a measured HT contention hole — CAM_THREADS=1 hurts.)
TRAIN_THREADS = 4     # torch threads during the t+ENTER phase: training frames are
                      # free/infinite, so capture losses cost nothing there — train at
                      # full speed. The capture-safe count is restored when the
                      # START QR arms data mode, where a missed dwell is a real loss.
OV_THREADS = 2        # OpenVINO inference threads for the worker's NN (0 = plain torch
                      # path). The i5-7300U saturates at its 2 PHYSICAL cores — HT adds
                      # nothing (test/igpu/bench_threads.py) — so 2 gives full NN speed
                      # while leaving the other two logical cores to capture + tuner.
                      # Engine: fp32 IR at setup, int8 (NNCF) once a CRC-proven field
                      # provides calibration crops, rebuilt after every tune. Training
                      # phases always read the LIVE torch net. Needs openvino+onnx+nncf
                      # on the Surface. docs/iGPU_CPU_performance.md.
TRAIN_EVERY = 20       # receiver fine-tune: run a chunk every N harvested frames
                      # (4 = fastest error descent; 12-20 = cooler, fresher data, long soaks)
TRAIN_STEPS = 200     # steps per chunk
TRAIN_BS = 256        # batch size per step
QCAP_TRAIN = 20        # drop training frames beyond this backlog (they are free/infinite)
QRAW = 50             # backlog under this -> queue RAW frames (skip the jpeg encode+
                      # decode roundtrip; the end table's "load" row IS that cost).
                      # 12 raw = ~100MB, safe on 4GB; deeper backlogs fall back to
                      # jpeg to keep RAM flat.

INSTANT_CROP_PICS = False       # once geometry is known, the capture thread crops each
                      # committed frame to the field's bounding box (+32px): smaller
                      # queue items, cheaper jpeg fallback, smaller --save archives.
MISC_IN_CAPTURE_THREAD = False  # once geometry is known, the capture thread also WARPS
                      # the frame — the worker receives ready fields (load/warp leave
                      # the worker's budget). Such frames are fast-path only: no full
                      # re-search is possible without the camera frame. Static rig.
RESEARCH_FAILS = 0    # re-run the FULL geometry search (~1.2s, the "re-search penalty")
                      # only after this many CONSECUTIVE fast-path CRC failures. 0 =
                      # never once geometry is seeded: junk frames cost ~0 instead of
                      # 1.2s each — but a BUMPED RIG then never re-locks (run dies).
                      # Static rig: 0. Anything that can move: 3-5.
AUTO_CULL_DEDUP_0_POOL = False   # drop never-stable dwell commits (TH_BLEND fallback,
                      # pool size 0) instead of decoding them — they are transition
                      # blends more often than frames; a culled REAL frame costs one
                      # fountain repair. False = keep the old best-effort behavior.
TH_CHANGE = 12.0      # dedup: new-content threshold (subsampled gray, mean abs diff)
TH_STABLE = 6.0       # dedup: per-band stability threshold (all 6 bands must be quiet)
TH_BLEND = 18.0       # dedup: a dwell that never settles still commits its calmest
                      # frame if under this (above = true transition blur, dropped)
ADAPT_BLOCK = 51      # adaptive threshold window for ring finding (glare-proof)
ADAPT_C = -5          # adaptive threshold offset

CAMERA_TYPE = "stable"     # "stable" = fixed rig: the cached homography holds across
                           # frames, probe coords are fixed. "unstable" (future,
                           # handheld) needs per-frame tracking; FA refuses it.
FA_ANALYZE_FRAMES = False   # frame analysis: probe-NN identifies every camera frame by
                           # its header (idx) and scores sharpness on probe glyphs
                           # spread across the field -> best frame per dwell, at ANY
                           # fps (only honored when CAMERA_TYPE == "stable")
FA_BATCH_MS = 1000          # probe batching window — one NN call per window; also how
                           # many frames the live version must hold (~15 at 30fps)
FA_INDEX_COUNT = 5         # header COPIES probed per frame (7 tiles each): majority
                           # over the copies + CRC-12 -> frame identity
FA_GLYPH_PATTERN = "grid" # probe spots: "cross" = center row + center column + both
                           # rectangle diagonals (~25 tiles); "grid" = the full lattice
                           # (~49). Coords derive from the CURRENT grid dims.
FA_ENABLED_IN_TRAINING = False  # FA also picks the training harvest (best shot per
                           # dwell — at high fps the pixel dedup feeds mostly blends).
                           # Probes with the LIVE torch net (the OV engine is stale
                           # mid-training); a window with zero CRC-named frames still
                           # feeds its best-margin shot (harvest CRC-gates it anyway).
                           # Needs a model warm enough to read headers (~<20% err) —
                           # cold bootstraps: legacy QR soak, or low FPS_TRAINING.

CAM_W, CAM_H = 1920, 1440
CAM_WARMUP = 120      # frames to discard at camera start (ISP shifts ~8px early on)
CAM_FOCUS = 290       # manual lens position, set AFTER warmup (0 = leave autofocus).
                      # Driver range 100-700, sharpness peak 290 (test/cam_sweep.py);
                      # fresh-session AF parks are a LOTTERY (visibly soft ~1 run in 3).
                      # Readback always claims 100 — the set applies anyway.
CAM_EXPOSURE = -5     # manual exposure (log2 seconds), set AFTER warmup (0 = auto).
                      # -5 = what a good AE picks; locking kills mid-run AE drift.
                      # -4 CLIPS the palette (yellow reads as white) — never brighter.
JPEG_Q = 85

MVER = "V2.2"         # model epoch folder models/<MVER>/ (V2.2 = mac rig, V2.1 alphabet);
MPREV = "V2.1"        # models/parents also resolve from here (cross-epoch warm-starts)
MODEL_PREFIX = "pc_L" # receiver base model when --model not given: f"pc_L{PX}"
TUNE_PREFIX = "rx_L"  # tuned model save name when --tune-name not given: f"rx_L{PX}"

NATIVE_TILES = True  # THE tile-resolution regime. False = every rung warps to the
                      # 28px canonical crop, NN reads 24x24 (r24's native size; sparser
                      # rungs get interpolation-PADDED -- same information, 2-3.5x the
                      # conv cost at r16/r12, see test/bench_nn_size.py). True = warp
                      # at the rung's NATIVE camera resolution (px+4 crop, NN input
                      # px x px): ~constant ~110ms/frame on the Surface at ANY rung.
                      # Models are input-size-bound: canonical and native weights are
                      # INCOMPATIBLE (load = shape crash; name native ones pc_L16n
                      # etc). Multi-rung ladders need canonical (one net, one size).
                      # "Native = rung px" assumes the ~1:1 rig (camera px ~ screen px).


def _coerce(old, v):
    if isinstance(old, bool): return v.lower() in ("1", "true", "yes")
    if isinstance(old, float): return float(v)
    if isinstance(old, int): return int(v)
    return v


def override(argv):   # apply key=value tokens; return the rest for the script to parse
    rest, g = [], globals()
    for a in argv:
        k, _, v = a.partition("=")
        if v and not a.startswith("-") and k.upper() in g:
            g[k.upper()] = _coerce(g[k.upper()], v)
        else:
            rest.append(a)
    return rest
