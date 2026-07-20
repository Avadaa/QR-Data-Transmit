# iphone_camera.md — the iPhone receiver's camera subsystem (2026-07-17)

How QR File Transmit drives the iPhone 14 Pro camera, what iOS honors that the Surface's
MSMF driver never did, the traps already paid for, and the truth-scored auto-calibration.
Code: `Camera.swift` + the calibration section of `LinkReceiver.swift` (mac,
`iOS Dev/QR File Transmit/Sources/`). Companion: docs/iPhone.md (the port), 
docs/surface_camera.md (the Surface's knob map — read it to appreciate the upgrade).

## Capture setup
- AVFoundation, `.hd1920x1080` BGRA at a pinned 30 fps (min=max frame duration), rear
  wide camera, frames copied out of the CVPixelBuffer on a dedicated queue (stride-aware).
- Unlike MSMF: **iOS 3A locks are real and readback is honest.** `lensPosition`,
  `exposureDuration`, `ISO`, WB gains — all reportable, all lockable
  (`setFocusModeLocked(lensPosition:)`, `setExposureModeCustom`, WB gains lock).
- `start()` is idempotent: the session keeps its input/output across stop/start —
  re-adding them is an AVFoundation error (this bug forced app restarts once). A restart
  resumes the session and resets all 3A to continuous for a fresh settle.

## The 3A doctrine (evolved through three failure generations)
1. **Never lock blind.** v1 froze 3A the instant a control QR appeared — locked whatever
   the lens happened to be doing (= the Surface's "AF park lottery", measured there as
   1-in-3 soft sessions). Live symptom: a whole run with zero decodable frames.
2. **Settle, then lock** (`settleAndLock`): one-shot AF at the frame center, poll
   `isAdjustingFocus` until convergence (2.5 s cap), settle 200 ms, then freeze
   focus+exposure+WB and log the values. Good sessions land lens ~0.44, 25 ms / ISO ~90
   on the mac rig. Still a (milder) lottery: one session settled at 0.404 and needed
   several frames before geometry caught (corr 0.52 vs the usual 0.8+).
3. **Calibrate by decoded truth** (`calRun`, auto after every settle-lock + CAL button):
   the transmitter's idle screen AND both control-QR screens display `train_img(0)` —
  a field whose every tile derives from seed 0. So while the transmitter holds any of
  those screens, sweep lens positions (0.25…0.7 + the AF pick), decode each setting with
  the live NN, and score = glyphs matching truth-of-frame-0, EXCLUDING the center 40%
  box the QR overlay may cover. Lock the argmax. Guards: a decoded header != 0 means the
  screen is NOT holding frame 0 (training/data running) -> abort + revert; nothing
  decodable at any lens -> report (likely inside the ~20 cm min focus distance, or
  framing). Lapvar is logged per setting but never decides — measured lesson from the
  Surface: **lapvar rewards clipping** (it once picked the exposure that turns yellow
  white). Decoded glyphs are the only metric that can't be gamed.

## Focus facts (iPhone 14 Pro main camera)
- `lensPosition` 0.0-1.0; min focus distance ~20 cm — closer than that NOTHING helps,
  move the phone back (CAL reports this as "nothing decodable at ANY focus").
- Locked exposure 1/40 s (25 ms) is the session norm; at 5 fps dwells (200 ms) motion
  blur is a non-issue on a propped phone.
- Sharpness gauge: variance-of-Laplacian on the full-res center window ("sharp" in the
  stat line, ~6000-7000 when good on this rig). The on-screen preview is a 1/4-res
  thumbnail and ALWAYS looks soft — judge focus by the number or the Z view, never by
  the small preview.

## Debug / operator tools
- **Z button** (preview panel): toggles the preview to a 1:1 320px center crop — the
  NN's-eye view of actual glyph pixels at sensor resolution.
- **AF button**: continuous-AF for 1.5 s, then settle-and-lock again (logs new values).
- **CAL button**: manual truth-scored sweep (transmitter must hold idle/QR screen).
- Debug mode additionally probes every camera frame's header ID at 30 fps (dbg lines:
  ids seen / shots per id / CRC fails / sharp) and dumps a rotating full-res jpg every
  10 s (`Documents/dbg{0,1,2}.jpg` — pull via `devicectl device copy from`).

## Measured channel notes (mac rig -> iPhone, 2026-07-17)
- Untuned loop_L16_best reads the phone channel at ~5.1% tile err; in-session training
  takes it to 0.2-0.4% within ~24 frames (t+ENTER doctrine unchanged from the Surface).
- 30 fps / 5 fps transmitter = 5-6 identified shots per transmitted frame; expected
  noCRC baseline ~3-8 per 60 frames (ISP transition blends — normal). Untuned-model
  frame loss was ~4%; after training it should be ~0 + fountain.
- Screen-glitch caveat: the mac panel occasionally flashes diagonal white/black line
  artifacts for a few hundred ms (compositor/pygame presentation bug, predates the
  phone). At 5 fps that costs 1-2 source frames -> run transmits with repair=6.
  If it persists, try pygame set_mode(..., vsync=1) on the mac.

## High-fps upgrade (2026-07-17 evening, from the external research in
## iphone_camera_specs.txt — read that file for sources and the full reasoning)
Adopted, phone side (`Camera.swift`): explicit 1080p60 activeFormat via
`.inputPriority` (presets silently enable videoHDR!), 60 fps capture,
`isGlobalToneMappingEnabled = TRUE` (counterintuitive: OFF means per-frame LOCAL tone
curves — adaptive nonstationarity the NN has to eat), EIS `.off` (warps frames — poison
for the cached homography), low-light boost off, geometric distortion correction off,
format specs (min exposure / ISO range) logged at start. settleAndLock now forces a
SHORT shutter: <= 8 ms with ISO scaled to preserve brightness (25ms@ISO90 ->
8ms@~ISO280) — blends kill the link, noise doesn't (clean-window math: clean fraction
= (dwell - exposure - readout)/dwell; 25 ms exposure = 30% clean at F=20, 8 ms + 60 fps
capture = guaranteed clean shots at F<=20).
Adopted, mac side (`transmit.py`): `vsync=1` + `present()` holds each frame for exactly
60/fps refreshes when fps divides 60 (clock.tick drifts against the panel scan — fps=8
made alternating 116.7/133.3 ms dwells, hidden jitter in the record run). OPERATOR
STEP: pin the panel to fixed 60 Hz in System Settings -> Displays (ProMotion adaptive
misbehaves for fixed-rate content). Use fps from {10, 12, 15, 20, 30} only.
Still to measure: actual sensor readout (flash the field at a known rate, count bands),
exposure/ISO through the CAL sweep.

## Future (designed, not built)
- Sweep exposure/ISO the same truth-scored way (the machinery is setting-agnostic).
- Persist per-lighting calibration results -> prior ranges per environment (the user's
  "optimal range" idea); score drift during a session -> auto re-CAL between phases.
