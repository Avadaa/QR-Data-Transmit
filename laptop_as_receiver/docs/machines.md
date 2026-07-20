# machines.md — the three-machine rig (roles, paths, access)

## PC — dev / training / heavy hardware
- GPU (cuda) — all offline training (pc_subtraining/), snapshot sweeps, chart making.
- historically also the transmitter (pass monitor=1 for the second display) — being
  replaced by the mac for controlled lighting; keeps the role for hostile-light tests.
- deploys to the other two: python deploy.py [model ...]
  (code always; named models -> Surface models/<config.MVER>. config.py ships everywhere —
  it IS the wire format; never let the machines drift apart.)

## MacBook — transmitter (M1 Pro 14", 2021)
- display: ALWAYS the Default scaled mode 1512x982 — the only integer (2x) mapping;
  every other mode fractionally resamples and blurs glyph edges.
  auto-brightness / True Tone / Night Shift OFF, away from the window. Brightness LOW
  — a bright panel BLOOMS on the camera (glyph edges bleed, whites clip, QRs unreadable;
  softness that looks like defocus). Drop it until the captured glyphs are sharp; too
  far = the camera slows and the receiver's pool histogram mode falls toward 0-1 (that
  histogram is the live gauge). Also disable Battery > "slightly dim on battery".
- GOTCHA: pygame needs the Aqua session — start transmit.py from a local terminal on
  the mac (or Screen Sharing), NOT through plain ssh.
- geometry: the field auto-scales to the screen (QR announces COLS/ROWS); on this
  panel r16 = 92x59 tiles = 14.0 KB/s at 5fps. PENDING: measure camera px/tile at
  the rig distance (want >=16 for native-16 models) before trusting NATIVE_TILES.

## Surface — receiver (2-core m3 tablet, 4 GB RAM, rear camera)
- runs receiver.py (data mode) and client.py (legacy training soaks); torch CPU.
- camera scripts NEED ssh -tt (MSMF cam.read hangs in warmup without a tty).
- camera 3A is LOCKED after warmup (config CAM_FOCUS=290 / CAM_EXPOSURE=-5, 0=auto):
  fresh-session autofocus parks are a lottery (visibly soft ~1 in 3, 2026-07-07 probe),
  and manual focus 290 beats even a good park. WB stays auto (driver refuses locks).
  Readback of focus/exposure always lies — trust test/cam_sweep.py frames, not get().
  Full knob map + re-check procedure: docs/surface_camera.md.
- kill scripts by PID (wmic CommandLine match), never taskkill python.exe.
- sleeps on AC unless: powercfg /change standby-timeout-ac 0
- models live in models\<config.MVER> (V2.2 = mac rig; V2.1 monitor-rig models still
  load as fallback/parents); ship with deploy.py <model>.

## The flow
  PC (train, GPU) --deploy.py--> mac (shows frames) --camera--> surface (decodes)
- run commands: docs/commands.md. wire format: docs/decode_practical.md.
- lighting doctrine: mac rig = reproducible development numbers; the old PC-monitor
  rig by the window = hostile-light stress testing. Never compare across rigs.
- mac-rig channel-mover is CAMERA SESSION STATE, not ambient drift (constant-light
  room): the ISP settles per session on whatever is on screen during warmup. Calibrate
  in-session (transmit.py t+ENTER) or warm up on the field, never on the black START
  screen. Details: decode_practical.md gotchas.
