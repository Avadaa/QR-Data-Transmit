# surface_camera.md — the Surface camera's real knobs (probed 2026-07-07)

The receiver's camera control used to be nothing: open MSMF, set 1920x1440, warm up,
pray. The prayer part was real — see "the autofocus lottery" below. This documents
what the Intel AVStream (INT3470) driver actually honors through cv2/MSMF, what we
locked and why, and how to re-run the checks.

## The one rule of this driver: READBACK LIES

`cam.get()` on focus/exposure returns the same numbers forever (focus "100",
exposure "-5") no matter what was set or where the lens physically is. `cam.set()`
return values are also only half-honest: out-of-range sets are refused (False), but
an accepted True doesn't prove anything moved either. **The only trustworthy output
of this camera is the frame.** Every conclusion below came from saved frames, judged
by a human, with variance-of-Laplacian (lapvar) as the machine-side hint.

## Knobs that work

| knob | cv2 prop | behavior |
|---|---|---|
| focus | CAP_PROP_FOCUS | range 100-700 (outside -> set refused). Writing a value silently switches to manual focus and STICKS. Blur bottom at 100-120 and 400+, sharp peak at **290**. |
| exposure | CAP_PROP_EXPOSURE | log2 seconds (-5 = 1/32s). Writing a value goes manual and sticks. -5 = what a good auto-exposure picks. |

## Knobs that do NOT work

- `CAP_PROP_AUTOFOCUS = 0` — refused. (Irrelevant: writing a focus value locks anyway.)
- `CAP_PROP_AUTO_WB` / `CAP_PROP_WB_TEMPERATURE` — unsupported (-1, sets refused).
  White balance stays auto; the ~6s AWB oscillation is still with us.
- `CAP_PROP_SHARPNESS` / gain / gamma — accepted or ignored, no visible effect.
- Restoring autofocus mid-session (`AUTOFOCUS=1`) re-parks the lens BADLY (~1900
  lapvar vs 6000+ for a good park). Only a fresh camera session re-hunts properly.
  Setting AE=1 and AF=1 together did re-hunt well once — not relied upon.

## The autofocus lottery (why we lock at all)

Fresh-session AF parks measured across five sessions on the identical rig:
lapvar 6146, 1874, 6581, 2103, ... — roughly **one session in three parks visibly
soft**, on a static rig, same scene, same light. Until 2026-07-07 every receiver run
drew from this lottery, and bad parks were mis-blamed on brightness/exposure state.
Manual focus 290 beats even the GOOD parks (8039 vs 6581, frozen-frame comparison).

## The human's choice (frozen-frame set, camfrozen/)

- **focus 290** — beats both auto baselines "manyfolds"; repeat-set stays sharp.
- **exposure -5** — matches a good AE pick, minus the mid-run drift.
- **exposure -4 rejected**: lapvar said it was the sharpest of all (8437) but it
  CLIPS the palette — yellow reads as white. Lapvar rewards clipping; the human
  vetoes it. This is exactly why frames get judged by eyes, not by the metric.

## What's shipped

config.py: `CAM_FOCUS = 290`, `CAM_EXPOSURE = -5` (0 = leave auto). Applied in
receiver.py / record.py / client.py / snap0.py right AFTER the CAM_WARMUP reads
(let auto settle first, then pin), followed by ~30 settle reads. snap0.py uses the
same locks on purpose: it must show what the receiver sees. Verified end-to-end:
fresh-session snap0 scored lapvar 8305 (peak zone).

Models note: anything trained on auto-park sessions sees a slightly different
channel than the locked camera — run t+ENTER calibration on the first post-lock
transfer, then sessions finally repeat.

## Re-running the checks

    # mac: freeze the display on one frame first (stop the training loop mid-frame)
    python test\cam_sweep.py <folder> 270 280 290 300 310   # focus sweep -> ../<folder>/
    python test\cam_frozen.py                               # full set: focus + exposure ladder

Frames land on the Surface; pull them to the PC (scp -r ... W:/Temp/renders/) and
LOOK at them — zoomed center crops (80x80 at 8x nearest-neighbor) make the call easy.

**Future checks must run against a STATIC frame.** With live training frames the
content changes between shots, so lapvar differences mean nothing (baseline once
ranged 2764-8020 on content alone — a 3x swing with zero camera change). The frozen
screen is what made 100 vs 290 vs auto comparable in the first place. And remember
the rig knob rule: if a re-check changes the operating point (new focus/exposure
values), the current model is invalid until recalibrated.
