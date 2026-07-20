# claude_handheld_analysis.md — the unstable-camera (handheld) approach (2026-07-18)

Synthesis of two Claude passes over the question "is handheld even doable?" —
the original feasibility argument plus an adversarial review of it — checked
against this project's measured data where the two disagreed. Scenario assumed
throughout: ~24" monitor (eventually a browser transmitter), phone held in hand,
5–20° turns/translations in all directions, r16 glyphs.

## Verdict

Doable, with honest conditions attached. Handheld is an ARCHITECTURE port
(FA per-frame naming + homography tracking), not a toggle on the current
receiver — the current build is correctly described as "marginal, works when
braced." Expected envelope when built: **controlled-light room, smooth motion,
warm-started model, r16 @ 1080p, fps 5–10, nsym64 ≈ 14–28 KB/s** (a 24" 1080p
panel's bigger grid pushes the ceiling to ~40 KB/s at 10 fps if the channel
holds). The three things most likely to make it fail outright — specular glare
sweeping a glossy panel, cold-model bootstrap, jerky motion — are testable
BEFORE building the tracker, and should be (see de-risk plan).

## What is measured vs what is assumed

The review flagged the light budget as the hardest problem, claiming 1/500 s
was "two stops shorter than the ISO-3580 run was actually using" → ISO ~14000
needed → format ceiling exceeded. **The log refutes the specific claim**: the
2026-07-18 stable-phone handheld-settings run ran the 1/500 override — its CAL
probed ISO 955→2389→2387 and 1432→3580 (values only consistent with the ~2 ms
shutter; the settled 16.6 ms/ISO 193 line prints the PRE-override 3A state) —
and trained to 0.88% err at ISO 2387–3580. So 1/500 + ISO ~2400–3600 + sub-1%
err is DEMONSTRATED at this rig's panel brightness, and the truth-scored CAL
kept choosing MORE ISO: sensor noise really is cheap for this classifier.

The DIRECTIONAL point survives, though: the budget is thin, not broken.
- ISO ceiling ≥3580 demonstrated; exact format maxISO never logged (do it).
- 1/1000 (for faster motion) needs ~7200 ISO — plausibly near/at the ceiling.
- Panel brightness can't buy stops freely: bloom clips glyphs (measured on the
  mac panel, machines.md). A matte office monitor at 1 m may bloom differently.
- A dim living room with a consumer-brightness panel is genuinely unproven.
  "Benches fine in the lab, fails in a dim room" is the realistic risk shape.

## Physics, corrected

**Density and readout must come from the same row.** The optimistic 27 px/tile
figure was 4K; the gentle ~8 ms readout / ~9 px shear figure was 1080p binned.
Handheld doctrine forces 1080p (4K doubles readout = doubles jello), so the
honest handheld numbers are: **~13–14 camera px/tile and ~9 px shear at 20°/s**
— exactly the density of the proven 1080p r16 record channel, adequate, no
headroom. Corollary: **handheld is r16-only.** r12c16 needs 4K density, which
buys double the rolling-shutter shear; that combination is not a handheld play.

**Motion blur**: at 1/500, 20°/s smears ~2 px, tremor sub-pixel. Solved by the
shutter override, now with measured backing (above). Blur was the fatal one and
it is paid for — at this brightness.

**Rolling shutter — the review's best insight, and it cuts in our favor:**
a constant-angular-rate sweep shears the frame linearly in row index, and a
linear shear is AFFINE — a homography absorbs it (the fitted quad is a slightly
wrong "pose" but a perfectly good decode geometry, and refine_quad fits
arbitrary quads). The genuinely non-projective residual is the ACCELERATION
term: flick starts, flick ends, correction jerks. So the receiver isn't waiting
for stillness — it's waiting for **constant velocity, which hands produce far
more often than they hold still**. Smooth 5–20° sweeps should decode per-frame;
jerks are the discards. This reframes dwell arithmetic: at 60 fps capture and
fps=10 display, 6 shots/frame, and "clean" = constant-velocity, not still.

**Perspective itself is a non-issue**: 20° tilt = 6% foreshortening; the
homography doesn't care. Far-edge px density and specular angle are what change.

**Focus (Z-drift)**: f/1.78 at 40–60 cm gives a few cm of DoF; today's locked
focus goes soft with no recovery. The 14 Pro's dual-pixel PDAF tracks focus
continuously without contrast-AF hunting — a policy change, not a hardware
problem. But not free (review is right): focus breathing = a small time-varying
magnification the tracker must eat, and refocus transitions are dropped frames.
Net positive — trades a hard failure (permanent defocus) for a soft coupling.
Alternative half-step: keep the lock, re-settle when sharpness/CRC-rate sags.

## Architecture: what actually has to change

Two static-rig assumptions are load-bearing in the current receiver:

1. **Dedup defines frame identity by scene stability.** Handheld, nothing is
   ever stable; identity gets guessed at dwell boundaries. Fix = **FA per-frame
   naming** (validated on the Surface at 10–15 fps: FA 1.2% vs positional 18%):
   every capture frame names itself by header probe, best shot per index wins,
   dedup is bypassed entirely. CAVEAT the original argument glossed: **the
   header rides the same photons** — a frame blurred/sheared past payload
   readability has an unreadable header too. FA's win is "pick the good shot
   when one exists," NOT "make brisk motion work." A dwell that is entirely
   mid-jerk loses all 6 name attempts and that display frame goes to fountain.
   That's what fountain is for — but budget repair accordingly (repair=12+).
   Possible softener: accept majority-vote idx WITHOUT CRC as a naming
   candidate (a wrong name only wastes one decode attempt; RS still arbitrates).

2. **Geometry is searched, not tracked.** The ring search is a bootstrap tool;
   re-searching per failed frame (current v1.2) is a band-aid. Track instead:
   refine the previous Hinv with a small local search each frame. **The gyro is
   TWO different features, not one** (review is right to split them):
   - *Inter-frame prior* (easy, cheap, first): CoreMotion rotation deltas seed
     the next frame's Hinv refinement. Standard, low-risk, ~free.
   - *Intra-frame de-shear* (hard, its own phase, probably unnecessary):
     un-warping rolling shutter WITHIN a frame needs per-row timestamps +
     tight gyro↔camera clock sync (sample-level AVFoundation plumbing). Only
     needed to decode mid-JERK — mid-sweep already works via the affine-shear
     argument. Scope it OUT of v1; note it as the last-resort lever.
   ARKit-class VIO exists on this chip at 60 fps, so compute is not the
   constraint; we just don't need most of it.

## The defining constraint: warm-start only

Promoted from "residual risk" to architectural fact (review is right).
Training harvest needs CRC-passing headers; reading headers under handheld
glare/noise needs a model already decent at handheld glare/noise. **There is no
cold path.** Handheld mode should refuse to bootstrap from a weak model and
demand a matured factory (rig-trained r16 fork, or a purpose-built handheld
factory — below). Whether the rig-trained model TRANSFERS to the moving-glare
distribution is the single biggest unknown in the whole plan — bigger than any
throughput question. Session training then adapts from that warm start (the
existing chunk pipeline is fine; FA_ENABLED_IN_TRAINING already solved the
"dedup feeds blends to training" version of this on the Surface).

## Failure modes, ranked by kill probability

1. **Specular glare sweeping the panel as viewpoint moves.** The NN sees a
   moving lighting field no rig session ever produced; augmentation + session
   training are the tools, transfer is unproven. Glossy panel = worst case.
2. **Cold/weak model** (see above) — by design refuse, not fail.
3. **Jerky motion** — not fatal (fountain absorbs), but sets the real goodput:
   if the user's hand produces mostly jerk and few constant-velocity windows,
   effective loss rate decides between "slow but works" and "dies."
4. Light budget in dim rooms / dimmer panels (proven only at rig brightness).
5. Z-drift (fixable, policy change).

## De-risk plan — cheapest experiment first, tracker LAST

**Stage 0 (build nothing): film handheld takes and replay offline.** Hold the
phone by hand over a running transmit (r16, 1080p60 video, 1/500 via a manual
camera app, fps=5 and 10, nsym=64, repair=12), with deliberate smooth sweeps
AND deliberate jerks, in both the rig room and a bright/glary setup. Run the
.MOVs through **sim_rx.py** (already exists; retries dwells, full session
semantics). This measures, in one afternoon, with zero new code:
- warm-start transfer (does the rig model read handheld frames at all),
- the glare distribution's actual damage,
- the constant-velocity-shear hypothesis (do mid-sweep frames decode with a
  per-frame homography?),
- realistic per-dwell clean-shot counts → FA's expected yield and the right
  repair budget.
**If Stage 0 fails on glare or transfer, stop: the tracker would be effort
spent on a channel that doesn't close.** If it passes:

**Stage 1**: FA naming port to the phone (per-frame header probe on ANE,
best-shot per idx, dedup bypass in handheld mode). This alone may carry
braced-to-lightly-moving use.
**Stage 2**: Hinv tracking with gyro inter-frame prior + continuous-AF policy.
**Stage 3** (only if the numbers demand): soft-decision RS erasures (2x error
budget, wire-compatible, parked in NOTES.txt), runner-up candidate retry,
intra-frame gyro de-shear, KLT/VIO.

Offline bonus from Stage 0 footage: harvest it with the PC pipeline (label_rx /
reharvest) and train a HANDHELD FACTORY model offline — directly attacks the
warm-start unknown with zero on-phone risk.

## Throughput expectations

| config                             | goodput      | status                    |
|------------------------------------|--------------|---------------------------|
| rig, r12c16 @ 30fps nsym32         | 126.6 KB/s   | measured record           |
| handheld, mac panel r16 nsym64 f5  | ~14 KB/s     | projected                 |
| handheld, mac panel r16 nsym64 f10 | ~28 KB/s     | projected                 |
| handheld, 24" 1080p r16 nsym64 f10 | ~41 KB/s     | projected (118x65 grid,   |
|                                    |              | ~13 px/tile at 50 cm)     |

Handheld and the density/rate frontier are different games; 1 MB in 30–70 s is
the right mental model, and "works at all, anywhere, in anyone's hand" is the
actual product feature — it's what separates this from a fixed-rig curiosity.
