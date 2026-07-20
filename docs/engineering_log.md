# engineering_log.md — the full dated engineering ledger

The project's running notes, verbatim (newest first). Dense by design: every
architectural decision, measured number, bug, trap and fix, in the order they
happened. If you are a future developer (or an LLM ingesting this repo), this
file plus docs/STATUS.md is the fastest way to inherit the project's entire
state of knowledge. Machine names/paths are sanitized; "the mac" = the
transmitter/build machine, "the Surface" = the first-generation laptop
receiver (see ../laptop_as_receiver/), "the phone" = the iPhone receiver.

2026-07-19  **PUBLISHABLE REPO TREE built: QR_data_transmit/QR_data_transmit/**
(17 MB, 62 files) — transmitter/ (transmit.py+config+glyphs+rs), web/, ios/
(full project from the mac: project.yml, Info.plist, Sources incl Demap.swift,
Assets incl both mlpackages + all weight bins; + tuned/ with the phone's
camera-tuned active_model_16/816/1216/3016), training/ (pictrain + the whole v3
pipeline + REPORT), models/ (glyph16_loop_L16_best, glyph12_loop_L12n,
demap16_nc8, demap16_nc12), docs/ (STATUS.md = honest tested-vs-not matrix:
v3.1 NEVER live-run, handheld NEVER succeeded, r8 CPU-only so far; WIRE.md =
full format spec v2+v3.1). READMEs per folder with real commands (ios build/
keychain/log-pull, transmit.py recipes, web usage, pretraining reqs + export +
load-to-phone). .gitignore excludes xcodeproj/pycache/v3 data/pt. NOT yet: git
init/license/push — user's call. pictrain copy's glyphs import repointed to
../../transmitter.

2026-07-19  **FIRST r8 (8c16) TRANSFER byte-perfect — but on CPU (693 ms/frame); two
ANE bugs found+fixed.** The run was the MAC FULLSCREEN transmitter (grid 185x114 =
T 21090, NOT web-1024's 15376): (a) T=21090 needs a 3-byte varint -> the equal-
length mint guard correctly refused -> "ANE engine unavailable"; (b) LATENT BUG:
aneDisabled was never reset in start() -> the failure stuck for the whole app
launch, so the NEXT sessions silently skipped ANE without a log line. Fixes
deployed: aneDisabled is per-session again (+ a log when skipped), and **CHUNKED
ANE MINTING** — grids past the 16383 cap mint a ceil(T/n)-batch model (2-byte
varint again) and predict in n chunks off the same Metal warp buffer (MetalWarp
outTiles pads the buffer; the tail is never read). Mac CoreML load-test: minted
T=6724/10545/15376 all predict OK (25/56/67 ms on the M1). Fullscreen-r8 = 2
predicts ~ 70-90 ms/frame expected on the phone -> 10 fps live; kpf 11651@nsym64
x20fps = 227 KB/s theoretical if the channel and rate hold. The r8 run itself:
CAL 12.2%, train 163 frames -> 3.81%, 8/9 + 1 rebuilt, md5 match at fps ~1.5
(CPU-paced). Also: CAL sweep on 8px glyphs is BELOW target by design (12%) — the
truth-scored sweep still picked the right lens; training carried it.

2026-07-19  8PX V2 (8c16) enabled: web px picker gained 8; phone px picker gained 8
(and the menu modelKey now forces canonical for ANY px without native weights —
30px previously produced a wrong native key unless "Scale to 16" was checked; the
toggle now shows only for r12). Everything else was already free: handleQR forces
scaled for px not in {12,16}, key 816 warm-starts from the best r16, ANE minting
covers the web-1024 8px grid (124x124 -> T=15376, JUST inside the 2-byte-varint
cap 16383; predict scales ~T: expect ~48ms -> 10-15fps live, 20 borderline).
Channel math: 8px at ~1.75 cam px/logical = ~14 cam px/tile at 4K = the proven
1080p-r16 density; kpf 8595@nsym64 -> 168 KB/s @20fps theoretical if the channel
holds. Deployed. NOTE: a FULL-size 8px field (mac grid or 4K-monitor browser)
would exceed T=16383 and the mint would refuse -> CPU decode; web-1024 is safe.

2026-07-19  **v3.1 ON THE PHONE — deployed, bit-exact.** Web defaults now = the
validated wire (v3 DMT, PAM-2, A=40, NC=8, px16, tfps15). Phone: main-menu wire
picker "v2 glyphs / v3.1 DMT" + nc 8/12 picker; when v3.1 + a V=3 QR (PAM-2 only),
the receiver decodes payload cells with a DEMAPPER (new Demap.swift, conv 3-32-64
-fc512-fc, 20x20 input, CPU batched over cores — ANE port is the next lever, so
fps<=5 for now) while the GLYPH net keeps header idx + corners (hybrid frame as
designed); RS/fountain/whitening identical (v3 grid: nblk from pay*3*NC/8, same
w-major interleave). Bundled factories demap16_nc8.bin (REAL-finetuned ft_v31,
0.40% BER) + demap16_nc12.bin (PC-pretrained, 0.00% synth — no real NC12 footage
yet); v3_ model namespace (v3_active/fork/factory_model_{key}_nc{n}.bin) behind
the SAME Save/Load/Factory buttons (route on the wire picker). R12 PIGGYBACK IS
FREE BY CONSTRUCTION: the demapper's 20x20 input IS the canonical C=20 window, so
a scaled r12 v3 cell is statistically identical to r16 training data — v3 px!=16
FORCES canonical (scale16 toggle hidden in v3), same model, keys 1216/3016 warm
from the same bundle. v3 sessions: training/CAL skipped (pretrained model decodes;
on-device demapper tuning = later), ANE mint skipped, handheld+v3 blocked.
VERIFIED: Swift Demap vs torch 0/384 bits differ (mac swiftc harness). Deployed
build + launch OK. First live test: web page (defaults), phone wire=v3.1, fps 5.

2026-07-19  **v3.1 VALIDATED ON REAL FOOTAGE — 0.17% BER, BEATS THE GLYPH RECORD AT
HALF THE FRAME RATE.** Same-day second clip (4K60 portrait, PAM-2 NC8 A40 tfps15,
127 s -> 1561 labeled frames zero gaps; portrait MOVs: ffmpeg auto-rotates the raw
pipe AND new cv2 auto-rotates its props — probe container dims+rotation via ffprobe
only, the mixed path double-corrects). Ladder: PC-only pretrain sub-0.2% in 1000
steps -> ZERO-SHOT on real 7.6% (v3.0 was 50.5%) -> 20 real frames 1.1% -> all
frames 0.40% -> + measured per-cell distortion correction **0.163%** (mean |shift|
1.0 px p90 2.0 — real lens distortion; applying the field to the ALREADY-trained
model at eval time is enough, retraining adds nothing => it's a static receiver
CAL). Spatial: 0.02-0.2% everywhere except extreme corners 1.6-2.6%. All 8 coeffs
0.1-0.3%. Wire verdict at measured BER: **~138 KB/s net @15 fps / ~277 @30 fps**
(web 62x62 grid, r0.83; glyph record 126.6) — through HEVC-compressed footage,
live ISP frames should be cleaner. Engineering next: phone demapper port (TileNet
family -> ANE minting path), bit-level coding (0.17% BER: even RS nsym32 has 4x
headroom), distortion CAL in receiver, NC=12@A40 capacity step. Full data:
test/pictrain/realv3/REPORT.md + results_v31.txt.

2026-07-19  **v3 REAL-FOOTAGE VERDICT (test/pictrain/realv3/REPORT.md): shipped
config DEAD, redesigned wire ALIVE.** Two phone vids of the web v3 transmitter
mined NN-free (training pays = seeded truth → frames identified by NCC vs
reconstructed truth fields, H refined sub-pixel against truth; 8 parallel ffmpeg-
pipe workers — cv2 HEVC SEEKING LIES, sequential reads only): 1601 labeled 4K
frames (zero gaps) + 492@1080p = 6M labeled cells. Battery: PAM-4/NC20/A20 = 41%
SER and NO lever moves it (lr/data/width/aug ±2pts; channel-limited); integer
jitter costs 9pts (jitter law re-proven on real data); noise/brightness aug HURTS
on real data (it already carries the channel); 1080p@0.89 cam px/logical dead
(53%). BUT the channel measurement says the THEORY survives: empirical PAM-2
(sign@|level|=3 = binary wire at today's peak, measured incl. HEVC codec noise):
NN 1.9%/2.6% on coeffs 0-1, 3-6% on 2-6 → redesigned v3.1 (PAM-2, 6-8 coeffs,
A 30-40) projects 175-500 KB/s vs the 126.6 record. Two structural finds: (1)
noise is STATIC not temporal (dwell-mean changes nothing → integration is a dead
lever; codec+demosaic/moire, and the codec part won't exist live — footage is a
LOWER bound); (2) global homography breaks the jitter law LOCALLY — corners 18-42%
err vs center 1.3-3%, per-block ±1px re-alignment recovers mid-field to 1.4-3%
(lens-distortion-shaped residual → v3 needs piecewise geometry + flip camera #6
GDC ON). Web page grew v3.1 knobs (v3 amp 20/30/40, PAM-2/4, NC 6-20; QR announces
A=/LV=); realv3/harvest.py takes V3A/V3LV/V3NC env for the next footage round.
NEEDED FROM THE RIG: ProRes 4K clip (isolates codec noise) + v3.1 clips per REPORT.

2026-07-19  **PICTURE-TRAINING BENCH (test/pictrain/) + THE v3 JITTER LAW.** Phone-
exact chunks (20-img batches, 120x256 steps, the lr tiers) on rendered frames:
v2-16 sub-0.2% in 100 imgs/5 chunks, v2-12 in 160/8 (0.2-0.3 s train/chunk on the
RTX — the loop_L* bootstrap doctrine reproduced in seconds). v3 demappers STALLED at
31%/48% under the same recipe — and the lr side quest (sidequest_results.txt) proved
it was NEITHER lr NOR size: const 1e-3, persistent Adam, hot tiers, 2x width, ~10x
params (8.9M) ALL plateau ~30% (10x even HURTS at 12px). ROOT CAUSE: the glyph
recipe's +-2px jitter-crop augmentation — glyphs are translation-tolerant, DMT is
NOT (a 2px shift flips high-freq DCT basis phases; half the coefficients become
undecodable in principle). Jitter removed (= the sub-pixel-geometry assumption the
rig actually meets, +-0.4px): **v3-16 sub-0.2% in 500 imgs / 25 chunks / 24 s at
const 1e-3**; phone tiers also converge but ~4x slower (they're fine-tuning tiers —
from-scratch wants 1e-3). v3-12 @ NC=20 reaches ~10% and keeps falling — the finest
12px-cell basis is intrinsically hard; for px12 drop NC or make px16 the v3 default
cell. THE LAW FOR v3: geometry budget is SUB-PIXEL and the demapper's augmentation
must model +-0.4px (bilinear subpixel shifts), never glyph-style integer jitter.

2026-07-19  **v3 ON THE WEB TRANSMITTER (experimental "wire" selector, v2 glyphs /
v3 DMT).** v3 sessions: payload cells carry NC=20 DCT coeffs x 3 ch PAM-4 (120 raw
bits/cell), HEADER + CORNERS STAY v2 GLYPHS (the phone's naming/geometry machinery
identifies every frame — hybrid frame), QRs announce V=3,NC=20, capacity/RS
computed per wire (px12@1024: nblk 392, kpf 74,872 B/frame @nsym64!), whitening +
fountain unchanged. TRAINING STAYS AND MATTERS MORE: train frames = PCG64(TRAIN_
SEED+i) PAM symbols — the seeded-truth doctrine carries to the demapper verbatim
(that's how the real-crop dataset gets built). JS IDCT ~60-120 ms/frame -> fps <=5
for v3 (per-pattern accumulation; separable IDCT is the speedup if ever needed).
PHONE GUARD: V>=3 in a control QR -> "footage/log mode" — trainingOver + calDone
forced (the glyph net must NEVER train on DMT cells = label poison; CAL scores
glyphs vs truth = meaningless). No phone demapper yet — v3 sessions are for filming
+ dataset collection toward test/v3/'s next step. (DEPLOYED 2026-07-19 evening:
mac sources already carried the v3 guard, web copy byte-identical via ~/windows
mount — build + devicectl install/launch completed when the rig came back.)

2026-07-19  **v3 "MODEM MODE" PoC (test/v3/): simulated 11-30x headroom over the
glyph wire, and the neural demapper is the ENABLER.** Same 12px cell, but instead
of a 6-bit glyph it carries NC=20 2D-DCT coefficients x 3 color channels, PAM-4
each (120 raw bits/cell); receiver crops canonical-style (20x20, C=20 doctrine
intact) and a ~1M-param conv net demaps. Channel sim mirrors the 4K rig (1.7 cam
px/logical px, PSF sigma 1.0, noise sigma 3, jitter +-0.4px, gradient, crosstalk).
RESULTS (30k steps, 50 s on the RTX PRO 6000): demapper mean SER 8.3% -> 56 loaded
bits/cell -> **1.39 MB/s @ web-1024 30fps, 3.86 MB/s @ mac-grid 60fps** (record:
126.6 KB/s). Classical DCT+pilot demod COLLAPSES to 4 bits/cell (28.7% SER) — the
jitter/crosstalk/gradient need a LEARNED equalizer (DeepRx in miniature). Clean
per-coeff frequency rolloff -> real bit-loading has more headroom. Caveats in the
README: pure sim (no moire/demosaic/tone curves/RS shear), no LDPC (tiered code-
rate accounting), overheads uncounted; even halved it's ~5x the record. Bonus: v3
cells are quiet pastel noise around mid-gray — less panel light, inherently
whitened, easy on the eyes (see closeup_glyph_vs_v3.png). Next: render v3 fields
from the web canvas, film with the phone, train on REAL crops.

2026-07-19  **6.4 MB MP3 BYTE-PERFECT — 65.68 KB/s, the long-transfer stack proven
end-to-end.** decoded 1745/1770 + 25 rebuilt (failed 7, dropped 0, repair 106/107),
MISSING none, md5 exact, file saved+openable. Every 2026-07-19 mechanism earned its
keep in one run: timestamp pacing (95.5 s ≈ nominal vs 104 tick-counted), END QR
seen (no timeout), 6 runtime tunes err pinned 0.29-0.49% across the whole hot run,
minted ANE T=6724, whitening, auto-saved active. USER-OBSERVED "counter stopped at
1745 while end frames ran" = DESIGN, not bug: the visible end frames are the 107
REPAIR frames, which never increment "decoded X/N" (sources only) — the 25 missed
sources (7 RS-fails + ~18 uncommitted dwells = 1.4% attrition, well inside the 6%
repair budget) get rebuilt in one shot at assemble. The counter jumping straight
from 1745 to complete at END is the fountain doing its job.

2026-07-19  TIMEOUT-vs-LIVE-TRANSMISSION BUG (user-spotted): the mp3 retry PROVED
runtime training (7 tunes, err pinned 0.20-0.39%, 1 fail in 1734 decodes!) and then
the receiver hung up at EXACTLY nominal+10s (104.0 s) while the screen was still
sending — the web's tick-counted pacing ran ~17% slow over 1877 frames (stretched
rAF ticks accumulate; ProMotion adaptivity/compositor hiccups), so the last 35
sources + all 107 repairs were still coming. TWO fixes: (a) receiver timeout is now
PROGRESS-based — finish("stalled") only after 12 s without a successful decode
(lastDecT under lock, set by both decode paths), absolute runaway guard at
2x nominal + 30 s; (b) web pacing is now TIMESTAMP-scheduled — frame i shows when
elapsed*fps-3 reaches i (frame 0 dwells 4x), so a late tick COMPRESSES the next
dwell instead of pushing the run; catch-up advances max 1 frame/tick (every frame
always shown); status line gained "drift N" (target-vs-shown frames, should hover
0-1). Total duration now matches nominal regardless of panel adaptivity.

2026-07-19  **RUNTIME TRAINING shipped (the long-transfer fix).** The 6.4 MB mp3 run
died of THERMAL CHANNEL DRIFT: best-ever session start (CAL 2.8%, train 0.24%,
minted-ANE T=6724 predict 16.8 ms, 60 KB/s @ ~20 fps) but the model was trained cold
in the first 15 s and never updated — err crossed the nsym64 wall ~75 s in, tail
1709-1769 + ALL 107 end-loaded repairs died, timeout ended it (decode times stayed
FLAT — compute never throttled; it's the sensor/ISP channel that moves when the
phone heats). Fix = the receiver.py doctrine ported: every rtEvery decoded frames
(Settings slider 0-100, default 50; 0=off) harvest the LAST rtTake (slider 1-50,
default 10 — freshest tracks drift), decoded payload re-encoded = free labels for
every tile (dataTruthSym), sampled 1200-tile crop harvest on the workQ (~10-15 ms),
then ONE .utility thread trains a clone (TRAIN_STEPS x TRAIN_BS, lr by terr) while
decode keeps the P-cores; swap on workQ + ANE re-patch (~0.3 s lean on backlog/
spill) + fresh cross-check; counting restarts AFTER completion. Log: "RUNTIME tune
#n: err X% (Y s, 1 thread)". Works in stable AND handheld paths (both decode hooks).
Repair-interleave (web side) still parked as the complementary lever.

2026-07-19  **FIRST BYTE-PERFECT REAL-FILE DELIVERY: thesis.pdf 576 KB, 275/275,
0 rebuilt (whitening erased the tail — previous run needed 9), md5 match, saved as
QR Transmit/thesis.pdf on the phone.** Minted-ANE at T=3844: cross-check 0/3844,
predict 11 ms (was 104 CPU); auto-save fired (active_model_16.bin). FILES-APP TRAP
fixed: INFOPLIST_KEY_UIFileSharingEnabled is NOT a recognized Xcode generated-plist
key — silently dropped (LSSupportsOpeningDocumentsInPlace went through), so the app
NEVER appeared under Files > On My iPhone (unnoticed since day 1: logs were always
pulled via devicectl). Fix: real Info.plist with both keys, INFOPLIST_FILE merged
with GENERATE_INFOPLIST_FILE, xcodegen regenerated. If the folder still doesn't
show: kill + reopen the Files app (it caches the provider list).

2026-07-19  **WIRE WHITENING (SCR=1) — content-shaped white floods found & fixed.**
First real-PDF web transfer decoded 255/275: the LAST ~5% of frames rendered 50-86%
WHITE glyphs (baseline 28%) and bloomed the camera dead. Cause: the wire had NO
scrambler — a PDF ends with its ASCII xref table (runs of 0x30), and 0x30-runs map
to 6-bit symbols where 3 of 4 glyphs are white. Diagnosed in test/webtx_debug/:
node runs the ACTUAL web-page functions vs a python transmit.py-reference — symbols
bit-identical (292/292 frames), so generation was correct and the CONTENT was the
flood; per-frame white%% stats nailed frames 262-274 (86% max at 274, + a 223-232
cluster). FIX: XOR each source row with PCG64(SCRAMBLE_SEED=5000 + idx) keystream
before repair/RS (web), unscramble at assembly only (phone, gated on the START QR's
SCR=1 — fountain/RS never see plaintext, old transmitters stay compatible). After:
white 26.4% mean / 28.6% MAX across all frames; round-trip asserted = original PDF.
Lesson for the ledger: line codes exist for a reason — any structured payload
(text, BMP, sparse binary) would have hit this; synthetic random payloads hid it
for the project's entire life.

2026-07-19  FILE DELIVERY: a real file selected on the web transmitter now announces
NAME=<sanitized filename> in the arm QR (string key, bypasses the Double plan dict;
web strips unsafe chars, caps base at 40, keeps the extension). On a COMPLETE
transfer (MISSING none = RS-proven) the phone saves/replaces the payload as
Documents/QR Transmit/<name> — visible in the Files app under On My iPhone >
QR File Transmit > QR Transmit; pdf/jpg/png/jpeg/avif open right there via QuickLook.
(iOS sandboxing: no app may write the system-wide Downloads folder — the app's shared
Documents is the closest legal equivalent.) Incomplete transfers log "file NOT saved
(N frames missing)". Also web px picker gained 12 (12c16 + minted ANE T=6724 at the
1024 field; native r12 stays a science option). 12px web math: 82x82, kpf 3629 @
nsym64 -> 17.7 KB/s @5fps.

2026-07-18  SPILL TIER (no more drop-and-forget): when the data backlog is full, the
frame JPEG-compresses (q85, ~2-3 MB, encoded on a .utility queue off the camera
thread) into a byte-capped spill (~200 MB ≈ 70 4K frames) and decodes LATE in
assemble() after END — the Surface's "capture never blocks, backlog drains after END"
doctrine, RAM-shaped for the phone. Drops now only happen past the spill cap. Log
line: "spill: N late frames decoded". writeJpg refactored into jpegData/jpegFrame.

2026-07-18  **ON-DEVICE ANE MINTING — the phone now builds an ANE model for ANY grid
at the TRAIN QR.** The mlprogram bakes T into its shapes, but T is only the BATCH dim
(weights are per-tile), so ANEField.reshape() mints a T-specific model from the
TileNet2 template with three edits: shape tokens in model.mil (16 sites, regex with
digit guards — no blob-offset collisions, verified), metadata.json, and THREE protobuf
varints in coremldata.bin (the model description predict validates against; 5244=FC28
-> newT, equal-length varint swap keeps framing — T must be 128..16383). PROVEN ON MAC
COREML BEFORE SHIPPING: minted T=3844 loads, predicts, and its per-tile logits are
bit-IDENTICAL to the template's. First-frame ANE-vs-CPU cross-check remains the
on-device guard (auto CPU fallback). Timing: engine now minted at the TRAIN QR
(grid known — the QRs already carry everything needed: PX/COLS/ROWS -> T), START only
re-patches the tuned weights (~0.3s, new anePatched flag; sessions without training
mint+patch at START). Kills yesterday's web-grid CPU bottleneck (T=3844 nn 104ms ->
~15ms expected). ALSO: **tuned models now auto-persist** — saveTuned writes the
px+set ACTIVE file when final training err < 5%, so the next session of that
px/scaled/handheld combo warm-starts from it automatically (loadNet logs "saved
tune"); rx_tuned_L* export unchanged. The old "training never writes model files"
doctrine is retired by design.

2026-07-18  ORIENTATION-FOLLOWING CAPTURE: the camera now delivers PORTRAIT buffers
(1080x1920) when the phone is held upright and landscape only when horizontal —
connection.videoRotationAngle mapped from UIDevice orientation (portrait 90 /
landscapeLeft 0 / landscapeRight 180 / upsideDown 270; faceUp keeps last), applied at
session start + live via orientationDidChange (a connection property — no session
rebuild). Natural aim for the 1:1 web field. Geometry always handled rotation; the
NEW hazard was buffer DIMS flipping mid-run on queued frames -> size guards
(f.count == fw*fh*4) added at tryDecode/trainFrame/hhDecode/hhRelock/dbgLock/calRun
(hhDecode's guard also fixed a latent hhInFlight leak on bare return). Mid-run
rotation = frames skipped until geometry re-locks; ANE falls to CPU via the existing
size guard, next session rebuilds the texture.

2026-07-18  **WEB TRANSMITTER v1 (shapes_n_colors/web/) + 30px BIG-GLYPH REGIME.** New
approach after the first handheld field run stalled at 8% train err: replace the mac
python transmitter with a browser page and go to big glyphs. web/index.html (+
qrcode.js, MIT qrcode-generator) is a full transmit.py port: settings panel persisted
in localStorage (px 16/20/24/30 default 30, field 1024, fps, train fps, nsym, repair%,
payload KB or file upload), phase flow idle field -> TRAIN QR -> training frames ->
START QR -> data -> END QR, Enter advances / Esc aborts. VERIFIED BIT-EXACT vs the
python reference (node harness): PCG64 (numpy default_rng chain — training truth,
repair masks, synth payload), crc32, RS encode, md5. Glyphs: blocky = nearest 4x4;
line shapes = binary capsule strokes at 4x + exact 4x4 box-average (cv2 supersample+
INTER_AREA equivalent; ink fractions match to 4 decimals except X/cutSlash ~2% off —
training absorbs). Pacing: rAF (compositor-vsynced) with AUTO-MEASURED panel Hz →
hold = round(Hz/fps) ticks, so 60 AND 120/ProMotion pace exactly; frame 0 gets 3
extra dwells; "late tick" counter + Hz readout in the status line. Field = 1:1 square
1024px LOGICAL pixels (image-rendering pixelated → integer 2x on retina = same
physical regime as pygame SCALED; "HiDPI 1:1" checkbox flips to device px), black
page + black bg + white ring = minimum panel light. wakeLock on, cursor hidden
(cursor-on-ring trap). Copied to mac ~/windows/.../web/ — open index.html in a
browser. RECEIVER: canonical-16 GENERALIZED past px 12 — model keys are now
px*100+16 ("30c16"=3016), any px other than 12/16 is FORCED scaled (no native
weights exist), 30c16 warm-starts from the best r16. px picker gained 30. At 1024px
field: 33x33 = 1089 tiles, 3 RS blocks, kpf 573@nsym64 → 2.8 KB/s @ 5fps. NO ANE
model for T=1089 → CPU decode (~10 ms at this tiny T — fine). 30px glyphs on the
mac panel = 60 device px ≈ 6 mm: the blur/shear robustness play for handheld.

2026-07-18  START-TRANSITION SIGSEGV FIXED (the "sometimes crashes when training stops
and the pre-live QR shows up" bug — crash reports 13:19 pre-v2 via dataFrame and
22:30 on v2 via hhDecode, IDENTICAL stacks: rx.worker -> aneDecode ->
MetalWarp.upload -> tex.replace -> AGX driver segfault). Cause: aneReady only rebuilt
the Metal warp when the GRID (T) changed between sessions, not when the CAMERA
RESOLUTION did — a 4K session followed by a 1080p session with the same grid (the #9
toggle, or handheld's forced 1080p after a 4K stable run) left a stale 3840x2160
texture whose tex.replace reads 33 MB from an 8 MB frame buffer. Fires exactly at
START because that's when ANE goes live and the first decode runs. Fix: aneReady
rebuilds metal+aneArr when mw.sw/sh != fw/fh (ANEField byte-search kept — only the
texture side rebuilds), plus a size guard in aneDecode (mismatch -> CPU decode,
never a driver overrun). Deployed.

2026-07-18  **HANDHELD v2 SHIPPED — full rework (FA naming + tracked geometry), fully
isolated from stable mode.** The dwell machinery is gone from handheld: every camera
frame (60fps) probes its 49 header tiles on the CPU net (~3ms, the debug-mode probe
promoted to production) -> (idx, logit-margin score); the best-scoring shot per idx
gets ONE full ANE decode ~50ms into the dwell (first shot = ISP blend, and the dwell
tail stays free to RETRY an RS fail — decode 25ms vs dwell 100ms). Geometry TRACKED:
last-good Hinv + a ±8/±16px translation shift ring (2 probes/frame, ring covered in
4 frames); full findQuad demoted to a throttled workQ fallback that doubles as the
bootstrap (probe-CRC verified, never full-decode); AF re-settle as last resort (>2s
unnamed + 2 failed re-locks, never during train). armed->data flips on the first RS
SUCCESS (the held START screen fails RS by design; data frame 0 shares header idx 0,
so armed attempts pace at 150ms and continue into data). Training feeder: probe-named
frames harvest at ~8/s through the untouched trainFrame. ISOLATION: model namespace
"hh" (hh_active/fork/factory_model_*.bin) — handheld sessions warm-start from the
best stable tune but SAVE apart; the handleQR/calRun reload condition now checks the
namespace, killing the in-memory-net leak between modes in BOTH directions (selfTest
also marks its stomped net so the next session always reloads — old trap fixed);
menu Save/Load/Factory follow the Handheld toggle; copy-capture forced in-session
(best-shot ledger retains frames; zero-copy pool retention = flaky camera);
auto-CAL suppressed (motion sweeps score -1 and the lapvar fallback locks garbage).
Native r12 handheld falls back to v1.2 (no ANE -> 343ms CPU can't hold the rate).
Concurrency: all HH decision state camera-thread-only; workQ answers via 3 mailboxes
under the existing lock (hhNewHinv / hhResults / hhRelockOutcome). Stable path is
byte-identical with the toggle off (every branch guarded). FIELD TEST recipe:
Handheld ON, transmit.py fps=10 nsym=64 repair=12; read the log for "HH: named x/y"
(target >=80% on smooth sweeps — constant-velocity shear is affine, jerks are the
discards), relocks s/f, af kicks, retries, frame-0 presence, md5.

2026-07-18  NSYM HANDSHAKE BUG (first nsym=64 phone run "succeeded" with a wrong md5,
MISSING: none): the receiver builds its grid at the FIRST control QR — the TRAIN QR,
which did not announce NSYM — so it fell back to 32 and IGNORED the START QR's
NSYM=64 (grid-exists guard). The killer: an nsym-64 codeword PASSES an nsym-32
syndrome check with zero errors (g64's roots contain g32's), so every frame decoded
"cleanly" while the receiver took 223 bytes/block instead of 191 — 32 parity bytes
per block assembled as data, every frame at the wrong kpf stride. Zero wire errors
needed; also halved correction to 16/block (the run's 4 RS failures). Diagnosis was
pure log arithmetic: SIZE/FRAMES = 1048576/366 = 2865.0 = 15x191 = nsym 64. FIXED
both ends: (a) receiver rebuilds the grid when a QR announces a different NSYM (only
kpf changes — geometry/model/training survive); (b) TRAIN QR now announces NSYM too,
which also plugs receiver.py's identical latent hole (its worker setup() runs at the
TRAIN QR with config.NSYM fallback). Non-training sessions never had the bug (grid
born at START). Everything handheld worked: QR scans, 0.88% training at 1/500
shutter ISO 3580, ANE live, 0 re-locks on the stable phone.

2026-07-18  HANDHELD CAMERA OVERRIDES (from docs/iphone_camera_handheld.txt; Settings
gained a bottom section active ONLY while Handheld is on, overwriting the numbered
toggles): (a) ultra-short shutter 1/250|1/500|1/1000 (default ON at 1/500) applied at
the 3A lock, ISO scaled+clamped — blur is linear in exposure and is THE handheld
killer; beats #8's gentle 8 ms; (b) force 1080p (default ON, overrides #9) — 4K
doubles sensor readout = rolling-shutter jello driven by the hand; (c) #1+#2
(explicit format + 60 fps) forced regardless of toggles — more shots per dwell =
more chances to land in a hand's micro-still-points. Stabilization deliberately
untouched: iPhone video OIS only comes BUNDLED with EIS frame-warp (non-rigid,
fights the homography) — parked as the experiment of last resort. Doctrine: brighten
the panel to pay the ISO bill, brace elbows (Z-drift through the f/1.78 depth of
field beats angular shake), retrain after toggling. Transmit side for handheld:
nsym=64 repair=12 fps=10.

2026-07-18  HANDHELD v1.1: control QRs decoupled from the dwell machinery. Live test
showed handheld NEVER reached TRAIN — control QRs are only scanned on HELD commits,
which require 1.5 s of per-band stability a hand never produces. Fix: in handheld
mode a raw camera frame is QR-scanned every 0.7 s (1.0 s during data — also fixes
END detection, which would otherwise only fire by timeout); handling factored into
handleQR() shared with the held-commit path. Costs ~1 Vision detect/s on the camera
thread (~2 dropped camera frames of 60 per scan — irrelevant).

2026-07-18  HANDHELD v1 shipped ("Handheld" toggle on the main menu next to Settings).
Trust-but-verify geometry: NO per-frame border hunt — the cached homography is tried
first and every decode's header CRC verifies it for free; on failure (hand moved past
tolerance) THAT frame runs the full ring search, re-locks, and continues. Applies to
both data and training paths (training previously just counted locked-geometry CRC
fails; now it re-searches too). End-of-run stat: "geometry re-locks: N". Candidate
attempts go through the ANE path. Handheld doctrine: EIS ON (#7 unchecked — the
bisection rejected EIS-off for exactly this), fps 10-15 to start, #8 if motion blur
shows in CAL. Escalation ladder if v1 struggles: header-probe-only candidate
arbitration (49 tiles vs full decode), Hinv motion extrapolation, CoreMotion gyro
prior, KLT tracking (the FA "unstable camera" future). NAMING NOTE: the Settings
"Zero-Copy Capture" toggle (camera->dedup, ON all along) is DIFFERENT from the
still-unimplemented decode-side texture-upload zero-copy (CVMetalTextureCache, the
5-12 ms "upload" row in the frame-time table).

2026-07-18  **R12c16 @ 30fps = 126.56 KB/s BYTE-PERFECT (nsym32) — 1 MB in 8.1 s.**
174/175 + 1 rebuilt, 0 failed, 0 dropped, err 1.03%. Leeway audit: upload 5.0 +
metal 2.3 + predict 17.5 + rs 2.0 = ~27 ms vs 33.3 ms slots = 19% headroom, never
fell behind. Predict carries ~4 ms live-marshaling over the idle bench. Options by
cost: (1) nsym=16 — free, ~135 KB/s, 2.4x safety at current err; (2) zero-copy
texture upload (5.0 -> ~0.5 ms) = insurance + enabler; (3) r10 canonical-16 =
~170 KB/s @ 20fps / ~250 @ 30 (needs generalized scale-to-16, TileNet10, and the
channel verdict on 10px glyphs at ~17 camera px — CAL will tell in one session);
(4) 60fps display now PROVABLY dead (predict 17.5 > 16.7 slot, plus the dedup wall).
Throughput frontier = tile density, not rate.

2026-07-18  **CANONICAL-16 VALIDATED SPECTACULARLY — R12c16@20fps = 94.83 KB/s
BYTE-PERFECT AT NSYM32 (new record).** First canonical session's FIRST chunk hit
0.44% err (native r12: started 7.1%, plateaued 2.5% over 32 chunks) — the r16
warm-start + 16x16 sampling of the 4K footprint is the whole game. TileNet12 ANE
live: cross-check 0/9348, predict 20-25 ms, decode ~32 ms total. Runs: 10fps 51.5
(174/175+1, 0.34% err), 20fps 94.83 (174/175+1, 0.78% err, 50ms slots). Fork flow
worked as designed: one session's warm restart RAISED err chunk-over-chunk (hot
phone + factory drift) -> abort, Load fork, next session started at CAL 4.0% and
sub-1% in 6 chunks. Ladder: r16 44/56/79 @ 15/20/30; r12c16 51/94.8 @ 10/20.
NEXT: r12c16@30fps ≈ 176 KB/s theoretical — decode 32ms vs 33ms slots, same razor
edge r16@30 survived; if drops appear, the texture upload (7-12ms) is the trim
(CVMetalTextureCache zero-copy finally has a customer).

2026-07-18  PHASE A: CANONICAL-16 FOR R12 ("Scale to 16px" checkbox next to the px
picker, shown when 12px selected; default off = native r12). Checked: r12 sessions
sample each 12px tile's ~20-camera-px footprint (4K) onto a 16x16 input — the r16
NETWORK SHAPE verbatim — so the model set "12c16" (active/fork/factory_model_1216.bin)
WARM-STARTS from the best r16 weights around (active_16 -> factory_16 -> bundle).
Grid grows a `scaled` mode (K=16, C=20 over px=12 — every path was already K/C-
parametric, incl. the Metal kernel whose C=20 convention matches). ANE: the mlprogram
bakes in the TILE COUNT, so TileNet12.mlpackage = same arch at T=9348 (built+validated
10/9348 noise flips); aneReady picks the model by grid T and rebuilds engines when the
grid changes between sessions. CPU decode note from the failed native-r12 run: nn was
343 ms/frame (9348 tiny per-tile GEMMs, hot phone) — ANE is a PREREQUISITE for live
r12, not a luxury. Also: 2026-07-18 late run showed native r12@4K trains to ~2.5-2.7%
plateau (32 chunks); nsym32 wall is ~2% -> all 57 attempts RS-failed. Canonical-16
targets sub-1%; even ~2% unlocks nsym32 r12@20fps ≈ 124 KB/s.
CAL GUARD shipped same build: calibrate() refuses while a sweep runs + button greys
out (two interleaved sweeps fought the lens and locked 0.250 — the "can't lock in" run).

2026-07-18  **30 fps RECORD: 78.67 KB/s BYTE-PERFECT** (r16 4K, 313/314 + 1 rebuilt,
0 failed 0 dropped, 1 MB in 13.0 s; decode 21 ms/frame in 33 ms slots; training
2 chunks to 0.00%). Pool histogram 0:288/0 — at 2-refresh dwells almost every commit
is the never-stable FALLBACK pick, and ALL of them decoded: with a 10 ms shutter on
a vsync-locked cadence every dwell has a clean-enough shot, so FA-style best-shot
picking stays unnecessary even at 30 fps. 60 fps FAILED ARCHITECTURALLY as expected
(0/314, ONE decode attempt): 1-refresh dwells mean consecutive camera frames never
look stable -> dwell-based dedup never commits. 60 fps would need per-frame FA
naming + <8 ms shutter + decode <16.7 ms (zero-copy texture upload). Parked.
Display-rate ladder now: 15 -> 44, 20 -> 56, 30 -> 79 KB/s, all byte-perfect.

2026-07-18  Chunk speed round 2 RESULTS: 3-5 s/chunk (was 12.6-14.7), 4 threads
still fastest. DECISION: keep fastest — race-to-idle beats frugality here (training
is ~10 s of a session whose thermals are dominated by minutes of 4K60 camera+ISP;
the 1-thread saving is ~10 J vs >100 J of camera). Encoded as a tie-break: the
picker takes the fewest threads within 10% of the best measured time, so it drifts
cooler automatically if thermal state ever narrows the gap. Batched-GEMM (#1)
PARKED: chunks already hide behind frame supply and r16@4K converges in 1-2 chunks;
revive only for r12-class channels, continuous online tuning in data mode, or
thermal-throttled chunk ballooning. GPU/MPSGraph stays the ceiling after that.

2026-07-18  Chunk speed round 2 (after 4-thread chunks measured 12.6-14.7 s =
ANTI-scaling; AMX is per-CLUSTER so 4 threads of tiny GEMMs serialize on 2 units):
(a) gauss() noise pool — augmentation reads a precomputed 256k-float ring (random
start per sample) instead of ~24M Box-Muller calls per chunk; (b) thread-count
self-A/B — first chunks probe 4 -> 2 -> 1 threads, then every later chunk uses the
measured fastest (chunkNth map, kept across sessions in-app); (c) chunk runs at
.userInteractive QoS (during TRAIN decode is idle and harvest is gated — nothing
else needs the cores). Chunk line shows threads+seconds+batch, so the winning config
is visible in the log. If still slow after this: batched-GEMM training (one big
sgemm per layer over the 256 batch — what AMX actually wants) is the next step,
GPU/MPSGraph after that.

2026-07-18  Pipelining follow-up: unbounded harvest STARVED the chunk (live: hundreds
of frames harvested, chunks ~20 s — gather+classify is ~150-200 ms/frame and ate 2-3
cores continuously once nothing blocked it; the old serial design had been
accidentally self-throttling). Fix: harvest gates at exactly the NEXT batch
(sinceChunk < TRAIN_EVERY) and reopens when the chunk consumes it. Chunk log now
includes the batch size: "(chunk Y s, T threads, batch N frames)".

2026-07-18  TRAINING PIPELINED + PARALLELIZED. Chunks now train a CLONE on their own
thread (runtime_tune doctrine): harvest keeps running on the worker at full rate, so
the next 20-frame batch accumulates DURING training and fires the instant the weight
swap lands — the serial gather->train->gather cycle is gone. The 256-sample batch is
split across all-but-two cores (6-core phone -> 4 threads; <=2 cores -> 1), per-thread
gradients summed into one Adam step. Each chunk logs its wall time:
"TRAIN frames=N err=X% (chunk Y s, T threads)". Known benign race: harvest may
overwrite a reservoir row mid-read (~1 torn sample in 30k — same acceptance as the
Surface). START mid-chunk discards the clone (live net never sees half a chunk).
4K r16 aside (same session): 4K costs ONLY +7 ms texture upload (10.6 vs 3.8;
predict/metal/rs unchanged, ~29 ms total) and makes the r16 channel PERFECT — first
ever TRAIN err=0.00% at 20 frames, factory read 2.9% cold. r12 RETIRED for now:
29% (1080p) -> 7% (4K) still sits on the nsym64 RS wall; r16@4K takes the same
throughput by RATE instead (30 fps ≈ 87 KB/s, decode 29 vs 33 ms slot — next test).

2026-07-18  CAL CRASH at r12 = LATENT Int(inf) TRAP, fixed everywhere. Crash report
(SIGTRAP in Geometry.findQuad from calRun): the orientation-scoring loop divided by
the homography's projective term with NO qd~0 guard — a defocused r12 blob (wide
sweep, lens 0.53) produced a near-singular quad -> Int(inf) traps in Swift. Same
unguarded Double->Int pattern lived in the edge-march AND the receiver's
gatherX/gatherCrops/gatherTiles (which CAL + geometry search feed with UNVALIDATED
candidate homographies). Fix: bounds checks moved into the double domain (NaN/inf
fail the comparison and take the miss path) + degenerate-side guard in the edge
march. r16 never tripped it because its blobs were always clean; blurry r12 sweeps
roll these dice constantly.

2026-07-18  FIRST R12 SESSION post-mortem: NOT a training problem — an OPTICS one.
CAL's best at r12 was 3017/7878 glyphs = 61.7% err at GOOD focus (lapvar 3680), the
cold loop_L12n harvest failed 2675 header CRCs vs 133 harvested, and training
plateaued at ~29% = the Surface's old r12 physics wall. Cause: at 1080p and this
distance an r12 tile is ~10 camera px — undersampled, no training can fix it.
Levers (all per-session, r16 config untouched): (a) move the phone ~25-30% closer
and re-CAL (free; r12 needs 16/12 the pixel density); (b) NEW camera toggle
#9 · 4K capture (3840x2160 = 2x linear px/tile at the same distance; default OFF so
r16 stays 1080p60; backlog caps now shrink automatically on 4K frames — 33 MB each);
(c) try #8 OFF at r12 (the short shutter's tripled ISO noise costs small glyphs
disproportionately; 10 fps dwells don't blend at 25 ms anyway). ALSO NOTED: Stop
mid-training DISCARDS the session's tuning — the tuned model only saves when START
arrives (or the session finishes). Train -> ENTER on the mac -> then stop/Save fork.

2026-07-18  Training chunk trigger is now time-OR-count: a chunk fires every 20
harvested frames as before, OR once >=4 frames have been waiting 20+ s since the
last chunk (or since harvest began). A cold model (the picture-trained r12 factory
especially) harvests slowly because header CRCs fail — waiting for 20 full frames
starved the very training that speeds the harvest up.

2026-07-18  R12 SUPPORT on the phone. Net.swift is now parametric over k (only the
fc layer's size changes: 32*(k/4)^2 -> 512 @ r16, 288 @ r12); all receiver paths
(gather/classify/crops/training/probe) take K/C from the Grid (K = px, C = px + 4 —
identical numbers at r16, so no behavior change there). Models are per-px sets:
active/fork/factory_model_{12,16}.bin in Documents; the QR's PX picks which set a
session loads and trains. Factory r12 = loop_L12n (bundle weights12.bin, raw fp32,
PICTURE-trained — expect a heavy first training session); factory is now
USER-OVERWRITABLE: main menu grew a 12/16 px picker for Save/Load/Factory plus an
"overwrite factory too" toggle (resets each app launch) so a camera-trained r12 can
replace loop_L12n as the phone's factory. ANE stays r16-only (TileNet2 bakes in
T=5244 + 16x16) — r12 decodes on the CPU (~50 ms/frame, fine at 5-10 fps; build a
TileNet12 mlpackage when r12 goes fast). Mac side needs nothing: transmit.py px=12
(grid 123x76 = 9348 tiles announced via QR). First-run suggestion:
  ../venv/bin/python transmit.py px=12 fps=10 nsym=64 repair=6

2026-07-18  ODD-INDEX DWELL LOSSES: ROOT CAUSE FOUND AND FIXED (mac-side). Three
morning runs at the new spot lost 25 frames — ALL odd indices (p ~ 2^-25), with exact
+24-frame (1.6 s) periodic segments; error rate/CAL/decode all healthy, all camera
configs affected equally. Measured on the rig over ssh: panel WAS correctly pinned
1512x982 @ fixed 60 Hz, transmit.py fix WAS running — but pygame vsync silently does
NOT engage on plain FULLSCREEN (measured 4.63 ms/flip = free-running; windowed SCALED
16.96 ms = engaged) and the clock.tick(15) fallback paces at 67.777 ms, not 66.667.
That +1.1 ms/frame drift walks the swap boundary across a refresh line every ~15-25
frames -> periodic one-refresh-short/long dwells = the metronomic losses (and, same
family, the dissolved-frame-at-~95% glitch). FIX: set_mode(desktop_size,
SCALED|FULLSCREEN, vsync=1) — verified on the rig: 15.91 ms/flip, vsync ENGAGED,
size honored. Deployed to the mac (surface offline). Sanity items for the next run:
verify CAL/lapvar stays sharp (SCALED renders via a texture path — confirm the 2x
Retina mapping stayed pixel-exact) and expect MISSING to go to ~none at 15 fps.
Repair-solve footnote: "+0 rebuilt" with 10 missing/10 repairs is EXPECTED (~29%
full-rank odds for n=n random GF(2)); with 6/10 it was ~3-6% bad luck, no bug.

2026-07-18  App QoL + the straddle fix: (a) Receive now fully resets the previous
run's stats (decCnt leaked — "decoded 295/314" survived into the next session's stat
box); (b) every run also writes Documents/QR-receiver-log-{yyyyMMdd-HHmmss}.txt
(persistent, visible in the Files app; log.txt still holds the full app session for
headless pulls); (c) camera bisection settings #1-8 are now Settings toggles (short
names, defaults = production 1+2+5+6; the session RECONFIGURES on next Receive when
toggles changed; #8 applies at the 3A lock — retrain after flipping it); (d) FIX:
trainChunk now aborts between steps when trainingOver flips — a chunk that straddled
START blocked the serial worker ~10 s and cost 6 dropped + 9 missing at the start of
the failed 2026-07-18 run (log order proved it: "data phase" before "TRAIN
frames=200"/"ANE decode LIVE").

2026-07-18  Settings page + ZERO-COPY CAPTURE toggle (default ON, applies at next
Receive start). ON: captureOutput hands the camera's CVPixelBuffer straight to dedup,
which samples its metrics from the locked buffer in place — the 8 MB copy happens
only when a dwell COMMITS and for the 3 fps preview (~20 copies/s instead of 60;
frees the CPU that predict's marshaling phases compete with). The pend candidate is
a lazy byte-provider closure retaining AT MOST ONE pool buffer — holding more
starves the ~6-deep capture pool (= the flaky-camera failure mode, deliberately
avoided). CAL routes to the copy path (wants fresh full frames); Debug forces copy
mode entirely (per-frame probing needs bytes). OFF = the original copy-every-frame
path, byte-identical behavior.

2026-07-18  22 fps run POST-MORTEM: decode fully vindicated (0 dropped, 2 failed,
predict 16.3 ms unchanged) but 90/314 frames MISSING = never committed by dedup —
the bottleneck is now capture/optics. Two causes: 22 does not divide 60 (present()
falls back to clock.tick -> dwells alternate 2/3 refreshes = 33/50 ms; missing list
shows the beat pattern) and the 16.6 ms locked exposure blends most of a 33 ms
dwell (pool histogram: 57 zero-pool commits vs none at 15 fps; train err 0.78% vs
0.15%). Fix ladder: fps=20 (free, every dwell = exactly 3 refreshes), then camera
#8 short shutter (<=8 ms, ISO scaled — the bisection kept it for exactly this
regime; its old drop failure was the now-fixed decode bottleneck), then FA-style
per-frame header naming if 30 fps still leaks dwells.

2026-07-17  DECODE ENGINE LADDER — measured on the SAME live task (mac -> iPhone,
15 fps display, r16 92x57, 1 MB transfer, in-session-trained model, phase times from
the receiver's own counters, ms per frame attempt):

| config                          | warp/gather      | nn (classify)   | rs | total | run outcome                     |
|---------------------------------|------------------|-----------------|----|-------|---------------------------------|
| CPU, workQ .userInitiated       | 14               | 78              | 1  | ~93   | 56 dropped, transfer FAILED     |
| CPU, workQ .userInteractive     | 7                | 44              | 0  | ~51   | 313/314+1, 43.4 KB/s, clean     |
| ANE, fp32 input  << PRODUCTION  | 6 (up 4.0+mtl 2.1) | 16.3 (predict)| 3  | ~25   | 313/314+1, 43.4 KB/s, clean     |
| ANE, fp16 end-to-end            | 6 (up 3.9+mtl 2.0) | 20.3 (predict)| 2  | ~28   | 313/314+1, 43.4 KB/s, clean     |

Reference points: idle-phone bench predict was 7.5 ms (ANE) — live predict carries
~9 ms of load/ANE-marshaling overhead; Surface OV int8 was 36.4 ms NN alone.
Lessons: (1) frame budget at 15 fps is 66 ms — the 93 ms config HAD to fail, the rest
are display-limited (identical 43.4 KB/s); decode headroom at PRODUCTION ~2.6x ->
20/30 fps display is the next lever, not decode. (2) fp16 END-TO-END LOST: native
fp16 input (build_mlmodel3.py + warpXh half-writing kernel) measured predict 20.3 vs
16.3 ms — Core ML's own fp32->fp16 ingest path is faster than feeding fp16 directly.
REVERTED to the fp32-input model (build_mlmodel2.py); warpXh stays in Warp.swift
unused. Both fp16 runs were symbol-exact (cross-check 0/5244). (3) All four rows
md5-matched except the first — errors never were the problem, throughput was.

2026-07-17  ANE CONFIRMED LIVE at 15 fps: cross-check 0/5244 tiles differ (weight
patch is symbol-exact), stages upload 4.0 + metal 2.1 + predict 16.3 + rs 3 ms
= ~25 ms/frame (CPU path was ~51). 313/314 + 1 rebuilt, 0 failed, 0 dropped, md5
match, 43.4 KB/s — display-limited now (~2.6x decode headroom at 15 fps). Two traps
found on the way: (1) coremltools INLINES consts < 10 elements into model.mil as hex
literals instead of the weight blob — bc (4-value color bias) was missing from
weight.bin and the all-or-nothing search disabled the engine ("ANE engine
unavailable" in the log; the first "43 KB/s ANE" run was actually pure CPU). Fix:
ANEField patches tiny consts into model.mil text (hex floats via %a), always from the
pristine copy. (2) The earlier CPU nn drop 78 -> 44 ms came from workQ QoS
.userInitiated -> .userInteractive (decode was losing P-core time to the camera
queue's 60 fps frame copies) — which also confirms the camera-flakiness theory:
decode at 100% CPU was starving capture. Model pinned .cpuAndNeuralEngine (GPU
belongs to Metal warp + UI).

2026-07-17  15 fps RECORD 43.4 KB/s (was 35). Per-run history at 15 fps: pre-fix
93 ms/frame -> 56 dropped, failed; QoS fix 51 ms -> clean; ANE 25 ms -> clean with
2.6x headroom. Next rungs: 20/30 fps display (both divide 60), then rung push.

2026-07-17  ANE decode wired LIVE into the phone receiver. Data-phase decode now runs
Metal warp -> Core ML ANE (bench E: 8.3 ms/frame vs ~50 ms CPU — the fix for the
high-fps drop failures). The bundled TileNet2.mlmodelc carries FACTORY weights, so at
every START the receiver patches the current (tuned) net into a Documents copy of the
compiled model: each tensor's offset in weights/weight.bin is found once by searching
for its factory fp16 byte pattern, then overwritten and the model reloaded (~0.3 s).
The first ANE-decoded frame is cross-checked against the CPU net; >1% tile
disagreement -> automatic CPU fallback with a log line. Training, CAL and debug
probing stay on the CPU net (live-weights doctrine, same as receiver.py's
nn_sym(live=True)).

2026-07-17  RESEARCH_FAILS is NOT ported to the phone. receiver.py re-runs the full
geometry search after RESEARCH_FAILS consecutive fast-path CRC failures; the phone's
LinkReceiver caches Hinv at the first CRC-proven decode and never invalidates it.
Fine while the rig is propped and static — but if the phone or the mac is nudged
mid-run, every later frame fails against the stale homography and the transfer dies
with no recovery. Port the failure counter if the rig ever stops being static.


##########  POSSIBLE TODOs (parked, gather here)  ##########

- SECOND ZERO-COPY (decode-side texture upload): CVMetalTextureCache wraps the camera
  CVPixelBuffer's IOSurface directly as an MTLTexture — kills the 5-12 ms (thermally
  wobbly) tex.replace upload of the committed frame's byte array, the shakiest line in
  the 30 fps budget. NOT the Settings "Zero-Copy Capture" toggle (that is camera->dedup,
  already shipped). Cost: decode-queue frames must RETAIN pool buffers -> needs a small
  in-flight cap + byte-copy fallback when the backlog deepens (pool starvation is the
  known failure mode). Build when 30 fps margins need insurance or r10c16@30 happens.
- RESEARCH_FAILS port: see the 2026-07-17 note above — the handheld trust-but-verify
  re-lock (v1, 2026-07-18) now covers the moved-rig case in handheld mode, but static
  mode still never re-searches after a CRC-fail streak. Port the counter if a propped
  rig ever gets nudged regularly.
- SOFT-DECISION RS ERASURES (receiver-side, wire-compatible): flag low-softmax-margin
  tiles as erasures — RS corrects 2x as many erasures as errors, doubling the effective
  budget without touching the transmitter. The designed-but-never-built v1 idea; the
  natural handheld robustness lever alongside runner-up-candidate retry per dwell.
- Fountain (LT/Raptor) coding is the elegant version, and it's not just extra copies. 
  Instead of transmitting frame chunks 0–99 directly, every transmitted frame carries a 
  random XOR-combination of source chunks (the combination recipe derives from the frame's 
  seed/index, so both ends know it). The property you get: the receiver doesn't need any 
  particular frame — it needs any ~102–105 distinct frames out of however many you show, 
  and it can reconstruct all 100 source chunks from whatever subset landed. The transmitter 
  just streams, say, 108 coded frames and stops. No feedback, no human in the loop, no 
  "which one died" question at all — a lost frame is automatically compensated by the 
  next coded frame that arrives.
