# ios/ — the receiver app ("QR File Transmit")

SwiftUI + AVFoundation + Accelerate + CoreML + Metal, zero third-party
dependencies, ~3500 lines. **Research build**: the UI is a developer console
(logs, benches, debug modes) — a product UI is future work.

## Build & install (Mac + Xcode required)

```bash
brew install xcodegen
cd ios
xcodegen generate                      # project.yml is the source of truth
DEV=<your-device-udid>                 # xcrun devicectl list devices
xcodebuild -project QRFileTransmit.xcodeproj -scheme QRFileTransmit \
  -configuration Debug -destination "id=$DEV" -allowProvisioningUpdates build
APP=~/Library/Developer/Xcode/DerivedData/QRFileTransmit-*/Build/Products/Debug-iphoneos/QRFileTransmit.app
xcrun devicectl device install app     --device "$DEV" "$APP"
xcrun devicectl device process launch  --device "$DEV" com.example.QRFileTransmit
```

Set your own `DEVELOPMENT_TEAM` in `project.yml` (free personal team works;
apps expire after 7 days — rebuild to refresh). First install of a given bundle
ID: enable Developer Mode, then **trust the developer profile on the phone** —
otherwise the launch fails with *"…its profile has not been explicitly trusted
by the user"*:

> **On the phone: Settings → General → VPN & Device Management → (under
> "Developer App") tap your Apple Development profile → Trust.**

(This is per app identity: a new bundle ID, or a reinstall after uninstalling,
needs trusting again.)

Headless/ssh builds: the keychain must be unlocked INSIDE the same command:
`security unlock-keychain -p "$PW" ~/Library/Keychains/login.keychain-db && xcodebuild …`
(one-time: `security set-key-partition-list -S apple-tool:,apple: -s`).

## Remote dev workflow (PC → Mac → iPhone), traps included

This project was developed by editing on a PC, shipping to a Mac over ssh, and
building onto a USB-connected iPhone. Everything that bit us, so it won't bite
you:

- **scp cannot target mac paths containing spaces** from Windows (every quoting
  variant mangles) — always stage via `/tmp` then `ssh 'cp /tmp/… "/path with
  spaces/"'`.
- **`devicectl install` requires the phone UNLOCKED** (it hangs or errors
  otherwise); confirm the device shows `connected` in
  `xcrun devicectl list devices`. First contact: accept the "Trust This
  Computer" prompt, enable Developer Mode (Settings → Privacy & Security,
  needs a restart), and after the first install trust the developer profile
  (Settings → General → VPN & Device Management).
- **`xcodegen generate` overwrites the .xcodeproj** — any signing team set in
  Xcode's GUI is wiped; keep `DEVELOPMENT_TEAM` pinned in `project.yml` (it is).
- **Adding a source file** requires re-running `xcodegen generate` (sources are
  globbed at generate time); pure edits don't.
- Pull anything out of the app sandbox headlessly:
  ```bash
  xcrun devicectl device copy from --device "$DEV" \
    --domain-type appDataContainer --domain-identifier com.example.QRFileTransmit \
    --source Documents/<file> --destination /tmp/<file>
  # useful <file>s: log.txt (whole app session), QR-receiver-log-*.txt (per run),
  # received_payload.bin, cal.jpg, dbg0-2.jpg, active_model_*.bin (tuned models!)
  # list what exists:
  xcrun devicectl device info files --device "$DEV" --domain-type appDataContainer \
    --domain-identifier com.example.QRFileTransmit --subdirectory Documents
  ```
  Note `copy from` OVERWRITES the destination silently — `rm -f` first if you
  need to detect a failed pull.
- Crash reports: `--domain-type systemCrashLogs` (grep for `QRFileTransmit-*.ips`;
  JSON after the first line — the stack usually names the guilty queue).
- App suspends on auto-lock mid-run; the app sets `isIdleTimerDisabled` while
  working, but keep the phone awake for manual tools.

## Using the app

Main menu: **Receive** (the real thing), Core demo/bench, frame viewer,
Settings, a **Handheld** toggle (experimental, never field-proven), a
**v2 glyphs / v3.1 DMT** wire picker, a px picker (8/12/16/30) + nc picker
(v3.1), and Save/Load/Factory model buttons that follow the pickers.

Rig rules learned the hard way: the phone must be **absolutely stable**
(tripod/propped — micro-shakes wreck training and recognition), and the **4K
capture toggle should be ON** (it greatly helps recognition even at 16 px
tiles; mandatory below 12 px).

A session: tap Receive (self-tests run once), aim at the transmitter's TRAIN
QR. The app locks focus/exposure on the field, auto-calibrates by decoding
seeded truth, trains during the training phase (v2 only), arms at the START QR,
decodes the stream, and on END fountain-rebuilds, verifies md5 and — if the QR
carried `NAME=` and nothing is missing — saves the file to
Files → On My iPhone → QR File Transmit → QR Transmit.

Per-run logs: `Documents/QR-receiver-log-<timestamp>.txt` (visible in Files;
pull headlessly with `xcrun devicectl device copy from --domain-type
appDataContainer --domain-identifier com.example.QRFileTransmit
--source Documents/log.txt --destination /tmp/log.txt`).

## The neural decoders (choices explained)

- **Glyph net** (`Net.swift`, ~90 k params, forward AND backward hand-rolled on
  Accelerate): 16×16×3 crop → 16-shape + 4-color heads. One net per "model
  key": native 16, native 12, or canonical `NNc16` keys (812/1216/3016-style)
  where any px is SAMPLED at 16×16 through the homography — this is why one
  architecture serves 8–30 px and why small-px models warm-start from the 16 px
  weights. Trains ON DEVICE (session training + runtime tuning during long
  transfers).
- **ANE path**: the same net as a CoreML mlprogram (`TileNet2/12`), weights
  patched into the compiled model per session, **re-minted on device for any
  grid size** (batch-dim rewrite; grids >16383 tiles predict in chunks).
  ~16 ms/frame vs ~50+ CPU. First frame is cross-checked vs the CPU net.
- **Demapper** (`Demap.swift`, ~1 M params, forward-only, CPU): v3.1 payload
  cells, 20×20 crop → 3×NC PAM-2 bits. Pretrained on PC (`training/`), shipped
  in `Assets/demap16_nc{8,12}.bin`. NOT yet on the ANE → keep v3.1 fps ≤5.
  **v3.1 has not had a live end-to-end run yet** (see docs/STATUS.md).

## Models on the device

`Documents/` holds per-key model sets: `active_model_<key>.bin` (what a session
warm-starts from; auto-saved when a session trains below 5 % err),
`model_fork_<key>.bin` (manual snapshots), `factory_model_<key>.bin`
(user-overridable factory), `hh_*` (handheld namespace), `v3_*_nc<n>`
(demapper namespace). You can copy models in/out via the Files app — same raw
fp32 format as `models/`.

`tuned/` in this folder holds the camera-tuned models pulled from the dev
phone (`active_model_16/816/1216/3016.bin`) — tuned to ONE rig's camera and
lighting; treat as warm starts, not universal weights.
