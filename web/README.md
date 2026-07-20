# web/ — browser transmitter

One self-contained page (`index.html` + bundled MIT `qrcode.js`). No server, no
network: open the file, fullscreen the stage, go. Verified **bit-exact** against
the Python transmitter (PCG64 / CRC32 / RS / md5 all match).

Usage: open `index.html`, settings persist in localStorage. Stage buttons on
top (idle → training qr → start training → arm qr → start live) or Enter to
advance, Esc to abort. Pick a real file to enable file delivery (`NAME=` in the
arm QR → the phone saves it into the Files app on a complete transfer).

Defaults = the current validated wire: **v3.1 DMT (PAM-2, A=40, NC=8), 16 px,
1024 px field, train 15 fps, data 5 fps, nsym 64**. For the PROVEN v2 path,
switch wire to "v2 glyphs" (px 12 @ 20–30 fps + nsym 32 is the record regime —
requires the receiver's 4K capture toggle).

Details that matter:
- Pacing is timestamp-scheduled on requestAnimationFrame with auto-measured
  panel Hz — 60 and 120 Hz panels both pace exactly; the status line shows
  "late/drift" counters (drift should hover 0–1).
- The field is a 1:1 LOGICAL-pixel square on black with a white ring — minimum
  panel light. "HiDPI 1:1" renders in device pixels on retina panels.
- v3 rendering costs 60–120 ms/frame of JS IDCT at NC=20 (less at NC=8) —
  effective fps caps below the picker value; the phone paces by content, so
  this only lowers throughput, never correctness.
- Keep the browser window undisturbed while streaming (wakeLock is requested;
  the cursor auto-hides over the stage).
