# transmitter/ — Python fullscreen transmitter

The original transmitter (`transmit.py` + `config.py`, `glyphs.py`, `rs.py`).
Minimal and keyboard-driven — but it's the path behind every speed record.

## Setup (venv)

Python 3.9–3.12. Four third-party packages (numpy, opencv-python, pygame,
segno); everything else is the standard library. `config.py` is pure settings.

```bash
python -m venv venv
# activate: Linux/macOS: source venv/bin/activate  |  Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Known-good tested set: Python 3.10, numpy 2.2.6, opencv-python-headless 4.11.0.86,
pygame 2.6.1, segno 1.6.6 (the `requirements.txt` floors also install cleanly on
Python 3.9 with earlier 1.x numpy, the original mac rig). OpenCV is *headless* and
pinned `<4.12` on purpose: 4.12+/5.x bundle their own SDL2, which collides with
pygame's on macOS (duplicate-class objc warnings → possible crashes) and the
transmitter never uses cv2's GUI functions anyway.

## Run

```bash
python transmit.py                      # defaults from config.py
python transmit.py px=12 fps=30 nsym=32              # the 126.6 KB/s record recipe
python transmit.py px=16 fps=10 nsym=64 repair=6 file=/path/to/payload.pdf
```

Flow: idle field → shows the TRAIN QR (receiver locks + calibrates) → ENTER
starts training frames → shows the START QR (receiver arms) → ENTER streams the
data frames → END QR holds. ESC aborts.

Key settings (command line `key=value` overrides `config.py`): `px` tile size,
`fps` data rate, `tfps` training rate, `nsym` RS parity bytes per 255-block
(64 = robust, 32 = fast channel), `repair` fountain percentage, `file=` real
payload (otherwise a seeded synthetic payload; md5 printed either way).

Traps already handled in code, don't undo them: pygame vsync only engages with
`SCALED|FULLSCREEN` (tick-paced fallback caused metronomic frame losses); the
mouse cursor is hidden (a cursor parked on the ring broke geometry); frame 0
dwells extra ticks (ISP transition ghosts); never flush with a black screen
(AE swing kills the first seconds).

The web transmitter (`../web/`) is a bit-exact port with a friendlier stage UI
and persisted settings — prefer it unless you need the mac/py rig automation.
