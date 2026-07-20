# commands.md — how to run everything (machines & roles: docs/machines.md)

## Settings
- config.py = single source of settings, SAME file on all machines. Any bare key=value
  on a command line overrides for that run (case-insensitive): fps=3 px=20 nsym=32
  train_every=20. --flags and positional args are separate and script-specific.
- after editing anything on the PC, ship it everywhere:
  python deploy.py [model ...]     # code -> surface + mac; models -> surface

## Data mode (transmit a payload)
    # mac (from the tx tmux session; field auto-scales to the screen):
    ../venv/bin/python transmit.py         # synthetic payload -> sent_payload.bin
    ../venv/bin/python transmit.py somefile.bin    # transmit a real file
    # PC window rig (stop output.py first; cap the field to the camera view):
    python transmit.py monitor=1 w_use=1800 h_use=1030 top_pad=0
    # console: ENTER = transmit, t+ENTER = training mode first; every stage gates on ENTER

    # Surface (own terminal; ssh needs -tt or the camera hangs in warmup):
    ssh -tt dev@<laptop-ip> "cd C:\Users\dev\Desktop\QR_data_transmit\shapes_n_colors && python receiver.py --save rx_frames"
    # no --model -> loads pc_L<PX> (PX from the START/TRAIN QR)
    # --model rx_L16     use a previously tuned model
    # --tune-name foo    name for the model saved after training mode (default rx_L<PX>)
    # --save <dir>       also write every kept frame (dir is WIPED each run)
- receiver prints received_payload.bin + md5; transmit printed the sent md5 — compare.
- fountain: transmit appends REPAIR PERCENT of sources (config, default 3%, floor 8)
  as XOR-combo frames; the receiver rebuilds any lost frames from them (needs ~lost+2).
  repair=6 for high-fps runs, repair=0 to disable. Losses beyond that: run it again.
- native tiles (see config.NATIVE_TILES): native_tiles=1 on any command switches the
  whole pipeline (receiver, train_sub, reharvest, label_rx) to native-resolution crops
  (~2x faster NN at r16). Needs models TRAINED native (pc_L16n etc) — sizes are
  incompatible with canonical models; transmit.py is unaffected either way.
- training mode: t+ENTER on PC -> receiver shows "TRAIN: frames=N err=X%"; ENTER when
  happy -> START QR -> receiver saves tuned model and receives with it immediately.
  Chunk pace: train_every=4 fastest descent, 12-20 for long/cool sessions.

## Legacy training mode (self-training client, SNC QR clock)
    # server: output.py [dwell_s] [start_n] [ladder] — grid auto-computed from the
    # screen and announced in the QR; only the rung is chosen
    python output.py 0.6 0 24,20,16
    # Surface client: client.py <model> [--init parent] [--ladder ...] [--save-frames N]
    python client.py surf_L24-16 --ladder 24,20,16 --save-frames 2
    # NO --init = resume by name; --init <parent> = warm-start RESET from parent
    # frames archive: frames/L<ladder>/ (ladder-bound!), 10 GB cap (FCAP in client.py)

    # mac-rig roles: legacy soak = pretraining + labeled frame archives (QR = labels
    # at any model quality). TRANSFERS use data-mode training instead — the camera's
    # per-session AE/AWB state differs run to run (see decode_practical.md), so tune
    # in-session:
    #   surface: python receiver.py --save rx_frames --model loop_L16_best --tune-name rx_L16
    #   mac:     ../venv/bin/python transmit.py   then t+ENTER, watch err, ENTER
    # first-time rig bootstrap: client.py <name> --init loop_L16n --ladder 16

## PC model training (offline, GPU)
    # 1. pull a frame archive from the Surface:
    scp -r "dev@<laptop-ip>:C:/Users/dev/Desktop/QR_data_transmit/shapes_n_colors/frames/L24-20-16-12-8" pc_subtraining/frames_L24-8/
    # 2. rebuild labeled shards (ALWAYS decode with the capture ladder):
    cd pc_subtraining && python reharvest.py [frames_dir] [set_name]
    # 3. train any subset model from shards (~1.5 min on GPU):
    python train_sub.py pc_L16 --ladder 16 --init surf_L24-8 --set night --steps 30000
    # outputs: models/<MVER>/<name>.npz + pc_subtraining/<name>/{log.csv,err.png,fails.json,meta.json}
    # (MVER in config.py: V2.2 = mac-rig epoch; parents/--init also resolve from MPREV)
    # 4. glyph error chart (compare mature models only, same rungs):
    python glyph_chart.py pc_L24-16/fails.json 16 ../docs/glyph_errors_V2.1.png "title"
    # 5. fine-tune from CAPTURED DATA-MODE frames (payload known -> free labels):
    python label_rx.py            # rx frames + seed -> data_day_r16.npz  (check wire
    python train_sub.py pc_L16d --ladder 16 --init pc_L16 --set day       # format inside!)
    # 6. ship any model to the Surface:
    scp models/V2.2/pc_L16d.npz dev@<laptop-ip>:C:/Users/dev/Desktop/QR_data_transmit/shapes_n_colors/models/V2.2/
    # (or just: python deploy.py pc_L16d)

## Surface fine-tune (no PC needed)
- data-mode: transmit.py training mode (above) — 1-2 min per new lighting condition.
- legacy: client.py record->train cycles (slow but fully autonomous).

## Diagnostics
    python plot.py <model>                          # error-over-time from a _log.csv
    ssh ... "python test\bench_rx.py rx_frames pc_L16d"     # per-stage timing over saved frames (no camera)
    ssh -tt ... "python test\bench_frame.py pc_L16 16 6"    # per-stage LIVE timing (legacy path, needs camera)
    ssh -tt ... "python test\diag_gate.py 24,20,16"         # per-frame QR/quad/corr check
    # payload diff after a run (PC):
    python -c "import numpy as np; s=np.fromfile('sent_payload.bin',np.uint8); r=np.fromfile('received_payload.bin',np.uint8); print((np.unpackbits(s)!=np.unpackbits(r)).mean())"

## Gotchas (the expensive ones)
- ssh camera scripts: ALWAYS -tt (MSMF cam.read hangs forever without a tty)
- kill remote scripts by PID (wmic ... CommandLine like '%script%'), never taskkill python.exe
- scp remote paths use forward slashes; PowerShell for $_, bash mangles it
- transmit and output.py both claim monitor 1 — one at a time
- Surface sleeps on AC unless: powercfg /change standby-timeout-ac 0
