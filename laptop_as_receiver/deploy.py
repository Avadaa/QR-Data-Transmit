# deploy.py — push the link code from the PC to both machines in one go.
# Usage: deploy.py [model ...]      (named models also ship to the Surface's models/<MVER>)
# Roles/paths: docs/machines.md. config.py goes EVERYWHERE — it is the wire format.
# A machine that is off is reported and skipped, the rest still get served.
import os, subprocess, sys
import config

HERE = os.path.dirname(os.path.abspath(__file__))
SURF = "dev@<laptop-ip>:C:/Users/dev/Desktop/QR_data_transmit/shapes_n_colors/"
MAC = "dev@<mac-ip>:/Users/dev/windows/QR_data_transmit/shapes_n_colors/"
SHARED = ["config.py", "glyphs.py", "rs.py"]

def scp(files, dest):
    r = subprocess.run(["scp", "-o", "ConnectTimeout=5"] + files + [dest], cwd=HERE)
    return r.returncode == 0

surf_ok = scp(SHARED + ["receiver.py", "client.py", "snap0.py", "record.py"], SURF)
print("surface: code ok" if surf_ok else "surface: OFFLINE, skipped")
mac_ok = scp(SHARED + ["transmit.py", "output.py"], MAC)
print("mac: code ok" if mac_ok else "mac: OFFLINE, skipped")
for m in sys.argv[1:]:
    if not surf_ok:
        print(f"surface: {m}.npz skipped (offline)")
        continue
    ok = scp([os.path.join("models", config.MVER, m + ".npz")], SURF + f"models/{config.MVER}/")
    print(f"surface: {m}.npz " + ("ok" if ok else "FAILED"))
