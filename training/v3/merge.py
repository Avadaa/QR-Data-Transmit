# merge.py — combine parallel worker outputs (data/<tag>w*/) into data/<tag>/, keeping
# the best-NCC shot per idx (workers overlap 0.6 s at slice boundaries).
# Run:  python merge.py [tag=4k]
import glob, os, shutil, sys
DIR = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "4k"
best = {}
for mf in sorted(glob.glob(f"{DIR}/data/{tag}w*/meta.csv")):
    for l in open(mf).read().splitlines()[1:]:
        r = l.split(",")
        seg, idx, ncc = int(r[0]), int(r[1]), float(r[2])
        if idx not in best or ncc > best[idx][0]:
            best[idx] = (ncc, os.path.dirname(mf), seg, l)
os.makedirs(f"{DIR}/data/{tag}", exist_ok=True)
out = open(f"{DIR}/data/{tag}/meta.csv", "w")
out.write("seg,idx,ncc,time,vframe,H\n")
for idx in sorted(best):
    ncc, d, seg, line = best[idx]
    shutil.copy(f"{d}/s{seg}f{idx:05d}.npy", f"{DIR}/data/{tag}/s0f{idx:05d}.npy")
    out.write("0," + line.split(",", 1)[1] + "\n")
out.close()
print(f"merged {len(best)} unique idx -> data/{tag}")
