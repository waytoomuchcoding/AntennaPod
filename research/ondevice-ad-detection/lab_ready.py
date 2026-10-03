"""Hand out labelling batches: episodes with lab/in/<ep>.txt, no lab/out/<ep>.json and not yet assigned.
usage: ./venv/bin/python lab_ready.py [batch_size] [max_batches]   -> prints one comma-separated batch per line and
       records them in lab/assigned.txt;   ./venv/bin/python lab_ready.py --status   -> counts
       ./venv/bin/python lab_ready.py --release   -> forget assignments whose output never appeared (after a crash)"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
A = f"{ROOT}/lab/assigned.txt"
assigned = set(open(A).read().split()) if os.path.exists(A) else set()
inp = {f[:-4] for f in os.listdir(f"{ROOT}/lab/in") if f.startswith("b_")}
out = {f[:-5] for f in os.listdir(f"{ROOT}/lab/out") if f.startswith("b_")}
if "--status" in sys.argv:
    print(f"transcribed {len(inp)}, labelled {len(out)}, assigned-not-done {len(assigned - out)}, "
          f"waiting {len(inp - out - assigned)}")
elif "--release" in sys.argv:
    open(A, "w").write("\n".join(sorted(assigned & out)) + "\n")
    print("released", len(assigned - out))
else:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    wait = sorted(inp - out - assigned)
    batches = [wait[i:i + size] for i in range(0, len(wait) - size + 1, size)][:nmax]
    with open(A, "a") as f:
        for b in batches:
            f.write("\n".join(b) + "\n")
            print(",".join(b))
