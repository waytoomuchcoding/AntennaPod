"""Before committing: move the full result files (whose "log" quotes transcript text) to logs/results_full/
and keep a copy without the log in results/. usage: python3 strip_logs.py results/<run> ..."""
import json, os, shutil, sys

for d in sys.argv[1:]:
    run = os.path.basename(d.rstrip("/"))
    os.makedirs(f"logs/results_full/{run}", exist_ok=True)
    for f in os.listdir(d):
        p = os.path.join(d, f)
        r = json.load(open(p))
        if "log" not in r:
            continue
        shutil.copy(p, f"logs/results_full/{run}/{f}")
        r.pop("log")
        json.dump(r, open(p, "w"), indent=1)
