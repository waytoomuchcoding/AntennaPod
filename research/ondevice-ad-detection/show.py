import json,sys
f=sys.argv[1]; a=float(sys.argv[2]) if len(sys.argv)>2 else 0; b=float(sys.argv[3]) if len(sys.argv)>3 else 1e9
w=int(sys.argv[4]) if len(sys.argv)>4 else 400
for l in open(f):
    s=json.loads(l)
    if a<=s['t']<=b:
        t=int(s['t']); print(f"{t//60:02d}:{t%60:02d} {s['text'][:w]}")
