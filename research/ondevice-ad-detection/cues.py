import json,sys,re
cue=re.compile(r"\.com|slash|promo code|sponsor|brought to you|support for|terms apply|try it|free trial|download|subscribe|listen to|available wherever|wherever you get|sign up|visit |offer|percent off|% off|order now|app store|this is an|podcast|peacock|iheart|nbc news|msnbc|dateline|commercial|after the break|we'll be right back|stay with us|welcome back|premium|insurance|shop|deal", re.I)
f=sys.argv[1]
for l in open(f):
    s=json.loads(l)
    if cue.search(s['text']):
        t=int(s['t']); print(f"{t//60:02d}:{t%60:02d} {s['text'][:150]}")
