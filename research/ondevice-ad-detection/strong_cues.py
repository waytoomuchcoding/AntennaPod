import json, re, sys
cue = re.compile(r"\.com|\.ai\b|dot com|slash|promo code|sponsor|brought to you|support for|support comes|terms apply|free trial|"
                 r"try it|wherever you get|this message comes|learn more at|visit |percent off|% off|sign up|app store|"
                 r"terms and conditions|restrictions apply|listen to .* on|available now|in theaters", re.I)
lines = [json.loads(l) for l in open(sys.argv[1])]
prev = -999
for s in lines:
    if cue.search(s["text"]):
        t = int(s["t"])
        gap = "" if t - prev < 90 else "\n"
        print(f"{gap}{t // 60:02d}:{t % 60:02d} {s['text'][:110]}")
        prev = t
