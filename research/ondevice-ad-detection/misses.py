import json,sys,adtest,rescore
run=sys.argv[1]; asr=sys.argv[2] if len(sys.argv)>2 else 'moonshine'
data=rescore.load_run(run,asr)
for ep,(lines,is_ad,blocks) in data.items():
    dur,gt=adtest.load_gt(ep)
    r=json.load(open(f"results/{run}/{ep}.json"))
    outs={e['block']:e['out'] for e in r['log'] if 'block' in e}
    for k,b in enumerate(blocks):
        s,e=lines[b[0]]['t'],lines[b[-1]]['e']
        inad=sum(adtest.overlap((s,e),(g0,g1)) for g0,g1,t in gt if t=='ad')/(e-s+1e-9)
        if (inad>0.5 and not is_ad[b[0]]) or (inad<0.1 and is_ad[b[0]] and not any(adtest.overlap((s,e),(g0,g1))>0 for g0,g1,t in gt)):
            kind='MISS' if inad>0.5 else 'FALSE'
            print(kind,ep,adtest.fmt(s),repr(outs.get(k))[:30],'|',' '.join(lines[i]['text'] for i in b)[:170])
