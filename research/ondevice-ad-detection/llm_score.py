"""Idea 8 (README 15.3): per-line ad probability from an LLM without generating text. The transcript around a block
of target lines is read once (llama-server keeps it in the KV cache: cache_prompt), then one short question per
target line is appended; the answer is a single token and the score is P(Yes) / (P(Yes) + P(No)).
Blocks of 20 target lines with 10 lines of context on each side (~1,200 tokens). Writes embp_llm/<name>/<ep>.json.
  llama.cpp/build/bin/llama-server -m models/gemma-4-E2B-it-Q4_K_M.gguf -ngl 99 -t 2 -c 4096 -np 1 --port 8094
usage: GT_DIR=gt_v2 ./venv/bin/python llm_score.py <name> <port> [labelled|unseen|all]
"""
import json, math, os, sys, time, urllib.request
import adtest
from emb_loo import eps
import eval_all

name, port = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else "all"
BLOCK, CTX = 20, 10
SYSTEM = ("You detect advertisements in podcast transcripts. An advertisement is a sponsor message, commercial, or a "
          "promo or trailer for a product, service, app, another podcast, a TV show or a movie. The podcast's own "
          "content (story, reporting, interviews, discussion, intro, credits) is not an advertisement.")
QUESTION = ("Line: \"{line}\"\n\nIs this line part of an advertisement, sponsor message or promo/trailer for "
            "another show or movie (not the podcast's own content)? Answer Yes or No.")


def post(path, body):
    req = urllib.request.Request(f"http://localhost:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def p_yes(prompt):
    r = post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": True})
    lp = {t["token"].strip().lower(): t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
    y, n = lp.get("yes", -30.0), lp.get("no", -30.0)
    return 1 / (1 + math.exp(n - y)), r.get("timings", {})


eps_list = (list(eps) if which in ("labelled", "all") else []) + (eval_all.UNSEEN if which in ("unseen", "all") else [])
meta = adtest.EPISODES
tot_t = tot_lines = tot_prompt = 0
for e in eps_list:
    path = f"embp_llm/{name}/{e}.json"
    if os.path.exists(path):
        continue
    lines = [l["text"] for l in adtest.load_lines("moonshine", e)]
    out, t0 = [], time.time()
    for s in range(0, len(lines), BLOCK):
        ctx = "\n".join(lines[max(0, s - CTX):s + BLOCK + CTX])
        head = (f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\nPodcast: {meta[e]['podcast']}, episode: {meta[e]['title']}"
                f"\n\nPart of the transcript:\n{ctx}\n\n")
        for line in lines[s:s + BLOCK]:
            p, tm = p_yes(head + QUESTION.format(line=line) + "<turn|>\n<|turn>model\n")
            out.append(round(p, 4)); tot_prompt += tm.get("prompt_n", 0)
    dt = time.time() - t0; tot_t += dt; tot_lines += len(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    dur = json.load(open(f"gt_auto/{e}.json"))["duration"] if e.startswith("x_") else eps[e]["dur"]
    print(f"  {name} {e}: {len(lines)} lines in {dt:.0f}s = {dt / dur * 3600:.0f}s per audio hour", flush=True)
print(f"done {name}: {tot_lines} lines in {tot_t:.0f}s, {tot_prompt} prompt tokens evaluated (after cache reuse)")
