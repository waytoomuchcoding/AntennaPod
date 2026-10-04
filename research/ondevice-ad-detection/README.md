# On-device podcast ad detection (research log)

This folder holds a research harness and the full record of one working session (2026-09-30 to 2026-10-01)
spent on replacing the fork's cloud ad detection (Gemini 3.5 Flash-Lite, see `AdSegmentIndexer.java`) with
models that run offline on the phone. Nothing here is wired into the app yet. Another agent should be able to
continue from this file alone: it lists what was tried, every number, what failed and why, and what to try next.

Target phones: the owner's **Pixel 10 Pro Fold** and a **Pixel 11**.

---

## 1. Summary

- **It works.** Moonshine speech recognition plus a small Gemma model in "copy mode" finds 54 of 55 ad breaks in
  13 labelled episodes (11.4 h of audio) with Gemma 4 E4B, F1 0.955 overall, held-out F1 0.967.
- **Best pipeline today:** Moonshine ASR → ~1,200-token transcript windows → the LLM copies one ad per call
  word for word → the copy is fuzzy-matched back to the transcript, replaced by `[advertisement removed]`, and the
  model is asked again (up to 6 rounds) → one short "is this an ad?" verify call per detected break → optional
  TF-IDF classifier veto. Per hour of audio: about 40 LLM calls, 44K input tokens, 1.9K output tokens.
- **Stand-in models.** Gemini Nano weights are not public. Pixel 10 runs Nano v3 (Gemma 3n architecture);
  Pixel 11 runs Nano v4 (built on Gemma 4 E2B "Fast" / E4B "Full"). We tested the open Gemma equivalents.
- **Recommendation:** bundle Moonshine (sherpa-onnx) and Gemma 4 E2B or E4B (LiteRT-LM) in the app rather than
  using AICore/ML Kit. The ML Kit Prompt API is foreground-only, quota-limited and capped at ~4K input /
  ~256 output tokens; the ML Kit speech API only accepts audio at real-time speed. The pipeline already fits
  the Prompt API limits, so switching later stays possible.
- **Things that did not help:** line-number ranges, 30 s yes/no blocks, sampling + voting, shifted windows,
  per-copy verification, longer context, focus windows with surrounding context, thinking mode, audio cues
  (speaker labels, pauses, music) in the prompt, and a more detailed system message. Details in section 7.

- **Update 2026-10-02 (section 13, new labels `gt_v2`, leave-one-show-out):** a 300M EmbeddingGemma line
  classifier + HMM finds all 54 breaks with 0 false alarms and **no LLM call** (all 13 eps F1 0.935, held-out
  0.920), beating Gemma 4 E2B copy + verify (0.917). Best overall: **E4B copy6 + EmbeddingGemma veto, held-out
  0.966, dev 0.969, 0 false alarms**. Repetition matching of already-confirmed ads is ~98% precise and covers
  ~25% of ad time even with 12 episodes. Distilling E4B labels from 35 other shows did not help.

### Headline results (copy + verify, Moonshine transcripts)

| Model (phone it stands in for) | Dev F1 (7 eps) | Held-out F1 (6 eps) | All 13: P / R / F1 [95% CI] | Breaks found | False alarms |
|---|---|---|---|---|---|
| Gemma 4 E4B (Pixel 11, Nano v4 Full) | 0.947 | 0.967 | 0.954 / 0.957 / **0.955** [0.93-0.97] | 54/55 | 2 |
| Gemma 4 E2B (Pixel 11, Nano v4 Fast) | 0.939 | 0.890 | 0.886 / 0.951 / **0.917** [0.86-0.96] | 54/55 | 4 |
| Gemma 3n E4B (Pixel 10, Nano v3) | 0.926 | 0.885 | 0.892 / 0.923 / **0.907** [0.87-0.94] | 51/55 | 6 |
| Gemma 3n E2B (Pixel 10, Nano v3) | 0.894 | 0.898 | 0.935 / 0.860 / **0.896** [0.83-0.95] | 50/55 | 2 |

With the classifier veto (leave-one-episode-out): E2B dev 0.939 → 0.944, held-out 0.890 → 0.910; E4B held-out
0.967 → 0.969 with 0 false alarms. The veto never cost a break.

Caveat: the prompts and strategies were developed on the 7 dev episodes. The 6 held-out episodes were added
later and never used for tuning, except where noted in section 7.12.

---

## 2. What the phones offer (research, 2026-09-30)

| Item | Finding |
|---|---|
| Access path | ML Kit GenAI APIs on top of AICore (`com.google.mlkit:genai-prompt:1.0.0-beta4`). No allowlist or Play distribution needed; sideloaded debug builds work. |
| Nano versions | nano-v3 on Pixel 9/10 (same architecture as Gemma 3n); nano-v4 on Pixel 11 and Galaxy Z Fold8/Flip8 (Gemma 4 E2B "Fast" and E4B "Full"). |
| Prompt API limits | ~4,000 input tokens, ~256 output tokens per call (docs disagree slightly; one page says 4,096 total / 1,024 per prompt). |
| Restrictions | Foreground only (`BACKGROUND_USE_BLOCKED`, even from a foreground service); per-app rate quota (`BUSY`) and daily battery quota (`PER_APP_BATTERY_USE_QUOTA_EXCEEDED`); not available with an unlocked bootloader (`FEATURE_NOT_FOUND`). |
| Thinking mode | `enableThinking = true` on the request, reasoning returned in `thoughtProcess`. Officially Nano v4+, but "can be tested through the Developer Preview program on any AICore-enabled device", so the Pixel 10 Pro Fold can likely use it through the preview. Unknown whether thinking tokens count against the 256-token output cap. |
| Speech recognition | `com.google.mlkit:genai-speech-recognition:1.0.0-alpha1`, Advanced mode (Gemini) on Pixel 10/11. File input must be raw 16 kHz PCM fed at real-time rate: a 90-minute episode takes 90 minutes. Unusable for batch processing. |
| Bundling instead | LiteRT-LM runs the same Gemma 4 files on CPU/GPU/NPU with no foreground rule or quota. Google publishes `gemma-4-E2B-it.litertlm` (2.6 GB, CPU), a GPU file, and builds compiled for Tensor G5 (Pixel 10) and G6 (Pixel 11) NPUs. Published Gemma 4 E2B speed on a Galaxy S26 Ultra: CPU 557 prefill / 47 decode tok/s, GPU 3,808 / 52 tok/s. |

Sources: developers.google.com/ml-kit/genai, .../prompt/android/get-started, .../prompt/android/thinking-mode,
.../speech-recognition/android, huggingface.co/litert-community/gemma-4-E2B-it-litert-lm,
androidauthority.com (Gemini Nano 4 benchmarks, bootloader article).

---

## 3. Environment and how to run

- Developed in an arm64 Lima VM (`vz` type: 6 vCPU, ~6 GB RAM, 4 GB swap, **no GPU**). Everything ran on CPU.
  GPU acceleration in Lima needs a new instance with `vmType: krunkit` (Lima 2.0+, Fedora guest with Mesa
  Venus); the VM type cannot be changed on an existing instance. LiteRT-LM's Linux library contains a
  Dawn/WebGPU backend that may work on Venus Vulkan (untested); llama.cpp can be built with `-DGGML_VULKAN=ON`.
- `./setup.sh` installs packages, downloads all models, builds llama.cpp, fetches the audio and transcribes it
  with Moonshine.
- **Audio, transcripts and prompt dumps are not committed** (copyrighted podcast content, and large). They are
  regenerated by `fetch_audio.py` and `transcribe.py`. On the original VM they are still in
  `~/Projects/ondevice-adtest/` (`wav/`, `tx/`, `bench/dump_*.jsonl`, `problem_cases.md`).
- **Dynamic ad insertion warning:** most feeds insert ads per download. A fresh download may carry different
  ads than the labelled audio from 2026-09-30, so `gt/*.json` may not match. Re-check labels (section 4) on any
  re-download, or keep the original WAVs.

Typical commands:

```bash
# full run of the best pipeline on the dev split, then the verify pass
./venv/bin/python adtest.py --model models/gemma-4-E2B-it.litertlm --split dev --strategy quotes \
    --arg mode=copy --arg iters=6 --arg min_words=5 --tag _copy6
./venv/bin/python verify.py models/gemma-4-E2B-it.litertlm gemma-4-E2B-it.litertlm__moonshine__quotes_copy6
python3 scoreboard.py            # every run with P/R/F1, 95% bootstrap CI, breaks, false alarms, tokens per hour

# Gemma 3n through llama.cpp (queue2.sh starts llama-server with --cache-ram 0, see gotchas)
echo 'models/gemma-3n-E2B-it-Q4_0.gguf|./venv/bin/python adtest.py --model "{spec}" --split test --strategy quotes --arg mode=copy --arg iters=6 --arg min_words=5 --tag _copy6_test' > q.txt
./queue2.sh q.txt

# fast iteration on 13 problem windows (2-5 min per variant on E2B)
./venv/bin/python bench.py models/gemma-4-E2B-it.litertlm my_variant budget=1200 dump=1
BENCH_ONLY="daily1 14:00" ./venv/bin/python bench.py models/gemma-4-E2B-it.litertlm one_case marker=0 dump=1
```

`--model` takes a `.litertlm` path (LiteRT-LM, in process) or `http://host:port#name` (llama-server OpenAI API).
`llm.py` enforces the Prompt API budget (<4,000 input tokens, ≤256 output tokens) and uses greedy decoding
(`top_k=1`, temperature 0, seed 1). `--temp/--seed` enable sampling; bench `think=N` enables thinking.

### File map

| File | Purpose |
|---|---|
| `adtest.py` | Core: line building, strategies (`ranges`, `blocks`, `quotes` incl. copy mode), interval conversion, scoring, CLI. |
| `llm.py` | LiteRT-LM and llama-server wrappers, token budget, sampling, thinking, `DUMP_PROMPTS` logging. |
| `transcribe.py` | sherpa-onnx ASR (moonshine, parakeet, whisper-tiny.en, whisper-base.en) with Silero VAD. |
| `verify.py` | One verify call per predicted interval; writes `<run>_verified`. |
| `bench.py` | Case bench: 13 windows around known problem spots and tricky true ads. |
| `clf*.py` | TF-IDF + logistic-regression line classifier: `clf.py` (train dev, test held-out), `clf_loo.py` (leave-one-episode-out + veto), `clf_probs.py` (per-line scores into `clfp/`), `clf_explain.py` (weights, worked example). |
| `audio_feats.py`, `diarize.py`, `postcues.py` | YAMNet music/speech/loudness per second (`feats/`), sherpa-onnx diarization (`diar/`), cue-based post-processing. |
| `vote.py`, `combine.py`, `rescore.py` | Offline voting, run combination, post-processing grid search. |
| `gt_check.py` | Second labeller via Gemini (cached results in `gt/check_*.json`; reads the key from `~/.config/antennapod-test/gemini-api-key`). |
| `scoreboard.py` | Re-scores every saved run against current labels. Output from the session: `results/scoreboard.txt`. |
| `gt/<ep>.json` | Ground-truth labels. `episodes.json` has titles, feeds and the dev/test split. |
| `results/<run>/<ep>.json` | Predictions, scores and token stats per run (model logs that quote transcript text removed). |
| `bench/results.jsonl` | Every bench run with per-case results. |
| `classifier-guide.html` | Explainer for the classifier (also published as a private claude.ai artifact). |
| `cues.py`, `show.py`, `misses.py`, `blockdev.py` | Labelling and error-analysis helpers; `blockdev.py` is the earlier block-prompt dev set. |

---

## 4. Data and labelling

13 episodes, 11.42 h of audio, 79 min of ads, 55 ad breaks. Dev = prompts were tuned on these. Test = added
later, never tuned on.

| Episode | Split | Show | Length | Ad time | Notes |
|---|---|---|---|---|---|
| daily1 | dev | The Daily | 27.8 min | 4.0 | stacked mid-roll; narrative Rinse ad at the end |
| daily2 | dev | The Daily | 31.0 | 4.2 | reporting quotes Kalshi TV ads (hard negative) |
| dateline1 | dev | Dateline NBC | 88.9 | 8.0 | 5 identical-pattern breaks |
| dateline2 | dev | Dateline NBC | 45.4 | 4.7 | |
| sysk1 | dev | Stuff You Should Know | 57.4 | 8.2 | hosts promote their own new podcast (self-promo) |
| crimejunkie1 | dev | Crime Junkie | 55.9 | 6.3 | host-read film promos (one break initially missed in labelling) |
| freak1 | dev | Freakonomics Radio | 59.8 | 9.6 | |
| planetmoney1 | test | Planet Money | 35.3 | 3.0 | NPR "This message comes from…" |
| conan1 | test | Conan O'Brien Needs A Friend | 73.7 | 8.6 | conversational host-read ads |
| morbid1 | test | Morbid | 83.2 | 7.2 | host chat about shows; long silent outro |
| huberman1 | test | Huberman Lab | 38.8 | 4.0 | long host-read sponsor reads |
| dateline3 | test | Dateline NBC | 45.8 | 5.2 | |
| dateline4 | test | Dateline NBC | 42.3 | 6.2 | Talking Dateline format |

Label format (`gt/<ep>.json`): `{"duration": s, "segments": [{"s": "mm:ss", "e": "mm:ss", "type": "ad"|"self_promo"}]}`.
Self-promo (show plugs, credits asks) is neutral in scoring: neither required nor penalised.

How labels were made: cue-phrase scan (`cues.py`) plus reading the transcript around every break, by hand.
Then a second labeller: Gemini 3.5 Flash (dev) and 3.5 Flash-Lite (test) read each full transcript once
(`gt_check.py`; about 13 successful requests in total, cached). Both agreed on every ad break after one fix:
the Crime Junkie 12:59–15:06 break (Carrie/Verity promos) had been missed by hand, and the on-device model was
the one that found it. The owner asked to keep Gemini API use minimal (free quota resets at midnight).

### Metrics

- Seconds-based precision/recall/F1 for ad time (self-promo time neutral).
- **Breaks found:** a labelled break counts as found when ≥50% of it is covered.
- **False alarms:** predicted intervals with <50% overlap with any ad or self-promo label.
- 95% CI: bootstrap over episodes (2,000 resamples).
- Predicted line flags become intervals by merging gaps ≤20 s and dropping intervals <10 s.

---

## 5. Speech recognition

All run through sherpa-onnx (Android-ready) with Silero VAD (speech segments ≤20 s; whisper ≤28 s) and merged
into lines of ≤25 words that never span a pause >1.5 s.

| Model | Size | Speed on this VM (avg, best) | F1 downstream (E2B copy+verify, dev) | False alarms |
|---|---|---|---|---|
| **Moonshine base int8** | 275 MB | 23×, 34× real time | **0.939** | 2 |
| Parakeet TDT 0.6B v2 int8 | 631 MB | 16×, 24× | 0.940 | 0 |
| Whisper base.en int8 | 433 MB | 8×, 10× | 0.906 | 3 |
| Whisper tiny.en int8 | 245 MB | 14×, 18× | 0.873 | 6 |

Moonshine is the pick (same accuracy as Parakeet, half the size, fastest). Parakeet dropped the middle of one
fast ad read in a spot check. Speeds were measured while LLM jobs shared the CPU.

---

## 6. The best pipeline in detail (copy mode)

1. Lines are concatenated into a word stream. Windows of ~1,200 tokens (≈7–8 min of speech), 30% overlap.
2. Each window gets the system message and the copy prompt below. The model copies one ad.
3. The first 8 and last 8 copied words are fuzzy-matched to the window (`find_quote`: in-order word hits,
   up to 2 skipped words, ≥50% of words must match; end search bounded to ~1.5× the copied length). Copies
   with <5 words at either end (`min_words=5`) or spanning >120 s are rejected.
4. Matched words are marked as ad, hidden, and replaced in the next round by a line `[advertisement removed]`.
   Up to 6 rounds per window; a window stops when the model answers NONE or nothing matches.
5. `verify.py`: one call per merged interval with the transcript text of that interval (≤350 words). If the
   answer is NONE, the interval is dropped. Verify removed 9 of 12 false alarms on dev for E2B.
6. Optional: classifier veto (keep an interval only if the classifier's smoothed score >0.15 somewhere inside it).

### Format sample: copy call

System message (all copy calls):
```
You find advertisements in podcast transcripts. An advertisement is a sponsor message or commercial that promotes a product, service, brand, website, app or another podcast or show to the listener, usually with a call to action such as a web address, a promo code, a free trial or where to listen. The podcast's own content (story, reporting, interviews, discussion, intro, credits) is not an advertisement, even when it talks about companies or products. You only copy words that appear in the transcript.
```
User prompt (round 2 of the Daily 1 stacked mid-roll; transcript shortened here, the real window is ~45 lines):
```
Below is part of the transcript of the podcast "The Daily", episode "Inside the Devastating Hack of the F.B.I.".

<transcript>
[...]
And then? On Monday. The day before the deadline comes to pass The hackers reach With a new message Twist the story on its
[advertisement removed]
If you like YouTube, you'll love YouTube Premium. Hi, I'm Sean Evans from Hot Ones and I want to tell you about YouTube Premium. [...]
Try YouTube Premium for two months free at youtube.com/premium. Trial eligibility varies, terms apply, cancel anytime.
Online checkout shouldn't feel like an obstacle course. [...] Learn more at link.com.
His friends call him the Oma Kase Authority, 17 counters ran. [...] We'll take the laundry, you take Mint.com slash omakase. It's time to
So Dustin, what did these hackers say right before the step? [...]
</transcript>

Find the first advertisement in this transcript and copy its complete text, word for word, exactly as it is written in the transcript. Output only the copied advertisement text. If the transcript contains no advertisement, answer NONE.
```
Output round 1: the first YouTube read (with a few words skipped; the matcher still placed it). Round 2: the
Sean Evans read **and** the Link ad as one copy. Round 3: `NONE`, so the narrative Rinse ad is missed. The
no-marker variant sees the same text without the `[advertisement removed]` line and gives identical outputs.

### Format sample: verify call
```
Here is a part of the transcript of the podcast "{podcast}", episode "{title}":

{text of the predicted interval}

Podcasts are interrupted by advertisements: a sponsor message or ad that tries to get the listener to buy, try, download, subscribe to or listen to a specific product, service, app, brand or other show, usually with a call to action such as a website, a promo code or "try it free". Reporting, interviews or discussion about companies, money or products is NOT an advertisement.

Does this part contain an advertisement? If it does, answer with the name of the advertised product or brand. If it does not, answer NONE. Answer with the name or NONE only.
```
(system message: "You detect advertisements in podcast transcripts. … Answer with the advertised brand, or NONE.")

### Cost and phone latency estimate

Per audio hour (E2B, dev): ~40 calls, 44K input tokens, 1.9K output tokens. At published S26 Ultra speeds that
is under a minute on GPU, about 2 minutes on CPU; Pixel Tensor chips are slower, so expect a few minutes.
Moonshine adds 1–2 minutes per audio hour on a phone CPU. On this VM's CPU, E2B took ~6 min per audio hour;
E4B ~2×; Gemma 3n via llama.cpp ~2–4 min per episode.

---

## 7. Everything tried, in order, with numbers

All numbers are Gemma 4 E2B on the 7 dev episodes unless marked otherwise. "Verified" = after `verify.py`.
Full table of every run: `results/scoreboard.txt`.

### 7.1 Line-number ranges (failed)
Numbered, timestamped lines; the model returns `FIRST-LAST` ranges. Smoke test on daily1: F1 0.14, 1/4 breaks.
The 2B model cannot map ads to line numbers in a long window and missed obvious YouTube Premium reads.

### 7.2 30-second blocks, one classification call per block
- `yesno` prompt (BEFORE/CURRENT/AFTER blocks, answer AD/CONTENT): daily1 F1 0.49, 4/4 breaks, 5 false alarms.
- `product` prompt ("name the advertised brand or NONE"): full dev F1 **0.739**, 22/31 breaks, 0 false alarms.
  Very conservative; missed blatant ads (Capital One, rug ads).
- Block prompt dev set (`blockdev.py`, 115 blocks): product recall 0.60 at 0% FP; no-context 0.58; "listfirst"
  0.78 at 1.4% FP; **system message + no context 0.93 at 4.3% FP**.
- `system` prompt on full dev: F1 **0.812**, 30/31 breaks, 18 false alarms (short blips).
- Boundary refinement (ask for first/last ad line among ~12 lines): no net gain (0.758 vs ~0.75).
- Post-processing grid (merge gap, padding, gap filling): ≤0.02 F1 change.
- Cost: 137 calls per audio hour.

### 7.3 Quote anchoring (owner's idea: quote start and end of each ad)
`START: <first words> || END: <last words>` per ad, fuzzy-matched back.
- Single pass, 1,500 tokens: precise boundaries (dateline1 F1 0.92) but echoed the prompt's example phrases.
- v2 without examples (1,200 tokens): F1 **0.727**, P 0.93, R 0.60. Often quoted only one ad of a stacked break.
- Union of blocks + quotes: 0.83; "confirm" combination: **0.88** (26/30 breaks, 3 FA).

### 7.4 Iterative removal (owner's idea: strip found ads and ask again)
- 3 passes with `[advertisement removed]` marker: first version over-extended (a 1-word quote `START: If`
  matched 3 minutes early; one quote pair spanned two breaks). Fixed with ≥3-word quotes and a 120 s span cap.
- Result: F1 **0.843** (P 0.94, R 0.77), up from 0.727 single pass.

### 7.5 Copy mode (owner's idea: copy the full ad text) — the winner
One ad per call, up to 6 rounds, ≥5-word ends.
- Copy alone: F1 **0.879** (R 0.953, 30/31 breaks, 11 FA). Copy + verify: **0.939** (P 0.925, R 0.953, 2 FA).
- The model sometimes copies `[advertisement removed]` itself, which ends that window's loop.
- No-marker variant (owner suggested): copy 0.849, + verify **0.897**; recall dropped on SYSK. Marker kept.
  It does not explain the Rinse miss: without markers that ad is still missed (section 6 sample).

### 7.6 Model comparison (copy + verify)
See the headline table. Gemma 3n runs used llama.cpp Q4_0 GGUFs, so they are less faithful to the phone than the
Gemma 4 LiteRT-LM runs. Gemma 4 E4B single-pass quotes on 6 episodes: 0.902 (vs E2B 0.727): E4B is clearly
stronger.

### 7.7 Sampling and voting (failed)
Temperature 0.7 (top-k 40), seeds 1–3: copy F1 0.888 / 0.878 / 0.866 (greedy 0.879). Voting 1-of-3 0.868,
2-of-3 0.869, 3-of-3 0.880, 4-run votes 0.870–0.874. Union + verify **0.925**, 2-of-3 + verify 0.922: both below
greedy + verify 0.939, at 3× cost. The errors are systematic, not random.

### 7.8 Shifted windows (failed)
Windows offset by half a window: copy 0.869; union with baseline + verify 0.925.

### 7.9 Per-copy verification (failed)
Verify each copied span before accepting (`verify_each`). First version (stop on reject) lost real ads after an
early false copy. Revised (skip and continue, stop after 2 rejects) + tight line ends: dev **0.898** (worse than
0.939), held-out 0.898 vs 0.890. Not adopted.

### 7.10 Audio cues: speakers, pauses, music
- YAMNet (4 MB, `feats/`): 28 min of audio in 1.4 s. Music lines up with ad breaks but also with intros, outros
  and music beds.
- Diarization (pyannote-3.0 segmentation + TitaNet-small, clustering threshold 0.95, `diar/`): ~13× real time.
  In daily1, hosts never spoke during ads and three voices spoke only in ads.
- As prompt annotations (`S1:` by talk-time rank, `[pause Ns]`, `[music]`): pauses+music **0.882**,
  speakers **0.884**, both worse than plain 0.939. The model copies across the markers into neighbouring content.
- As post-processing: snapping edges to pauses/speaker changes 0.939 → ≤0.945; dropping detections spoken only by
  the main voice **0.829** (kills host-read ads). Conclusion: not worth shipping diarization or YAMNet for this.

### 7.11 Held-out evaluation and a harness fix
- Held-out (6 new episodes) results are in the headline table. E2B's errors were mostly over-extension into
  outros and credits, and copying host chatter in late rounds of a window (koala facts in Morbid's outro).
- Tight line ends (an ad line ends 2 s after its last word instead of at the next line start; `--tight-ends`)
  were added after a 90 s silence was counted as ad. Only used in the `copy6ve_*` runs.

### 7.12 Case bench (fast iteration)
`bench.py` runs only 13 windows: 8 problem spots (1 known miss, 7 false-positive spots) and 5 tricky true ads.
Copy stage only (no verify, no veto). A case passes when it has <10 s false positive and <10 s missed ad time.

| Variant | Search tokens | Context each side | Passed | FP s | FN s | Calls | Think tokens | Minutes |
|---|---|---|---|---|---|---|---|---|
| baseline | 1200 | 0 | 6/13 | 228 | 62 | 50 | 0 | 4.6 |
| marker fallback (re-ask without markers if only the marker is copied) | 1200 | 0 | 6/13 | 324 | 62 | 70 | 0 | 5.5 |
| detailed system message `sysv2` | 1200 | 0 | 6/13 | 393 | 85 | 54 | 0 | 5.3 |
| longer context | 2500 | 0 | 3/13 | 369 | 104 | 55 | 0 | 8.2 |
| longer context | 3500 | 0 | 6/13 | 275 | 346 | 54 | 0 | 10.0 |
| focus (search area in `<search>` tags) | 600 | 1200 | 8/13 | 139 | 121 | 30 | 0 | 4.7 |
| focus | 1200 | 1000 | 7/13 | 159 | 90 | 42 | 0 | 6.6 |
| shorter single window* | 600 | 0 | 7/13 | 208 | 50 | 41 | 0 | 2.6 |
| shorter single window* | 400 | 0 | 8/13 | 163 | 40 | 38 | 0 | 2.0 |
| shorter single window* | 250 | 0 | 6/13 | 278 | 63 | 35 | 0 | 1.5 |
| thinking, 512-token budget | 1200 | 0 | 6/13 | 304 | 84 | 45 | 22,952 | 17.1 |
| small search + context | 400 | 600 | 6/13 | 472 | 57 | 34 | 0 | 3.8 |
| small search + context | 400 | 300 | 2/13 | 477 | 76 | 45 | 0 | 3.6 |
| sliding 400 over a 1,200 region | 400 | 0 | 5/13 | 441 | 96 | 111 | 0 | 4.5 |
| sliding 400 + classifier veto (strip vetoed spans) | 400 | 0 | 8/13 | 238 | 138 | 109 | 0 | 4.4 |
| sliding 400 + thinking 512 | 400 | 0 | 4/13 | 197 | 211 | 94 | 40,133 | 25.4 |
| sliding 400 + veto + thinking 512 | 400 | 0 | 7/13 | 107 | 220 | 94 | 40,133 | 25.4 |

\* The shorter single windows cover less transcript than baseline (one centred window), so part of their gain is
less text to get wrong. With the same 1,200-token region, sliding 400-token windows were worse (5/13, 441 s FP):
small windows with no ad still produce a copy instead of NONE.

Findings from the bench:
- Longer context hurts the 2B model (owner's hunch, confirmed).
- With a small search area and lots of context, the model copies ads from the context area and ignores the
  `<search>` instruction; the matcher then forced those copies onto unrelated words. A context-rejection fix
  changed nothing (identical numbers), so that diagnosis was incomplete. Dropped at the owner's suggestion.
- **Thinking**: the model re-quotes the transcript line by line in its reasoning and hits the 512-token budget
  before reaching the ad (median 360 words of thought). It became more conservative (fewer FP, many more misses)
  at 4–6× the time. A fair test needs ≥1,500 thinking tokens, which is roughly an hour of phone compute per audio
  hour. Not recommended unless run overnight.
- The classifier veto in the copy loop removes false positives but also strips real ads it scores low, which are
  then never found.

### 7.13 Line classifier (TF-IDF + logistic regression)
Words and word pairs of each line plus a 5-line context window, ~80K features, `class_weight="balanced"`,
self-promo lines excluded from training, 3-line moving average, threshold 0.15, same interval rules. Explained in
detail with a worked example in `classifier-guide.html`.

| Evaluation | F1 | Breaks | False alarms |
|---|---|---|---|
| Trained on dev, tested on held-out, threshold 0.5 / 0.25 / 0.15 | 0.455 / 0.732 / **0.795** | 9 / 17 / 21 of 24 | 1 / 1 / 4 |
| Leave-one-episode-out, dev / held-out (alone) | 0.838 / 0.795 | 31/31, 24/24 | 9, 9 |
| As a veto on LLM copy+verify | see headline section | never cost a break | halves the remaining FAs |

Strongest weights: ad side `com`, `your`, `slash`, `com slash`, `october`, `podcast`, `visit`; content side
`was`, `he`, `that`, `they`, `she`, `yeah`. It partly memorises current sponsors (e.g. films opening in October)
and needs retraining as sponsors change. The 0.15 threshold was chosen after looking at held-out results, so
those numbers are a little optimistic. Pickled size 2.5 MB; a top-features cut should get it to a few hundred KB.

---

## 8. Known error cases (good regression tests)

| Case | Type | What happens |
|---|---|---|
| daily1 27:12 Rinse | miss | Narrative ad ("a woman is sitting across from a toilet…") only names the brand at the end. Every LLM variant misses it; the leave-one-out classifier scores it 0.07–0.26 (above its 0.15 threshold for most lines). |
| daily1 14:41 / daily2 15:05 Rinse (same ad mid-roll) | miss | Same ad at the end of a stacked break; round 3 answers NONE. In daily2 the classifier scores it 0.77–0.93. |
| daily2 02:08 Kalshi | false positive | Reporting plays real Kalshi TV ad clips. Arguable even for humans; the classifier scores it ~0.1. |
| daily1 09:21 hackers | false positive (copy stage) | "Hey, look, we extort victims…" quoted in a news story; removed by verify. |
| morbid1 03:43 KIFF / 80:05 koala | false positive | Host chat ("listen to it on spotify", "koala world on netflix"); late rounds of a window copy something instead of NONE. |
| freak1 56:17 "Better in Person" | debatable | Show plugs its own TV show; partly in a self-promo label. |
| conan1 70:41 credits | false positive | Credits and subscribe plug merged with the final ad. |

---

## 9. Gotchas hit during the session

- **`pkill -f` / `pgrep -f` self-match:** a waiter or kill command whose own command line contains the pattern
  matches itself (killed the shell 4 times, and waiters never fired). Put the pattern in a script file, or wait
  on a PID.
- **llama-server OOM:** the prompt cache defaults to 8 GB host RAM (`--cache-ram`); llama-server grew to 4.9 GB
  and was OOM-killed twice. `queue2.sh` passes `--cache-ram 0`.
- Editing a shell script while bash runs it: the running process keeps the old (deleted) file.
- `tail -F | grep` waiters must also look for errors, or a crash looks like "still running".
- The redroid Android container shares this VM's RAM; another session may be using it.
- Gemini 3.1 Pro returned 429 on the owner's key (free tier); 3.5 Flash worked with retries on 503.
- `adtest.py` main defaults to the dev split and only episodes that have labels and transcripts.

---

## 10. Next steps

Ideas from the owner at the end of the session, not yet tried:

1. **In-context examples.** Put 2–3 short worked examples in the prompt: a host-read ad, a narrative ad that
   names the brand only at the end (like Rinse), and a hard negative (reporting that quotes an ad, or hosts
   plugging their own show) with the answer NONE. Keep the total under ~4K input tokens.
2. **Two-pass structure.** Pass 1: normalise the ASR text into clean, numbered sentences (fix broken fragments,
   merge VAD splits). Pass 2: label each numbered sentence with one topic word and AD / CONTENT. Ad breaks are
   then runs of sentences labelled AD, and the topic words give a cheap signal for where the topic jumps (ad) and
   returns (content). Watch the ~256-token output cap: about 40 sentences × ~5 tokens per call.
3. Thinking is probably not worth it (section 7.12).

Other open items:
- Validate the bench winners on full episodes (dev and held-out) before adopting anything.
- Make the copy loop say NONE more readily in windows without ads (prompt line such as "most sections contain no
  ads"), and only strip vetoed spans when the classifier is very sure (score <0.05).
- Embedding-based classifier (EmbeddingGemma 300M via LiteRT) to generalise to new sponsors.
- More held-out episodes once a GPU VM is available (owner asked not to expand the set on this CPU-only VM).
- Port to the app: sherpa-onnx (Moonshine) + LiteRT-LM (Gemma 4 E2B/E4B) in a background worker; the
  classifier as a weight table in Kotlin. Optionally an ML Kit Prompt API backend for phones with Nano v4.

---

## 11. Session chronology (short)

1. Researched ML Kit GenAI / AICore access, limits and Pixel model versions (section 2).
2. Built the harness on a CPU-only arm64 VM; downloaded 7 episodes (The Daily ×2, Dateline ×2 as requested,
   SYSK, Crime Junkie, Freakonomics), 4 ASR models, Gemma 4 E2B/E4B (LiteRT-LM) and Gemma 3n E2B/E4B (GGUF).
3. Labelled the episodes by hand. Ranges and block prompts plateaued around F1 0.74–0.81.
4. Owner's ideas: quote anchoring → iterative removal → full-text copy. Copy + verify reached 0.939 (E2B).
5. Model and ASR comparisons; sampling, voting, offsets and per-copy verify did not help.
6. Added 6 held-out episodes and a cloud second labeller; fixed one missed label; E4B held-out 0.967.
7. Audio cues (diarization, YAMNet) as annotations and post-processing: no gain.
8. Case bench for fast iteration: context length, focus windows, short windows, thinking, sliding + veto.
9. TF-IDF classifier as a veto: small, consistent gain; guide published as a private artifact.

---

## 12. Session 2 (2026-10-01, Lima `krunkit` VM with GPU) — status and handoff

New VM: Fedora 44 arm64, 6 vCPU, 5.9 GB RAM, Vulkan through Venus (`Virtio-GPU Venus (Apple M4)`).
The old VM's WAVs were not available, so everything was re-downloaded.

### Re-downloaded audio has different ads → new labels in `gt_v2/`
- 12 of 13 episodes changed length (−54 s to +221 s; only dateline3 within 2 s). Ad loads are different
  (e.g. daily1's Rinse mid-roll is now Harvey; the Rinse narrative end ad is now Vanta). **`gt/` matches only
  the old audio; `gt_v2/` matches the audio downloaded on 2026-10-01.** Section 8's error cases no longer exist
  in the new audio.
- `gt_v2/`: 13 episodes, 11.63 h, 54 ad breaks, 91.8 ad minutes. Labelled by hand from the Moonshine
  transcripts: `strong_cues.py` (cue scan) plus reading around every hit and every old break position (DAI slots
  stay at the same story positions). Same conventions as before: network cross-promos for other podcasts are
  `ad`; the show's own plugs/credits asks are `self_promo`. No cloud second labeller was run.
- `adtest.py` reads labels from `$GT_DIR` (default `gt`). Use `GT_DIR=gt_v2` for every run on the new audio;
  old `results/` runs only make sense with the default `gt`.

### GPU: not working yet (VM wedged)
- `LITERT_BACKEND=gpu` (new in `llm.py`) with `gemma-4-E2B-it-gpu.litertlm` hung during engine creation
  (no output after 10 min). The process then got stuck in the kernel (`D` state in `exit_mm`) and cannot be killed.
- Afterwards llama.cpp built with `-DGGML_VULKAN=ON` hung in `virtio_gpu_vram_mmap` on `--list-devices`, and
  `vulkaninfo` hangs too: the guest GPU is wedged until the VM restarts. dmesg showed
  `virtio_gpu_dequeue_ctrl_func ... response 0x1200` errors from early boot.
- Side effect: `ps`, `pgrep`, `pkill` block forever on the stuck process (`__access_remote_vm`). Avoid them while
  it exists; read `/proc/<pid>/stat` directly instead.
- RAM was not the cause (4.8 GB available, no swap use during the hang).
- **After restart, test the GPU in this order, each with `timeout -s KILL` in the background:** `vulkaninfo
  --summary` → `llama.cpp/build/bin/llama-server --list-devices` → `llama-bench -m
  models/gemma-4-E2B-it-Q4_K_M.gguf -ngl 99 -p 1200 -n 64` → only then LiteRT-LM GPU. If llama.cpp Vulkan
  works, run the pipeline through `queue2.sh` (add `-ngl 99`). Watch RAM: `gemma-4-E4B-it-Q4_K_M.gguf` is
  5.0 GB on a 5.9 GB VM.

### Re-baseline on `gt_v2` (CPU)
| Run (Gemma 4 E2B, all 13 eps) | P | R | F1 | Breaks | False alarms |
|---|---|---|---|---|---|
| copy6 | 0.836 | 0.946 | 0.887 | 54/54 | 17 |
| copy6 + verify | 0.888 | 0.949 | **0.917** | 54/54 | 4 |

- Speed on this VM's CPU: 2.6 min per audio hour (copy 30 min + verify 1.7 min for 11.6 h).
- `clf_edges.py`: TF-IDF line classifier retrained leave-one-episode-out on `gt_v2`, used to trim interval
  edges. With the threshold picked on dev, held-out F1 0.885 → **0.914**. Per-line scores in `clfp_gt_v2/`.
- Moonshine on this VM's CPU: 27–37× real time (all 13 episodes in ~25 min while other jobs ran).
- `runq.sh [queue.txt]` runs the uncommented lines of a queue file one by one (marks each `# done:`), logging to
  `logs/runq.txt` and `logs/runq_out.txt`. Defaults `GT_DIR=gt_v2`.
- The VM rebooted at ~20:09 during the `shots=1` run; the partial result was discarded and the queue restarted
  at 20:40. Remaining queue: shots + verify, `label` strategy, E4B copy6 + verify.
- LiteRT-LM logs `XNNPack weight cache could neither be loaded from or saved to .litert-cache/...` when that
  folder is missing; create it (`mkdir .litert-cache`) to skip re-packing weights on every start.

### Few-shot examples (`shots=1`, E2B, CPU, `gt_v2`)
| Run | P | R | F1 | Breaks | False alarms |
|---|---|---|---|---|---|
| copy6 shots | 0.869 | 0.918 | 0.893 | 52/54 | 9 |
| copy6 shots + verify | 0.915 | 0.921 | **0.918** | 52/54 | 1 |

Same F1 as the baseline (0.917): fewer false alarms (1 vs 4) but two breaks missed. Not a clear win.

### Per-line labelling (`--strategy label`, chunk 30, context 6, topic words, E2B, CPU, `gt_v2`)
P 0.829, R 0.735, **F1 0.779**, 45/54 breaks, 23 false alarms, 33 min for 11.6 h. Clearly worse than copy mode;
like the line-number ranges in 7.1, the 2B model cannot keep per-line labels straight. Dropped.

### GPU (Vulkan through Venus, after the reboot)
- `vulkaninfo --summary` and `llama-server --list-devices` work after the reboot (Venus, 16 GB shared).
  LiteRT-LM GPU was **not** retried (it wedged the VM last time).
- `llama-bench` Gemma 4 E2B Q4_K_M `-ngl 99`: **515 tok/s prefill, 39 tok/s decode** (CPU queue running at the
  same time).
- **Gotcha:** llama.cpp's Gemma 4 chat template thinks by default. Every call spent all 256 output tokens in
  `reasoning_content` and returned empty content (F1 0; that run was deleted). `llm.py` now sends
  `chat_template_kwargs: {enable_thinking: false}`.
- `queue2.sh` takes extra server flags from `LLAMA_ARGS`; GPU runs use `LLAMA_ARGS="-ngl 99"`.

| Run (E2B, all 13 eps, `gt_v2`) | Backend | P | R | F1 | Breaks | FA | LLM min per audio hour |
|---|---|---|---|---|---|---|---|
| copy6 | LiteRT-LM CPU (int4 `.litertlm`) | 0.836 | 0.946 | 0.887 | 54/54 | 17 | 2.56 |
| copy6 | llama.cpp Vulkan (Q4_K_M) | 0.800 | 0.957 | 0.871 | 53/54 | 14 | 2.07 |
| copy6 + verify | LiteRT-LM CPU | 0.888 | 0.949 | **0.917** | 54/54 | 4 | |
| copy6 + verify | llama.cpp Vulkan | 0.833 | 0.960 | 0.892 | 53/54 | 6 | |

The GPU is only ~20% faster than LiteRT-LM on CPU here (the CPU queue shared the machine during the GPU run), and
the Q4_K_M GGUF scores a bit lower than the LiteRT-LM file (different quantisation and chat template). For
phone estimates, the LiteRT-LM numbers stay the reference.

Why the GPU gain is small (investigated, not a bug):
- The host is a **Mac mini with a base Apple M4** (confirmed by the owner): 10-core GPU, Metal 4, 16 GiB unified
  memory shared with macOS and this VM, 120 GB/s. Native
  Metal on that chip does ~221 tok/s prefill and ~24 tok/s decode on Llama 2 7B Q4_0 (llama.cpp Apple Silicon
  thread, github.com/ggml-org/llama.cpp/discussions/4167). Scaled to E2B's ~2.3B compute parameters that is roughly
  ~640 prefill / ~55–60 decode, so the measured 515 / 39 is ~65–80% of native.
- The guest path is Mesa Venus → virglrenderer → MoltenVK → Metal. Red Hat measured Venus at 75–80% of native
  llama.cpp speed because every Vulkan call is serialised through virtio-gpu (developers.redhat.com, 2025-09-18).
  MoltenVK also exposes no cooperative matrix or integer dot product (`int dot: 0`, `matrix cores: none` in the
  llama.cpp log), so llama.cpp's Vulkan backend uses its slowest matmul path. Running Vulkan on Apple GPUs
  natively (Asahi, M2 Max) was 6× slower than Metal for prefill in llama.cpp issue #10982.
- Flash attention and ubatch 256/512/1024 change nothing (466–513 prefill, 37–40 decode): the limit is the
  translation layer and the small GPU, not llama.cpp settings.
- The CPU side is strong: 6 vCPUs of M4 with `i8mm`, `bf16` and `asimddp` exposed, which XNNPack's int8 kernels use.
  A base M4's CPU and GPU are close in LLM throughput, unlike phones (S26 Ultra: GPU prefill ~7× CPU).
- **Clean benchmark (02:20, idle machine, `logs/clean_bench*.txt`)**, E2B Q4_K_M, llama.cpp, 1200-token prompt:
  pure CPU (`-ngl 0 -nopo 1`, 6 threads) **92 tok/s prefill / 40 decode**; GPU (`-ngl 99`) **518 / 43**. So the GPU
  is 5.6× faster than llama.cpp's CPU path at reading prompts (decode is memory-bound and equal). Note `-ngl 0`
  alone still offloads large prompt batches to the GPU (404 / 27 tok/s), so use `-nopo 1` for CPU numbers.
  The small end-to-end gain (2.56 → 2.07 min per audio hour) is because **LiteRT-LM's CPU path is itself fast**
  (XNNPack int8 kernels using `i8mm`): per call it is only ~20% slower than llama.cpp on the GPU, i.e. several
  times faster than llama.cpp on the CPU.
- Conclusion: this VM cannot predict phone GPU speedups. For phone numbers use Google's published LiteRT-LM figures
  or run on a Pixel. Red Hat's "API remoting" krunkit build (forwards ggml calls to host Metal) reaches near-native
  speed if real Apple GPU numbers are ever needed.

### Gemma 4 E4B on `gt_v2` (CPU)
| Run | P | R | F1 | Breaks | False alarms |
|---|---|---|---|---|---|
| E4B copy6 | 0.954 | 0.967 | **0.960** | 54/54 | 3 |
| E4B copy6 + verify | 0.957 | 0.964 | 0.960 | 53/54 | 3 |

E4B needs no verify pass (it costs a break). ~5 min LLM time per audio hour on this CPU (2× E2B). On the 7 dev
episodes the copy stage alone: E4B 0.963 / 1 false alarm vs E2B 0.908 / 9.

### Not yet measured
- LiteRT-LM on the GPU (`LITERT_BACKEND=gpu`); CPU reference (old VM): E2B ~6 min per audio hour, E4B ~2×.

### Options added in session 2 (both tested above: shots = no gain, label = worse)
- `--arg shots=1` (quotes/copy strategy): appends four invented worked examples to the system message (host-read
  ad, narrative ad naming the brand at the end, reporting that quotes an ad → NONE, show credits/plugs → NONE).
  Invented text so the held-out split stays clean.
- `--strategy label` (`chunk=30 context=6 topic=1`): numbered lines, the model writes `N | topic | AD/C` for
  every line (30 lines ≈ 240 output tokens, under the 256 cap). `topic=0` drops the topic word.

### Environment notes (Fedora)
- `dnf install gh git ffmpeg-free cmake gcc gcc-c++ python3-devel vulkan-headers vulkan-loader-devel glslc
  glslang spirv-tools spirv-headers-devel aria2`; the venv works on Python 3.14.
- Hugging Face downloads through `huggingface_hub` ran at ~235 KB/s here; `dl_models.sh` uses `aria2c -x16`
  (~48 MB/s). Gemma 4 GGUFs: `unsloth/gemma-4-E2B-it-GGUF`, `unsloth/gemma-4-E4B-it-GGUF` (Q4_K_M).
- `transcribe_all.sh` transcribes every `wav/*.wav` without a transcript.

---

## 13. Session 2, night of 2026-10-01: cheap classifiers instead of (or next to) the LLM

Goal: something much smaller and faster than Gemma that finds ads well, or at least vetoes the LLM's false alarms.
Everything on `gt_v2`. **Evaluation is leave-one-show-out** (`GROUP=show`): all episodes of a show are held out
together, because same-day downloads of one show carry the same inserted ads (Daily ×2, Dateline ×4), which let
leave-one-episode-out memorise them. (It turned out to matter little: TF-IDF test F1 0.796 → 0.793.) Thresholds
and smoothing knobs are picked on the 7 dev episodes and applied unchanged to the 6 held-out ones.

Scripts: `emb_loo.py` (features + logistic regression, writes per-line scores to `embp_gt_v2_show/<name>/`),
`seq_smooth.py` (score → intervals: 3-line average, hysteresis, 2-state HMM/Viterbi), `repeat_match.py`,
`gemma_probe.py` (Gemma hidden states via `llama-server --embeddings --pooling none`), `fetch_extra.py`,
`night.sh` (resumable overnight chain). Logs in `logs/emb_*`, `logs/seq_*`, `logs/repeat_match.txt`.

### 13.1 Line classifiers (each line + a 5-line window, both embedded; logistic regression)
| Features | Size | Line AUC dev / test | Line AP dev / test | Alone, test F1 (ma3) | Breaks | FA |
|---|---|---|---|---|---|---|
| TF-IDF words + pairs (old classifier) | ~2.5 MB | 0.996 / 0.981 | 0.972 / 0.910 | 0.793 | 20/23 | 2 |
| Model2Vec potion-base-8M (static) | 30 MB | 0.991 / 0.956 | 0.941 / 0.827 | 0.704 | 17/23 | 5 |
| all-MiniLM-L6-v2 | 22M params | 0.986 / 0.974 | 0.932 / 0.884 | 0.819 | 20/23 | 4 |
| **bge-small-en-v1.5** | 33M params | 0.992 / **0.989** | 0.961 / **0.953** | **0.880** | 23/23 | 6 |

### 13.2 Smoothing (score → intervals), held-out test F1 of the classifier alone
| Features | 3-line average | Hysteresis | HMM (Viterbi) |
|---|---|---|---|
| TF-IDF | 0.793 (2 FA) | 0.803 (4 FA) | 0.822 (1 FA) |
| MiniLM | 0.819 (4 FA) | 0.821 (3 FA) | 0.878 (0 FA) |
| **bge-small** | 0.880 (6 FA) | 0.907 (1 FA) | **0.917 (23/23 breaks, 1 FA)** |

**bge-small + HMM alone matches the LLM:** held-out F1 0.917 vs 0.888 for E2B copy6 + verify on the same 6 episodes
(E4B: 0.96). It is a 33M-parameter encoder (~35 MB int8) plus a logistic regression and a 2-state HMM, i.e.
milliseconds per line on a phone CPU, no LLM call at all. HMM knobs picked on dev: alpha 2, switch 0.001, bias 1.

### 13.3 As a veto on E2B copy6 + verify (drop LLM intervals the classifier does not overlap)
Held-out: 0.888 / 3 FA → **0.931 / 0 FA** with TF-IDF or bge-small (no break lost); MiniLM and Model2Vec lose one
break. Dev stays 0.938 (TF-IDF) or 0.931–0.934 (embeddings, which drop 1–2 dev breaks). The veto can only remove
LLM intervals, so it caps out once the false alarms are gone.

### 13.4 Repetition matching (`repeat_match.py`)
Word 6-gram hashes of the ads the LLM confirmed in the *other* 12 episodes; a line matches when ≥50% of its 6-grams
are in that database. Alone: **precision 0.976, 0 false alarms**, recall 0.25 (15/54 breaks); 13 of the LLM's 60
detected breaks are already covered, so a phone could skip the LLM there. Using every line of other episodes
instead (no LLM filter) adds intros/credits: precision 0.90, 6 false alarms. This grows with the user's library.

### 13.5 EmbeddingGemma 300M and the Gemma hidden-state probe
| Features | Line AUC dev / test | Line AP dev / test | Test F1, 3-line avg | Test F1, HMM | Dev F1, HMM |
|---|---|---|---|---|---|
| **EmbeddingGemma 300M** (`unsloth/embeddinggemma-300m`, "Classification" prompt) | 0.998 / **0.991** | 0.983 / **0.962** | 0.920 (1 FA) | **0.920 (23/23, 0 FA)** | 0.945 (31/31, 0 FA) |
| Gemma 4 E2B final-layer probe (mean + last token per line, via llama.cpp) | 0.988 / 0.983 | 0.962 / 0.925 | 0.877 | 0.880 | 0.937 |

- **EmbeddingGemma + HMM alone (no LLM): all 13 episodes F1 0.935, 54/54 breaks, 0 false alarms** (E2B copy6 +
  verify: 0.917, 4 FA). ~1 s per 4 lines on this busy VM CPU in fp32 (unoptimised; batch of line + window texts).
  HMM knobs picked on dev: alpha 1, switch 0.01, bias -1.
- Ablation (HMM, dev / test F1): line embedding only 0.941 / 0.912 (1 FA each); 5-line window only 0.876 / 0.896;
  window cut to 256 or 128 Matryoshka dims 0.886 / 0.910 and 0.876 / 0.889; both 0.945 / 0.920. The HMM supplies
  most of the context, so a phone could embed only each line (5× fewer tokens) and lose little.
- The final-layer probe is worse than a purpose-built embedding model. A mid-layer probe needs a truncated GGUF:
  `--override-kv gemma4.block_count` fails because per-layer arrays (`feed_forward_length`, …) must have 35 entries.
- Ensembles (averaged line probabilities, HMM): EmbeddingGemma + bge-small test 0.940 / 0 FA (dev 0.937); adding
  TF-IDF or the probe does not help. Chosen after seeing test numbers and within noise (6 held-out episodes), so
  EmbeddingGemma alone stays the principled pick.

### 13.6 Combinations with the LLM (all 0 false alarms unless noted)
| Pipeline | Dev F1 | Test F1 | Breaks | LLM calls per audio hour |
|---|---|---|---|---|
| E2B copy6 + verify | 0.938 (1 FA) | 0.888 (3 FA) | 54/54 | ~35 |
| E2B copy6 + verify + EmbeddingGemma veto | 0.938 (1 FA) | 0.931 | 54/54 | ~35 |
| EmbeddingGemma + HMM alone | 0.945 | 0.920 | 54/54 | 0 |
| EmbeddingGemma + HMM, then E2B verify per interval (`clf_run.py` + `verify.py`) | all 13: 0.938 | | 54/54 | ~5 |
| E4B copy6 | 0.967 (1 FA) | 0.959 (2 FA) | 54/54 | ~33 (2× slower calls) |
| **E4B copy6 + EmbeddingGemma veto** | **0.969** | **0.966** | **54/54** | ~33 |

Takeaways: on a Pixel 11 class phone, E4B copy + an EmbeddingGemma veto is the most accurate (0 false alarms in
11.6 h). Where the LLM is unavailable or too slow (Pixel 10, background limits), EmbeddingGemma + HMM alone is
within ~0.03–0.04 F1 of E4B at a tiny fraction of the compute, and beats E2B.

### 13.7 Distillation from the E4B teacher (`fetch_extra.py`, `night.sh`, `distill.py`, `finetune.py`)
- Data: the newest 20–90 min episode of 35 other shows (none in the labelled set), 36.1 h, downloaded 2026-10-01/02
  (`episodes_extra.json`, empty labels in `gt_extra/`). Moonshine transcripts in ~70 min. Teacher: E4B copy6 (no
  verify), 4.7 min LLM per audio hour, 176 breaks, 12.8% of the time marked ad (labelled set: 13.2%). Teacher
  results (logs stripped) in `results/gemma-4-E4B-it.litertlm__moonshine__quotes_teacher/`.
- Student = the same logistic regression + HMM, every setting picked on dev, scored on the 13 human-labelled
  episodes:

| Features | Trained on | Dev F1 (HMM) | Test F1 (HMM) | Test line AP |
|---|---|---|---|---|
| EmbeddingGemma | human labels, other shows (13.5) | **0.945** | **0.920** | 0.962 |
| EmbeddingGemma | teacher labels, 35 new shows only | 0.908 | 0.850 | 0.915 |
| EmbeddingGemma | both | 0.929 | 0.907 | 0.941 |
| bge-small | human labels | 0.921 | 0.917 | 0.953 |
| bge-small | teacher only / both | 0.927 / 0.936 | 0.839 / 0.861 | 0.865 / 0.917 |

- **Distilled data made the linear student worse.** Not because the teacher's labels are poor: on the 13
  labelled episodes, E4B's labels agree with the human ones on 99.5% of lines, and a student trained on them
  (leave-one-show-out) scores 0.935 / 0.904 vs 0.945 / 0.920 (`logs/teacher_label_diag.txt`). The teacher also
  marks 40% of self-promo lines as ad, which the human labels leave neutral. The likely cause is distribution: the 12 labelled
  shows are closer to each other (same networks, same day's ad campaigns) than to the 35 new shows. A linear
  model on frozen embeddings already has enough data at 13 episodes; more distant data dilutes it.
- **Fine-tuned bge-small** (`finetune.py`: whole encoder + linear head, input = (line, 5-line window) pair, 1 epoch
  on the 15,325 teacher-labelled lines, 37 min on this CPU): test line AP **0.936** (frozen bge-small on the same
  teacher labels: 0.865), dev AP 0.975. After HMM: dev 0.926, test 0.851 (23/23 breaks, 2 FA). Fine-tuning clearly
  helps per line, but the teacher-only domain gap remains; it does not beat frozen EmbeddingGemma on human labels.
- Next for distillation: fine-tune on teacher + human labels (human weighted up), or pre-train on teacher data and
  then fine-tune on the human-labelled shows (needs leave-one-show-out folds: ~9 × 40 min on this CPU); fine-tune
  EmbeddingGemma itself (300M) on a GPU machine.

---

## 14. Session 3 (2026-10-02 morning): speed, GPU, boundaries, inserted-ad alignment

### 14.1 EmbeddingGemma speed and size
- Files: full precision safetensors 1.21 GB; **Q8_0 GGUF 334 MB** (`ggml-org/embeddinggemma-300M-GGUF`); Google's
  QAT LiteRT builds run in <200 MB RAM and take <15 ms per 256-token input on a Pixel's EdgeTPU (Google blog).
  The trained part (logistic regression + HMM) is ~20 KB.
- Per audio hour (~425 lines of 32 tokens; 5-line windows of 153 tokens), idle VM:
  PyTorch fp32 CPU 18 s lines only, 69 s windows, **87 s both**; llama.cpp Q8_0 on the Vulkan GPU (`egemma_gguf.py`,
  CPU busy with transcription) **24 s both**. Accuracy of Q8_0 features: test line AP 0.961 (fp32 0.962), HMM dev
  0.942 / test 0.906 (fp32 0.945 / 0.920), 54/54 breaks, 0 false alarms.
- Whole classifier pipeline per audio hour on this VM: Moonshine ~1.5–2 min + embedding 0.4–1.5 min + classifier
  <1 s. Phone estimate for the embedding step: ~5–10 s on the TPU, ~20–60 s on CPU (not measured).

### 14.2 Boundaries (`boundary_err.py`, `edge_refine.py`)
Per labelled break (all 13 eps), ad seconds left playing / content seconds wrongly skipped: EmbeddingGemma+HMM
8.0 / 5.0, E2B copy+verify 5.2 / 12.2, E4B copy 3.4 / 4.8. Most error is at **starts** (classifier starts >5 s
late on 24% of breaks). Lines are ~7 s (median 21 words), so line-level edges are coarse.
- Classifier-guided edge refinement of LLM intervals (extend while the next line scores >ext, trim while <trim,
  knobs on dev): E4B dev 0.967 → 0.975 (ad left 3.5 → 1.1 s/break) but test unchanged (0.959 → 0.957); E2B
  0.888 → 0.904 test. Not a robust win.
- Shorter lines (`LINE_WORDS=12`, median 3.9 s; VAD segments limit it to ~3 s): test 0.914 vs 0.906, but it trades
  ad seconds left (9.2 → 5.5) for wrongly skipped content (5.3 → 10.4). Granularity is not the bottleneck; the
  classifier's judgement at transitions is. Next idea: ask the LLM about a small window around each classifier
  edge only (~2 short calls per break).

### 14.3 Which feeds insert ads per download (`dai_survey.py`, `logs/dai_survey.txt`)
1-byte range requests with two User-Agents, compared with our earlier downloads: **46 of 48 episodes are
dynamically stitched** (simplecast, megaphone, art19, triton, acast, PRX dovetail, flightcast, cloudfront). Only Lore
(libsyn) and Acquired (transistor) were identical. Big shows rarely ship purely baked-in audio; host-read ads may
still be baked in alongside the inserted ones.

### 14.4 Free labels from inserted ads (`fetch_variants.py`, `dai_align.py`)
Downloading the same episode with other User-Agents gives identical content with different inserted ads. Aligning
the word streams (difflib, matches ≥4 words) and keeping base stretches ≥15 s that have no counterpart:

| Rule (daily1, conan1, sysk1, planetmoney1; 2 extra copies each) | Precision | Ad time found | Breaks | FA |
|---|---|---|---|---|
| differs from **every** copy | **0.97** | 0.48 | 8/16 | 1 |
| differs from **any** copy | 0.81 | 0.79 | 15/16 | 15 |
| any copy, ≥8 s | 0.70 | 0.80 | 16/16 | 41 |

All 16 breaks in these episodes are inserted (none baked in). Short false stretches come from ASR differences when
the same audio is segmented at a different offset; requiring a difference from every copy removes them but loses
slots where two copies got the same ad. Use: high-precision positive labels for new shows at the cost of downloads
and transcription only, combined with the classifier/E4B for the rest. `variants_extra.sh` fetches and transcribes
two copies of each of the 35 extra episodes for this.

### 14.5 Audio embeddings instead of (or next to) the transcript (`audio_emb.py`, `emb_concat.py`)
Same per-line layout as the text features ([line audio, mean of lines i-2..i+2]), same logistic regression, HMM and
leave-one-show-out evaluation, so numbers compare directly with 13.1–13.5.

| Features | Size | Line AUC dev / test | Line AP dev / test | HMM dev F1 | HMM test F1 (breaks) |
|---|---|---|---|---|---|
| log-mel + loudness stats ("production style") | 0 | 0.634 / 0.389 | 0.168 / 0.090 | 0.371 | 0.062 |
| Whisper-base encoder, frames mean-pooled | 20M (encoder) | 0.988 / 0.961 | 0.937 / 0.856 | 0.898 | 0.741 (19/23) |
| **Gemma 4 E2B audio encoder** (305M) | 610 MB bf16 | 0.977 / 0.952 | 0.896 / 0.827 | 0.873 | 0.659 (14/23) |
| *text: EmbeddingGemma (13.5)* | 300M | 0.998 / 0.991 | 0.983 / 0.962 | 0.945 | **0.920 (23/23)** |
| text + Whisper, features joined | | 0.998 / 0.990 | 0.989 / 0.968 | 0.950 | 0.890 (22/23) |
| text + Gemma audio, features joined | | 0.996 / 0.984 | 0.982 / 0.936 | 0.950 | 0.879 (22/23) |
| text + Whisper, scores averaged | | | | 0.943 | 0.763 (19/23) |

- Gemma 4's audio encoder: only its weights were fetched from `google/gemma-4-E2B-it` (public, not gated) with an
  HTTP range request (614 MB of the 10.2 GB file; offsets from the safetensors header, `models/gemma4_e2b_audio/`),
  loaded into `transformers`' `Gemma4AudioModel` (305M params, strict load). 30 s chunks; ~4.6 min per audio hour
  on this CPU (fp32, mostly uncontended). Use `curl -r`, not aria2c: aria2c ignored the Range header and pulled
  the whole file.
- **Audio does not beat the transcript, and adding it hurts on held-out shows.** Audio embeddings separate ads well
  within the shows they were trained on (dev AUC 0.98–0.99) but generalise worse to unseen shows: they learn each
  show's sound (voices, mics, music beds), while the transcript carries the generic ad language (offers, URLs,
  promo codes). Raw spectral statistics are worse than chance on unseen shows. Joining text and audio slightly
  improves dev and line AP but lowers held-out recall.
- Not tried: CLAP and AST (general audio-event models; branches exist in `audio_emb.py`), and audio models
  trained on many more shows, which might fix the generalisation gap.

### 14.6 Automatic labels by aligning the audio of two downloads (`audio_align.py`, owner's idea)
Better than the transcript diff of 14.4: no ASR noise. Loudness envelope in 10 ms buckets for both copies; copy A
is cut into 2 s pieces, each matched into copy B by normalised cross-correlation; **content = runs of ≥3 pieces
with the same time shift** (copies are re-encoded, so content correlates 0.93–0.99, not 1.0; ads land at random
shifts). A stretches with no steady match (≥10 s) are inserted ads; edges are refined bucket by bucket while the
copies agree (≤4 dB). Flat/silent B windows are excluded from matching (they broke the normalisation). ~3–60 s per
episode, audio only, no transcription needed.

| 4 labelled eps (daily1, conan1, sysk1, planetmoney1) | Precision | Ad time found | Breaks | FA |
|---|---|---|---|---|
| transcript diff, any copy (14.4) | 0.81 | 0.79 | 15/16 | 15 |
| **audio alignment, 2 extra copies, ≥10 s** | **0.985** | 0.79 | 15/16 | **0** |
| audio alignment, 5 extra copies | 0.977 | 0.81 | 15/16 | 0 |

- Edges: median error 0.0 s (start) / -0.3 s (end); most within 1–2 s of the hand labels. Large misses (+27 to
  +76 s late starts, and one sysk1 break) are ads that were identical in all 6 downloads: baked in or the same
  campaign everywhere. More copies barely help, so 2 extra copies are enough.
- `gt_auto/`: labels for the 34 extra episodes (copies `ua1`, `ua2`; `fetch_variants.py`), 172 min. Plausibility
  vs E4B: E4B marks 83% of the auto-labelled time as ad (90–100% on most shows). Three episodes look broken (E4B
  agreement 23–60%, fragmented stretches): Casefile, Revisionist History, Hidden Brain. Lore and Acquired have no
  inserted ads. The labels cover ~53% of E4B's ad time (the rest is baked in or shared), so they measure recall,
  not precision.

### 14.7 Recall on 29 unseen shows (auto labels, `logs/eval_auto.txt`)
29 episodes (34 minus the 5 above), 145 min of inserted ads in 134 stretches. Classifier trained on all 13
human-labelled episodes with the dev-picked HMM knobs:

| Method | Inserted-ad time found | Stretches found (≥50% covered) |
|---|---|---|
| EmbeddingGemma + HMM (no LLM) | **0.865** | 103/134 |
| E4B copy6 | 0.917 | 112/134 |

For comparison, held-out recall on the human-labelled shows: EmbeddingGemma 0.884, E4B 0.973. The classifier
generalises to genuinely new shows with little loss; E4B stays ~5 points ahead in recall.

---

## 15. Session 3 (2026-10-02 afternoon): fixing disguised ads, without the LLM where possible

### 15.0 What the worst misses are (`logs/worst_cases.txt`)
Per hand-labelled break (54), ad seconds still playing — EmbeddingGemma+HMM: 24 breaks 0–2 s, 14 at 2–5 s, 2 at
5–10 s, 5 at 10–20 s, **6 at 20–40 s, 3 at 40–64 s**; E4B copy6: 46 breaks under 5 s, worst 26 s. Both wrongly skip
~4 min of content in 11.6 h. The bad cases are almost all the **first ad of a break when it does not sound like an
ad**: movie/TV trailers inside true-crime shows (the "Verity" trailer in 3 Dateline episodes, a "Sheriff Country"
promo opening Morbid) and skit-style reads (a LifeLock dialogue in Conan). The classifier catches the break only from
the second ad on.

### 15.1 Idea 1: off-topic features (`topic_feats.py`; one evaluation for everything: `eval_all.py`)
`eval_all.py` reports, for any feature or score set: dev/test F1 (leave-one-show-out, HMM knobs on dev), breaks with
≥20 s of ad left (of 54), and recall on the 29 unseen auto-labelled shows (model trained on all 13 episodes).

| Features (logistic regression + HMM) | Dev F1 | Test F1 | ≥20 s left | Unseen recall (stretches) |
|---|---|---|---|---|
| EmbeddingGemma (13.5) | 0.945 | 0.920 | 8 | 0.865 (103/134) |
| **+ episode-centred window vector** (`egemma_ctr`) | 0.942 | 0.924 | **5** | **0.902** (108/134) |
| + novelty scalars (cosine to episode mean, previous/next ~2 min) | 0.946 | 0.926 | 8 | 0.883 |
| + both | 0.947 | 0.926 | 5 | 0.899 |
| *E4B copy6, for reference* | 0.967 | 0.959 | 0 | 0.917 (112/134) |

Subtracting each episode's mean window embedding removes "what this show always sounds like", so a trailer in a crime
show stands out. Needs the whole transcript first (fine for downloaded episodes). False alarms stay at 0–1.

### 15.2 Idea 2: bidirectional GRU over the line sequence (`seq_model.py`)
Linear 2304→128, BiGRU, per-line output, random 60–200 line crops, human labels only (leave-one-show-out), HMM on top.

| Model (egemma_ctr features) | Dev F1 | Test F1 | ≥20 s left | Unseen recall |
|---|---|---|---|---|
| logistic regression (15.1) | 0.942 | 0.924 | 5 | 0.902 |
| GRU 64, 40 epochs | 0.918 | 0.916 | 7 | 0.909 |
| GRU 32, 15 epochs | 0.942 | 0.933 | 6 | 0.906 |

A tie with the logistic regression; 13 training episodes are too few for it to learn break structure. Worth
retrying with more labelled episodes (e.g. the auto labels as training data).

### 15.3 Idea 8: LLM yes/no probability per line (`llm_score.py`)
E2B Q4_K_M on the Vulkan GPU via llama-server: 40 transcript lines are read once (KV cache), then one question per
target line ("Is this line part of an advertisement...? Answer Yes or No."), score = P(Yes)/(P(Yes)+P(No)) from
the first token's top log-probabilities. ~190–220 s per audio hour on this GPU (slower than copy mode here: ~70
question tokens per line + per-request overhead). **Bug found and fixed:** top-k tokens include "Yes" and "yes";
a dict keyed by the lower-cased token let the rare variant overwrite the likely one and inverted the scores (line
AUC 0.19). Rerun pending.

---

## 16. Bulk LLM-labelled training data (owner: "500 hours, a hoard of Haiku subagents")
No API credits, so labelling is done by Claude Haiku subagents of this session reading transcript files.

- `labtools.py`: exports `lab/in/<ep>.txt` (numbered, timestamped lines) and imports the labellers'
  `lab/out/<ep>.json` (line ranges, type ad/self_promo) into `gt_<name>/` label files. `lab/INSTRUCTIONS.md`
  holds the labelling rules (same conventions as `gt_v2`: other-show/movie promos are ads, the show's own plugs and
  credits are self-promo). `lab_ready.py` hands out batches and tracks assignments (`lab/assigned.txt`).
- `bulk_shows.py`: 673 episodes from 342 shows (iTunes search over ~10 genre terms before reaching 521 h; every
  labelled/evaluation show excluded), 20–90 min each, up to 2 per show. Hosts: podtrac 233, pdst.fm 71,
  megaphone 42, acast 40, pscrb 38, clrtpod 28, mgln 21, libsyn 21, simplecast 18, anchor 15, buzzsprout 15, …
- `bulk.py` (daemon, resumable): downloads two copies (two User-Agents), transcribes one with Moonshine, aligns the
  audio of both (inserted-ad labels → `gt_auto_bulk/`), exports the labelling input, then deletes the audio.
  At most 8 episodes ahead of transcription; downloads pause below 4 GB free disk. Log: `logs/bulk.txt`.

### 16.1 Haiku calibration on the 13 human-labelled episodes (`logs/haiku_calibration.txt`)
4 Haiku agents, 3–4 episodes each (80–120K tokens and 2.5–5.5 min per agent). Haiku labels scored against `gt_v2`:

| Labeller | P | R | F1 | Breaks | FA | Ad left / content skipped per break |
|---|---|---|---|---|---|---|
| Claude Haiku (whole transcript, one read) | 0.927 | 0.955 | 0.941 | 53/54 | 1 | 4.6 s / 7.7 s |
| Gemma 4 E4B copy6 (on-device) | 0.957 | 0.969 | 0.963 | 54/54 | 3 | 3.1 s / 4.4 s |

Haiku is good but not better than E4B: looser edges, one missed break (daily2), over-marking in sysk1 and
planetmoney1 (P 0.77–0.81). Usable as training labels, ideally combined with the alignment labels.

### 16.2 Labeller choice: Gemini 3.5 Flash-Lite with the app's own prompt (`gemini_label.py`)
Haiku subagents were stopped (too much of the owner's Claude quota per episode; a 3-run vote test was abandoned
half way). `gemini_label.py` reproduces `AdSegmentIndexer.detectAdsInWindow` exactly (model, prompt, JSON schema,
20-min windows every 15 min, 5 s merge) on our Moonshine transcripts instead of Gemini's own transcription (saves
quota). Key from `~/.config/antennapod-test/gemini-api-key` (mode 600, never committed). Calibration on the 13
human-labelled episodes (`logs/gemini_calibration.txt`), 48 calls, ~3.5 min:

| Labeller | P | R | F1 | Breaks | FA | Ad left / content skipped per break |
|---|---|---|---|---|---|---|
| **Gemini 3.5 Flash-Lite, app prompt** | **0.987** | 0.965 | **0.976** | 53/54 | 0 | 3.6 s / **1.3 s** |
| Gemma 4 E4B copy6 (on-device) | 0.957 | 0.969 | 0.963 | 54/54 | 3 | 3.1 s / 4.4 s |
| Claude Haiku (subagent) | 0.927 | 0.955 | 0.941 | 53/54 | 1 | 4.6 s / 7.7 s |

~15 s and ~20K input tokens per episode. Runs as a daemon (`gemini_label.py --bulk`) next to `bulk.py`, labelling
each bulk episode as soon as its transcript exists (`gt_gemini/`); sleeps an hour when the daily quota runs out.

### 16.3 More (Gemini-labelled) data for the linear classifier (`train_bulk.py`, first 80 bulk episodes, 66 h)
EmbeddingGemma Q8_0 features (GPU), episode-centred, logistic regression + HMM; `eval_all.py` metrics.

| Trained on | Dev F1 | Test F1 | ≥20 s left | Unseen recall |
|---|---|---|---|---|
| 13 human-labelled episodes | **0.941** | **0.921** | **5** | **0.897** |
| human + 80 bulk (Gemini labels) | 0.910 | 0.896 | 11 | 0.887 |
| 80 bulk only | 0.895 | 0.876 | 12 | 0.867 |
| human + 80 bulk (Gemini ∪ alignment labels) | 0.924 | 0.919 | 10 | 0.869 |
| 80 bulk only (union labels) | 0.908 | 0.897 | 8 | 0.867 |

More machine-labelled data from other shows does not help the linear classifier (same as the E4B distillation in
13.7). Gemini covers only 82% of the alignment-detected inserted-ad time in the bulk episodes (399/536 stretches;
its recall on the 13 human-labelled episodes is 0.965), and bulk episodes have fewer ad lines (7% vs 13%). Next:
the sequence model, which looked data-limited (15.2), queued after the bulk transcription.

### 16.4 Why the session kept restarting: out-of-memory kills (owner spotted it)
`dmesg`: repeated `Out of memory: Killed process ... (python)` on the 5.9 GB VM, with swap 2+ GB full. The Moonshine
transcriber (sherpa-onnx / onnxruntime) alone peaks at **~4.6 GB** (610 MB after loading; memory grows with each
decoded segment until it plateaus; independent of thread count and of total audio length, 10 min ≈ 30 min). With
10 s instead of 20 s maximum speech segments (`MAX_SPEECH_S=10`) the peak is ~3.8 GB at the same speed. Every
experiment run next to it (GPU embedding server, training, evaluation) pushed the VM into the OOM killer, which also
took down the agent session and all its jobs.
Fixes: `bulk.py` runs the transcriber with `MAX_SPEECH_S=10` in its own systemd scope (`MemoryMax=4500M`,
`MemorySwapMax=1G`) with `oom_score_adj 1000`, so an overrun only kills that transcription; heavy experiments run
strictly one at a time after the transcription (`queue_after_bulk.sh`, log `logs/queue.txt`).

### 16.5 Fine-tuned small BERT on the Gemini-labelled data (`ft_bulk.py`)
bge-small-en-v1.5 (33M, BERT architecture), whole model fine-tuned 1 epoch on 80 bulk episodes only (28,035 lines,
Gemini ∪ alignment labels, no human labels; 3,972 s on 6 CPU threads with transcription stopped, peak ~3.5 GB RSS),
scored on the 13 human-labelled and 29 unseen episodes (none of their shows in training). `eval_all.py` metrics:

| Model | Labels | Dev F1 | Test F1 | ≥20 s left | Unseen recall | Line AP dev / test |
|---|---|---|---|---|---|---|
| **bge-small fine-tuned** | 80 bulk (Gemini) | 0.922 | **0.915** (23/23, 1 FA) | 7 | 0.885 | 0.967 / 0.948 |
| same, dynamic int8 | 80 bulk (Gemini) | 0.917 | 0.879 (23/23, 2 FA) | 9 | 0.886 | 0.967 / 0.940 |
| bge-small fine-tuned (13.7) | 35 shows (E4B) | 0.926 | 0.851 | – | – | – / 0.936 |
| EmbeddingGemma + LR | 80 bulk (Gemini) | 0.895 | 0.876 | 12 | 0.867 | |
| EmbeddingGemma + LR (best so far) | 13 human eps | 0.941 | 0.921 | 5 | 0.897 | |

Fine-tuning uses the larger machine-labelled set where a linear model on frozen embeddings could not, and Gemini
labels beat E4B labels as teacher. Without any human labels it comes within ~0.01 F1 of the human-trained
EmbeddingGemma classifier. Inference: 29 ms/line fp32, 14 ms/line dynamic int8 on this CPU (~6 s per audio hour).
Quick dynamic quantisation costs some held-out F1; static/QAT export for the phone should be tested.
