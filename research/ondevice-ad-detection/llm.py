"""Thin wrappers so the classifier can run on LiteRT-LM (the runtime Android/AICore-style apps use)
or on llama.cpp's server (for GGUF models). Both use greedy decoding (top_k=1), like a deterministic
Prompt API config, and enforce the Prompt API budget: <4000 input tokens, <=256 output tokens."""
import json, os, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_INPUT_TOKENS = 4000
MAX_OUTPUT_TOKENS = 256


class LiteRT:
    def __init__(self, path, threads=6):
        import litert_lm
        self.m = litert_lm
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        backend = litert_lm.Backend.GPU() if os.environ.get("LITERT_BACKEND", "cpu") == "gpu" else litert_lm.Backend.CPU()
        self.engine = litert_lm.Engine(path, backend=backend, max_num_tokens=4096 + 512,
                                       cache_dir=os.path.join(ROOT, ".litert-cache"))
        self.name = path.rsplit("/", 1)[-1]
        self.stats = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0}
        self.sampling = None

    def count(self, text):
        return len(self.engine.tokenize(text))

    def generate(self, prompt, max_out=MAX_OUTPUT_TOKENS, system=None):
        n = self.count(prompt) + (self.count(system) if system else 0)
        assert n < MAX_INPUT_TOKENS, f"prompt too long: {n}"
        t0 = time.time()
        conv = self.engine.create_conversation(
            sampler_config=(self.m.SamplerConfig(top_k=40, temperature=self.sampling[0], seed=self.sampling[1])
                            if self.sampling else self.m.SamplerConfig(top_k=1, temperature=0.0, seed=1)),
            thinking_config=(self.m.ThinkingConfig(enable_thinking=True, thinking_token_budget=self.thinking)
                             if getattr(self, "thinking", 0) else self.m.ThinkingConfig(enable_thinking=False)),
            system_message=system, max_output_tokens=max_out)
        try:
            resp = conv.send_message(prompt, max_output_tokens=max_out + getattr(self, "thinking", 0))
        finally:
            conv.close()
        text = "".join(c.get("text", "") for c in resp.get("content", [])
                       if isinstance(c, dict) and c.get("type", "text") == "text")
        self.last_response = resp
        thought = (resp.get("channels") or {}).get("thought", "")
        if thought:
            self.stats["think_tokens"] = self.stats.get("think_tokens", 0) + self.count(thought)
        if os.environ.get("DUMP_PROMPTS"):
            with open(os.environ["DUMP_PROMPTS"], "a") as f:
                f.write(json.dumps({"system": system, "prompt": prompt, "out": text,
                                    "thought": (resp.get("channels") or {}).get("thought", "")}) + "\n")
        self.stats["calls"] += 1
        self.stats["in_tokens"] += n
        self.stats["out_tokens"] += self.count(text) if text else 0
        self.stats["seconds"] += time.time() - t0
        return text


class LlamaServer:
    def __init__(self, url, name):
        self.url = url.rstrip("/")
        self.name = name
        self.stats = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "seconds": 0.0}
        self.sampling = None

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=3600).read())

    def count(self, text):
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def generate(self, prompt, max_out=MAX_OUTPUT_TOKENS, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t0 = time.time()
        # Gemma 4's llama.cpp chat template thinks by default and spends the whole output budget on it.
        r = self._post("/v1/chat/completions", {"messages": msgs, "max_tokens": max_out,
                                                "chat_template_kwargs": {"enable_thinking": False},
                                                "temperature": self.sampling[0] if self.sampling else 0,
                                                "top_k": 40 if self.sampling else 1,
                                                "seed": self.sampling[1] if self.sampling else 1})
        n = r["usage"]["prompt_tokens"]
        assert n < MAX_INPUT_TOKENS + 50, f"prompt too long: {n}"
        self.stats["calls"] += 1
        self.stats["in_tokens"] += n
        self.stats["out_tokens"] += r["usage"]["completion_tokens"]
        self.stats["seconds"] += time.time() - t0
        return r["choices"][0]["message"]["content"] or ""


def load(spec):
    if spec.endswith(".litertlm"):
        return LiteRT(spec)
    url, name = spec.split("#", 1)
    return LlamaServer(url, name)
