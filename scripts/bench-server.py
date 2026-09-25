#!/usr/bin/env python3
"""Server benchmark / purity harness for the FA band WMMA patch (used for bench/results/linux).

Starts llama-server per run, sends one greedy /completion, records timings and tokens.

Environment:
  MODEL            GGUF to load (the results used Qwen3.8-27B Q4_K_M)
  UPSTREAM_SERVER  llama-server of a reference build (e.g. upstream llama.cpp)
  PATCHED_SERVER   llama-server of rdna-boosts + this patch
  CORPUS_DIR       a llama.cpp checkout; its docs/, tools/ and src/ files form the prompt
  OUT_DIR          where results go (default ./bench-out)
Usage: bench-server.py [depth ...]   (default: 20000 60000 110000)
"""
import json, os, subprocess, sys, time, urllib.request, glob

OUT = os.environ.get("OUT_DIR", os.path.abspath("bench-out"))
os.makedirs(OUT, exist_ok=True)

MODEL = os.environ["MODEL"]
BIN = {
    "upstream": os.environ["UPSTREAM_SERVER"],
    "rb":       os.environ["PATCHED_SERVER"],
}
CORPUS = os.environ["CORPUS_DIR"]
PORT = 8199
BASE = ["-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "131072",
        "-ub", "512", "-b", "2048", "--parallel", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "--no-webui"]
MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]


def http(path, body=None, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start(binary, env_band, mtp, log):
    env = dict(os.environ)
    env.pop("GGML_HIP_FA_BAND_WMMA", None)
    env.pop("GGML_HIP_FA_BAND_WMMA_SPLIT", None)
    if env_band:
        band, _, split = env_band.partition(":")
        env["GGML_HIP_FA_BAND_WMMA"] = band
        if split:
            env["GGML_HIP_FA_BAND_WMMA_SPLIT"] = split
    args = [binary] + BASE + (MTP if mtp else [])
    f = open(log, "w")
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError(f"server exited, see {log}")
        try:
            if http("/health", timeout=5).get("status") == "ok":
                return p
        except Exception:
            pass
        if time.time() - t0 > 900:
            p.kill(); raise RuntimeError("load timeout")
        time.sleep(2)


def stop(p):
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
    time.sleep(3)


def run(name, binary, env_band, mtp, prompt_tokens, n_predict=512):
    out = os.path.join(OUT, name + ".json")
    if os.path.exists(out):
        return json.load(open(out))
    p = start(binary, env_band, mtp, os.path.join(OUT, name + ".log"))
    try:
        r = http("/completion", {"prompt": prompt_tokens, "n_predict": n_predict, "temperature": 0,
                                 "top_k": 1, "seed": 1, "cache_prompt": False, "ignore_eos": True,
                                 "return_tokens": True})
    finally:
        stop(p)
    json.dump(r, open(out, "w"))
    return r


def tokenize_prompt(text):
    # tokenize once with the upstream server (same vocab for every build)
    p = start(BIN["upstream"], None, False, os.path.join(OUT, "tokenize.log"))
    try:
        toks = http("/tokenize", {"content": text, "add_special": False})["tokens"]
    finally:
        stop(p)
    return toks


def summary(name, r):
    t = r["timings"]
    acc = t.get("draft_n_accepted") or 0
    steps = t["predicted_n"] - acc
    return (f"{name:28s} prompt {t['prompt_n']:6d} @ {t['prompt_per_second']:7.1f} t/s | gen {t['predicted_n']} @ "
            f"{t['predicted_per_second']:6.2f} t/s | draft {acc}/{t.get('draft_n')} | "
            f"{t['predicted_ms']/max(steps,1):6.2f} ms/step ({steps} steps)")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


if __name__ == "__main__":
    tok_file = os.path.join(OUT, "prompt-tokens.json")
    if not os.path.exists(tok_file):
        files = sorted(glob.glob(os.path.join(CORPUS, "docs/**/*.md"), recursive=True))
        files += sorted(glob.glob(os.path.join(CORPUS, "tools/**/*.md"), recursive=True))
        files += sorted(glob.glob(os.path.join(CORPUS, "src/*.cpp")))
        text = ""
        for fn in files:
            text += f"\n\n===== full/{os.path.relpath(fn, CORPUS)} =====\n" + open(fn, errors="replace").read()
            if len(text) > 700_000:
                break
        json.dump(tokenize_prompt(text), open(tok_file, "w"))
    toks = json.load(open(tok_file))
    print("prompt tokens available:", len(toks), flush=True)

    depths = [int(d) for d in (sys.argv[1:] or ["20000", "60000", "110000"])]
    results = {}
    for d in depths:
        pt = toks[:d]
        for cfg, binary, band in (("upstream", BIN["upstream"], None), ("rb-stock", BIN["rb"], None), ("rb-D4", BIN["rb"], "4:64")):
            name = f"{cfg}-mtp-d{d}"
            results[name] = run(name, binary, band, True, pt)
            print(summary(name, results[name]), flush=True)
    # purity at the middle depth: patched and stock, MTP vs plain
    d = depths[len(depths) // 2]
    pt = toks[:d]
    for cfg, band in (("rb-D4", "4:64"), ("rb-stock", None)):
        name = f"{cfg}-plain-d{d}"
        results[name] = run(name, BIN["rb"], band, False, pt)
        print(summary(name, results[name]), flush=True)
        fd = first_diff(results[f"{cfg}-mtp-d{d}"]["tokens"], results[name]["tokens"])
        print(f"PURITY {cfg} d{d}: " + ("PASS (identical)" if fd is None else f"FAIL first diff at {fd}"), flush=True)
