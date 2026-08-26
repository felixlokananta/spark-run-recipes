#!/usr/bin/env python3
"""Needle-in-a-haystack for the vLLM endpoint.

Tests whether uncalibrated fp8 KV (q/k/v scale 1.0) degrades long-range
retrieval.  Uses the server's own /tokenize and /detokenize so needle depth is
measured in real model tokens, not characters.

Sequential by design: concurrent long prefills starve each other's decode.
"""
import json, sys, time, urllib.request, argparse

BASE = "http://localhost:8000"
MODEL = "qwen3.8-27b-uncensored"
CORPUS = "/home/felix/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt"

# One distinct code per cell so a lucky guess, a prefix-cache hit, or
# cross-cell memorisation cannot produce a false pass.
CODES = ["7391-KESTREL-42", "5628-OBSIDIAN-17", "9043-MARLIN-88", "2175-FULCRUM-63",
         "6802-LANTERN-29", "4417-QUARRY-51", "8360-VESPER-74", "1594-THICKET-36",
         "3728-HALYARD-95", "7043-CINDER-08", "5981-PLINTH-47", "2604-BRAMBLE-72",
         "9137-SEXTANT-13", "4850-TALLOW-69", "6293-WICKER-24", "8716-GANTRY-57",
         "1462-SORREL-81", "3079-KEYSTONE-45", "5534-MERIDIAN-90", "7168-ASHLAR-32"]

NEEDLE = ("\n\nImportant administrative note: the secret access code for the "
          "Vestibule Archive is {code}. Do not lose it.\n\n")
QUESTION = ("\n\nQuestion: What is the secret access code for the Vestibule "
            "Archive? Reply with only the code and nothing else.")


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[8192, 32768, 65536, 131072])
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default="niah-results.json")
    args = ap.parse_args()

    text = open(args.corpus, encoding="utf-8", errors="ignore").read()
    ids = post("/tokenize", {"model": MODEL, "prompt": text})["tokens"]
    print(f"corpus: {len(ids):,} tokens", flush=True)

    need = max(args.contexts)
    if len(ids) < need:
        reps = -(-need // len(ids))
        ids = (ids * reps)[:need]
        print(f"corpus tiled x{reps} -> {len(ids):,} tokens", flush=True)

    results, cell = [], 0
    print(f"\n{'ctx':>8} {'depth':>6} {'real':>8} {'ok':>4} {'lat':>7}  answer", flush=True)
    for ctx in args.contexts:
        for depth in args.depths:
            code = CODES[cell % len(CODES)]
            cell += 1
            cut = int(len(ids[:ctx]) * depth / 100)
            head = post("/detokenize", {"model": MODEL, "tokens": ids[:cut]})["prompt"]
            tail = post("/detokenize", {"model": MODEL, "tokens": ids[cut:ctx]})["prompt"]
            body = head + NEEDLE.format(code=code) + tail + QUESTION
            real = len(post("/tokenize", {"model": MODEL, "prompt": body})["tokens"])

            t0 = time.time()
            r = post("/v1/chat/completions", {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content":
                     "Answer using only information found in the document provided."},
                    {"role": "user", "content": body}],
                "temperature": 0, "max_tokens": 768})
            lat = time.time() - t0
            msg = r["choices"][0]["message"]
            ans = (msg.get("content") or "").strip()
            think = (msg.get("reasoning_content") or "")
            ok = code in ans
            if not ok and not ans and code in think:
                ok = None  # only found inside reasoning, answer truncated
            results.append({"ctx": ctx, "depth": depth, "real_tokens": real,
                            "code": code, "ok": bool(ok), "answer": ans[:200],
                            "reasoning_tokens": len(think.split()),
                            "finish": r["choices"][0].get("finish_reason"),
                            "latency_s": round(lat, 1),
                            "usage": r.get("usage")})
            mark = "PASS" if ok else ("TRUNC" if ok is None else "FAIL")
            print(f"{ctx:>8} {depth:>5}% {real:>8,} {mark:>5} {lat:>6.1f}s  "
                  f"{ans[:60]!r}", flush=True)
            json.dump(results, open(args.out, "w"), indent=1)

    print(f"\n{'ctx':>8} " + " ".join(f"{d:>5}%" for d in args.depths))
    for ctx in args.contexts:
        row = [next((x for x in results if x["ctx"] == ctx and x["depth"] == d), None)
               for d in args.depths]
        print(f"{ctx:>8} " + " ".join(
            ("  ok  " if c and c["ok"] else " FAIL ") if c else "  --  " for c in row))
    n = sum(1 for x in results if x["ok"])
    print(f"\n{n}/{len(results)} passed -> {args.out}")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
