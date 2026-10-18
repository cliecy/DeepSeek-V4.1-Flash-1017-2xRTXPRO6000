#!/usr/bin/env python3
"""Evaluation harness for the DeepSeek-V4.1-Flash-1017 EXL3 deployment (OpenAI-compatible /v1, stdlib + optional math_verify).

usage: eval1017.py --base URL --model NAME [--key K] --out DIR TASK [options]
tasks:
  speed     --ctx 2000,32000,... --conc 1,2,4 [--think both|on|off] [--gen 256]
            prefill: one uncached request per stream (max_tokens=1) -> prefill tok/s and TTFT
            decode : streamed reply on the now-warm prefix, ignore_eos, decode tok/s = (n-1)/(t_last-t_first) per stream
  needle    --ctx 32000,131072,524288,1040000 --depth 10,50,90
  gsm8k | math500 | humaneval | mmlupro | ceval | tools   [--think on|off] [--conc N] [--limit N] [--max-tokens N]
Every request/response is appended to DIR/<task>.jsonl; DIR/<task>.summary.json holds the score."""
import argparse, concurrent.futures as cf, glob, json, os, random, re, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True); ap.add_argument("--key", default="x")
ap.add_argument("--out", required=True); ap.add_argument("task")
ap.add_argument("--ctx", default="2000,32000,131072"); ap.add_argument("--conc", default="1")
ap.add_argument("--depth", default="10,50,90"); ap.add_argument("--think", default="off")
ap.add_argument("--gen", type=int, default=256); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=0); ap.add_argument("--data", default="samples")   # make_samples.py output
ap.add_argument("--corpus", default="/usr/lib/python3*/**/*.py")
A = ap.parse_args()
BASE = A.base.rstrip("/"); os.makedirs(A.out, exist_ok=True)
H = {"Content-Type": "application/json", "Authorization": f"Bearer {A.key}"}
LOCK = threading.Lock()


def log(task, rec):
    with LOCK, open(os.path.join(A.out, f"{task}.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def kw(think):
    return {"chat_template_kwargs": {"thinking": think}} if think is not None else {}


def chat(messages, max_tokens, think=False, retries=3, **extra):
    body = {"model": A.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw(think), **extra}
    for i in range(retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers=H)
            d = json.loads(urllib.request.urlopen(req, timeout=7200).read())
            return time.time() - t0, d
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e.read().decode()[:300] if hasattr(e, "read") else str(e)
            if i == retries - 1:
                raise RuntimeError(err)
            time.sleep(10)


def chat_stream(messages, max_tokens, think=False, ignore_eos=True):
    body = {"model": A.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos, **kw(think)}
    t0 = time.time(); first = last = None; usage = None; nchunks = 0
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")):
                    now = time.time(); first = first or now; last = now; nchunks += 1
    n = usage["completion_tokens"]
    return dict(ttft=first - t0, decode_tps=(n - 1) / (last - first) if last and last > first else 0.0, n=n,
                ptok=usage["prompt_tokens"], chunks=nchunks)


# ----------------------------------------------------------------------------------------------------------- corpus
_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        files = sorted(glob.glob(A.corpus, recursive=True))
        random.Random(7).shuffle(files)
        parts, n = [], 0
        for f in files:
            try:
                s = open(f, errors="ignore").read()
            except OSError:
                continue
            parts.append(f"# ==== {os.path.relpath(f, '/usr/lib')}\n{s}"); n += len(s)
            if n > 5_000_000:
                break
        _CORPUS = "\n".join(parts)
    return _CORPUS


CHARS_PER_TOK = 3.3   # measured on this tokenizer for Python source, refined by the first response


def doc_for(tokens, seed):
    c = corpus(); n = int(tokens * CHARS_PER_TOK); off = random.Random(seed).randrange(0, max(1, len(c) - n))
    s = c[off:off + n]
    while len(s) < n:
        s += c[: n - len(s)]
    return s


# ----------------------------------------------------------------------------------------------------------- speed
def speed():
    res = []
    for ctx in [int(x) for x in A.ctx.split(",")]:
        for conc in [int(x) for x in A.conc.split(",")]:
            for think in ([False, True] if A.think == "both" else [A.think == "on"]):
                seeds = [int(time.time() * 1000) % 10**9 + i for i in range(conc)]
                q = ("\n\nYou are reviewing the source above. Explain in detail what the code does, module by module."
                     if not think else "\n\nThink carefully, then explain in detail what the code above does, module by module.")
                msgs = [[{"role": "user", "content": f"[{s}]\n" + doc_for(ctx, s) + q}] for s in seeds]
                bar = threading.Barrier(conc)

                def one(i):
                    bar.wait(); t, d = chat(msgs[i], 1, think)
                    pf = dict(prefill_s=t, ptok=d["usage"]["prompt_tokens"])
                    bar.wait(); st = chat_stream(msgs[i], A.gen, think)
                    return {**pf, **st}
                with cf.ThreadPoolExecutor(conc) as ex:
                    rs = list(ex.map(one, range(conc)))
                rec = dict(ctx=ctx, conc=conc, think=think, ptok=rs[0]["ptok"],
                           prefill_tps_agg=sum(r["ptok"] for r in rs) / max(r["prefill_s"] for r in rs),
                           ttft_uncached_s=max(r["prefill_s"] for r in rs),
                           decode_tps_per_stream=sum(r["decode_tps"] for r in rs) / conc,
                           decode_tps_agg=sum(r["decode_tps"] for r in rs),
                           ttft_warm_s=max(r["ttft"] for r in rs), streams=rs)
                log("speed", rec); res.append(rec)
                print(f"[speed] ctx={rec['ptok']:>9,} conc={conc} think={think!s:5} | prefill {rec['prefill_tps_agg']:8,.0f} tok/s "
                      f"(TTFT uncached {rec['ttft_uncached_s']:6.1f}s) | decode {rec['decode_tps_per_stream']:6.1f}/stream "
                      f"{rec['decode_tps_agg']:6.1f} agg | warm TTFT {rec['ttft_warm_s']:.2f}s", flush=True)
    json.dump(res, open(os.path.join(A.out, "speed.summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- needle
def needle():
    res = []
    for ctx in [int(x) for x in A.ctx.split(",")]:
        for depth in [int(x) for x in A.depth.split(",")]:
            code = f"{random.Random(ctx * 100 + depth).randrange(10**5, 10**6)}"
            d = doc_for(ctx - 200, ctx + depth)
            cut = d.rfind("\n", 0, int(len(d) * depth / 100)) + 1
            needle_line = f"# NOTE: the secret deployment passphrase is CRANE-{code}. Remember it.\n"
            doc = d[:cut] + needle_line + d[cut:]
            msgs = [{"role": "user", "content": doc + "\n\nSomewhere in the source above there is a comment with a secret "
                     "deployment passphrase. What is it? Answer with the passphrase only."}]
            try:
                t, r = chat(msgs, 64, False)
                txt = r["choices"][0]["message"].get("content") or ""
                rec = dict(ctx=ctx, depth=depth, ptok=r["usage"]["prompt_tokens"], seconds=t, found=f"CRANE-{code}" in txt,
                           answer=txt.strip()[:100])
            except Exception as e:
                rec = dict(ctx=ctx, depth=depth, error=str(e)[:300], found=False)
            log("needle", rec); res.append(rec)
            print(f"[needle] ctx={rec.get('ptok', ctx):>9,} depth={depth}% found={rec['found']} "
                  f"{rec.get('seconds', 0):.0f}s {rec.get('answer', rec.get('error', ''))!r}", flush=True)
    json.dump(res, open(os.path.join(A.out, "needle.summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- quality
def load(name):
    rows = [json.loads(l) for l in open(os.path.join(A.data, f"{name}.jsonl"))]
    return rows[: A.limit] if A.limit else rows


def last_number(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"; depth -= s[k] == "}"
        if depth == 0:
            return s[j + 1:k]
    return None


def _strip_math(x):
    """Drop presentation-only parts: units/\\text wrappers, a leading 'var =', thousands commas, \\left/\\right."""
    x = re.sub(r"\\(?:text|textbf|mbox|mathrm)\{([^{}]*)\}", r"\1", x)
    x = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", "", x)
    x = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", x)
    x = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", x)
    x = re.sub(r"\^\{?\\circ\}?|\\%|\$", "", x)
    x = re.sub(r"\s*(?:inches|inch|cm|meters|units|degrees|dollars)(?:\^?\{?2\}?)?\s*$", "", x.strip())
    return x.strip()


def math_eq(pred, gold):
    """math_verify with signal timeouts disabled (they raise in worker threads), on raw and unit-stripped forms."""
    if pred is None:
        return False
    from math_verify import parse, verify
    for p, g in ((pred, gold), (_strip_math(pred), _strip_math(gold))):
        try:
            if verify(parse("$" + g + "$", parsing_timeout=None), parse("$" + p + "$", parsing_timeout=None), timeout_seconds=None):
                return True
        except Exception:
            pass
        mc = re.fullmatch(r"\(?([A-E])\)?", p.strip()) and re.fullmatch(r"\(?([A-E])\)?", g.strip())
        if mc or re.sub(r"\s", "", p) == re.sub(r"\s", "", g):
            if re.sub(r"[()\s]", "", p) == re.sub(r"[()\s]", "", g):
                return True
    return False


def run_code(code, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py"); open(p, "w").write(code)
        base = ["python3", "-I", p]
        sandbox = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
                   "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc/alternatives", "/etc/alternatives", "--tmpfs", "/tmp",
                   "--bind", d, d, "--proc", "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent", "--chdir", d]
        cmd = sandbox + base if shutil_which("bwrap") else base
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               preexec_fn=lambda: __import__("resource").setrlimit(__import__("resource").RLIMIT_AS, (4 << 30, 4 << 30)))
            return r.returncode == 0, (r.stderr or r.stdout)[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def shutil_which(x):
    import shutil
    return shutil.which(x)


LET = "ABCDEFGHIJ"


def choice_letter(s, n):
    s = s.strip()
    pats = [r"(?:answer|答案)\s*(?:is|是|为|:|：)?\s*\(?([A-J])\)?", r"\(([A-J])\)", r"\b([A-J])\b"]
    for p in pats:
        m = re.findall(p, s, re.I)
        m = [x.upper() for x in m if x.upper() in LET[:n]]
        if m:
            return m[-1]
    return None


def task_item(task, x, think):
    mt = A.max_tokens or (16384 if think else 2048)
    if task == "gsm8k":
        q = x["question"] + "\n\nSolve step by step. End with a line of the form 'Answer: <number>'."
        t, r = chat([{"role": "user", "content": q}], mt, think)
        out = r["choices"][0]["message"].get("content") or ""
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", out)
        pred = m[-1].replace(",", "") if m else last_number(out)
        ok = num_eq(pred, x["answer"])
    elif task == "math500":
        q = x["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        t, r = chat([{"role": "user", "content": q}], mt, think)
        out = r["choices"][0]["message"].get("content") or ""
        pred = boxed(out); ok = math_eq(pred, x["answer"])
    elif task == "humaneval":
        q = ("Complete the following Python function. Return the complete function (with any imports it needs) in a single "
             "```python code block and nothing else.\n\n```python\n" + x["prompt"] + "```")
        t, r = chat([{"role": "user", "content": q}], mt, think)
        out = r["choices"][0]["message"].get("content") or ""
        m = re.findall(r"```(?:python)?\n(.*?)```", out, re.S)
        code = m[-1] if m else out
        if f"def {x['entry_point']}" not in code:
            code = x["prompt"] + code
        ok, err = run_code(x["prompt"] + "\n" + code + "\n\n" + x["test"] + f"\n\ncheck({x['entry_point']})\n")
        pred = err if not ok else "pass"
    elif task in ("mmlupro", "ceval"):
        opts = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(x["options"]))
        if task == "ceval":
            q = f"以下是中国考试的单项选择题，请选出正确答案。\n\n{x['question']}\n{opts}\n\n请在最后一行用“答案：X”的格式给出选项字母。"
        else:
            q = (f"The following is a multiple choice question about {x['category']}.\n\n{x['question']}\n{opts}\n\n"
                 "Think step by step, then finish with 'The answer is (X)' where X is the letter of the correct option.")
        t, r = chat([{"role": "user", "content": q}], mt, think)
        out = r["choices"][0]["message"].get("content") or ""
        pred = choice_letter(out, len(x["options"])); ok = pred == x["answer"]
    msg = r["choices"][0]["message"]
    return dict(id=x["id"], ok=bool(ok), pred=pred, gold=x.get("answer"), seconds=round(t, 1),
                completion_tokens=r["usage"]["completion_tokens"], finish=r["choices"][0].get("finish_reason"),
                reasoning_chars=len(msg.get("reasoning_content") or msg.get("reasoning") or ""), output=out)


def quality(task):
    think = A.think == "on"
    rows = load(task); tag = f"{task}-think{'on' if think else 'off'}"
    done = {}
    p = os.path.join(A.out, f"{tag}.jsonl")
    if os.path.exists(p):   # resume
        for l in open(p):
            r = json.loads(l); done[str(r["id"])] = r
    todo = [x for x in rows if str(x["id"]) not in done]
    t0 = time.time()

    def go(x):
        try:
            rec = task_item(task, x, think)
        except Exception as e:
            rec = dict(id=x["id"], ok=False, error=str(e)[:300])
        log(tag, rec)
        return rec
    with cf.ThreadPoolExecutor(int(A.conc)) as ex:
        for i, rec in enumerate(ex.map(go, todo), 1):
            done[str(rec["id"])] = rec
            if i % 10 == 0:
                print(f"[{tag}] {len(done)}/{len(rows)} acc so far {sum(r['ok'] for r in done.values()) / len(done):.3f}", flush=True)
    recs = list(done.values())
    s = dict(task=task, think=think, n=len(recs), correct=sum(r["ok"] for r in recs), acc=sum(r["ok"] for r in recs) / len(recs),
             errors=sum("error" in r for r in recs), truncated=sum(r.get("finish") == "length" for r in recs),
             mean_completion_tokens=sum(r.get("completion_tokens", 0) for r in recs) / len(recs), wall_s=round(time.time() - t0))
    json.dump(s, open(os.path.join(A.out, f"{tag}.summary.json"), "w"), indent=1)
    print(f"[{tag}] DONE acc={s['acc']:.3f} ({s['correct']}/{s['n']}) truncated={s['truncated']} errors={s['errors']} "
          f"mean_tokens={s['mean_completion_tokens']:.0f} wall={s['wall_s']}s", flush=True)


# ----------------------------------------------------------------------------------------------------------- tools
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "IATA code"}, "destination": {"type": "string", "description": "IATA code"},
                                                      "date": {"type": "string", "description": "YYYY-MM-DD"}, "passengers": {"type": "integer"}}, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query on the sales database",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
TOOL_CASES = [
    ("What's the weather in Paris right now, in celsius?", "get_weather", lambda a: "paris" in a.get("city", "").lower() and a.get("unit", "celsius") == "celsius"),
    ("东京现在天气怎么样？用华氏度。", "get_weather", lambda a: "tokyo" in a.get("city", "").lower() or "东京" in a.get("city", "")),
    ("Find me flights from SFO to JFK on 2026-11-03 for 2 people.", "search_flights", lambda a: a.get("origin") == "SFO" and a.get("destination") == "JFK" and a.get("date") == "2026-11-03" and int(a.get("passengers", 0)) == 2),
    ("帮我查一下 2026年12月24日 从北京首都机场(PEK)飞上海虹桥(SHA)的航班。", "search_flights", lambda a: a.get("origin") == "PEK" and a.get("destination") == "SHA" and a.get("date") == "2026-12-24"),
    ("How many orders did we have in March 2026? The table is orders(id, created_at, total).", "run_sql", lambda a: "count" in a.get("query", "").lower() and "orders" in a.get("query", "").lower()),
    ("Email bob@example.com with subject 'Lunch' and tell him I'm running 10 minutes late.", "send_email", lambda a: a.get("to") == "bob@example.com" and "lunch" in a.get("subject", "").lower() and "10" in a.get("body", "")),
    ("What is 1234 * 5678 - 91011? Use the calculator.", "calculator", lambda a: all(t in a.get("expression", "").replace(" ", "") for t in ("1234", "5678", "91011"))),
    ("请用计算器算一下 (3.5 + 4.25) * 12 等于多少。", "calculator", lambda a: "3.5" in a.get("expression", "") and "12" in a.get("expression", "")),
    ("Tell me a one-sentence joke about cats.", None, None),
    ("Is it raining in London? Also email alice@example.org the answer with subject 'Weather'.", "get_weather", lambda a: "london" in a.get("city", "").lower()),
]


def tools():
    res = []
    for q, name, check in TOOL_CASES:
        try:
            t, r = chat([{"role": "user", "content": q}], 1024, A.think == "on", tools=TOOLS, tool_choice="auto")
            m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
            if name is None:
                ok = not calls and bool((m.get("content") or "").strip())
            else:
                c0 = calls[0]["function"] if calls else None
                args = json.loads(c0["arguments"]) if c0 else {}
                ok = bool(c0) and c0["name"] == name and bool(check(args))
            rec = dict(q=q, expect=name, ok=ok, calls=[c["function"] for c in calls], content=(m.get("content") or "")[:200], seconds=round(t, 1))
        except Exception as e:
            rec = dict(q=q, expect=name, ok=False, error=str(e)[:300])
        log("tools", rec); res.append(rec)
        print(f"[tools] {'PASS' if rec['ok'] else 'FAIL'} {q[:50]!r} -> {rec.get('calls', rec.get('error'))}", flush=True)
    s = dict(n=len(res), correct=sum(r["ok"] for r in res))
    json.dump(s, open(os.path.join(A.out, "tools.summary.json"), "w"))
    print(f"[tools] DONE {s['correct']}/{s['n']}")


def rescore_math():
    """Re-grade saved math500 outputs with the current math_eq (no new requests); rewrites the .jsonl and .summary.json."""
    for think in ("off", "on"):
        tag = f"math500-think{think}"; p = os.path.join(A.out, f"{tag}.jsonl")
        if not os.path.exists(p):
            continue
        recs = {}
        for l in open(p):
            r = json.loads(l); recs[str(r["id"])] = r
        for r in recs.values():
            r["ok"] = math_eq(boxed(r.get("output") or ""), r["gold"]); r["pred"] = boxed(r.get("output") or "")
        with open(p, "w") as f:
            for r in recs.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = json.load(open(os.path.join(A.out, f"{tag}.summary.json")))
        s.update(correct=sum(r["ok"] for r in recs.values()), acc=sum(r["ok"] for r in recs.values()) / len(recs), regraded=True)
        json.dump(s, open(os.path.join(A.out, f"{tag}.summary.json"), "w"), indent=1)
        print(f"[{tag}] regraded acc={s['acc']:.3f} ({s['correct']}/{s['n']})")
        for r in recs.values():
            if not r["ok"]:
                print(f"   miss {r['id']}: pred={boxed(r.get('output') or '')!r} gold={r['gold']!r} finish={r.get('finish')}")


if A.task == "speed":
    speed()
elif A.task == "needle":
    needle()
elif A.task == "tools":
    tools()
elif A.task == "rescore-math":
    rescore_math()
else:
    quality(A.task)
