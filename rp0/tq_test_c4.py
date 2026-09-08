import json, threading, time, urllib.request, urllib.error

KEY = "genesis-local"
API = "http://127.0.0.1:18001/v1/chat/completions"
personas = ["fantasy ranger", "cyberpunk netrunner", "gothic monster hunter", "space opera pilot"]
res = {}

def build(i):
    filler = "Detailed scene history, prior turns, world state: " * 780
    msgs = [{"role": "system", "content": f"You narrate a {personas[i]} campaign. " + filler},
            {"role": "user", "content": "What happens next in one paragraph?"}]
    return msgs

def turn(i):
    body = json.dumps({"model": "/root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4",
                       "messages": build(i), "max_tokens": 256, "temperature": 0.7}).encode()
    try:
        t0 = time.perf_counter()
        d = json.loads(urllib.request.urlopen(urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY}), timeout=600).read())
        dt = time.perf_counter() - t0
        ntok = d["usage"]["completion_tokens"]
        res[i] = (ntok - 1) / dt if dt > 0 else 0
    except urllib.error.HTTPError as e:
        errbody = e.read().decode(errors="replace")[:220]
        print(f"user {i}: HTTP {e.code}: {errbody}", flush=True)
        res[i] = 0
    except Exception as e:
        print(f"user {i}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
        res[i] = 0

ths = [threading.Thread(target=turn, args=(i,)) for i in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
print("RESULT c4-64K:", [f"{res.get(i,0):.1f}" for i in range(4)], flush=True)
