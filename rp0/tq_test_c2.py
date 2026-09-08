import json, time, urllib.request, threading

KEY = "genesis-local"
API = "http://127.0.0.1:18001/v1/chat/completions"
personas = ["fantasy ranger", "cyberpunk netrunner", "gothic monster hunter"]

def build(i):
    filler = "Detailed scene history, prior turns, world state: " * 780
    msgs = [{"role": "system", "content": f"You narrate a {personas[i]} campaign. " + filler},
            {"role": "user", "content": "The story continues. 2 paragraphs."}]
    body = json.dumps({"model": "/root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4", "messages": msgs, "max_tokens": 256, "temperature": 0.7}).encode()
    urllib.request.urlopen(urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY}), timeout=900).read()

t0 = time.perf_counter()
ths = [threading.Thread(target=build, args=(i,)) for i in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
print(f"3x64K contexts built in {time.perf_counter()-t0:.0f}s", flush=True)

def turn(i, res):
    msgs = [{"role": "system", "content": f"You narrate a {personas[i]} campaign."},
            {"role": "user", "content": "Continue the scene in one paragraph (150 words)."}]
    body = json.dumps({"model": "/root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4", "messages": msgs, "max_tokens": 160, "temperature": 0.7, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    first = None; last = None; ntok = 0
    for raw in urllib.request.urlopen(req, timeout=300):
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            ntok = d["usage"]["completion_tokens"]
        ch = d.get("choices") or []
        if ch and ch[0].get("delta", {}).get("content"):
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
    res[i] = ((ntok - 1) / (last - first)) if first and last and last > first else 0

res = {}
ths = [threading.Thread(target=turn, args=(i, res)) for i in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
print("RESULT c2-64K:", [f"{res[i]:.1f}" for i in range(2)], flush=True)
import subprocess
r = subprocess.run(["/venv/main/bin/python", "/root/qs18001.py"], capture_output=True, text=True)
print("C1:", r.stdout.strip().splitlines()[-1], flush=True)
print("TQ-DONE", flush=True)
