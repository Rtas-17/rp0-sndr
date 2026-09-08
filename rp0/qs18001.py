import json, time, urllib.request
KEY=open("/root/qs2/api_key.txt").read().strip()
body=json.dumps({"model":"/root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4","prompt":"Count from one to twenty slowly:","max_tokens":60,"temperature":0}).encode()
req=urllib.request.Request("http://127.0.0.1:18001/v1/completions",data=body,headers={"Content-Type":"application/json","Authorization":"Bearer genesis-local"})
t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(req,timeout=120).read()); dt=time.perf_counter()-t0
c=d["usage"]["completion_tokens"]
print("C1 single-stream: %.1f t/s" % (c/dt))
