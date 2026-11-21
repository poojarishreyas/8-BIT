# end-to-end: real model + real tools through server.py, same agent loop as the UI
import json, re, time, urllib.request, os, subprocess, sys
os.environ["NO_BROWSER"]="1"
here=os.path.dirname(os.path.abspath(__file__))
srv=subprocess.Popen([sys.executable, os.path.join(here,"..","chat","server.py")]); time.sleep(1.5)
SYS=open(os.path.join(here,"_sysprompt.txt"),encoding="utf-8").read()
def post(path, body):
    r=urllib.request.Request("http://127.0.0.1:8787"+path, json.dumps(body).encode(), {"Content-Type":"application/json"})
    return urllib.request.urlopen(r, timeout=180)
def chat(msgs):
    raw=""
    with post("/api/chat", {"messages":msgs,"stream":True,"max_tokens":600,"temperature":0.1}) as f:
        for line in f:
            line=line.decode().strip()
            if line.startswith("data:") and line[5:].strip()!="[DONE]":
                j=json.loads(line[5:]); d=j.get("choices") and j["choices"][0]["delta"].get("content")
                if d: raw+=d
    return raw
Q=["ಕುವೆಂಪು ಅವರ ಬಗ್ಗೆ ಹೇಳಿ","18% GST on ₹2,450 eshtu?","Mysuru weather today"]
try:
  for q in Q:
    api=[{"role":"user","content":q}]; t0=time.time(); steps=[]
    for it in range(5):
        out=chat([{"role":"system","content":SYS}]+api)
        calls=[json.loads(m) for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",out,re.S)]
        if not calls: break
        api.append({"role":"assistant","content":out}); resp=""
        for c in calls:
            name,args=c["name"],c.get("arguments",{})
            if name in ("set_reminder","get_datetime"): res={"ok":True,"result":{"saved":True,**args}}
            else: res=json.load(post("/api/tool",{"name":name,"args":args}))
            steps.append(f"{name}({json.dumps(args,ensure_ascii=False)})->{'ok' if res.get('ok') else 'ERR'}")
            resp+=f'<tool_response>{{"name": "{name}", "result": {json.dumps(res.get("result",res),ensure_ascii=False)[:3500]}}}</tool_response>\n'
        api.append({"role":"user","content":resp.strip()})
    print(f"\nQ: {q}  ({time.time()-t0:.1f}s)\n  steps: {steps or 'none'}\n  A: {out.strip()[:260]}")
finally:
  srv.terminate()
