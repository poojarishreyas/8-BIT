import json, time, urllib.request
U="https://bar-gonna-answered-earrings.trycloudflare.com/v1/chat/completions"
K="IULAqgTEfJNwk0YiiBoCaSsMq8Vd_0G_WhD0cPyyoIc"
TOOLS=[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
def call(msgs, tools=None, tool_choice=None, max_tokens=300):
    b={"model":"vajram-8b","messages":msgs,"max_tokens":max_tokens,"temperature":0}
    if tools: b["tools"]=tools
    if tool_choice: b["tool_choice"]=tool_choice
    r=urllib.request.Request(U,json.dumps(b).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json","User-Agent":"probe"})
    t=time.time()
    try:
        with urllib.request.urlopen(r,timeout=120) as f: j=json.load(f)
    except urllib.error.HTTPError as e: return {"HTTP":e.code,"body":e.read().decode()[:400]}, time.time()-t
    return j, time.time()-t
j,dt=call([{"role":"user","content":"What's the weather in Mysuru right now?"}],TOOLS,"auto")
print("AUTO:",round(dt,1),"s"); print(json.dumps(j,ensure_ascii=False)[:1200])
j,dt=call([{"role":"user","content":"What's the weather in Mysuru right now?"}],TOOLS,"required")
print("\nREQUIRED:",round(dt,1),"s"); print(json.dumps(j,ensure_ascii=False)[:800])
