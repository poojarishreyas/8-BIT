import json, re, time, urllib.request
U="https://bar-gonna-answered-earrings.trycloudflare.com/v1/chat/completions"
K="IULAqgTEfJNwk0YiiBoCaSsMq8Vd_0G_WhD0cPyyoIc"
TOOLS=[
 {"name":"get_weather","description":"Current weather for a city","parameters":{"city":"string"}},
 {"name":"web_search","description":"Search the web for recent information","parameters":{"query":"string"}},
 {"name":"bus_route","description":"Find BMTC bus routes between two places in Bengaluru","parameters":{"from":"string","to":"string"}},
 {"name":"mandi_price","description":"Today's market (mandi) price of a crop in a district","parameters":{"crop":"string","district":"string"}},
 {"name":"set_reminder","description":"Set a reminder","parameters":{"text":"string","time":"string"}},
 {"name":"calculator","description":"Evaluate a math expression","parameters":{"expression":"string"}},
]
SYS=("You are an assistant that can use tools. Available tools (JSON):\n"+json.dumps(TOOLS,ensure_ascii=False)+
"\n\nIf a tool is needed, reply with ONLY:\n<tool_call>{\"name\": \"<tool>\", \"arguments\": {...}}</tool_call>\n"
"Rules:\n- You DO have these tools. Never say you lack a tool or cannot check live data; call the tool instead.\n"
"- Any question about weather, buses, crop prices, reminders, arithmetic, news, sports results or anything recent MUST be a tool call.\n"
"- Translate argument values to English (togari bele = toor dal, Kalburgi = Kalaburagi). Times as YYYY-MM-DD HH:MM or 'tomorrow 07:00'.\n"
"- Only chit-chat or general knowledge gets a normal answer, in the SAME language and script as the user.\n\n"
"Examples:\nUser: Bengaluru alli male barutta?\nAssistant: <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Bengaluru\"}}</tool_call>\n"
"User: ಶಿವಾಜಿನಗರದಿಂದ ಹೆಬ್ಬಾಳಕ್ಕೆ ಬಸ್?\nAssistant: <tool_call>{\"name\": \"bus_route\", \"arguments\": {\"from\": \"Shivajinagar\", \"to\": \"Hebbal\"}}</tool_call>\n"
"User: 23% of 4500?\nAssistant: <tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"0.23*4500\"}}</tool_call>\n"
"User: Sanje 6 ge milk tagolakke nenapisu\nAssistant: <tool_call>{\"name\": \"set_reminder\", \"arguments\": {\"text\": \"Buy milk\", \"time\": \"today 18:00\"}}</tool_call>\n"
"User: ನಮಸ್ಕಾರ!\nAssistant: ನಮಸ್ಕಾರ! ಹೇಳಿ, ಏನು ಸಹಾಯ ಬೇಕು?")
CASES=[
 ("en","What's the weather in Mysuru right now?","get_weather",{"city":"mysuru"}),
 ("kn","ಮೈಸೂರಿನಲ್ಲಿ ಈಗ ಹವಾಮಾನ ಹೇಗಿದೆ?","get_weather",{"city":"mysuru"}),
 ("kg","Majestic inda Whitefield ge bus yavudu?","bus_route",{"from":"majestic","to":"whitefield"}),
 ("kn","ಮಂಡ್ಯ ಜಿಲ್ಲೆಯಲ್ಲಿ ಇವತ್ತು ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?","mandi_price",{"crop":"tomato","district":"mandya"}),
 ("kg","Naale belige 7 ge doctor appointment reminder haaki","set_reminder",{}),
 ("en","What is 1845 * 37?","calculator",{}),
 ("kg","IPL 2026 final yaaru gellidru?","web_search",{}),
 ("kn","ಬೆಂಗಳೂರಿನಿಂದ ಹುಬ್ಬಳ್ಳಿಗೆ ಹೋಗುವ ರೈಲುಗಳ ಬಗ್ಗೆ ಇತ್ತೀಚಿನ ಸುದ್ದಿ ಹುಡುಕಿ","web_search",{}),
 ("kn","ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?",None,{}),
 ("kg","Kalburgi alli togari bele rate eshtu ide?","mandi_price",{"crop":"toor","district":"kalaburagi"}),
]
def chat(msgs):
    b={"model":"vajram-8b","messages":msgs,"max_tokens":200,"temperature":0}
    r=urllib.request.Request(U,json.dumps(b).encode(),{"Authorization":"Bearer "+K,"Content-Type":"application/json","User-Agent":"probe"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=120) as f: j=json.load(f)
    return j["choices"][0]["message"]["content"] or "", time.time()-t
def parse(txt):
    m=re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",txt,re.S) or re.search(r"(\{\s*\"name\".*\})",txt,re.S)
    if not m: return None,False
    try: return json.loads(m.group(1)),True
    except Exception: return None,"badjson"
res=[]
for lang,q,exp,args in CASES:
    out,dt=chat([{"role":"system","content":SYS},{"role":"user","content":q}])
    call,ok=parse(out)
    name=call.get("name") if call else None
    a=json.dumps(call.get("arguments",{}),ensure_ascii=False) if call else ""
    right=(name==exp)
    argok=all(v in a.lower() for v in args.values()) if call else (exp is None)
    res.append((lang,right,argok,ok))
    print(f"[{lang}] {'OK ' if right else 'BAD'} exp={exp} got={name} args={a[:90]} json={ok} {dt:.1f}s")
    if not call: print("     text:",out[:160].replace("\n"," "))
print("\nTOOL CHOICE: %d/%d  ARGS: %d/%d" % (sum(r[1] for r in res),len(res),sum(r[2] for r in res),len(res)))
# multi-turn: tool result -> final answer
msgs=[{"role":"system","content":SYS},{"role":"user","content":"ಮೈಸೂರಿನಲ್ಲಿ ಈಗ ಹವಾಮಾನ ಹೇಗಿದೆ?"},
      {"role":"assistant","content":"<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Mysuru\"}}</tool_call>"},
      {"role":"user","content":"<tool_response>{\"city\":\"Mysuru\",\"temp_c\":27,\"condition\":\"light rain\",\"humidity\":78}</tool_response>"}]
out,dt=chat(msgs); print("\nFOLLOW-UP (%.1fs): %s" % (dt,out[:300].replace("\n"," ")))
