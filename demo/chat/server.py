"""Local server for the demo chatbot.

- Serves index.html
- /api/chat   relays chat requests to the hosted model (adds the key, streams tokens)
- /api/tool   runs the agent's tools: weather, Wikipedia, page reader, calculator, date/time
- /api/health checks the model is reachable

Run:  python server.py      then open http://127.0.0.1:8787
Change the endpoint/key/model below, or set CHAT_BASE_URL / CHAT_API_KEY / CHAT_MODEL.
Standard library only, no installs needed.
"""
import ast
import datetime
import html
import json
import math
import operator
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_URL = os.environ.get("CHAT_BASE_URL", "https://tattoo-durham-titanium-belong.trycloudflare.com/v1").rstrip("/")
API_KEY = os.environ.get("CHAT_API_KEY", "IULAqgTEfJNwk0YiiBoCaSsMq8Vd_0G_WhD0cPyyoIc")
MODEL = os.environ.get("CHAT_MODEL", "vajram-8b")
PORT = int(os.environ.get("PORT", "8787"))
HERE = os.path.dirname(os.path.abspath(__file__))
UA = "ArbokDemo/1.0 (hackathon demo)"

MODEL_HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json", "User-Agent": UA}


# ----------------------------------------------------------------- tools
def get_json(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def q(s):
    return urllib.parse.quote(str(s))


WMO = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
       61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
       71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers", 81: "Showers",
       82: "Violent showers", 85: "Snow showers", 86: "Snow showers", 95: "Thunderstorm",
       96: "Thunderstorm with hail", 99: "Thunderstorm with hail"}


def tool_get_weather(args):
    city = str(args.get("city", "")).strip()
    if not city:
        raise ValueError("city is required")
    g = get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&name=" + q(city))
    if not g.get("results"):
        return {"error": "city not found: " + city}
    p = g["results"][0]
    w = get_json(
        "https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=2"
        % (p["latitude"], p["longitude"]))
    c, d = w["current"], w["daily"]
    return {"city": p["name"], "region": p.get("admin1"), "country": p.get("country"),
            "temp_c": c["temperature_2m"], "feels_like_c": c["apparent_temperature"],
            "humidity_pct": c["relative_humidity_2m"], "wind_kmh": c["wind_speed_10m"],
            "condition": WMO.get(c["weather_code"], "Unknown"),
            "today_max_c": d["temperature_2m_max"][0], "today_min_c": d["temperature_2m_min"][0],
            "rain_chance_pct": d["precipitation_probability_max"][0],
            "tomorrow_max_c": d["temperature_2m_max"][1],
            "tomorrow_rain_chance_pct": d["precipitation_probability_max"][1],
            "source": "Open-Meteo"}


def strip_html(s):
    s = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def tool_search_wikipedia(args):
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    lang = args.get("lang", "")
    if lang not in ("en", "kn"):
        lang = "kn" if re.search(r"[ಀ-೿]", query) else "en"
    base = "https://%s.wikipedia.org" % lang
    s = get_json(base + "/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch=" + q(query))
    hits = s.get("query", {}).get("search", [])
    out = []
    for i, h in enumerate(hits[:3]):
        title = h["title"]
        item = {"title": title, "url": base + "/wiki/" + q(title.replace(" ", "_")),
                "summary": strip_html(h.get("snippet", ""))}
        if i < 2:
            try:
                sm = get_json(base + "/api/rest_v1/page/summary/" + q(title.replace(" ", "_")))
                item["summary"] = (sm.get("extract") or item["summary"])[:900]
            except Exception:  # noqa: BLE001
                pass
        out.append(item)
    return {"query": query, "lang": lang, "results": out, "source": "Wikipedia"}


def tool_fetch_page(args):
    url = str(args.get("url", "")).strip()
    if not re.match(r"^https?://", url):
        raise ValueError("url must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read(600_000)
        charset = r.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, "replace")
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    body = strip_html(text)
    return {"url": url, "title": strip_html(m.group(1)) if m else "", "text": body[:3500], "chars": len(body)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUN = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp", "floor", "ceil")}
_FUN.update(abs=abs, round=round, min=min, max=max)
_CONST = {"pi": math.pi, "e": math.e}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN:
        return _FUN[n.func.id](*[_ev(a) for a in n.args])
    if isinstance(n, ast.Name) and n.id in _CONST:
        return _CONST[n.id]
    raise ValueError("unsupported expression")


def tool_calculator(args):
    expr = str(args.get("expression", "")).strip()
    clean = expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "").replace("₹", "")
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", clean)
    clean = re.sub(r"(?i)\bof\b", "*", clean)
    val = _ev(ast.parse(clean, mode="eval").body)
    if isinstance(val, float):
        val = round(val, 10)
        if val.is_integer():
            val = int(val)
    return {"expression": expr, "result": val}


def tool_get_datetime(_args):
    now = datetime.datetime.now()
    return {"local": now.strftime("%A, %d %B %Y, %I:%M %p"), "iso": now.isoformat(timespec="seconds")}


TOOLS = {"get_weather": tool_get_weather, "search_wikipedia": tool_search_wikipedia,
         "fetch_page": tool_fetch_page, "calculator": tool_calculator, "get_datetime": tool_get_datetime}


# ----------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/metrics", "/metrics.html"):
            page = "metrics.html" if path.startswith("/metrics") else "index.html"
            with open(os.path.join(HERE, page), "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/health":
            t0 = time.time()
            try:
                req = urllib.request.Request(BASE_URL + "/models", headers=MODEL_HEADERS)
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read()
                self._json(200, {"ok": True, "ms": int((time.time() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001
                self._json(200, {"ok": False, "error": str(e)[:200]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "bad json"})
            return
        if path == "/api/tool":
            name, args = body.get("name"), body.get("args") or {}
            t0 = time.time()
            if name not in TOOLS:
                self._json(200, {"ok": False, "error": "unknown tool: %s" % name})
                return
            try:
                res = TOOLS[name](args if isinstance(args, dict) else {})
                self._json(200, {"ok": True, "result": res, "ms": int((time.time() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001
                self._json(200, {"ok": False, "error": str(e)[:300], "ms": int((time.time() - t0) * 1000)})
            return
        if path != "/api/chat":
            self._json(404, {"error": "not found"})
            return
        body["model"] = MODEL
        req = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
                                     headers=dict(MODEL_HEADERS, Accept="text/event-stream"), method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            self._json(e.code, {"error": e.read().decode("utf-8", "replace")[:500]})
            return
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": "model unreachable: " + str(e)[:300]})
            return
        self.send_response(200)
        self.send_header("Content-Type", r.headers.get("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            read = getattr(r, "read1", None) or r.read
            while True:
                chunk = read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            r.close()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("=" * 52)
    print("  Demo chat running at  " + url)
    print("  Model endpoint:       " + BASE_URL)
    print("  Press Ctrl+C to stop")
    print("=" * 52)
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
