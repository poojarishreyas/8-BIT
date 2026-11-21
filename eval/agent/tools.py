"""Agent tool set (8 tools: 4 harness + 4 MCP-style local services) with deterministic mock results.

Schemas use the OpenAI/Qwen function format accepted by `apply_chat_template(tools=...)`.
Mocks return fixed JSON so scoring depends only on the model's calls, not on live services.
"""
from __future__ import annotations

import json


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = [
    fn("web_search", "Search the web and return result titles, URLs and snippets.", {"query": S}, ["query"]),
    fn("web_fetch", "Fetch a web page and return its text.", {"url": S}, ["url"]),
    fn("read_file", "Read a local text file.", {"path": S}, ["path"]),
    fn("ask_user_question", "Ask the user a clarifying question when required information is missing.",
       {"question": S}, ["question"]),
    fn("get_weather", "Current weather or forecast for a city in Karnataka/India.",
       {"city": S, "date": {"type": "string", "description": "today, tomorrow, or YYYY-MM-DD"}}, ["city"]),
    fn("bmtc_bus_eta", "Next BMTC bus arrivals at a Bengaluru bus stop.",
       {"stop": S, "route": {"type": "string", "description": "route number, e.g. 500D"}}, ["stop"]),
    fn("mandi_price", "Today's mandi (APMC) price for a crop in a market.",
       {"commodity": S, "market": S}, ["commodity"]),
    fn("kannada_wiki", "Look up a topic on Kannada Wikipedia and return the summary.", {"topic": S}, ["topic"]),
]
NAMES = [t["function"]["name"] for t in TOOLS]

MOCK = {
    "web_search": lambda a: {"results": [{"title": f"Result for {a.get('query', '')}", "url": "https://example.com/1",
                                          "snippet": "..."}]},
    "web_fetch": lambda a: {"text": "Page text (mock)."},
    "read_file": lambda a: {"text": "file contents (mock)"},
    "ask_user_question": lambda a: {"answer": "(user did not answer)"},
    "get_weather": lambda a: {"city": a.get("city"), "date": a.get("date", "today"), "temp_c": 24, "sky": "cloudy", "rain_mm": 2},
    "bmtc_bus_eta": lambda a: {"stop": a.get("stop"), "arrivals": [{"route": a.get("route", "500D"), "minutes": 7}]},
    "mandi_price": lambda a: {"commodity": a.get("commodity"), "market": a.get("market", "Bengaluru"),
                              "modal_price_rs_per_quintal": 1850},
    "kannada_wiki": lambda a: {"summary": f"{a.get('topic')} ಕುರಿತು ವಿಕಿಪೀಡಿಯ ಸಾರಾಂಶ (mock)."},
}


def call_mock(name: str, args: dict) -> str:
    f = MOCK.get(name)
    return json.dumps(f(args) if f else {"error": f"unknown tool {name}"}, ensure_ascii=False)


def load_extra(path: str | None) -> list[dict]:
    """Extra tool schemas (e.g. the remaining harness tools, to reach 52) from a JSON list file."""
    if not path:
        return []
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))
