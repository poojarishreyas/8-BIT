"""Build Stage C (instruction + tool-call) data as chat jsonl: {"messages": [...], "tools": [...]?, "source": ...}.

Sources (all open, ungated):
  - ai4bharat/indic-align Dolly, Kannada-script column (kan_Knda). The romanized column (kan_Latn) has
    question and answer swapped, so it is not used.
  - ai4bharat/indic-align Dolly, English column (keeps English instruction following)
  - CohereLabs/aya_dataset, language == Kannada (human-written)
  - NousResearch/hermes-function-calling-v1 single-turn (English tool calls, many tool schemas)
  - Templated Kannada / Kanglish / English calls for OUR 8 agent tools (eval/agent/tools.py), with mock
    results and a short templated answer. These use different slot values from eval/agent/queries_example.jsonl,
    and any query that also appears in --exclude (the hand-written agent test set) is dropped.

Usage: python -m data.build_sft --out data/processed/sft/sft_v1.jsonl --exclude eval/agent/queries_v1.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from eval.agent.tools import TOOLS, call_mock


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def dolly(lang: str, n: int, rng) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open("datasets/ai4bharat/indic-align/indicalign-instruct/dolly/Dolly.parquet", "rb") as f:
        rows = pq.read_table(f, columns=[lang]).column(0).to_pylist()
    out = []
    for r in rows:
        if isinstance(r, str):  # older dumps stored the turns as a string
            try:
                r = ast.literal_eval(r)
            except (ValueError, SyntaxError):
                continue
        turns = r or []
        msgs = []
        for t in turns:
            if len(t) == 2 and t[0] and t[1]:
                msgs += [{"role": "user", "content": nfc(t[0])}, {"role": "assistant", "content": nfc(t[1])}]
        if msgs:
            out.append({"messages": msgs, "source": f"indic-align-dolly-{lang}"})
    rng.shuffle(out)
    return out[:n]


def aya_kannada() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open("datasets/CohereLabs/aya_dataset/data/train-00000-of-00001.parquet", "rb") as f:
        t = pq.read_table(f, columns=["language", "inputs", "targets"]).to_pylist()
    return [{"messages": [{"role": "user", "content": nfc(r["inputs"])}, {"role": "assistant", "content": nfc(r["targets"])}],
             "source": "aya-kannada"} for r in t if r["language"] == "Kannada"]


def hermes(n: int, rng) -> list[dict]:
    from huggingface_hub import hf_hub_download

    d = json.loads(Path(hf_hub_download("NousResearch/hermes-function-calling-v1", "func-calling-singleturn.json",
                                        repo_type="dataset")).read_text(encoding="utf-8"))
    out = []
    for x in d:
        c = {m["from"]: m["value"] for m in x["conversations"]}
        sysm = c["system"]
        i = sysm.rfind("<tools>")
        j = sysm.find("</tools>", i)
        try:
            tools = json.loads(sysm[i + len("<tools>"):j])
            calls = []
            for blk in c["gpt"].split("<tool_call>")[1:]:
                calls.append(json.loads(blk.split("</tool_call>")[0]))
        except (json.JSONDecodeError, ValueError):
            continue
        if not calls:
            continue
        out.append({"tools": tools, "source": "hermes-fc-singleturn", "messages": [
            {"role": "user", "content": c["human"]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": k["name"], "arguments": k.get("arguments", {})}} for k in calls]}]})
    rng.shuffle(out)
    return out[:n]


# ------------------------------------------------------------------ templated calls for our tools
CITIES = [("ಮಂಡ್ಯ", "Mandya", "mandya"), ("ಹಾಸನ", "Hassan", "hassan"), ("ಉಡುಪಿ", "Udupi", "udupi"),
          ("ಬೆಳಗಾವಿ", "Belagavi", "belagavi"), ("ಶಿವಮೊಗ್ಗ", "Shivamogga", "shivamogga"), ("ಕಲಬುರಗಿ", "Kalaburagi", "kalaburagi"),
          ("ತುಮಕೂರು", "Tumakuru", "tumkur"), ("ಮಂಗಳೂರು", "Mangaluru", "mangalore"), ("ದಾವಣಗೆರೆ", "Davanagere", "davangere")]
STOPS = [("ಮೆಜೆಸ್ಟಿಕ್", "Majestic"), ("ಕೆ.ಆರ್. ಮಾರುಕಟ್ಟೆ", "KR Market"), ("ಹೆಬ್ಬಾಳ", "Hebbal"), ("ಜಯನಗರ", "Jayanagar"),
         ("ಬನಶಂಕರಿ", "Banashankari"), ("ಮಾರತ್ತಹಳ್ಳಿ", "Marathahalli")]
ROUTES = ["335E", "201R", "G-3", "45G", "V-500CA", "KIA-8"]
CROPS = [("ಟೊಮೆಟೊ", "tomato"), ("ರಾಗಿ", "ragi"), ("ಅಡಿಕೆ", "arecanut"), ("ತೆಂಗಿನಕಾಯಿ", "coconut"), ("ಭತ್ತ", "paddy"), ("ಮೆಣಸಿನಕಾಯಿ", "chilli")]
MARKETS = [("ಕೋಲಾರ", "Kolar"), ("ಚಿತ್ರದುರ್ಗ", "Chitradurga"), ("ಶಿವಮೊಗ್ಗ", "Shivamogga"), ("ಹಾವೇರಿ", "Haveri")]
TOPICS = [("ಹಂಪಿ", "Hampi"), ("ಬಸವಣ್ಣ", "Basavanna"), ("ದ.ರಾ. ಬೇಂದ್ರೆ", "Bendre"), ("ಜೋಗ ಜಲಪಾತ", "Jog Falls"),
          ("ಕಿತ್ತೂರು ಚೆನ್ನಮ್ಮ", "Kittur Chennamma"), ("ಯಕ್ಷಗಾನ", "Yakshagana")]


def templated(rng, n_each: int) -> list[dict]:
    T = {t["function"]["name"]: t for t in TOOLS}
    out = []

    def ex(lang, q, name, args, answer):
        res = call_mock(name, args)
        out.append({"source": f"templated-{lang}", "tools": TOOLS, "messages": [
            {"role": "user", "content": nfc(q)},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]},
            {"role": "tool", "content": res},
            {"role": "assistant", "content": nfc(answer)}]})

    for _ in range(n_each):
        kn, en, lat = rng.choice(CITIES)
        day_kn, day_en, day_lat = rng.choice([("ಇವತ್ತು", "today", "ivattu"), ("ನಾಳೆ", "tomorrow", "naale")])
        ex("kn", rng.choice([f"{day_kn} {kn}ದಲ್ಲಿ ಹವಾಮಾನ ಹೇಗಿದೆ?", f"{kn} ಹವಾಮಾನ {day_kn} ಹೇಳಿ"]), "get_weather",
           {"city": en, "date": day_en}, f"{kn}: 24°C, ಮೋಡ ಕವಿದ ವಾತಾವರಣ, ಸ್ವಲ್ಪ ಮಳೆ ಸಾಧ್ಯತೆ.")
        ex("kanglish", rng.choice([f"{day_lat} {lat} alli weather hegide?", f"{lat} ge {day_lat} male barutta?"]),
           "get_weather", {"city": en, "date": day_en}, f"{en} nalli 24°C, cloudy, swalpa male chance ide.")
        skn, sen = rng.choice(STOPS)
        r = rng.choice(ROUTES)
        ex("kn", f"{skn} ನಿಲ್ದಾಣಕ್ಕೆ {r} ಬಸ್ ಯಾವಾಗ ಬರುತ್ತದೆ?", "bmtc_bus_eta", {"stop": sen, "route": r},
           f"{r} ಬಸ್ {skn} ನಿಲ್ದಾಣಕ್ಕೆ ಸುಮಾರು 7 ನಿಮಿಷದಲ್ಲಿ ಬರುತ್ತದೆ.")
        ex("kanglish", f"{sen} stop ge {r} bus eshtu hottige barutte?", "bmtc_bus_eta", {"stop": sen, "route": r},
           f"{r} bus {sen} ge innu 7 nimisha alli barutte.")
        ckn, cen = rng.choice(CROPS)
        mkn, men = rng.choice(MARKETS)
        ex("kn", f"{mkn} ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ {ckn} ಬೆಲೆ ಎಷ್ಟು?", "mandi_price", {"commodity": cen, "market": men},
           f"{mkn} ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ {ckn} ಬೆಲೆ ಕ್ವಿಂಟಾಲ್‌ಗೆ ₹1,850.")
        ex("kanglish", f"{men} market alli {cen} rate eshtu ivattu?", "mandi_price", {"commodity": cen, "market": men},
           f"{men} market alli {cen} quintal ge ₹1,850.")
        tkn, ten = rng.choice(TOPICS)
        ex("kn", f"{tkn} ಬಗ್ಗೆ ಸ್ವಲ್ಪ ಮಾಹಿತಿ ಕೊಡಿ", "kannada_wiki", {"topic": tkn}, f"{tkn} ಕುರಿತು ವಿಕಿಪೀಡಿಯ ಸಾರಾಂಶ ಇಲ್ಲಿದೆ.")
        ex("en", f"What's the weather in {en} {day_en}?", "get_weather", {"city": en, "date": day_en},
           f"{en}: 24°C and cloudy, with a small chance of rain.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/sft/sft_v1.jsonl")
    ap.add_argument("--n-dolly-kn", type=int, default=6000)
    ap.add_argument("--n-dolly-en", type=int, default=2000)
    ap.add_argument("--n-hermes", type=int, default=1500)
    ap.add_argument("--n-templated-rounds", type=int, default=80, help="each round makes 8 examples")
    ap.add_argument("--exclude", nargs="*", default=["eval/agent/queries_example.jsonl"],
                    help="agent test-set jsonl files whose queries must not appear in training")
    a = ap.parse_args()
    rng = random.Random(0)
    ex = dolly("kan_Knda", a.n_dolly_kn, rng) + dolly("eng_Latn", a.n_dolly_en, rng) + aya_kannada() + \
        hermes(a.n_hermes, rng) + templated(rng, a.n_templated_rounds)
    bad = set()
    for p in a.exclude:
        if Path(p).exists():
            bad |= {nfc(json.loads(l)["query"]) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}
    seen, keep = set(), []
    for e in ex:
        q = e["messages"][0]["content"]
        if q in bad:
            continue
        key = json.dumps(e["messages"], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            keep.append(e)
    rng.shuffle(keep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in keep) + "\n", encoding="utf-8")
    c = Counter(e["source"] for e in keep)
    (out.with_suffix(".stats.json")).write_text(json.dumps(dict(total=len(keep), by_source=c, excluded_queries=len(bad)), indent=1))
    print(out, len(keep), dict(c))


if __name__ == "__main__":
    main()
