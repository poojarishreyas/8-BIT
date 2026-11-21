"""Build tests/data/identity_corpus.jsonl: 1,000+ English / JSON / code / tool-call strings that must
tokenize to identical IDs under the original and the extended tokenizer.

Deterministic (seed 0); no network. English comes from Python's stdlib docstrings, code from stdlib
source, JSON / tool calls are generated. Regenerate with:
  python -m tests.build_identity_corpus
"""
from __future__ import annotations

import ast
import json
import random
import sysconfig
from pathlib import Path

OUT = Path(__file__).parent / "data" / "identity_corpus.jsonl"
STDLIB_FILES = ["json/__init__.py", "argparse.py", "textwrap.py", "csv.py", "dataclasses.py",
                "functools.py", "pathlib/__init__.py", "statistics.py", "string.py", "heapq.py"]

TOOLS = ["get_weather", "bmtc_bus_eta", "mandi_price", "web_search", "web_fetch", "read_file", "wiki_lookup",
         "ask_user_question", "todo_write"]
ARGS = {"city": ["Bengaluru", "Mysuru", "Hubballi", "New York"], "query": ["best dosa near me", "KSRTC timings",
        "tomato price today", "python asyncio tutorial"], "route": ["500D", "335E", "G-4"], "limit": [1, 5, 10],
        "unit": ["metric", "imperial"], "url": ["https://example.com/a?b=1&c=2", "http://kn.wikipedia.org/wiki/X"]}


def english_and_code(root: Path) -> tuple[list[str], list[str]]:
    eng, code = [], []
    for rel in STDLIB_FILES:
        p = root / rel
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                d = ast.get_docstring(node)
                if d:
                    eng += [s.strip() for s in d.split("\n\n") if 20 < len(s.strip()) < 400]
        lines = src.splitlines()
        for i in range(0, len(lines) - 6, 7):
            chunk = "\n".join(lines[i:i + 6])
            if chunk.strip():
                code.append(chunk)
    return eng, code


def tool_calls(rng: random.Random, n: int) -> tuple[list[str], list[str]]:
    js, tc = [], []
    for _ in range(n):
        keys = rng.sample(list(ARGS), rng.randint(1, 3))
        args = {k: rng.choice(ARGS[k]) for k in keys}
        name = rng.choice(TOOLS)
        js.append(json.dumps(args, indent=rng.choice([None, 2])))
        call = {"name": name, "arguments": args}
        tc.append(rng.choice([
            f"<tool_call>\n{json.dumps(call)}\n</tool_call>",
            json.dumps({"tool_calls": [{"id": f"call_{rng.randrange(10**6)}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}]}),
            f'{{"role": "tool", "name": "{name}", "content": {json.dumps(json.dumps(args))}}}',
        ]))
    return js, tc


def main() -> None:
    rng = random.Random(0)
    eng, code = english_and_code(Path(sysconfig.get_paths()["stdlib"]))
    js, tc = tool_calls(rng, 300)
    misc = ["", " ", "\n\n", "  leading", "trailing  ", "tab\tsep", "naïve café résumé", "Ünïcödé — “quotes” ‘x’",
            "12345678901234567890", "3.14159 1e-9 0xFF", "email@test.com", "C:\\path\\to\\file.txt", "<b>html</b> &amp;",
            "emoji 😀👍🏽 ok", "math: x² + y² = z²", "hello\r\nworld"]
    rows = ([("english", s) for s in eng[:500]] + [("code", s) for s in code[:400]] +
            [("json", s) for s in js] + [("tool_call", s) for s in tc] + [("misc", s) for s in misc])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for kind, s in rows:
            f.write(json.dumps({"kind": kind, "text": s}, ensure_ascii=False) + "\n")
    from collections import Counter

    print(OUT, len(rows), Counter(k for k, _ in rows))


if __name__ == "__main__":
    main()
