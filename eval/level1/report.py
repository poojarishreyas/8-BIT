"""Part B report from results/level1_matrix.csv (one command regenerates everything):

  results/kanglish_tax.csv + .md      English vs Kannada vs romanized (IndicXlit) vs hand-written Kanglish
  results/charts/1_parity.png         parity vs English, baselines vs bigo_final (FLORES devtest)
  results/charts/2_ablation.png       Kannada tokens/word as layers are added (16K new tokens)
  results/charts/3_vocab_sweep.png    parity vs number of new tokens, ours vs standard BPE extension
  results/charts/4_kanglish_tax.png   grouped bars per tokenizer x form
  results/charts/5_token_viz.png      same sentence under Qwen3 vs ours, token boundaries coloured
  results/failures.md + failures.csv  where bigo_final is no better / worse, with counts per category

Usage: python -m eval.level1.report [--final bigo_final] [--split flores_devtest]
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
CH = RES / "charts"
# reference palette (dataviz skill, light mode): ours = series-1 blue, standard extension = series-2 orange,
# other baselines = recessive neutral. Text never wears series colour.
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2"
NICE = {"cl100k_base": "GPT-4 (cl100k)", "o200k_base": "GPT-4o (o200k)", "llama3": "Llama-3", "gemma3": "Gemma-3",
        "sarvam1": "Sarvam-1", "indictok256k": "indictok-256k", "qwen3": "Qwen3", "bigo_final": "Ours (bigo_final)"}


def style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=12)
    ax.set_title(title, loc="left", fontsize=16, color=INK, pad=14, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=12)


def nice(name: str, m: pd.DataFrame) -> str:
    if name in NICE:
        return NICE[name]
    nv = m.loc[m.tokenizer == name, "new_vocab"].dropna()
    return name


def get(m, tok, split, form, col):
    r = m[(m.tokenizer == tok) & (m.split == split) & (m.form == form)]
    return float(r[col].iloc[0]) if len(r) and str(r[col].iloc[0]) not in ("", "nan") else None


# ------------------------------------------------------------------ B2 Kanglish tax
def kanglish_tax(m: pd.DataFrame, final: str, split: str) -> pd.DataFrame:
    toks = [t for t in ["cl100k_base", "o200k_base", "llama3", "gemma3", "sarvam1", "indictok256k", "qwen3",
                        "std_16k", final] if t in set(m.tokenizer)]
    rows = []
    for t in toks:
        r = dict(tokenizer=t, new_vocab=m.loc[m.tokenizer == t, "new_vocab"].iloc[0],
                 eng_tokens_per_sent=get(m, t, split, "eng_Latn", "tokens_per_sent"),
                 kan_Knda_parity=get(m, t, split, "kan_Knda", "parity"),
                 kan_Latn_xlit_parity=get(m, t, split, "kan_Latn_xlit", "parity"),
                 kanglish_parity=get(m, t, "kanglish_handwritten", "kanglish", "parity"),
                 kan_Knda_fertility=get(m, t, split, "kan_Knda", "fertility"),
                 kan_Latn_xlit_fertility=get(m, t, split, "kan_Latn_xlit", "fertility"),
                 kanglish_fertility=get(m, t, "kanglish_handwritten", "kanglish", "fertility"),
                 kanglish_n_sent=get(m, t, "kanglish_handwritten", "kanglish", "n_sent"))
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "kanglish_tax.csv", index=False)
    fmt = lambda v: "pending" if v is None or pd.isna(v) else f"{v:.2f}×"
    lines = [f"# Kanglish tax (parity = tokens ÷ English tokens for the same sentence; {split} + hand-written Kanglish)", "",
             "| tokenizer | new vocab | Kannada script | romanized (IndicXlit) | hand-written Kanglish |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {NICE.get(r['tokenizer'], r['tokenizer'])} | {r['new_vocab'] if str(r['new_vocab']) != 'nan' else '—'} | "
                     f"{fmt(r['kan_Knda_parity'])} | {fmt(r['kan_Latn_xlit_parity'])} | {fmt(r['kanglish_parity'])} |")
    n = rows[-1]["kanglish_n_sent"] if rows else None
    lines += ["", f"Hand-written Kanglish set: {int(n) if n else 0} sentences (`data/processed/test/kanglish_handwritten.txt`)."
              + (" **Template only (10 lines) — team set pending; numbers are provisional.**" if n and n < 100 else ""),
              "Source: `results/level1_matrix.csv` (python -m eval.level1.matrix && python -m eval.level1.report)."]
    (RES / "kanglish_tax.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


# ------------------------------------------------------------------ B3 charts
def charts(m: pd.DataFrame, final: str, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CH.mkdir(parents=True, exist_ok=True)
    k = m[(m.split == split) & (m.form == "kan_Knda")]

    # 1 parity bars
    names = [t for t in ["cl100k_base", "llama3", "qwen3", "o200k_base", "gemma3", "sarvam1", "indictok256k", "std_16k", final]
             if t in set(k.tokenizer)]
    d = k.set_index("tokenizer").loc[names].sort_values("parity")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    cols = [BLUE if t == final else ORANGE if t.startswith("std_") else GRAY for t in d.index]
    lab = [NICE.get(t, "Qwen3 + standard BPE ext. (16K)" if t == "std_16k" else t) for t in d.index]
    ax.barh(lab, d.parity, color=cols, height=0.62, edgecolor=SURF, linewidth=2)
    for y, v in enumerate(d.parity):
        ax.text(v + 0.08, y, f"{v:.2f}×", va="center", fontsize=12, color=INK)
    ax.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(1.02, len(d) - 0.45, "parity with English", fontsize=10, color=INK2)
    style(ax, "Token cost of Kannada relative to the same English text", "tokens ÷ English tokens (FLORES-200 devtest)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CH / "1_parity.png", facecolor=SURF)
    plt.close(fig)

    # 2 ablation: tokens/word as layers are added (16K)
    steps = [("Qwen3", "qwen3"), ("+ standard\nBPE ext.", "std_16k"), ("+ L1 split\nrule", "l1z_16k"),
             ("+ L2 akshara\nunits", "l12z_16k"), ("+ L3 suffix\nsplits", "l123z_16k"), ("+ L4 parity\n(= ours)", "l1234_16k")]
    steps = [(a, b) for a, b in steps if b in set(k.tokenizer)]
    v = [float(k.loc[k.tokenizer == b, "fertility"].iloc[0]) for _, b in steps]
    ak = [k.loc[k.tokenizer == b, "akshara_split_rate"].iloc[0] for _, b in steps]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    cols = [GRAY, ORANGE] + [BLUE] * (len(steps) - 2)
    ax.bar([a for a, _ in steps], v, color=cols, width=0.6, edgecolor=SURF, linewidth=2)
    for x, (val, a) in enumerate(zip(v, ak)):
        ax.text(x, val + 0.15, f"{val:.2f}", ha="center", fontsize=13, color=INK, fontweight="bold")
        ax.text(x, val + 0.75, f"aksharas split {float(a):.1%}", ha="center", fontsize=9, color=INK2)
    style(ax, "Kannada tokens per word, adding one layer at a time (16K new tokens)", "", "tokens / word (FLORES devtest)")
    ax.set_ylim(0, max(v) * 1.18)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CH / "2_ablation.png", facecolor=SURF)
    plt.close(fig)

    # 3 vocab sweep
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    for fam, pref, col, lbl in (("std", "std_", ORANGE, "Qwen3 + standard BPE extension"),
                                ("ours", "l123z_", BLUE, "Qwen3 + ours (layers 1–3)")):
        pts = sorted((float(r.new_vocab) / 1000, float(r.parity)) for r in k.itertuples()
                     if r.tokenizer.startswith(pref) and str(r.new_vocab) not in ("", "nan"))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, marker="o", ms=8,
                    markeredgecolor=SURF, markeredgewidth=2, label=lbl)
            ax.text(pts[-1][0] + 1.5, pts[-1][1], f"{pts[-1][1]:.2f}×", va="center", fontsize=12, color=INK)
    l4 = sorted((float(r.new_vocab) / 1000, float(r.parity)) for r in k.itertuples() if r.tokenizer.startswith("l1234_"))
    if l4:
        ax.plot([p[0] for p in l4], [p[1] for p in l4], ls="none", marker="D", ms=9, color=BLUE, markeredgecolor=INK,
                label="Qwen3 + ours with L4 (l1234)")
    q = k.loc[k.tokenizer == "qwen3", "parity"]
    if len(q):
        ax.axhline(float(q.iloc[0]), color=GRAY, lw=1.5, ls=(0, (4, 3)))
        ax.text(1, float(q.iloc[0]) - 0.35, f"Qwen3 original {float(q.iloc[0]):.2f}×", fontsize=11, color=INK2)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (1, 2)))
    style(ax, "More vocabulary only helps if the split rule lets Kannada words stay whole",
          "new Kannada tokens appended to Qwen3 (thousands)", "Kannada ÷ English tokens (FLORES devtest)")
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(CH / "3_vocab_sweep.png", facecolor=SURF)
    plt.close(fig)

    # 4 Kanglish tax grouped bars
    toks = [t for t in ["qwen3", "std_16k", final] if t in set(m.tokenizer)]
    forms = [("kan_Knda", split, "Kannada script", BLUE), ("kan_Latn_xlit", split, "romanized (IndicXlit)", ORANGE),
             ("kanglish", "kanglish_handwritten", "hand-written Kanglish", AQUA)]
    forms = [f for f in forms if any(get(m, t, f[1], f[0], "parity") for t in toks)]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    w = 0.8 / max(1, len(forms))
    for i, (form, sp, lbl, col) in enumerate(forms):
        vals = [get(m, t, sp, form, "parity") or 0 for t in toks]
        xs = [j + (i - (len(forms) - 1) / 2) * w for j in range(len(toks))]
        ax.bar(xs, vals, width=w * 0.92, color=col, label=lbl, edgecolor=SURF, linewidth=2)
        for x, val in zip(xs, vals):
            if val:
                ax.text(x, val + 0.08, f"{val:.2f}×", ha="center", fontsize=11, color=INK)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(toks)), [NICE.get(t, "Qwen3 + std BPE ext." if t == "std_16k" else t) for t in toks])
    style(ax, "The Kanglish tax: tokens ÷ English for the same content", "", "tokens ÷ English tokens")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CH / "4_kanglish_tax.png", facecolor=SURF)
    plt.close(fig)


def token_viz(final_path: str, sentence: str) -> None:
    """HTML with coloured token spans for Qwen3 vs ours, rendered by headless Edge (shapes Kannada correctly)."""
    from tokenizer.registry import load

    rows = []
    for label, spec in (("Qwen3", "qwen3"), ("Ours", final_path)):
        tok = load(spec)
        pieces = tok.token_bytes(tok.normalize(sentence))
        spans, buf = [], b""
        for p in pieces:  # byte pieces may split characters; group until they decode
            buf += p
            try:
                spans.append(buf.decode("utf-8"))
                buf = b""
            except UnicodeDecodeError:
                spans.append(None)  # placeholder: this token ends mid-character
        cells = []
        alt = 0
        for s in spans:
            alt ^= 1
            if s is None:
                cells.append(f'<span class="t t{alt} part">·</span>')
            else:
                cells.append(f'<span class="t t{alt}">{html.escape(s).replace(" ", "&nbsp;")}</span>')
        rows.append(f'<div class="row"><div class="lab">{label}<b>{len(pieces)} tokens</b></div><div class="toks">{"".join(cells)}</div></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:{SURF};font-family:'Nirmala UI','Segoe UI',sans-serif;color:{INK}}}
.wrap{{padding:36px 44px;width:1500px}} h1{{font-size:30px;margin:0 0 6px}} p{{color:{INK2};font-size:19px;margin:0 0 26px}}
.row{{display:flex;gap:24px;align-items:flex-start;margin:20px 0}} .lab{{width:140px;font-size:22px;font-weight:600}}
.lab b{{display:block;font-size:17px;font-weight:400;color:{INK2}}} .toks{{flex:1;font-size:34px;line-height:2.1}}
.t{{display:inline-block;padding:2px 3px;margin:3px 2px;border-radius:5px;white-space:pre}} .t0{{background:#d6e6f8}} .t1{{background:#fbe0d3}}
.part{{color:{INK2};font-size:20px}}
</style></head><body><div class="wrap"><h1>Same sentence, two tokenizers</h1>
<p>Each shaded box is one token. “·” marks a token that ends in the middle of a character (raw UTF-8 bytes).</p>
<div style="font-size:30px;margin-bottom:10px">{html.escape(sentence)}</div>{''.join(rows)}</div></body></html>"""
    CH.mkdir(parents=True, exist_ok=True)
    hp = CH / "5_token_viz.html"
    hp.write_text(page, encoding="utf-8")
    edge = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"] if Path(p).exists()), None)
    if edge:
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={CH / '5_token_viz.png'}",
                        "--window-size=1600,1100", hp.resolve().as_uri()], check=False, capture_output=True, timeout=60)


# ------------------------------------------------------------------ B5 failure analysis (counts)
VIRAMA = "\u0ccd"


def train_word_freq(path: Path, cache: Path) -> Counter:
    if cache.exists():
        c = Counter()
        for line in cache.read_text(encoding="utf-8").splitlines():
            w, n = line.rsplit("\t", 1)
            c[w] = int(n)
        return c
    c = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            c.update(unicodedata.normalize("NFC", line).split())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(f"{w}\t{n}" for w, n in c.items()), encoding="utf-8")
    return c


def is_kn(w): return any("\u0c80" <= ch <= "\u0cff" for ch in w)
def is_lat(w): return bool(re.search(r"[A-Za-z]", w))


def failures(final_path: str) -> None:
    from tokenizer.registry import load

    toks = {"qwen3": load("qwen3"), "std_16k": load(str(ROOT / "artifacts/ext/std_16k/tokenizer.json")),
            "bigo_final": load(final_path)}
    freq = train_word_freq(ROOT / "data/processed/train/kan_Knda_mix.txt", ROOT / "artifacts/cache/kan_mix_wordfreq.tsv")
    strip = lambda w: w.strip(".,;:!?\"'()[]{}“”‘’—-")
    files = [ROOT / "data/raw/test/flores_devtest_kan_Knda.txt", ROOT / "data/raw/test/in22gen_kan_Knda.txt"]
    kl = ROOT / "data/processed/test/kanglish_handwritten.txt"
    cats: dict[str, list[str]] = {k: [] for k in ["all_kannada", "rare_kannada (train freq<5)", "loanword (ends in virama)",
                                                  "number", "english_in_kannada_line", "kanglish_word"]}
    mixed_lines, n_lines = [], 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            n_lines += 1
            ws = [strip(w) for w in unicodedata.normalize("NFC", line).split()]
            if any(is_kn(w) for w in ws) and any(is_lat(w) for w in ws):
                mixed_lines.append(line)
            for w in ws:
                if not w:
                    continue
                if re.fullmatch(r"[\d\u0ce6-\u0cef.,]+", w) and re.search(r"[\d\u0ce6-\u0cef]", w):
                    cats["number"].append(w)
                elif is_kn(w) and not is_lat(w):
                    cats["all_kannada"].append(w)
                    if freq.get(w, 0) < 5:
                        cats["rare_kannada (train freq<5)"].append(w)
                    if w.rstrip("\u200c").endswith(VIRAMA):
                        cats["loanword (ends in virama)"].append(w)
                elif is_lat(w):
                    cats["english_in_kannada_line"].append(w)
    if kl.exists():
        cats["kanglish_word"] = [strip(w) for l in kl.read_text(encoding="utf-8").splitlines() for w in l.split() if strip(w)]
    ntok = {n: {} for n in toks}
    rows, examples = [], {}
    for cat, ws in cats.items():
        if not ws:
            continue
        cnt = {n: [len(t.encode(" " + w)) for w in ws] for n, t in toks.items()}
        q, s, o = cnt["qwen3"], cnt["std_16k"], cnt["bigo_final"]
        worse_std = sum(a > b for a, b in zip(o, s))
        nobetter_q = sum(a >= b for a, b in zip(o, q))
        rows.append(dict(category=cat, words=len(ws), tokens_per_word_qwen3=sum(q) / len(ws), tokens_per_word_std16k=sum(s) / len(ws),
                         tokens_per_word_ours=sum(o) / len(ws), ours_no_better_than_qwen3=nobetter_q,
                         ours_no_better_than_qwen3_pct=nobetter_q / len(ws), ours_worse_than_std16k=worse_std,
                         ours_worse_than_std16k_pct=worse_std / len(ws), ours_single_token_pct=sum(x == 1 for x in o) / len(ws)))
        bad = sorted({(w, a, b, c) for w, a, b, c in zip(ws, o, s, q) if a > b or a >= c}, key=lambda x: -(x[1] - x[2]))[:8]
        examples[cat] = bad
    mixed_q = sum(len(toks["qwen3"].encode(l)) for l in mixed_lines)
    mixed_o = sum(len(toks["bigo_final"].encode(l)) for l in mixed_lines)
    with (RES / "failures.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["# Failure analysis: where bigo_final is no better (or worse)", "",
         "Words from FLORES-200 devtest + IN22-Gen Kannada (and the hand-written Kanglish file), each tokenized with a",
         "leading space as in running text. *No better than Qwen3* = ours uses as many or more tokens than the original;",
         "*worse than std* = more tokens than the standard BPE extension at 16K (the fair baseline). Counts, not anecdotes.", "",
         "| category | words | Qwen3 tok/word | std ext tok/word | ours tok/word | ours ≥ Qwen3 | ours > std ext | ours = 1 token |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['category']} | {r['words']:,} | {r['tokens_per_word_qwen3']:.2f} | {r['tokens_per_word_std16k']:.2f} | "
                 f"{r['tokens_per_word_ours']:.2f} | {r['ours_no_better_than_qwen3']:,} ({r['ours_no_better_than_qwen3_pct']:.1%}) | "
                 f"{r['ours_worse_than_std16k']:,} ({r['ours_worse_than_std16k_pct']:.1%}) | {r['ours_single_token_pct']:.1%} |")
    L += ["", f"Mixed-script lines (Kannada + Latin letters): {len(mixed_lines)} of {n_lines}; "
          f"tokens Qwen3 {mixed_q:,} → ours {mixed_o:,} ({(mixed_o / mixed_q if mixed_q else 0):.2f}×). "
          "English words inside them tokenize exactly as in Qwen3 by design (identity test).", "",
          "## Worst examples per category (ours vs std ext vs Qwen3 tokens)", ""]
    for cat, ex in examples.items():
        if ex:
            L.append(f"**{cat}**: " + ", ".join(f"`{w}` {a}/{b}/{c}" for w, a, b, c in ex))
            L.append("")
    L += ["Notes:", "- Numbers: Qwen3's split rule gives every digit its own token (`\\p{N}`); we keep that rule, so numbers",
          "  are unchanged by design.",
          "- English inside Kannada and romanized/Kanglish words: Latin text is untouched in bigo_final (English identity),",
          "  so these get no benefit — the Kanglish tax remains (see kanglish_tax.md).",
          "- Loanwords ending in virama and rare words are where the akshara inventory and suffix list run out.",
          "", "Source: `python -m eval.level1.report` → results/failures.csv"]
    (RES / "failures.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="bigo_final")
    ap.add_argument("--final-path", default=str(ROOT / "artifacts/final/bigo_final/tokenizer.json"))
    ap.add_argument("--split", default="flores_devtest")
    ap.add_argument("--viz-sentence", default="ಬೆಂಗಳೂರಿನಲ್ಲಿ ನಾಳೆ ಮಳೆ ಬರುವ ಸಾಧ್ಯತೆ ಇದೆ, ಆದ್ದರಿಂದ ಕೊಡೆ ತೆಗೆದುಕೊಂಡು ಹೋಗಿ.")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    m = pd.read_csv(RES / "level1_matrix.csv")
    if "tax" not in a.skip:
        kanglish_tax(m, a.final, a.split)
    if "charts" not in a.skip:
        charts(m, a.final, a.split)
    if "viz" not in a.skip:
        token_viz(a.final_path, a.viz_sentence)
    if "failures" not in a.skip:
        failures(a.final_path)
    print("wrote", sorted(p.name for p in RES.glob("*.md")), sorted(p.name for p in CH.glob("*.png")) if CH.exists() else [])


if __name__ == "__main__":
    main()
