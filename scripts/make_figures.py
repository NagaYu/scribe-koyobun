"""図の生成（目玉 2 枚 + 補助図）。

figure1_hard_by_kind: hard 集合で、gold 種別（かな正/漢字正）ごとの正解率をシステム別に。
  一律置換は『漢字正』側をほぼ 0% にしてしまう＝構造的に半分外すことを可視化する。
figure2_overflag_curve: 指摘数 vs 過剰指摘率 の曲線（Scribe は確信度閾値スイープ）。
  一律置換(A)が高い過剰指摘の一点に留まるのに対し、Scribe は低い過剰指摘で指摘できる。
figure3_by_category: 語種別（形式名詞/補助動詞/実質名詞/本動詞）の正解率内訳。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scribe.model import UsageClassifier  # noqa: E402
from scribe.norms import load_norms  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("scribe_eval", ROOT / "scripts" / "evaluate.py")
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

FIG_DIR = ROOT / "figures"

# 日本語フォント
for cand in ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "YuGothic",
             "IPAexGothic", "TakaoGothic", "AppleGothic", "Arial Unicode MS"]:
    if cand in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["svg.fonttype"] = "path"   # 文字をパス化して閲覧側フォント非依存に
plt.rcParams["axes.unicode_minus"] = False

# 配色（一律置換=グレー系、Scribe=青系）
C_UNIFORM = "#c04a3b"
C_JOYO = "#9aa0a6"
C_SCRIBE = "#2b6cb0"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  figures/{name}.svg, .png")


def figure1_hard_by_kind(results):
    sys_keys = ["A_prh_uniform", "B_joyo_checker", "D_scribe"]
    labels = [results["labels"][k] for k in sys_keys]
    kana = [results["systems"][k]["hard_by_goldkind"].get("kana", 0) * 100 for k in sys_keys]
    kanji = [results["systems"][k]["hard_by_goldkind"].get("kanji", 0) * 100 for k in sys_keys]

    x = range(len(sys_keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar([i - w / 2 for i in x], kana, w, label="かな正（形式名詞・補助動詞）", color="#7fb3d5")
    b2 = ax.bar([i + w / 2 for i in x], kanji, w, label="漢字正（実質名詞・本動詞）", color=C_UNIFORM)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("正解率 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("hard 集合：同一語の両用法での正解率\n一律置換は『漢字正』を構造的に外す", fontsize=12)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ax.annotate("一律置換は漢字用法を\nほぼ全て誤る", xy=(0 + w / 2, kanji[0]),
                xytext=(0.35, 42), fontsize=9, color=C_UNIFORM,
                arrowprops=dict(arrowstyle="->", color=C_UNIFORM))
    _save(fig, "figure1_hard_by_kind")


def figure2_overflag_curve(results):
    norms = load_norms()
    records = _eval.build_eval_records(norms)
    clf = UsageClassifier(norms=norms)

    # 各出現の (pred, conf, surface_kind, gold)
    cache = {}
    rows = []
    for r in records:
        t = r["text"]
        if t not in cache:
            cache[t] = {(p.occurrence.span.start, p.occurrence.span.end): (p.kind.value, p.confidence)
                        for p in clf.classify_text(t) if p.occurrence}
        pred, conf = cache[t].get(tuple(r["span"]), (r["surface_kind"], 1.0))
        rows.append((pred, conf, r["surface_kind"], r["gold"]))

    correct_total = sum(1 for _, _, sk, g in rows if sk == g)

    def point_at(tau):
        flags = 0
        overflag = 0
        for pred, conf, sk, g in rows:
            eff = pred if (pred == sk or conf >= tau) else sk
            if eff != sk:
                flags += 1
                if sk == g:  # 正しい表記なのに指摘 = 過剰指摘
                    overflag += 1
        return flags, (overflag / correct_total if correct_total else 0) * 100

    taus = [i / 100 for i in range(50, 101, 2)]
    pts = [point_at(t) for t in taus]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # ベースラインの点
    def baseline_point(pred_fn):
        flags = overflag = 0
        for i, (pred, conf, sk, g) in enumerate(rows):
            p = pred_fn(sk)
            if p != sk:
                flags += 1
                if sk == g:
                    overflag += 1
        return flags, (overflag / correct_total if correct_total else 0) * 100

    a_pt = baseline_point(lambda sk: "kana")          # prh 一律 かな
    b_pt = baseline_point(lambda sk: sk)              # 常用漢字チェック（無指摘）

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ys, "-o", color=C_SCRIBE, ms=4, label="(D) Scribe（確信度閾値スイープ）")
    ax.scatter([a_pt[0]], [a_pt[1]], color=C_UNIFORM, s=90, zorder=5, marker="X",
               label="(A) prh 一律置換")
    ax.scatter([b_pt[0]], [b_pt[1]], color=C_JOYO, s=90, zorder=5, marker="s",
               label="(B) 常用漢字チェック")
    ax.set_xlabel("指摘数（多いほど網羅的）")
    ax.set_ylabel("過剰指摘率 (%)  ＝ 正しい表記を誤指摘した割合")
    ax.set_title("指摘数 vs 過剰指摘率\nScribe は多く指摘しても過剰指摘が低い", fontsize=12)
    ax.annotate(f"prh: 過剰指摘 {a_pt[1]:.0f}%", xy=a_pt, xytext=(a_pt[0] - 34, a_pt[1] + 4),
                color=C_UNIFORM, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_UNIFORM))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(-3, max(a_pt[1] + 12, 20))
    _save(fig, "figure2_overflag_curve")


def figure3_by_category(results):
    cats = ["formal_noun", "substantive", "aux_verb", "main_verb"]
    cat_ja = {"formal_noun": "形式名詞\n(かな)", "substantive": "実質名詞\n(漢字)",
              "aux_verb": "補助動詞\n(かな)", "main_verb": "本動詞\n(漢字)"}
    sys_keys = ["A_prh_uniform", "D_scribe"]
    colors = {"A_prh_uniform": C_UNIFORM, "D_scribe": C_SCRIBE}

    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, k in enumerate(sys_keys):
        vals = [results["systems"][k]["by_usage"].get(c, 0) * 100 for c in cats]
        off = (j - 0.5) * w
        bars = ax.bar([i + off for i in x], vals, w, label=results["labels"][k], color=colors[k])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([cat_ja[c] for c in cats])
    ax.set_ylabel("正解率 (%)")
    ax.set_ylim(0, 112)
    ax.set_title("語種別の正解率内訳", fontsize=12)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    _save(fig, "figure3_by_category")


def main():
    results = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    print("図を生成:")
    figure1_hard_by_kind(results)
    figure2_overflag_curve(results)
    figure3_by_category(results)


if __name__ == "__main__":
    main()
