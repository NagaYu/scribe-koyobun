"""評価ハーネス（目玉）。比較条件 A〜E を同一データで測る。

比較条件:
  (A) prh/textlint 相当の一律置換（監視語を常に かな へ）… 現行の実務標準。
  (B) 常用漢字表ベースの規則チェック（表内字は許容＝書かれたまま）… 使い分けに触れない。
  (C) 汎用 LLM（プロンプトのみ）… API があるときのみ。無ければ N/A。
  (D) Scribe（量子化前）… ヒューリスティック、または --model 指定でニューラル。
  (E) Scribe（量子化後）… 量子化モデルがあれば --model-quant で指定。無ければ D と同値注記。

評価軸:
  (1) hard 集合の正解率（同一語の両用法。主張の核）
  (2) 全体の適合率・再現率（誤り検出）
  (3) 過剰指摘率（正しい表記を誤りとして指摘した割合）
  (4) 語種別の内訳
  (5) 規則層とモデル層の寄与の分離
  (6) CPU 応答時間

結果は benchmarks/results.json に書き出し、make_figures.py が図を作る。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.analysis import _kind_of
from scribe.collect import build_seed_labeled, load_rule_cases
from scribe.datamodel import Kind
from scribe.flip import expand_with_flips
from scribe.model import UsageClassifier
from scribe.norms import load_norms
from scribe.rules import RuleLayer

ROOT = Path(__file__).resolve().parent.parent


def surface_kind(lemma: str, surface: str, norms) -> str:
    lex = norms.lexeme(lemma)
    return _kind_of(surface, lex).value


def build_eval_records(norms):
    """反転込みの評価レコード。各出現に gold・現在表記種・hard・flipped を持たせる。"""
    positives = build_seed_labeled(norms)
    data = expand_with_flips(positives, norms)
    records = []
    for x in data:
        records.append({
            "text": x.text,
            "span": (x.span.start, x.span.end),
            "lemma": x.lemma,
            "category": x.category,
            "gold": x.gold_kind,
            "surface_kind": surface_kind(x.lemma, x.surface, norms),
            "is_hard": x.is_hard,
            "flipped": x.flipped,
        })
    return records


# --- 予測器（システムごと） -------------------------------------------
def predictor_prh_uniform(records, norms) -> Dict[int, str]:
    # 監視語を常に かな へ（prh の一律置換の代表設定）
    return {i: "kana" for i, _ in enumerate(records)}


def predictor_joyo_checker(records, norms) -> Dict[int, str]:
    # 常用漢字表内はそのまま許容 → 書かれた表記を受け入れる（使い分けに踏み込まない）
    return {i: r["surface_kind"] for i, r in enumerate(records)}


def predictor_scribe(records, norms, model_dir: Optional[str]) -> Dict[int, str]:
    clf = UsageClassifier(model_dir=model_dir, norms=norms)
    # text ごとに一括判定してスパンで対応づけ
    by_text: Dict[str, List] = {}
    preds: Dict[int, str] = {}
    cache: Dict[str, Dict[tuple, str]] = {}
    for i, r in enumerate(records):
        t = r["text"]
        if t not in cache:
            m = {}
            for p in clf.classify_text(t):
                if p.occurrence:
                    m[(p.occurrence.span.start, p.occurrence.span.end)] = p.kind.value
            cache[t] = m
        preds[i] = cache[t].get(tuple(r["span"]), r["surface_kind"])
    return preds


def predictor_llm(records, norms, caller: Optional[Callable]) -> Optional[Dict[int, str]]:
    """汎用 LLM（プロンプトのみ）。caller が無ければ None（N/A）。

    caller(text, span, lemma) -> "kana"|"kanji" を渡せば評価に組み込める。
    再現性のため既定では未実行（API キー要）。
    """
    if caller is None:
        return None
    return {i: caller(r["text"], r["span"], r["lemma"]) for i, r in enumerate(records)}


# --- 指標 --------------------------------------------------------------
def metrics(records, preds: Dict[int, str]) -> dict:
    n = len(records)
    correct = sum(1 for i, r in enumerate(records) if preds[i] == r["gold"])

    hard = [(i, r) for i, r in enumerate(records) if r["is_hard"]]
    hard_correct = sum(1 for i, r in hard if preds[i] == r["gold"])

    # 検出（誤り = surface_kind != gold; flag = pred != surface_kind; 正検出 = flag かつ pred==gold）
    tp = fp = fn = 0
    over_flag = 0
    correct_occ = 0
    for i, r in enumerate(records):
        is_error = r["surface_kind"] != r["gold"]
        flagged = preds[i] != r["surface_kind"]
        if is_error:
            if flagged and preds[i] == r["gold"]:
                tp += 1
            else:
                fn += 1
        else:
            correct_occ += 1
            if flagged:
                fp += 1
                over_flag += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # 語種別（実質名詞/本動詞は gold=kanji 側として集計）
    by_cat = defaultdict(lambda: [0, 0])
    for i, r in enumerate(records):
        key = r["category"]
        by_cat[key][1] += 1
        if preds[i] == r["gold"]:
            by_cat[key][0] += 1
    # gold 種別（かな正/漢字正）でも分ける（目玉図用）
    by_goldkind = defaultdict(lambda: [0, 0])
    for i, r in enumerate(records):
        by_goldkind[r["gold"]][1] += 1
        if preds[i] == r["gold"]:
            by_goldkind[r["gold"]][0] += 1

    # 用法別（カテゴリ×正しい種類）: 形式名詞/実質名詞/補助動詞/本動詞
    _usage = {("formal_noun", "kana"): "formal_noun", ("formal_noun", "kanji"): "substantive",
              ("aux_verb", "kana"): "aux_verb", ("aux_verb", "kanji"): "main_verb"}
    by_usage = defaultdict(lambda: [0, 0])
    for i, r in enumerate(records):
        key = _usage.get((r["category"], r["gold"]), r["category"])
        by_usage[key][1] += 1
        if preds[i] == r["gold"]:
            by_usage[key][0] += 1

    return {
        "n": n,
        "accuracy": correct / n,
        "hard_n": len(hard),
        "hard_accuracy": (hard_correct / len(hard)) if hard else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "over_flag_rate": over_flag / correct_occ if correct_occ else 0.0,
        "by_category": {k: v[0] / v[1] for k, v in by_cat.items()},
        "by_usage": {k: v[0] / v[1] for k, v in by_usage.items()},
        "by_goldkind": {k: v[0] / v[1] for k, v in by_goldkind.items()},
        "hard_by_goldkind": _hard_by_goldkind(records, preds),
    }


def _hard_by_goldkind(records, preds):
    d = defaultdict(lambda: [0, 0])
    for i, r in enumerate(records):
        if not r["is_hard"]:
            continue
        d[r["gold"]][1] += 1
        if preds[i] == r["gold"]:
            d[r["gold"]][0] += 1
    return {k: v[0] / v[1] for k, v in d.items()}


# --- (5) 規則層とモデル層の寄与分離 -----------------------------------
def rule_layer_contribution() -> dict:
    layer = RuleLayer()
    cases = load_rule_cases()
    total_expected = 0
    caught = 0
    false_pos = 0
    for c in cases:
        findings = {(f.surface, f.recommended_surface, f.clause_id)
                    for f in layer.check(c["text"]) if f.needs_change}
        for e in c["expect"]:
            total_expected += 1
            if (e["surface"], e["recommended"], e["clause"]) in findings:
                caught += 1
        if not c["expect"]:
            false_pos += len(findings)
    return {
        "rule_expected": total_expected,
        "rule_caught": caught,
        "rule_recall": caught / total_expected if total_expected else 0.0,
        "rule_false_positive_on_clean": false_pos,
    }


# --- (6) CPU 応答時間 -------------------------------------------------
def latency(records, norms, model_dir: Optional[str]) -> dict:
    clf = UsageClassifier(model_dir=model_dir, norms=norms)
    texts = list({r["text"] for r in records})
    # ウォームアップ
    for t in texts[:5]:
        clf.classify_text(t)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for t in texts:
            clf.classify_text(t)
    dt = (time.perf_counter() - t0) / (reps * len(texts))
    return {"method": clf.method, "n_texts": len(texts), "ms_per_sentence": dt * 1000}


def main():
    ap = argparse.ArgumentParser(description="Scribe 評価ハーネス")
    ap.add_argument("--model", help="Scribe(D) 用の学習済みモデル。無ければヒューリスティック。")
    ap.add_argument("--model-quant", help="Scribe(E) 用の量子化モデル。")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results.json"))
    args = ap.parse_args()

    norms = load_norms()
    records = build_eval_records(norms)

    systems = {}
    systems["A_prh_uniform"] = metrics(records, predictor_prh_uniform(records, norms))
    systems["B_joyo_checker"] = metrics(records, predictor_joyo_checker(records, norms))

    llm_preds = predictor_llm(records, norms, caller=None)  # 既定 N/A
    if llm_preds is not None:
        systems["C_llm_prompt"] = metrics(records, llm_preds)
    else:
        systems["C_llm_prompt"] = {"status": "N/A（API キー未設定のため未実行）"}

    d_preds = predictor_scribe(records, norms, args.model)
    systems["D_scribe"] = metrics(records, d_preds)

    if args.model_quant:
        systems["E_scribe_quant"] = metrics(records, predictor_scribe(records, norms, args.model_quant))
    else:
        # 量子化モデル未指定: ヒューリスティックには量子化差が無いため D と同値。
        e = dict(systems["D_scribe"])
        e["note"] = "量子化モデル未指定のため D と同値（ニューラル学習後は export.py で量子化版を生成し --model-quant で計測）"
        systems["E_scribe_quant"] = e

    result = {
        "dataset": {
            "total_occurrences": len(records),
            "hard_occurrences": sum(1 for r in records if r["is_hard"]),
            "correct_examples": sum(1 for r in records if not r["flipped"]),
            "flipped_examples": sum(1 for r in records if r["flipped"]),
            "source": "synthetic-seed (CC0)",
        },
        "systems": systems,
        "rule_layer": rule_layer_contribution(),
        "latency": latency(records, norms, args.model),
        "labels": {
            "A_prh_uniform": "(A) prh 一律置換",
            "B_joyo_checker": "(B) 常用漢字チェック",
            "C_llm_prompt": "(C) 汎用LLM",
            "D_scribe": "(D) Scribe 量子化前",
            "E_scribe_quant": "(E) Scribe 量子化後",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out}")
    # 要約表示
    for k, v in systems.items():
        if "accuracy" in v:
            print(f"  {k:18s} 全体acc={v['accuracy']:.2f} hard acc={v['hard_accuracy']:.2f} "
                  f"P={v['precision']:.2f} R={v['recall']:.2f} 過剰指摘={v['over_flag_rate']:.2f}")
        else:
            print(f"  {k:18s} {v.get('status')}")
    print(f"  規則層 recall={result['rule_layer']['rule_recall']:.2f} "
          f"（誤指摘 {result['rule_layer']['rule_false_positive_on_clean']}）")
    print(f"  応答時間={result['latency']['ms_per_sentence']:.2f} ms/文 ({result['latency']['method']})")


if __name__ == "__main__":
    main()
