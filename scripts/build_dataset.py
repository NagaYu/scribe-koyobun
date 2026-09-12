"""build_dataset.py — 文脈つき用法判定データを構築し、HF Dataset へ push する。

実証する主張: 「hard 集合の正解率」を測るためのデータを、hard 集合を分離した形で公開する。
出典と利用条件をカードに明記する（本文の再配布はしない方針。同梱データは合成種文=CC0）。

生成物（data/hf_dataset/ に保存、--push で Hub へ）:
  - train / validation / test: 正例 + 反転誤りの用法判定データ（スパン単位）。
  - hard: 同一語が両用法で出現する事例だけを集めた診断用スプリット（主要指標）。
分割は doc_id 単位で行い、反転対と原文が異なるスプリットに散らばらないようにする（リーク防止）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scribe.collect import KNOWN_SOURCES, build_seed_labeled  # noqa: E402
from scribe.flip import expand_with_flips  # noqa: E402
from scribe.norms import load_norms  # noqa: E402

OUT_DIR = ROOT / "data" / "hf_dataset"


def to_record(x) -> dict:
    return {
        "text": x.text,
        "target_start": x.span.start,
        "target_end": x.span.end,
        "surface": x.surface,
        "lemma": x.lemma,
        "category": x.category,
        "gold_kind": x.gold_kind,          # "kana" | "kanji"
        "label": 1 if x.gold_kind == "kanji" else 0,
        "clause_id": x.clause_id,
        "flipped": x.flipped,
        "is_hard": x.is_hard,
        "doc_id": x.doc_id,
        "source": x.source,
        "license": x.license,
    }


def split_by_doc(records: List[dict], seed: int = 20260101) -> Dict[str, List[dict]]:
    by_doc = defaultdict(list)
    for r in records:
        by_doc[r["doc_id"]].append(r)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(docs)
    n = len(docs)
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.1))
    test_docs = set(docs[:n_test])
    val_docs = set(docs[n_test:n_test + n_val])
    out = {"train": [], "validation": [], "test": []}
    for d in docs:
        split = "test" if d in test_docs else "validation" if d in val_docs else "train"
        out[split].extend(by_doc[d])
    return out


def build():
    norms = load_norms()
    positives = build_seed_labeled(norms)
    all_labeled = expand_with_flips(positives, norms)
    records = [to_record(x) for x in all_labeled]
    splits = split_by_doc(records)
    splits["hard"] = [r for r in records if r["is_hard"]]
    return records, splits


def save_local(splits: Dict[str, List[dict]]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # データセットカードの出典情報
    card = {
        "note": "文脈つき公用文表記（漢字/かな使い分け）用法判定データ。hard スプリットは同一語の両用法。",
        "license": "CC0-1.0（同梱の合成種文）",
        "redistribution": "官公庁文書の本文は再配布しない。実データ学習は各自の環境で collect.py 経由。",
        "known_sources_for_real_training": [
            {"name": s.name, "url": s.url, "license": s.license,
             "redistributable": s.redistributable, "note": s.note}
            for s in KNOWN_SOURCES
        ],
    }
    (OUT_DIR / "sources.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


def push(repo_id: str, splits: Dict[str, List[dict]], private: bool):
    from datasets import Dataset, DatasetDict
    dd = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
    dd.push_to_hub(repo_id, private=private)
    print(f"push 完了: https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="REPO_ID", help="HF Hub のデータセットリポジトリへ push")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    records, splits = build()
    save_local(splits)
    print(f"保存: {OUT_DIR}")
    for name, rows in splits.items():
        print(f"  {name:12s} {len(rows):4d} 件"
              + ("  ← 主要指標(同一語の両用法)" if name == "hard" else ""))

    if args.push:
        push(args.push, splits, args.private)
    else:
        print("（--push REPO_ID で Hub へ公開）")


if __name__ == "__main__":
    main()
