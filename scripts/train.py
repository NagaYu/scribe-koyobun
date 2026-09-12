"""train.py — UsageClassifier(ニューラル)の学習。

実証する主張: 「hard 集合の正解率」＋「速度」。0.1〜0.2B の日本語 BERT を token 分類として
微調整し、監視語の各出現に『あるべき表記』(O / TARGET-KANA / TARGET-KANJI)を付与する。
入力の表層は正誤どちらもあり得る（反転誤りを含む）ため、モデルは表層ではなく文脈から学ぶ。

既定モデル: google-bert/bert-base-multilingual-cased（約 178M=0.18B。fast トークナイザで
 offset_mapping が使えるため採用。日本語専用 BERT を使う場合はオフセット整合を自前で実装する）。
ベースモデルの取得にネットワークが必要。学習後は export.py で ONNX/GGUF/MLX に書き出す。

使い方:
  python scripts/train.py --data data/hf_dataset --out artifacts/scribe-usage \
      [--model tohoku-nlp/bert-base-japanese-v3] [--epochs 8] [--push REPO_ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ["O", "TARGET-KANA", "TARGET-KANJI"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_split(data_dir: Path, name: str) -> List[dict]:
    path = data_dir / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_token_labels(example: dict, tokenizer, max_length: int = 256) -> dict:
    """1 事例をトークナイズし、対象スパンのサブトークンにラベルを付ける。

    対象スパンに重なるサブトークンだけ TARGET-KANA / TARGET-KANJI、他は O。
    表層に依存しないよう、ラベルは gold_kind から決める（反転誤りでも正解ラベルは正しい種類）。
    """
    enc = tokenizer(example["text"], truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    gold_label = "TARGET-KANJI" if example["gold_kind"] == "kanji" else "TARGET-KANA"
    s, e = example["target_start"], example["target_end"]
    labels = []
    for (a, b) in enc["offset_mapping"]:
        if a == b:  # 特殊トークン
            labels.append(-100)
        elif (a < e) and (b > s):  # 対象スパンと重なる
            labels.append(LABEL2ID[gold_label])
        else:
            labels.append(LABEL2ID["O"])
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def compute_span_accuracy(model, tokenizer, examples: List[dict], device: str) -> Dict[str, float]:
    """対象スパンの予測種類 vs gold_kind で正解率（hard 診断に使う）。"""
    import torch
    model.eval()
    correct = total = hard_correct = hard_total = 0
    for ex in examples:
        enc = tokenizer(ex["text"], return_offsets_mapping=True, return_tensors="pt",
                        truncation=True, max_length=256)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        s, e = ex["target_start"], ex["target_end"]
        idx = next((i for i, (a, b) in enumerate(offsets) if a != b and a < e and b > s), None)
        if idx is None:
            continue
        pred = int(logits[idx].argmax())
        pred_kind = "kanji" if pred == LABEL2ID["TARGET-KANJI"] else "kana"
        ok = (pred_kind == ex["gold_kind"])
        correct += ok
        total += 1
        if ex.get("is_hard"):
            hard_correct += ok
            hard_total += 1
    return {
        "span_accuracy": correct / total if total else 0.0,
        "hard_span_accuracy": hard_correct / hard_total if hard_total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hf_dataset"))
    ap.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "scribe-usage"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--push", metavar="REPO_ID", help="学習済みモデルを Hub へ push")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments)

    data_dir = Path(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID)

    def make_ds(name):
        rows = load_split(data_dir, name)
        feats = [build_token_labels(r, tokenizer) for r in rows]
        return Dataset.from_list(feats), rows

    train_ds, _ = make_ds("train")
    val_ds, _ = make_ds("validation")

    collator = DataCollatorForTokenClassification(tokenizer)
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=10,
        load_best_model_at_end=True, report_to=[])
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, processing_class=tokenizer)
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for split in ("test", "hard"):
        try:
            rows = load_split(data_dir, split)
        except FileNotFoundError:
            continue
        m = compute_span_accuracy(model, tokenizer, rows, device)
        print(f"[{split}] span_acc={m['span_accuracy']:.3f} hard_span_acc={m['hard_span_accuracy']:.3f}")

    if args.push:
        model.push_to_hub(args.push)
        tokenizer.push_to_hub(args.push)
        print(f"push 完了: https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
