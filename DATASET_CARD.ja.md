---
language:
  - ja
license: cc0-1.0
pretty_name: Scribe 公用文表記（漢字/かな使い分け）用法判定データ
tags:
  - japanese
  - koyobun
  - orthography
  - token-classification
task_categories:
  - token-classification
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/hf_dataset/train.jsonl
      - split: validation
        path: data/hf_dataset/validation.jsonl
      - split: test
        path: data/hf_dataset/test.jsonl
      - split: hard
        path: data/hf_dataset/hard.jsonl
---

# Scribe 用法判定データセット

公用文における**文脈依存の漢字/かな使い分け**を判定するためのスパン単位データ。
`scripts/build_dataset.py` で生成します。

## 主張との対応

このデータの中心は **`hard` スプリット**です。`hard` は「**同一語が両用法（かな/漢字）で
コーパス内に出現する**」事例だけを集めたもので、一律置換が原理的に片方を必ず誤る集合です。
ここでの正解率が主要指標になります。

## フィールド

| フィールド | 説明 |
|---|---|
| `text` | 文（原文） |
| `target_start` / `target_end` | 対象語の文字オフセット（半開区間） |
| `surface` | 対象語の表層（反転誤りでは誤った表記） |
| `lemma` | 対象語の代表形（`こと`, `いただく` 等） |
| `category` | `formal_noun` / `aux_verb` |
| `gold_kind` | 規範上正しい種類 `kana` / `kanji`（文脈で決まる。表層とは独立） |
| `label` | `0`=かな, `1`=漢字（`gold_kind` の数値化） |
| `clause_id` | 根拠条項（`C-KEISHIKI-1` 等） |
| `flipped` | 反転で作った誤り事例か |
| `is_hard` | hard 集合に属するか |
| `doc_id` | 分割の単位（原文と反転対は同一スプリットに入る） |
| `source` / `license` | 出典・利用条件 |

## 構築方法

1. **正例**: 公用文体で自作した種文（`scribe/collect.py` の `SEED_SENTENCES`）に、対象語の正しい種類を注記。
   スパンは形態素解析器の検出結果から導出し、注記との整合を検証（記述ミスを排除）。
2. **反転誤り**（`scribe/flip.py`）: 対象語だけを反対表記へ機械的に反転し、**位置と正解を保った**負例を生成。
   表層だけが変わり文脈は同じなので、モデルは表層の暗記ではなく文脈から学ばざるを得ません。
3. **hard 付与**（`scribe/collect.py:mark_hard`）: 同一語が両用法で出現するものを hard として分離。
4. **分割**: `doc_id` 単位で train/validation/test に分割（反転対のリーク防止）。`hard` は診断用の別スプリット。

## ライセンスと再配布

- 同梱データは**すべて合成種文（CC0-1.0）**で、官公庁文書の本文は含みません。
- 実データからの学習を行う場合は、`scribe/collect.py` の `DocumentSource`／`KNOWN_SOURCES` を参照し、
  各文書の**利用条件を取得時点で必ず再確認**してください。再配布不可のソースは本文を保存せず、
  その場の学習にのみ用いる設計です。出典・利用条件・再配布可否は `data/hf_dataset/sources.json` に記録されます。

## 規範の出典

- 「公用文作成の考え方」（令和4年1月7日 文化審議会建議）
- 「公用文における漢字使用等について」（平成22年内閣訓令第1号）別紙 — 仮名で書く語句／漢字で書く語句の語例一覧
- 「送り仮名の付け方」（内閣告示）／「常用漢字表」（内閣告示第2号）

本データセットは上記の**条項番号への対応づけ**を持ちますが、規範本文の逐語転載は行いません。

## 限界

- 種文は少数の合成文であり、自然文の分布を代表しません。絶対精度の主張には実データ学習が必要です。
- 対象語は現状 `data/norms/lexicon.json` の範囲（形式名詞・補助動詞の代表語）に限定されます。
