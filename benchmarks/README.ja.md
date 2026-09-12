# ベンチマーク

`python scripts/evaluate.py` が `results.json` を生成し、`python scripts/make_figures.py` が
`../figures/` に図を描きます。

## 比較条件

| ID | システム | 実装 |
|---|---|---|
| A | prh / textlint 相当の一律置換 | 監視語を文脈を見ず常に かな へ（`predictor_prh_uniform`） |
| B | 常用漢字表ベースの規則チェック | 表内字は書かれたまま許容＝使い分けに踏み込まない（`predictor_joyo_checker`） |
| C | 汎用 LLM（プロンプトのみ） | `predictor_llm` に caller を渡すと計測。既定は API 未設定で N/A |
| D | Scribe（量子化前） | `UsageClassifier`（`--model` 未指定ならヒューリスティック層） |
| E | Scribe（量子化後） | `--model-quant` に ONNX int8 を指定。未指定なら D と同値注記 |

## 評価軸

1. **hard 集合の正解率** — 同一語の両用法（主張の核）。`systems.*.hard_accuracy`, `hard_by_goldkind`
2. **全体の適合率・再現率** — 誤り検出の precision / recall / f1
3. **過剰指摘率** — 正しい表記を誤りとして指摘した割合（`over_flag_rate`）。実務では致命的
4. **語種別の内訳** — 形式名詞 / 実質名詞 / 補助動詞 / 本動詞（`by_usage`）
5. **規則層とモデル層の寄与分離** — 規則層 recall と誤指摘（`rule_layer`）、モデル層は上記精度
6. **CPU 応答時間** — `latency.ms_per_sentence`

## 検出・過剰指摘の定義

各出現について、正しい種類を `gold`、書かれている種類を `surface_kind` とする。

- **誤り**: `surface_kind != gold`（反転誤りがこれに当たる）
- **指摘**: システムの予測 `pred != surface_kind`（＝表記変更を提案）
- **正検出(TP)**: 誤りを指摘し、かつ `pred == gold`（正しい向きに直した）
- **過剰指摘**: `surface_kind == gold`（正しい表記）なのに指摘した割合

## 結果の読み方（重要）

同梱の数値は**合成種文（CC0）**による診断です。目的は
**「一律置換は同一語を一方に倒す以上、両用法混在の hard 集合で構造的に約半分を必ず誤る」**
という**構造的事実**の提示であり、自然文コーパスにおける絶対精度の主張ではありません。
D/E の高い値は、種文が統語的手掛かりを含むよう設計されているためでもあります。
実データでの精度は `train.py` によるニューラル学習を要し、ヒューリスティックの被覆ギャップ
（特に意味依存の実質名詞判別、図3の「実質名詞」列）を埋める役割がニューラル版にあります。

## 再現

```bash
python scripts/evaluate.py                      # ヒューリスティックで D/E を計測
python scripts/evaluate.py --model artifacts/scribe-usage \
                           --model-quant exports/onnx-int8   # 学習後のニューラル/量子化を計測
python scripts/make_figures.py
```
