# Benchmarks

`python scripts/evaluate.py` writes `results.json`; `python scripts/make_figures.py` renders the figures into `../figures/`.

## Systems compared

| ID | System | Implementation |
|---|---|---|
| A | prh / textlint-style uniform replacement | monitored words always → kana, context-blind (`predictor_prh_uniform`) |
| B | Jōyō-kanji rule check | in-list kanji accepted as written = never engages the distinction (`predictor_joyo_checker`) |
| C | General-purpose LLM (prompt only) | pass a `caller` to `predictor_llm` to measure; N/A by default (no API) |
| D | Scribe (pre-quantization) | `UsageClassifier` (heuristic layer unless `--model` is given) |
| E | Scribe (post-quantization) | pass ONNX int8 via `--model-quant`; equals D when omitted |

## Axes

1. **Hard-set accuracy** — both usages of the same word (the core claim). `systems.*.hard_accuracy`, `hard_by_goldkind`
2. **Overall precision / recall** — of error detection (precision / recall / f1)
3. **Over-flag rate** — share of already-correct spellings that were flagged (`over_flag_rate`); costly in practice
4. **Accuracy by usage type** — formal noun / substantive / auxiliary verb / main verb (`by_usage`)
5. **Rule-layer vs. model-layer contribution** — rule recall & false positives (`rule_layer`); model accuracy above
6. **CPU latency** — `latency.ms_per_sentence`

## Definitions of detection / over-flagging

For each occurrence, let `gold` be the correct kind and `surface_kind` the written kind.

- **Error**: `surface_kind != gold` (the flipped examples)
- **Flag**: prediction `pred != surface_kind` (i.e., a change is proposed)
- **True positive**: an error is flagged and `pred == gold` (corrected in the right direction)
- **Over-flag**: `surface_kind == gold` (already correct) yet flagged

## How to read the numbers (important)

The bundled numbers are a diagnostic on **synthetic seed text (CC0)**. The purpose is to present the **structural fact**
that *uniform replacement, by collapsing a word to one spelling, must miss about half of a hard set where both usages
coexist* — not to claim absolute accuracy on natural text. The high D/E values partly reflect that the seed sentences
contain the syntactic cues by design. Accuracy on real text requires neural training via `train.py`, whose job is to
fill the heuristic's coverage gaps (notably meaning-dependent substantive nouns — the "substantive" column of Figure 3).

## Reproduce

```bash
python scripts/evaluate.py                      # measure D/E with the heuristic layer
python scripts/evaluate.py --model artifacts/scribe-usage \
                           --model-quant exports/onnx-int8   # measure the trained/quantized neural model
python scripts/make_figures.py
```
