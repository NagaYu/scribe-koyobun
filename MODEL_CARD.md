---
language:
  - ja
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: google-bert/bert-base-multilingual-cased
tags:
  - japanese
  - koyobun
  - orthography
  - kanji-kana
  - government-writing
  - token-classification
metrics:
  - accuracy
model-index:
  - name: scribe-usage-classifier
    results:
      - task:
          type: token-classification
          name: Context-dependent kanji/kana usage judgment
        metrics:
          - type: accuracy
            name: test span accuracy (held-out by document)
            value: 0.818
          - type: accuracy
            name: test hard-set span accuracy
            value: 0.800
---

# scribe-usage-classifier

Token-classification model that judges **context-dependent kanji/kana usage** in Japanese official documents:
given each occurrence of a monitored word, it decides whether the norm writes it in **kana** or **kanji** based on
the word's **syntactic role** — a distinction that dictionary/regex replacement cannot make.

- Part of the **Scribe** project · 💻 [GitHub](https://github.com/NagaYu/scribe-koyobun) ·
  🤗 [Space](https://huggingface.co/spaces/NagaYu/scribe-koyobun) · 📚 [Dataset](https://huggingface.co/datasets/NagaYu/scribe-koyobun-usage)
- Base model: `google-bert/bert-base-multilingual-cased` (~178M params, within the 0.1–0.2B target).
  Chosen because its **fast tokenizer** returns `offset_mapping`, which the span-labeling pipeline needs.

## Labels

Per subtoken: `O` / `TARGET-KANA` / `TARGET-KANJI`. Subtokens overlapping a monitored word's span receive the
usage label; the input surface may be correct or wrong (the training data includes flipped errors), so the model
must learn from **context, not surface**.

## How to use

The model is meant to be driven through the Scribe pipeline, which finds the monitored words and attaches the
normative clause as rationale:

```python
from scribe import ScribeChecker
checker = ScribeChecker(model_dir="NagaYu/scribe-usage-classifier")
for f in checker.check("確認する事がある。重い物を運ぶ。"):
    print(f.surface, "→", f.recommended_surface, "|", f.message)
```

Raw transformers use is also possible (`AutoModelForTokenClassification`), but you then need to locate the target
spans yourself; the Scribe repo does this with a morphological analyzer.

## Training data

Fine-tuned on the **Scribe usage-judgment dataset** ([NagaYu/scribe-koyobun-usage](https://huggingface.co/datasets/NagaYu/scribe-koyobun-usage)),
a **synthetic seed corpus (CC0)** of official-style sentences plus flip-generated errors, split by document.
The dataset separates a **hard** split (words appearing in both usages) as the headline diagnostic.

## Evaluation

Span-level accuracy (predicted kind vs. gold kind):

| split | span accuracy | hard-set span accuracy |
|---|---|---|
| test (held out by document) | **0.818** | **0.800** |
| hard (diagnostic, overlaps train) | 0.947 | 0.947 |

Evaluate `test` for generalization; `hard` cross-cuts the training split and is optimistic. For the project-level
comparison against uniform replacement (prh) and a jōyō-kanji checker, see the
[benchmarks](https://github.com/NagaYu/scribe-koyobun/tree/main/benchmarks).

## Intended use & limitations

- **Intended**: assisting drafters of Japanese official/public documents by flagging context-dependent kanji/kana
  choices with a cited clause of the norm; output is phrased as "the norm would write it this way," never as a verdict
  on the writer.
- **Proof-of-concept scale**: trained on a small synthetic corpus. It demonstrates that the distinction is learnable
  from context, but is **not** production-grade on natural text. For real deployment, train on a licensed real corpus
  (see `scribe/collect.py`) and expand the monitored vocabulary (`data/norms/lexicon.json`).
- **Coverage**: limited to the monitored formal nouns and auxiliary verbs; other orthographic issues (numbers,
  punctuation, okurigana) are handled by Scribe's deterministic rule layer, not this model.
- The base model is multilingual; a dedicated Japanese encoder may improve quality if paired with a custom
  offset aligner.

## License

Apache-2.0. The normative documents' rights belong to their issuers; this model references clause numbers only.
