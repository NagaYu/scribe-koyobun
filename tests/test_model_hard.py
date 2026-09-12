"""hard 集合で両用法を区別できることを検証する（主張の核）。

一律置換は同一語をどちらか一方に必ず倒すため、両用法が混在する hard 集合では
原理的に片方を必ず誤る。Scribe（ここではヒューリスティック層）は文脈で振り分けるので、
(1) 同一語について kana/kanji の両方を実際に出力し、(2) 一律置換より高い正解率を出す。
"""
from collections import defaultdict

from tests.helpers import predict_kind


def _uniform_accuracy(items, force):
    """一律置換ベースライン: 全て force('kana'|'kanji') に倒す。"""
    correct = sum(1 for x in items if x.gold_kind == force)
    return correct / len(items)


def test_hard_set_exists(seeds):
    hard = [x for x in seeds if x.is_hard]
    assert len(hard) >= 10
    # hard 集合の各語は両用法を含む
    kinds = defaultdict(set)
    for x in hard:
        kinds[x.lemma].add(x.gold_kind)
    for lemma, ks in kinds.items():
        assert ks == {"kana", "kanji"}, f"{lemma} が両用法を含まない: {ks}"


def test_heuristic_distinguishes_both_usages(seeds, heuristic):
    """同一語について kana と kanji の両方を実際に出力できる（一方に潰れていない）。"""
    hard = [x for x in seeds if x.is_hard]
    outputs = defaultdict(set)
    for x in hard:
        pred = predict_kind(heuristic, x)
        assert pred is not None
        outputs[x.lemma].add(pred)
    collapsed = [lemma for lemma, outs in outputs.items() if len(outs) < 2]
    # 大半の語で両方の出力が現れること（一律置換ではあり得ない挙動）
    assert len(collapsed) <= 1, f"両用法に振り分けられていない語: {collapsed}"


def test_heuristic_beats_uniform_on_hard_set(seeds, heuristic):
    hard = [x for x in seeds if x.is_hard]
    correct = sum(1 for x in hard if predict_kind(heuristic, x) == x.gold_kind)
    scribe_acc = correct / len(hard)
    uniform_acc = max(_uniform_accuracy(hard, "kana"), _uniform_accuracy(hard, "kanji"))
    # 一律置換は best-case でも hard 集合の多数派止まり。Scribe はそれを明確に上回る。
    assert scribe_acc > uniform_acc + 0.15, f"scribe={scribe_acc:.2f} uniform={uniform_acc:.2f}"
    assert scribe_acc >= 0.8, f"hard 正解率が低い: {scribe_acc:.2f}"


def test_context_only_not_surface(seeds, heuristic, norms):
    """反転して表層を変えても、文脈が同じなら判定(gold)が変わらないことを確認する。"""
    from scribe.flip import make_flipped
    checked = 0
    for x in seeds:
        flipped = make_flipped(x, norms)
        if flipped is None:
            continue
        pred_orig = predict_kind(heuristic, x)
        pred_flip = predict_kind(heuristic, flipped)
        # 表層が違っても正解は同じなので、正しく判定できていれば予測も一致するはず
        if pred_orig == x.gold_kind:
            assert pred_flip == x.gold_kind, f"表層依存の疑い: {x.text!r} / {flipped.text!r}"
            checked += 1
    assert checked > 0
