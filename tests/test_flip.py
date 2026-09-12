"""反転で作った誤りの位置が復元できることを検証する。

主張『hard 集合の正解率』の教師信号の健全性: 反転は表層だけを変え、位置(span)と
正解(gold_kind)を保つ。反転後のスパンから正しい表層を復元できる。
"""
from scribe.flip import expand_with_flips, flip_all_in_text, make_flipped, restore
from scribe.norms import load_norms


def test_flip_preserves_position_and_recovers(seeds, norms):
    n = 0
    for lab in seeds:
        flipped = make_flipped(lab, norms)
        if flipped is None:  # kana_only は反転不能
            continue
        n += 1
        # 1) 反転位置の文字列が反転後の表層と一致する
        assert flipped.text[flipped.span.start:flipped.span.end] == flipped.surface
        # 2) 表層は元と異なる（実際に誤りへ変わった）
        assert flipped.surface != lab.surface
        # 3) 正解種類は不変（文脈で決まるため）
        assert flipped.gold_kind == lab.gold_kind
        assert flipped.flipped is True
        # 4) 反転誤りから正しい表層を復元できる
        recovered = restore(flipped, norms)
        expected = norms.lexeme(lab.lemma).kana if lab.gold_kind == "kana" \
            else norms.lexeme(lab.lemma).kanji_primary
        assert recovered == expected
    assert n > 0, "反転可能な事例が無い"


def test_flip_only_changes_target_span(seeds, norms):
    for lab in seeds:
        flipped = make_flipped(lab, norms)
        if flipped is None:
            continue
        before_orig = lab.text[:lab.span.start]
        before_flip = flipped.text[:flipped.span.start]
        assert before_orig == before_flip  # 前方は不変
        after_orig = lab.text[lab.span.end:]
        after_flip = flipped.text[flipped.span.end:]
        assert after_orig == after_flip    # 後方は不変


def test_expand_adds_flips_but_not_for_kana_only(seeds, norms):
    expanded = expand_with_flips(seeds, norms)
    flippable = [x for x in seeds if not norms.lexeme(x.lemma).kana_only]
    assert len(expanded) == len(seeds) + len(flippable)


def test_flip_all_in_text(norms):
    from scribe.analysis import find_targets
    from scribe.datamodel import LabeledSpan
    text = "確認することがある。"
    labs = [LabeledSpan(text=text, span=o.span, surface=o.surface, lemma=o.lemma,
                        category=o.category, gold_kind="kana", clause_id=o.kana_clause)
            for o in find_targets(text, norms)]
    assert labs, "対象語が検出できていない"
    flipped_text = flip_all_in_text(text, labs, norms)
    assert "こと" not in flipped_text and "事" in flipped_text
