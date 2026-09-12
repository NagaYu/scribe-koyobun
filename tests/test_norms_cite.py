"""規範の構造化と根拠提示の整合性を検証する。"""
from scribe.cite import ClauseCitation
from scribe.datamodel import Finding, Kind, Layer, Span


def test_partition_deterministic_vs_context(norms):
    part = norms.partition_summary()
    assert set(part["deterministic"]) >= {"R-NUM-1", "R-PUNC-1", "R-OKURI-1", "R-BRACKET-1"}
    assert set(part["context"]) >= {"C-KEISHIKI-1", "C-HOJODOUSHI-1"}
    # 交わりが無い（同じ条項が両方に入らない）
    assert not (set(part["deterministic"]) & set(part["context"]))


def test_every_lexeme_clause_exists(norms):
    for lex in norms.lexemes:
        assert norms.clause(lex.kana_clause) is not None, lex.lemma
        assert norms.clause(lex.kanji_clause) is not None, lex.lemma


def test_every_clause_source_exists(norms):
    for c in norms.clauses:
        assert c.source in norms.sources, f"{c.id} の出典 {c.source} が未定義"
        assert norms.source_citation(c.id)  # 空でない出典表記


def test_citation_is_non_negating():
    citer = ClauseCitation()
    f = Finding(
        span=Span(6, 7), surface="事", lemma="こと", category="formal_noun",
        current_kind=Kind.KANJI, recommended_kind=Kind.KANA, recommended_surface="こと",
        needs_change=True, layer=Layer.MODEL, clause_id="C-KEISHIKI-1", confidence=0.85,
    )
    msg = citer.explain(f)
    assert "規範では" in msg
    assert "C-KEISHIKI-1" in msg
    # 否定的な断定語を含まない
    for bad in ("間違い", "誤りです", "ダメ", "不正解"):
        assert bad not in msg


def test_citation_unknown_clause_is_safe():
    citer = ClauseCitation()
    assert citer.cite(None) is None
    assert citer.cite("NO-SUCH") is None
