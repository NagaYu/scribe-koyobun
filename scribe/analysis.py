"""形態素解析の薄いラッパと、判定対象語の検出。

実証する主張: 「文脈依存の判定」。UsageClassifier のヒューリスティックと RuleLayer が
共通で使う統語情報（品詞・活用形・直前トークン）をここで一元的に得る。fugashi が無い
環境でも壊れないよう、辞書引きだけの縮退動作を用意する（結果は落ちるが動く）。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from .datamodel import Kind, Span, TargetOccurrence
from .norms import Norms, load_norms


@dataclass
class Token:
    surface: str
    pos1: str        # 品詞大分類（名詞/動詞/助詞…）
    pos2: str        # 品詞中分類
    c_form: str      # 活用形（連体形-一般 等）
    lemma: str       # 語彙素（辞書見出し）
    start: int       # 文字オフセット [start, end)
    end: int

    @property
    def is_yougen(self) -> bool:
        """用言（活用する自立語）か。連体修飾の手掛かり。"""
        return self.pos1 in ("動詞", "形容詞", "形状詞", "助動詞")

    @property
    def is_rentai(self) -> bool:
        return "連体形" in (self.c_form or "")


class Analyzer:
    """fugashi(+unidic-lite) を優先。無ければ辞書引きのみの縮退版。"""

    def __init__(self) -> None:
        self._tagger = None
        self._degraded = False
        try:
            import fugashi  # noqa
            self._tagger = fugashi.Tagger()
        except Exception:
            self._degraded = True

    @property
    def degraded(self) -> bool:
        return self._degraded

    def tokens(self, text: str) -> List[Token]:
        if self._tagger is None:
            return self._degraded_tokens(text)
        out: List[Token] = []
        cursor = 0
        for w in self._tagger(text):
            surf = w.surface
            idx = text.find(surf, cursor)
            if idx < 0:
                idx = cursor
            start, end = idx, idx + len(surf)
            cursor = end
            f = w.feature
            out.append(Token(
                surface=surf,
                pos1=getattr(f, "pos1", "") or "",
                pos2=getattr(f, "pos2", "") or "",
                c_form=getattr(f, "cForm", "") or "",
                lemma=(getattr(f, "lemma", None) or surf),
                start=start,
                end=end,
            ))
        return out

    def _degraded_tokens(self, text: str) -> List[Token]:
        """辞書が無い場合: 文字単位の極簡易トークン（品詞は不明）。"""
        return [Token(ch, "", "", "", ch, i, i + 1) for i, ch in enumerate(text)]


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    return Analyzer()


def _kind_of(surface: str, lex) -> Kind:
    if surface == lex.kana:
        return Kind.KANA
    if surface in lex.kanji:
        return Kind.KANJI
    # 表記ゆれ（送り仮名付きの漢字形など）は漢字扱い
    if any(surface.startswith(k[0]) for k in lex.kanji if k):
        return Kind.KANJI
    return Kind.OTHER


def find_targets(text: str, norms: Optional[Norms] = None,
                 analyzer: Optional[Analyzer] = None) -> List[TargetOccurrence]:
    """監視語彙(lexicon)に該当する出現を検出する。

    トークンの品詞で対象語種を絞り込み、複合語（『事故』『時間』『仕事』等、一語として
    切り出される語）を誤って拾わないようにする。これにより過剰指摘率(3)を抑える。
    """
    norms = norms or load_norms()
    analyzer = analyzer or get_analyzer()
    toks = analyzer.tokens(text)
    out: List[TargetOccurrence] = []

    for tok in toks:
        lex = _match_lexeme(tok, norms)
        if lex is None:
            continue
        kind = _kind_of(tok.surface, lex)
        if kind is Kind.OTHER:
            continue
        out.append(TargetOccurrence(
            span=Span(tok.start, tok.end),
            surface=tok.surface,
            lemma=lex.lemma,
            category=lex.category,
            current_kind=kind,
            kana_clause=lex.kana_clause,
            kanji_clause=lex.kanji_clause,
            kana_only=lex.kana_only,
        ))
    return out


def _match_lexeme(tok: Token, norms: Norms):
    """トークンが監視語彙のどれかに一致するか。品詞で妥当性を確認する。"""
    for lex in norms.lexemes:
        surface_hit = (tok.surface == lex.kana) or (tok.surface in lex.kanji)
        lemma_hit = tok.lemma in (lex.kanji + [lex.kana, lex.lemma])
        if not (surface_hit or lemma_hit):
            continue
        # 品詞での妥当性チェック（複合語の内部一致を除外）
        if lex.category == "formal_noun":
            if tok.pos1 and tok.pos1 != "名詞":
                continue
            # 複合語の一部（例: 事故=名詞/固有でない一語）は surface が一致しない限り除外
            if not surface_hit:
                continue
        elif lex.category in ("aux_verb",):
            if tok.pos1 and tok.pos1 not in ("動詞", "補助記号", "助動詞"):
                # ください等は動詞扱いされないことがあるため surface 一致は許容
                if not surface_hit:
                    continue
        return lex
    return None
