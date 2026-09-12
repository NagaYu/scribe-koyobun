"""flip.py — 反転による誤り生成（位置と語種別が既知）。

実証する主張: 「hard 集合の正解率」の教師信号づくり。正しい表記の事例(正例)に対し、
対象語だけを反対の表記へ機械的に反転して、位置が既知の誤り事例(負例)を作る:
  形式名詞のかな（こと/とき/ところ/もの）→ 漢字へ、
  実質名詞の漢字（事/時/所/物）→ かなへ、
  補助動詞のかな（〜ていただく/ください/みる/おく）→ 漢字へ。
反転位置(span)と対象語種別(category)・正しい種類(gold_kind)を厳密なラベルとして記録する。

反転は表層だけを変え、文脈は保つ。これにより「同じ文脈で表層だけ違う」対が生まれ、
モデルが表層の暗記ではなく文脈から判定しているかを検証できる（テストで位置復元を確認）。
"""

from __future__ import annotations

from typing import List, Optional

from .datamodel import LabeledSpan, Span
from .norms import Norms, load_norms


def correct_surface(lemma: str, gold_kind: str, norms: Optional[Norms] = None) -> str:
    """規範上正しい表層を返す。"""
    norms = norms or load_norms()
    lex = norms.lexeme(lemma)
    if lex is None:
        raise KeyError(f"未知の対象語: {lemma}")
    return lex.kana if gold_kind == "kana" else lex.kanji_primary


def opposite_surface(lemma: str, gold_kind: str, norms: Optional[Norms] = None) -> Optional[str]:
    """反転先（誤り）の表層を返す。反転できない語（kana_only）は None。"""
    norms = norms or load_norms()
    lex = norms.lexeme(lemma)
    if lex is None:
        raise KeyError(f"未知の対象語: {lemma}")
    if lex.kana_only or not lex.kanji:
        return None
    return lex.kanji_primary if gold_kind == "kana" else lex.kana


def make_flipped(labeled: LabeledSpan, norms: Optional[Norms] = None) -> Optional[LabeledSpan]:
    """正例 1 件を反転し、位置既知の誤り事例を作る。

    gold_kind（正しい種類）は保持したまま、text 中の対象スパンだけを反対表記に置換する。
    返り値の surface は誤った表層、flipped=True。反転不能語(kana_only)は None を返す。
    """
    norms = norms or load_norms()
    wrong = opposite_surface(labeled.lemma, labeled.gold_kind, norms)
    if wrong is None:
        return None
    s, e = labeled.span.start, labeled.span.end
    new_text = labeled.text[:s] + wrong + labeled.text[e:]
    new_span = Span(s, s + len(wrong))
    return LabeledSpan(
        text=new_text,
        span=new_span,
        surface=wrong,
        lemma=labeled.lemma,
        category=labeled.category,
        gold_kind=labeled.gold_kind,     # 正解は不変（文脈で決まる）
        clause_id=labeled.clause_id,
        flipped=True,
        is_hard=labeled.is_hard,
        doc_id=labeled.doc_id,
        source=labeled.source + "+flip",
        license=labeled.license,
    )


def restore(labeled: LabeledSpan, norms: Optional[Norms] = None) -> str:
    """反転で作った誤り事例から、正しい表層を復元する（テストの検証点）。"""
    return correct_surface(labeled.lemma, labeled.gold_kind, norms)


def flip_all_in_text(text: str, targets: List[LabeledSpan],
                     norms: Optional[Norms] = None) -> str:
    """文中の全対象語を一括反転（一律置換ベースラインの入力や、誤り混入文書の作成に使う）。

    右から置換して先行スパンのオフセットを崩さない。
    """
    norms = norms or load_norms()
    out = text
    for lab in sorted(targets, key=lambda x: x.span.start, reverse=True):
        wrong = opposite_surface(lab.lemma, lab.gold_kind, norms)
        if wrong is None:
            continue
        out = out[:lab.span.start] + wrong + out[lab.span.end:]
    return out


def expand_with_flips(positives: List[LabeledSpan],
                      norms: Optional[Norms] = None) -> List[LabeledSpan]:
    """正例集合に反転負例を加えて返す（正例 + 反転誤り）。

    各正例につき最大 1 件の反転事例を生成する。表層は違っても gold_kind は同じなので、
    分類器は文脈を見ざるを得ない。
    """
    norms = norms or load_norms()
    out: List[LabeledSpan] = list(positives)
    for p in positives:
        flipped = make_flipped(p, norms)
        if flipped is not None:
            out.append(flipped)
    return out
