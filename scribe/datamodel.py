"""Scribe 全体で共有するデータ構造。

ここでは判定の入出力を表す軽量なデータクラスだけを定義する。どの主張を実証する
コードでもないが、規則層(RuleLayer)とモデル層(UsageClassifier)が同じ語彙で結果を
やり取りできるようにすることで、評価(5)の『規則層とモデル層の寄与の分離』を機械的に
集計可能にする土台になる。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Kind(str, Enum):
    """対象語の表記の種類。"""

    KANA = "kana"
    KANJI = "kanji"
    OTHER = "other"  # 送り仮名・数字など、漢字/かなの二択でないもの


class Layer(str, Enum):
    """指摘を出した層。評価(5)で寄与を分離するために使う。"""

    RULE = "rule"      # RuleLayer（規則で確定）
    MODEL = "model"    # UsageClassifier（文脈依存の判定）


@dataclass(frozen=True)
class Span:
    """文中の文字オフセット区間（半開区間 [start, end)）。"""

    start: int
    end: int

    def text_of(self, s: str) -> str:
        return s[self.start:self.end]


@dataclass
class Finding:
    """1 件の指摘。

    needs_change=False の Finding は「規範上は現状の表記でよい」ことを積極的に示す
    ためにも使える（過剰指摘率(3)を測るには、正しい表記を『正しい』と判定できたことを
    記録する必要があるため）。CLI/アプリでは needs_change=True のものだけを既定で表示する。
    """

    span: Span
    surface: str                 # 現在の表記（原文どおり）
    lemma: str                   # 語の代表形
    category: str                # formal_noun / aux_verb / number ...
    current_kind: Kind           # 現在の表記の種類
    recommended_kind: Kind       # 規範上あるべき種類
    recommended_surface: str     # 置き換え候補（あれば）
    needs_change: bool
    layer: Layer
    clause_id: Optional[str] = None       # 根拠となる条項（ClauseCitation が肉付け）
    confidence: float = 1.0               # 0..1。規則層は 1.0、モデル層は確信度。
    message: str = ""                     # 利用者向けの説明（否定しない言い回し）
    profile_note: Optional[str] = None    # 組織プロファイルによる上書きがあった場合の明示
    source: str = ""                      # どの規則/モデルが出したか（デバッグ用）

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = {"start": self.span.start, "end": self.span.end}
        d["current_kind"] = self.current_kind.value
        d["recommended_kind"] = self.recommended_kind.value
        d["layer"] = self.layer.value
        return d


@dataclass
class TargetOccurrence:
    """文中で見つかった判定対象語の 1 出現（モデル層への入力単位）。"""

    span: Span
    surface: str
    lemma: str
    category: str
    current_kind: Kind
    kana_clause: Optional[str] = None
    kanji_clause: Optional[str] = None
    kana_only: bool = False


@dataclass
class LabeledSpan:
    """データセットの 1 事例（用法判定の教師信号）。

    flip.py はここに『反転で作った誤りかどうか(flipped)』と『反転位置(span)』を記録する。
    is_hard は『同一語がコーパス内で両用法で出現する』hard 集合に属するかどうか。
    """

    text: str
    span: Span
    surface: str
    lemma: str
    category: str
    gold_kind: str               # "kana" / "kanji": 規範上正しい種類
    clause_id: str
    flipped: bool = False        # 反転により誤りへ変えた事例か
    is_hard: bool = False
    doc_id: str = ""
    source: str = ""
    license: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = {"start": self.span.start, "end": self.span.end}
        return d

    @staticmethod
    def from_dict(d: dict) -> "LabeledSpan":
        sp = d["span"]
        return LabeledSpan(
            text=d["text"],
            span=Span(sp["start"], sp["end"]),
            surface=d["surface"],
            lemma=d["lemma"],
            category=d["category"],
            gold_kind=d["gold_kind"],
            clause_id=d["clause_id"],
            flipped=d.get("flipped", False),
            is_hard=d.get("is_hard", False),
            doc_id=d.get("doc_id", ""),
            source=d.get("source", ""),
            license=d.get("license", ""),
        )
