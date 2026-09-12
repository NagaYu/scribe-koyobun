"""RuleLayer — 規則で確定できる表記を判定する層。

実証する主張: 「規則層とモデル層の寄与の分離」＋「過剰指摘の少なさ」。
数字・単位・括弧・句読点・確定的な送り仮名など、辞書と正規表現で確実に解ける項目を
ここで処理し、モデルには渡さない。規則が確実な箇所でモデルの確率的判断を仰がないことで、
文脈依存の判定にモデルの容量を集中させ、かつ規則で防げる誤指摘を排除する。

各規則は確信度 1.0 の Finding を返す。組織プロファイルにより一部の既定（読点の字種など）を
切り替えられる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .datamodel import Finding, Kind, Layer, Span

# 確定的に誤りとみなせる送り仮名（本則違反の代表例）。網羅ではなく高確度のものに限定する。
OKURIGANA_FIXES: Dict[str, str] = {
    "行なう": "行う", "行なっ": "行っ", "行なわ": "行わ", "行ない": "行い", "行ないます": "行います",
    "表わす": "表す", "表わし": "表し", "表わさ": "表さ", "表わせ": "表せ",
    "現われ": "現れ", "現わす": "現す", "現わし": "現し",
    "著わす": "著す", "著わし": "著し",
    "断わる": "断る", "断わっ": "断っ", "断わり": "断り",
    "起る": "起こる", "起っ": "起こっ", "起り": "起こり",
    "終る": "終わる", "終っ": "終わっ", "終り": "終わり",
    "少い": "少ない",
    "全て": "全て",  # 参考: 副詞「すべて」はかな推奨だが常用漢字表内のため規則層では触れない
}
# 「全て」は常用漢字だが副詞用法はかな推奨 — 文脈依存なので規則層では扱わない（誤登録防止のため上で無効化）
OKURIGANA_FIXES.pop("全て", None)


@dataclass
class RuleConfig:
    """組織プロファイルで切り替える規則の既定。"""
    comma: str = "、"          # 読点の字種（"、" or "，"）。府省庁・自治体で異なる。
    fix_fullwidth_digits: bool = True
    fix_fullwidth_latin: bool = True
    fix_halfwidth_brackets: bool = True
    fix_punctuation: bool = True
    fix_okurigana: bool = True


class RuleLayer:
    """確定項目の規則判定。"""

    def __init__(self, config: Optional[RuleConfig] = None):
        self.config = config or RuleConfig()

    def check(self, text: str) -> List[Finding]:
        findings: List[Finding] = []
        c = self.config
        if c.fix_okurigana:
            findings += self._okurigana(text)
        if c.fix_fullwidth_digits:
            findings += self._fullwidth_digits(text)
        if c.fix_fullwidth_latin:
            findings += self._fullwidth_latin(text)
        if c.fix_halfwidth_brackets:
            findings += self._halfwidth_brackets(text)
        if c.fix_punctuation:
            findings += self._punctuation(text)
        findings.sort(key=lambda f: f.span.start)
        return findings

    # --- 個別規則 -----------------------------------------------------
    def _mk(self, span: Span, surface: str, rec: str, clause: str, msg: str,
            category: str, source: str) -> Finding:
        return Finding(
            span=span, surface=surface, lemma=surface,
            category=category, current_kind=Kind.OTHER, recommended_kind=Kind.OTHER,
            recommended_surface=rec, needs_change=(surface != rec), layer=Layer.RULE,
            clause_id=clause, confidence=1.0, message=msg, source=source,
        )

    def _okurigana(self, text: str) -> List[Finding]:
        out = []
        for wrong, right in OKURIGANA_FIXES.items():
            for m in re.finditer(re.escape(wrong), text):
                out.append(self._mk(
                    Span(m.start(), m.end()), wrong, right, "R-OKURI-1",
                    f"送り仮名は本則により「{right}」と書きます。", "okurigana", "rule:okurigana"))
        return out

    def _fullwidth_digits(self, text: str) -> List[Finding]:
        out = []
        for m in re.finditer(r"[０-９]+", text):
            surface = m.group()
            rec = surface.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            out.append(self._mk(
                Span(m.start(), m.end()), surface, rec, "R-NUM-1",
                "横書きの公用文では算用数字（半角）を用います。", "number", "rule:fullwidth_digit"))
        return out

    def _fullwidth_latin(self, text: str) -> List[Finding]:
        out = []
        for m in re.finditer(r"[Ａ-Ｚａ-ｚ]+", text):
            surface = m.group()
            rec = "".join(chr(ord(ch) - 0xFEE0) for ch in surface)
            out.append(self._mk(
                Span(m.start(), m.end()), surface, rec, "R-UNIT-1",
                "単位・英字は半角に統一します。", "unit", "rule:fullwidth_latin"))
        return out

    def _halfwidth_brackets(self, text: str) -> List[Finding]:
        out = []
        for m in re.finditer(r"[()]", text):
            surface = m.group()
            rec = "（" if surface == "(" else "）"
            out.append(self._mk(
                Span(m.start(), m.end()), surface, rec, "R-BRACKET-1",
                "括弧は全角（　）に統一します。", "bracket", "rule:halfwidth_bracket"))
        return out

    def _punctuation(self, text: str) -> List[Finding]:
        out = []
        comma = self.config.comma
        # 全角ピリオド '．' を句点に流用している箇所 → 。
        for m in re.finditer(r"．", text):
            out.append(self._mk(
                Span(m.start(), m.end()), "．", "。", "R-PUNC-1",
                "句点は「。」を用います。", "punctuation", "rule:period"))
        # 半角カンマ ',' → 読点（既定は「、」。プロファイルで「，」に切替可）
        for m in re.finditer(r",", text):
            out.append(self._mk(
                Span(m.start(), m.end()), ",", comma, "R-PUNC-1",
                f"読点は「{comma}」を用います。", "punctuation", "rule:comma"))
        # プロファイルが「、」の場合に「，」が混在していれば「、」へ（逆も同様）
        other = "，" if comma == "、" else "、"
        for m in re.finditer(re.escape(other), text):
            out.append(self._mk(
                Span(m.start(), m.end()), other, comma, "R-PUNC-1",
                f"読点は「{comma}」に統一します。", "punctuation", "rule:comma_style"))
        return out
