"""ClauseCitation — 指摘に規範の条項を根拠として対応づける。

実証する主張: 「根拠提示」。実務価値の中心は『なぜ直すのか』を人に説明できること。
各指摘に、規範のどの条項に基づくか（id・見出し・要約・出典表記）を添えて返す。

物言いの原則（制約より）: 利用者の文章を否定しない。出力は「規範ではこうなります」に
留める。プロファイルによる上書きがある場合はその旨を明示する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .datamodel import Finding, Kind
from .norms import Norms, load_norms


@dataclass
class Citation:
    clause_id: str
    title: str
    summary: str
    source: str        # 出典表記（文書名・年月日・該当箇所）
    note: str = ""

    def one_line(self) -> str:
        return f"[{self.clause_id}] {self.title}｜{self.source}"


# 種類 → 表記名（説明文生成用）
_KIND_JA = {Kind.KANA: "平仮名", Kind.KANJI: "漢字", Kind.OTHER: "この表記"}


class ClauseCitation:
    def __init__(self, norms: Optional[Norms] = None):
        self.norms = norms or load_norms()

    def cite(self, clause_id: Optional[str]) -> Optional[Citation]:
        if not clause_id:
            return None
        c = self.norms.clause(clause_id)
        if c is None:
            return None
        return Citation(
            clause_id=c.id,
            title=c.title,
            summary=c.summary,
            source=self.norms.source_citation(c.id),
            note=c.note,
        )

    def explain(self, finding: Finding) -> str:
        """『なぜ直すのか』の説明文（否定しない言い回し）。"""
        cit = self.cite(finding.clause_id)
        if cit is None:
            return finding.message or "規範に照らした表記の候補です。"

        if finding.profile_note:
            # 組織プロファイル優先の明示
            return (f"組織の表記規則により「{finding.recommended_surface}」を推奨します"
                    f"（{finding.profile_note}）。参考: {cit.one_line()}")

        rec_ja = _KIND_JA.get(finding.recommended_kind, "この表記")
        if finding.recommended_kind in (Kind.KANA, Kind.KANJI):
            cat_ja = {
                "formal_noun": "形式名詞として",
                "substantive": "実質的な意味の語として",
                "aux_verb": "補助動詞として",
                "main_verb": "本来の意味の動詞として",
            }.get(finding.category, "")
            return (f"「{finding.surface}」は{cat_ja}用いられているため、"
                    f"規範では{rec_ja}で「{finding.recommended_surface}」と書きます。"
                    f"（根拠: {cit.one_line()}）")
        # 規則層（数字・句読点など）
        return f"{finding.message}（根拠: {cit.one_line()}）"

    def annotate(self, finding: Finding) -> Finding:
        """Finding に説明文を書き込んで返す（message が空なら埋める）。"""
        finding.message = self.explain(finding)
        return finding
