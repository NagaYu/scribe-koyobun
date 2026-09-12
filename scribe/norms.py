"""規範の構造化とアクセス層。

実証する主張: 「根拠提示」。指摘を規範のどの条項に基づくのか対応づけられるように、
条項(clauses.json)と対象語彙(lexicon.json)を読み込み、
  - 規則で解けるもの(rule_type="deterministic") と
  - 文脈依存のもの(rule_type="context")
を機械的に仕分けた一覧を提供する。この仕分けが Scribe 全体の設計の土台であり、
RuleLayer が扱う範囲と UsageClassifier が扱う範囲の境界を定義する。

規範は絶対ではない: 条項には出典(source_locator)を持たせ、本文は短い要約に留める
（原文の逐語転載を避ける）。厳格さは StyleProfile / severity で切り替える前提。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "norms"


@dataclass(frozen=True)
class Clause:
    """規範の 1 条項。"""

    id: str
    title: str
    rule_type: str            # "deterministic" | "context"
    category: str
    source: str               # sources のキー
    source_locator: str       # 一次資料の該当箇所
    summary: str              # 短い要約（逐語転載しない）
    counterpart: Optional[str] = None  # 対になる条項（かな/漢字の裏表）
    note: str = ""
    examples: List[str] = field(default_factory=list)

    @property
    def is_context_dependent(self) -> bool:
        return self.rule_type == "context"


@dataclass(frozen=True)
class Lexeme:
    """文脈依存判定の対象語。"""

    lemma: str
    kana: str
    kanji: List[str]
    category: str             # formal_noun / aux_verb / aux_adj ...
    kana_clause: str
    kanji_clause: str
    contested: bool = False   # 同一語が両用法で頻出 → hard 集合の中核
    kana_only: bool = False   # 常に仮名（漢字用法が存在しない）
    note: str = ""

    @property
    def kanji_primary(self) -> str:
        return self.kanji[0] if self.kanji else self.kana


class Norms:
    """条項と語彙のレジストリ。"""

    def __init__(self, clauses: Dict[str, Clause], lexemes: Dict[str, Lexeme], sources: dict):
        self._clauses = clauses
        self._lexemes = lexemes
        self.sources = sources

    # ---- 条項アクセス -------------------------------------------------
    def clause(self, clause_id: str) -> Optional[Clause]:
        return self._clauses.get(clause_id)

    @property
    def clauses(self) -> List[Clause]:
        return list(self._clauses.values())

    def deterministic_clauses(self) -> List[Clause]:
        """規則で解ける条項（RuleLayer の担当範囲）。"""
        return [c for c in self._clauses.values() if c.rule_type == "deterministic"]

    def context_clauses(self) -> List[Clause]:
        """文脈依存の条項（UsageClassifier の担当範囲）。"""
        return [c for c in self._clauses.values() if c.rule_type == "context"]

    # ---- 語彙アクセス -------------------------------------------------
    @property
    def lexemes(self) -> List[Lexeme]:
        return list(self._lexemes.values())

    def lexeme(self, lemma: str) -> Optional[Lexeme]:
        return self._lexemes.get(lemma)

    def contested_lexemes(self) -> List[Lexeme]:
        """両用法を持つ語（hard 集合の候補）。"""
        return [x for x in self._lexemes.values() if x.contested]

    def source_citation(self, clause_id: str) -> str:
        """条項 → 出典表記の文字列（カード/README/出力の脚注に使う）。"""
        c = self.clause(clause_id)
        if not c:
            return ""
        src = self.sources.get(c.source, {})
        title = src.get("title", c.source)
        date = src.get("date", "")
        return f"{title}（{date}）{c.source_locator}".strip("　 ")

    def partition_summary(self) -> Dict[str, List[str]]:
        """規則で解けるもの / 文脈依存のもの の仕分け一覧。設計の土台であり
        README とテストで参照する。"""
        return {
            "deterministic": [c.id for c in self.deterministic_clauses()],
            "context": [c.id for c in self.context_clauses()],
        }


def _load(data_dir: Path = _DATA_DIR) -> Norms:
    clauses_raw = json.loads((data_dir / "clauses.json").read_text(encoding="utf-8"))
    lex_raw = json.loads((data_dir / "lexicon.json").read_text(encoding="utf-8"))

    clauses: Dict[str, Clause] = {}
    for c in clauses_raw["clauses"]:
        clauses[c["id"]] = Clause(
            id=c["id"],
            title=c["title"],
            rule_type=c["rule_type"],
            category=c["category"],
            source=c["source"],
            source_locator=c.get("source_locator", ""),
            summary=c.get("summary", ""),
            counterpart=c.get("counterpart"),
            note=c.get("note", ""),
            examples=c.get("examples", []),
        )

    lexemes: Dict[str, Lexeme] = {}
    for e in lex_raw["entries"]:
        lexemes[e["lemma"]] = Lexeme(
            lemma=e["lemma"],
            kana=e["kana"],
            kanji=e.get("kanji", []),
            category=e["category"],
            kana_clause=e["kana_clause"],
            kanji_clause=e["kanji_clause"],
            contested=e.get("contested", False),
            kana_only=e.get("kana_only", False),
            note=e.get("note", ""),
        )

    return Norms(clauses, lexemes, clauses_raw["_meta"]["sources"])


@lru_cache(maxsize=4)
def load_norms(data_dir: Optional[str] = None) -> Norms:
    """規範レジストリを読み込む（キャッシュ付き）。"""
    return _load(Path(data_dir) if data_dir else _DATA_DIR)
