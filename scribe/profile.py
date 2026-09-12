"""StyleProfile — 組織ごとの表記規則を上書き設定として読み込む。

実証する主張: 「組織規則の尊重と明示」。自治体独自ルールや企業の用字用語集を上書き設定
として読み込み、規範と衝突する場合は組織側を優先し、その旨(profile_note)を明示する。
また文書の種類に応じて厳格さ(strictness)を切り替え、過剰指摘を抑えられるようにする
（規範を絶対視しない）。

YAML 例は profiles/ を参照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .datamodel import Finding, Kind, Layer, Span
from .rules import RuleConfig

# strictness → モデル指摘を表示する最小確信度。緩いほど閾値が上がり指摘が減る。
_STRICTNESS_THRESHOLD = {"strict": 0.5, "normal": 0.6, "loose": 0.75}


@dataclass
class TermRule:
    """組織の用字用語集による語単位の書換ルール。"""
    find: str
    use: str
    note: str = ""


@dataclass
class Override:
    """特定の対象語を常に指定表記に倒す上書き（規範より優先）。"""
    force: str          # "kana" | "kanji"
    note: str = ""


@dataclass
class StyleProfile:
    name: str = "default"
    comma: str = "、"
    strictness: str = "normal"     # strict | normal | loose（文書種別で切替）
    overrides: Dict[str, Override] = field(default_factory=dict)  # lemma -> Override
    term_rules: List[TermRule] = field(default_factory=list)
    disabled_categories: set = field(default_factory=set)
    disabled_clauses: set = field(default_factory=set)

    # --- 派生値 -------------------------------------------------------
    @property
    def min_confidence(self) -> float:
        return _STRICTNESS_THRESHOLD.get(self.strictness, 0.6)

    def to_rule_config(self) -> RuleConfig:
        cfg = RuleConfig(comma=self.comma)
        # カテゴリ無効化を規則層にも反映
        if "number" in self.disabled_categories:
            cfg.fix_fullwidth_digits = False
        if "unit" in self.disabled_categories:
            cfg.fix_fullwidth_latin = False
        if "bracket" in self.disabled_categories:
            cfg.fix_halfwidth_brackets = False
        if "punctuation" in self.disabled_categories:
            cfg.fix_punctuation = False
        if "okurigana" in self.disabled_categories:
            cfg.fix_okurigana = False
        return cfg

    def override_for(self, lemma: str) -> Optional[Override]:
        return self.overrides.get(lemma)

    def is_category_enabled(self, category: str) -> bool:
        return category not in self.disabled_categories

    def is_clause_enabled(self, clause_id: Optional[str]) -> bool:
        return clause_id not in self.disabled_clauses

    def term_findings(self, text: str) -> List[Finding]:
        """用字用語集による指摘（組織規則。規範に条項が無くても出す）。"""
        out: List[Finding] = []
        for tr in self.term_rules:
            start = 0
            while True:
                idx = text.find(tr.find, start)
                if idx < 0:
                    break
                out.append(Finding(
                    span=Span(idx, idx + len(tr.find)),
                    surface=tr.find, lemma=tr.find, category="org_term",
                    current_kind=Kind.OTHER, recommended_kind=Kind.OTHER,
                    recommended_surface=tr.use, needs_change=(tr.find != tr.use),
                    layer=Layer.RULE, clause_id=None, confidence=1.0,
                    message=f"組織の用字用語により「{tr.use}」を用います。",
                    profile_note=tr.note or f"{self.name} 用字用語集", source="profile:term",
                ))
                start = idx + len(tr.find)
        return out


def load_profile(path: str) -> StyleProfile:
    """YAML から StyleProfile を読み込む。"""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    overrides = {}
    for lemma, spec in (data.get("overrides") or {}).items():
        if isinstance(spec, str):
            overrides[lemma] = Override(force=spec)
        else:
            overrides[lemma] = Override(force=spec["force"], note=spec.get("note", ""))
    term_rules = [TermRule(find=t["find"], use=t["use"], note=t.get("note", ""))
                  for t in (data.get("term_rules") or [])]
    return StyleProfile(
        name=data.get("name", Path(path).stem),
        comma=data.get("comma", "、"),
        strictness=data.get("strictness", "normal"),
        overrides=overrides,
        term_rules=term_rules,
        disabled_categories=set(data.get("disabled_categories", []) or []),
        disabled_clauses=set(data.get("disabled_clauses", []) or []),
    )


def default_profile() -> StyleProfile:
    return StyleProfile()
