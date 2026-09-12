"""ScribeChecker — 4 つのコアを束ねる本体オーケストレーション。

処理順（設計の土台）:
  1. RuleLayer: 規則で確定する項目を処理（数字・単位・括弧・句読点・送り仮名）。
     ここで確実な箇所はモデルに渡さない。
  2. StyleProfile.term_findings: 組織の用字用語集による指摘。
  3. UsageClassifier: 文脈依存の漢字/かな使い分けを判定 → Finding 化。
     組織プロファイルの上書きを適用（規範と衝突する場合は組織優先＋明示）。
     strictness に応じて低確信度の指摘は抑制（過剰指摘を減らす）。
  4. ClauseCitation: すべての指摘に根拠条項を添える。

実証する主張: 「根拠提示」「過剰指摘の少なさ」「規則層とモデル層の寄与分離」を
一つの出力にまとめる。出力は利用者の文章を否定せず『規範ではこうなります』に留める。
"""

from __future__ import annotations

from typing import List, Optional

from .analysis import Analyzer, get_analyzer
from .cite import ClauseCitation
from .datamodel import Finding, Kind, Layer
from .model import UsageClassifier, UsagePrediction
from .norms import Lexeme, Norms, load_norms
from .profile import StyleProfile, default_profile
from .rules import RuleLayer


def _common_suffix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return a[len(a) - i:] if i else ""


def recommended_surface(lex: Lexeme, surface: str, current: Kind, target: Kind) -> str:
    """対象語の表層を、活用語尾を保ったまま目的の字種へ変換する。

    形式名詞は 1 形態素なので辞書形で置換。動詞は漢字語幹とかな語幹を送り仮名の共通接尾で
    切り分け、活用語尾を保って入れ替える（例: 頂い→いただい, ください→下さい）。
    不規則で変換が定まらない場合は辞書形を返す。
    """
    if current == target:
        return surface
    if lex.category == "formal_noun":
        return lex.kana if target == Kind.KANA else lex.kanji_primary
    kana_dict, kanji_dict = lex.kana, lex.kanji_primary
    suf = _common_suffix(kana_dict, kanji_dict)
    kana_stem = kana_dict[: len(kana_dict) - len(suf)]
    kanji_stem = kanji_dict[: len(kanji_dict) - len(suf)]
    try:
        if current == Kind.KANJI:  # 漢字 → かな
            if kanji_stem and surface.startswith(kanji_stem):
                return kana_stem + surface[len(kanji_stem):]
        else:                      # かな → 漢字
            if kana_stem and surface.startswith(kana_stem):
                return kanji_stem + surface[len(kana_stem):]
    except Exception:
        pass
    return kana_dict if target == Kind.KANA else kanji_dict


class ScribeChecker:
    def __init__(self, profile: Optional[StyleProfile] = None,
                 model_dir: Optional[str] = None,
                 norms: Optional[Norms] = None,
                 analyzer: Optional[Analyzer] = None):
        self.norms = norms or load_norms()
        self.analyzer = analyzer or get_analyzer()
        self.profile = profile or default_profile()
        self.rules = RuleLayer(self.profile.to_rule_config())
        self.classifier = UsageClassifier(model_dir=model_dir, norms=self.norms,
                                          analyzer=self.analyzer)
        self.citer = ClauseCitation(self.norms)

    @property
    def method(self) -> str:
        return self.classifier.method

    def check(self, text: str, include_ok: bool = False) -> List[Finding]:
        findings: List[Finding] = []

        # 1. 規則層
        findings += self.rules.check(text)
        # 2. 組織の用字用語集
        findings += self.profile.term_findings(text)
        # 3. 文脈依存の用法判定
        for pred in self.classifier.classify_text(text):
            f = self._to_finding(pred)
            if f is None:
                continue
            findings.append(f)

        # 4. 根拠を付与
        for f in findings:
            self.citer.annotate(f)

        findings.sort(key=lambda f: f.span.start)
        if include_ok:
            return findings
        return [f for f in findings if f.needs_change]

    # --- 内部 ---------------------------------------------------------
    def _to_finding(self, pred: UsagePrediction) -> Optional[Finding]:
        occ = pred.occurrence
        if occ is None:
            return None
        if not self.profile.is_category_enabled(occ.category):
            return None

        lex = self.norms.lexeme(occ.lemma)
        rec_kind = pred.kind
        clause = pred.clause_id
        profile_note = None
        confidence = pred.confidence

        # 組織プロファイルの上書き（規範より優先）
        ov = self.profile.override_for(occ.lemma)
        if ov is not None:
            forced = Kind.KANA if ov.force == "kana" else Kind.KANJI
            if forced != rec_kind:
                profile_note = ov.note or f"{self.profile.name} の表記規則"
            rec_kind = forced
            clause = lex.kana_clause if forced == Kind.KANA else lex.kanji_clause
            confidence = 1.0  # 組織規則は確定

        if not self.profile.is_clause_enabled(clause):
            return None

        rec_surface = recommended_surface(lex, occ.surface, occ.current_kind, rec_kind)
        needs = (occ.current_kind != rec_kind)

        # strictness による抑制: 上書きが無く確信度が閾値未満で、かつ変更を要する指摘は出さない
        if profile_note is None and needs and confidence < self.profile.min_confidence:
            return None

        return Finding(
            span=occ.span, surface=occ.surface, lemma=occ.lemma, category=occ.category,
            current_kind=occ.current_kind, recommended_kind=rec_kind,
            recommended_surface=rec_surface, needs_change=needs, layer=Layer.MODEL,
            clause_id=clause, confidence=confidence, message="",
            profile_note=profile_note, source=f"model:{pred.method}",
        )
