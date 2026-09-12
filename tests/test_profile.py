"""組織プロファイルが規範より優先されることを検証する。

主張『組織規則の尊重と明示』: 規範が漢字を求める箇所でも、組織プロファイルが仮名を
指定していれば組織側を優先し、その旨(profile_note)を明示する。
"""
from pathlib import Path

from scribe.checker import ScribeChecker
from scribe.datamodel import Kind
from scribe.profile import default_profile, load_profile

_PROFILE = str(Path(__file__).resolve().parent.parent / "profiles" / "jichitai.yaml")

# 「重い物」は規範上は実質名詞なので漢字（物）。jichitai は もの→かな を強制する。
CONFLICT_TEXT = "重い物を運搬する。"


def _finding_for(findings, surface):
    for f in findings:
        if f.surface == surface:
            return f
    return None


def test_default_profile_follows_norm_kanji():
    checker = ScribeChecker(profile=default_profile())
    findings = checker.check(CONFLICT_TEXT, include_ok=True)
    f = _finding_for(findings, "物")
    assert f is not None
    assert f.recommended_kind == Kind.KANJI     # 規範どおり漢字
    assert f.needs_change is False              # 既に正しい → 過剰指摘しない
    assert f.profile_note is None


def test_org_profile_overrides_norm():
    checker = ScribeChecker(profile=load_profile(_PROFILE))
    findings = checker.check(CONFLICT_TEXT)
    f = _finding_for(findings, "物")
    assert f is not None, "組織上書きにより指摘が出るはず"
    assert f.recommended_kind == Kind.KANA          # 組織規則で かな
    assert f.recommended_surface == "もの"
    assert f.needs_change is True
    assert f.profile_note, "組織優先である旨が明示されていない"
    assert "規範では" not in f.message              # 否定でなく組織規則として説明
    assert "組織" in f.message


def test_org_term_rule():
    checker = ScribeChecker(profile=load_profile(_PROFILE))
    findings = checker.check("子供と障害者に配慮する。")
    surfaces = {(f.surface, f.recommended_surface) for f in findings}
    assert ("子供", "子ども") in surfaces
    assert ("障害者", "障がい者") in surfaces


def test_comma_style_is_profile_driven():
    # 既定は「、」。半角カンマは「、」へ。
    checker = ScribeChecker(profile=default_profile())
    findings = checker.check("第一に,第二に検討する。")
    f = _finding_for(findings, ",")
    assert f is not None and f.recommended_surface == "、"
