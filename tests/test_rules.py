"""規則層が確定的項目を取りこぼさないことを検証する。

主張『過剰指摘の少なさ』『規則層とモデル層の寄与分離』に対応: 規則で確実なもの
（数字・単位・括弧・句読点・確定送り仮名）は必ず拾い、正しい文には何も出さない。
"""
from scribe.collect import load_rule_cases
from scribe.rules import RuleLayer


def test_rule_layer_catches_all_expected():
    layer = RuleLayer()
    cases = load_rule_cases()
    assert cases, "rule_cases.jsonl が空"
    for case in cases:
        findings = layer.check(case["text"])
        got = {(f.surface, f.recommended_surface, f.clause_id) for f in findings if f.needs_change}
        for exp in case["expect"]:
            key = (exp["surface"], exp["recommended"], exp["clause"])
            assert key in got, f"取りこぼし: {key} in {case['text']!r} 実際={got}"


def test_rule_layer_no_false_positive_on_clean_text():
    layer = RuleLayer()
    for case in load_rule_cases():
        if case["expect"]:
            continue
        findings = [f for f in layer.check(case["text"]) if f.needs_change]
        assert findings == [], f"正しい文に誤指摘: {case['text']!r} -> {[f.surface for f in findings]}"


def test_rule_layer_does_not_touch_context_words():
    """規則層は こと/事 等の文脈依存語には触れない（モデル層の担当）。"""
    layer = RuleLayer()
    findings = layer.check("確認する事がある。住む所を探す。")
    surfaces = {f.surface for f in findings}
    assert "事" not in surfaces and "所" not in surfaces


def test_all_rule_findings_are_rule_layer():
    layer = RuleLayer()
    for f in layer.check("３名で行なう(重要)．"):
        assert f.layer.value == "rule"
        assert f.confidence == 1.0
        assert f.clause_id is not None
