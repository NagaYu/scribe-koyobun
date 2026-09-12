"""app.py — Scribe の Gradio Space。

文書を貼ると、Scribe の指摘（推奨表記＋規範の条項の根拠）を表示し、
prh(辞書・正規表現の一律置換)の結果と並置する。

必須のデモ: 「一律置換だと実質名詞まで平仮名にしてしまう」実例を用意する。
既定の例文には『重い物を運ぶ』『住む所を確保する』のような実質名詞を含め、
prh が誤って かな 化する一方 Scribe は文脈で漢字を保つことを示す。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from scribe.analysis import find_targets
from scribe.checker import ScribeChecker
from scribe.datamodel import Finding, Kind
from scribe.norms import load_norms
from scribe.profile import default_profile, load_profile

_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
_NORMS = load_norms()

EXAMPLE_TEXT = (
    "申請の内容を確認する事が必要である。事故のとき，速やかに御連絡下さい。\n"
    "重い物を運搬する場合は，３名で行なうこと。住む所を確保しておく。\n"
    "正しいものと認められる書類を提出していただく。"
)


def _profile(name: str):
    if name and name != "（既定：公用文作成の考え方）":
        p = _PROFILE_DIR / f"{name}.yaml"
        if p.exists():
            return load_profile(str(p))
    return default_profile()


def apply_findings(text: str, findings: List[Finding]) -> str:
    """needs_change の指摘を反映して修正後テキストを作る（右から置換）。"""
    out = text
    for f in sorted([f for f in findings if f.needs_change], key=lambda x: x.span.start, reverse=True):
        out = out[:f.span.start] + f.recommended_surface + out[f.span.end:]
    return out


def prh_uniform_rewrite(text: str) -> Tuple[str, List[str]]:
    """prh 相当の一律置換: 監視語（形式名詞・補助動詞）が漢字なら文脈を見ず かな へ。

    実質名詞・本動詞（漢字が正しい）まで かな 化してしまう箇所を『過剰変換』として列挙する。
    """
    checker = ScribeChecker(profile=default_profile())
    scribe_findings = {(f.span.start, f.span.end): f for f in checker.check(text, include_ok=True)
                       if f.layer.value == "model"}
    occs = find_targets(text, _NORMS)
    edits = []
    over = []
    for occ in occs:
        if occ.current_kind != Kind.KANJI:
            continue
        lex = _NORMS.lexeme(occ.lemma)
        if lex.kana_only:
            continue
        # 一律に かな へ
        kana = lex.kana
        edits.append((occ.span.start, occ.span.end, kana))
        sf = scribe_findings.get((occ.span.start, occ.span.end))
        # Scribe が『漢字が正しい』と判断している箇所 = 一律置換の過剰変換
        if sf is not None and sf.recommended_kind == Kind.KANJI:
            over.append(f"「{occ.surface}」→「{kana}」は誤り（実質語のため漢字が正しい）")
    out = text
    for s, e, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        out = out[:s] + rep + out[e:]
    return out, over


def _render_findings(text: str, findings: List[Finding]) -> str:
    if not findings:
        return "**指摘なし**（規範に照らして修正候補は見つかりませんでした）"
    rows = ["| # | 種別 | 現在 → 推奨 | 確信度 | 根拠 |", "|---|---|---|---|---|"]
    for i, f in enumerate(findings, 1):
        layer = "規則" if f.layer.value == "rule" else "文脈"
        conf = f"{f.confidence:.0%}" if f.layer.value == "model" else "—"
        change = f"`{f.surface}` → `{f.recommended_surface}`" if f.needs_change else f"`{f.surface}`(可)"
        cid = f.clause_id or ""
        rows.append(f"| {i} | {layer} | {change} | {conf} | {cid} |")
    # 詳細説明
    detail = ["", "#### 根拠の詳細"]
    for i, f in enumerate(findings, 1):
        if f.needs_change:
            detail.append(f"{i}. {f.message}")
    return "\n".join(rows + detail)


def analyze(text: str, profile_name: str):
    profile = _profile(profile_name)
    checker = ScribeChecker(profile=profile)
    findings = checker.check(text)
    scribe_md = f"**判定器: {checker.method}｜プロファイル: {profile.name}**\n\n" + _render_findings(text, findings)
    scribe_fixed = apply_findings(text, findings)

    prh_fixed, over = prh_uniform_rewrite(text)
    prh_md = ["**prh（辞書・正規表現の一律置換）**", "", "修正後テキスト:", "", f"> {prh_fixed}".replace("\n", "  \n> ")]
    if over:
        prh_md += ["", "⚠️ **過剰変換（実質語まで かな 化）**:"]
        prh_md += [f"- {o}" for o in over]
    else:
        prh_md += ["", "（この文では過剰変換は検出されませんでした）"]
    return scribe_md, scribe_fixed, "\n".join(prh_md)


def build_demo():
    import gradio as gr

    profiles = ["（既定：公用文作成の考え方）"] + sorted(
        p.stem for p in _PROFILE_DIR.glob("*.yaml"))

    with gr.Blocks(title="Scribe｜公用文の表記判定") as demo:
        gr.Markdown(
            "# Scribe — 公用文の表記判定\n"
            "辞書・正規表現では解けない **文脈依存の漢字/かな使い分け** を判定し、"
            "**規範の条項** を根拠として返します。右側は prh 相当の一律置換で、"
            "**実質名詞まで平仮名にしてしまう** 例を確認できます。")
        with gr.Row():
            inp = gr.Textbox(label="文書を貼り付け", value=EXAMPLE_TEXT, lines=6)
        with gr.Row():
            prof = gr.Dropdown(profiles, value=profiles[0], label="組織プロファイル（規範と衝突時は組織優先）")
            btn = gr.Button("判定する", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Scribe の指摘（根拠つき）")
                scribe_out = gr.Markdown()
                gr.Markdown("**Scribe 修正後**")
                scribe_fixed = gr.Textbox(label="", lines=4)
            with gr.Column():
                gr.Markdown("### prh（一律置換）との比較")
                prh_out = gr.Markdown()
        btn.click(analyze, [inp, prof], [scribe_out, scribe_fixed, prh_out])
        demo.load(analyze, [inp, prof], [scribe_out, scribe_fixed, prh_out])
    return demo


if __name__ == "__main__":
    build_demo().launch()
