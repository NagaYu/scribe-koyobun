"""scribe CLI — `scribe check doc.txt --profile jichitai`

実証する主張: 「根拠提示」を実務の入口で提供する。指摘ごとに、現在の表記・推奨表記・
根拠条項・説明を並べて返す。--profile で組織規則を優先し、その旨を明示する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .checker import ScribeChecker
from .datamodel import Finding
from .profile import StyleProfile, default_profile, load_profile

_PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


def resolve_profile(name: Optional[str], strictness: Optional[str]) -> StyleProfile:
    if not name:
        prof = default_profile()
    elif Path(name).exists():
        prof = load_profile(name)
    else:
        candidate = _PROFILE_DIR / f"{name}.yaml"
        if candidate.exists():
            prof = load_profile(str(candidate))
        else:
            raise SystemExit(f"プロファイルが見つかりません: {name}（{candidate} も不在）")
    if strictness:
        prof.strictness = strictness
    return prof


def _context(text: str, f: Finding, width: int = 12) -> str:
    s = max(0, f.span.start - width)
    e = min(len(text), f.span.end + width)
    before = text[s:f.span.start]
    tgt = text[f.span.start:f.span.end]
    after = text[f.span.end:e]
    return f"…{before}〖{tgt}〗{after}…".replace("\n", " ")


def render_text(text: str, findings: List[Finding], method: str) -> str:
    lines = [f"■ Scribe 判定（判定器: {method}）  指摘 {len(findings)} 件"]
    if not findings:
        lines.append("  規範に照らして修正候補は見つかりませんでした。")
        return "\n".join(lines)
    for i, f in enumerate(findings, 1):
        layer = "規則" if f.layer.value == "rule" else "文脈"
        arrow = f"「{f.surface}」→「{f.recommended_surface}」" if f.needs_change else f"「{f.surface}」(現状で可)"
        conf = f" 確信度{f.confidence:.0%}" if f.layer.value == "model" else ""
        lines.append(f"\n{i}. [{layer}]{conf} {arrow}")
        lines.append(f"   位置: {_context(text, f)}")
        lines.append(f"   {f.message}")
    return "\n".join(lines)


def cmd_check(args: argparse.Namespace) -> int:
    if args.text is not None:
        text = args.text
    else:
        p = Path(args.path)
        if not p.exists():
            raise SystemExit(f"ファイルが見つかりません: {p}")
        text = p.read_text(encoding="utf-8")

    profile = resolve_profile(args.profile, args.strictness)
    checker = ScribeChecker(profile=profile, model_dir=args.model)
    findings = checker.check(text, include_ok=args.show_ok)

    if args.json:
        out = {
            "method": checker.method,
            "profile": profile.name,
            "findings": [f.to_dict() for f in findings],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(render_text(text, findings, checker.method))
    return 0


def cmd_norms(args: argparse.Namespace) -> int:
    from .norms import load_norms
    norms = load_norms()
    part = norms.partition_summary()
    print("規則で解けるもの (RuleLayer):")
    for cid in part["deterministic"]:
        c = norms.clause(cid)
        print(f"  {cid}  {c.title}")
    print("\n文脈依存のもの (UsageClassifier):")
    for cid in part["context"]:
        c = norms.clause(cid)
        print(f"  {cid}  {c.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scribe", description="公用文の表記判定（文脈依存の漢字/かな使い分け）")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="文書を判定する")
    c.add_argument("path", nargs="?", help="対象ファイル（.txt）")
    c.add_argument("--text", help="ファイルの代わりに直接テキストを渡す")
    c.add_argument("--profile", help="組織プロファイル名（profiles/<name>.yaml）またはパス")
    c.add_argument("--strictness", choices=["strict", "normal", "loose"], help="厳格さの上書き")
    c.add_argument("--model", help="学習済みモデルのディレクトリ/HFリポジトリ（無ければヒューリスティック）")
    c.add_argument("--json", action="store_true", help="JSON で出力")
    c.add_argument("--show-ok", action="store_true", help="修正不要（現状で可）の判定も表示")
    c.set_defaults(func=cmd_check)

    n = sub.add_parser("norms", help="規則で解けるもの/文脈依存のものの仕分けを表示")
    n.set_defaults(func=cmd_norms)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
