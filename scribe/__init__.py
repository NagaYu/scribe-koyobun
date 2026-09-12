"""Scribe — 公用文の文脈依存の漢字/かな使い分けを判定し、規範の条項を根拠として返す。

主張（README・評価と対応）:
  - hard 集合の正解率: 同一語の両用法を文脈で判別する（一律置換は原理的に片方を必ず誤る）。
  - 過剰指摘の少なさ: 規則で確実な箇所を規則層で処理し、文脈判定は確信度で抑制する。
  - 根拠提示: 指摘ごとに規範のどの条項に基づくかを添える。
  - 速度: 0.1〜0.2B の小型モデルを量子化し CPU で即応する。
"""

from .checker import ScribeChecker
from .datamodel import Finding, Kind, Layer, LabeledSpan, Span, TargetOccurrence
from .model import UsageClassifier, HeuristicUsageClassifier
from .norms import Norms, load_norms
from .profile import StyleProfile, default_profile, load_profile
from .rules import RuleLayer

__version__ = "0.1.0"

__all__ = [
    "ScribeChecker",
    "UsageClassifier",
    "HeuristicUsageClassifier",
    "RuleLayer",
    "StyleProfile",
    "default_profile",
    "load_profile",
    "Norms",
    "load_norms",
    "Finding",
    "Kind",
    "Layer",
    "Span",
    "LabeledSpan",
    "TargetOccurrence",
    "__version__",
]
