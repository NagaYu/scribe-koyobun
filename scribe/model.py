"""UsageClassifier — 文脈依存の漢字/かな使い分けを判定する本体。

実証する主張: 「hard 集合の正解率」。対象語そのものではなく、その語の統語的な働き
（連体修飾・て接続・格関係・修飾語の意味クラス）から漢字/かなを判定する。一律置換が
同一語をどちらか一方に必ず倒すのに対し、ここでは同じ語を文脈ごとに振り分ける。

二つの実装を持つ:
  - HeuristicUsageClassifier: fugashi の統語情報に基づく規則ベースの近似。学習済み
    モデルが無くても動く軽量ベースライン。網羅性には限界がある（特に『所/物/時』の
    意味的な判別）ことを明示する。この限界こそがニューラル版の存在理由。
  - NeuralUsageClassifier: HuggingFace の token 分類モデル（目標 0.1〜0.2B、CPU 即応）。
    学習(scripts/train.py)・書き出し(scripts/export.py)の対象。実データから統語的手掛かりを
    学習し、ヒューリスティックの被覆ギャップを埋める production 経路。

UsageClassifier はモデルがあればニューラル、無ければヒューリスティックを選ぶ facade。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .analysis import Analyzer, Token, find_targets, get_analyzer
from .datamodel import Kind, TargetOccurrence
from .norms import Norms, load_norms


@dataclass
class UsagePrediction:
    kind: Kind                 # 推定される正しい表記の種類
    confidence: float          # 0..1
    clause_id: str             # 根拠条項
    method: str                # "heuristic" | "neural"
    occurrence: Optional[TargetOccurrence] = None


# --- ヒューリスティックが使う意味クラス手掛かり（語彙特化・被覆は限定的） -----------
# これらは「本来は実質語（漢字）」を示す統語・意味の目印。ニューラル版はこれらを
# データから学習する前提で、ここでは代表的なものだけを列挙する。
_PHYSICAL_ADJ = {"重い", "軽い", "固い", "堅い", "柔らかい", "新しい", "古い",
                 "大きい", "小さい", "鋭い", "丸い", "熱い", "冷たい", "白い", "赤い"}
_PHYSICAL_VERB = {"運ぶ", "置く", "持つ", "作る", "壊す", "捨てる", "拾う", "食べる",
                  "投げる", "包む", "並べる", "動かす", "渡す"}
_PLACE_VERB_PREV = {"住む", "勤める", "泊まる", "働く", "暮らす", "宿る"}
_TEMPORAL_VERB = {"来る", "過ぎる", "迫る", "移す", "経つ", "至る", "訪れる", "待つ"}
_KOTO_SUBSTANTIVE_HEAD = {"重大", "次第", "真相", "大小", "軽重", "善悪", "是非", "発端", "顛末"}
_DEMONSTRATIVE = {"その", "この", "あの", "かの"}


class HeuristicUsageClassifier:
    """統語情報に基づく規則ベースの用法判定（学習不要のベースライン）。

    補助動詞（て接続 vs 格関係）は統語だけで高精度に判別できる。形式名詞は連体修飾で
    かな側を高確度に確認できる一方、実質名詞（所/物/時）の意味的判別は限定的な手掛かり
    でしか拾えない — この被覆ギャップがニューラル版の担当領域であることを設計上明示する。
    """

    method = "heuristic"

    def __init__(self, norms: Optional[Norms] = None, analyzer: Optional[Analyzer] = None):
        self.norms = norms or load_norms()
        self.analyzer = analyzer or get_analyzer()

    # --- 公開 API -----------------------------------------------------
    def classify_text(self, text: str) -> List[UsagePrediction]:
        toks = self.analyzer.tokens(text)
        targets = find_targets(text, self.norms, self.analyzer)
        return [self._classify(occ, toks) for occ in targets]

    def classify_occurrence(self, text: str, occ: TargetOccurrence) -> UsagePrediction:
        return self._classify(occ, self.analyzer.tokens(text))

    # --- 内部 ---------------------------------------------------------
    def _token_index(self, toks: Sequence[Token], occ: TargetOccurrence) -> int:
        for i, t in enumerate(toks):
            if t.start == occ.span.start:
                return i
        # フォールバック: 包含関係で探す
        for i, t in enumerate(toks):
            if t.start <= occ.span.start < t.end:
                return i
        return -1

    def _classify(self, occ: TargetOccurrence, toks: Sequence[Token]) -> UsagePrediction:
        i = self._token_index(toks, occ)
        prev = toks[i - 1] if i - 1 >= 0 else None
        nxt = toks[i + 1] if 0 <= i and i + 1 < len(toks) else None
        following = list(toks[i + 1:i + 5])

        if occ.kana_only:
            return UsagePrediction(Kind.KANA, 0.98, occ.kana_clause, self.method, occ)

        if occ.category == "aux_verb":
            return self._classify_aux(occ, prev)
        if occ.category in ("formal_noun",):
            return self._classify_formal(occ, prev, nxt, following)
        # aux_adj など未対応カテゴリは既定でかな
        return UsagePrediction(Kind.KANA, 0.5, occ.kana_clause, self.method, occ)

    def _classify_aux(self, occ: TargetOccurrence, prev: Optional[Token]) -> UsagePrediction:
        # 補助動詞: 直前が「て/で」接続 → 補助 → かな
        if prev is not None and (
            (prev.pos2 == "接続助詞" and prev.surface in ("て", "で"))
            or prev.surface.endswith(("て", "で"))
        ):
            return UsagePrediction(Kind.KANA, 0.9, occ.kana_clause, self.method, occ)
        # 直前が格助詞（を/に/が/へ/と）→ 目的語・対象をとる本動詞 → 漢字
        if prev is not None and prev.pos2 == "格助詞":
            return UsagePrediction(Kind.KANJI, 0.85, occ.kanji_clause, self.method, occ)
        # 直前がサ変名詞・接頭辞（御記入ください／連絡いただく）→ 補助 → かな
        if prev is not None and prev.pos1 in ("名詞", "接頭辞"):
            return UsagePrediction(Kind.KANA, 0.75, occ.kana_clause, self.method, occ)
        # 既定: 公用文の依頼表現は補助動詞（かな）に倒れやすい
        return UsagePrediction(Kind.KANA, 0.55, occ.kana_clause, self.method, occ)

    def _classify_formal(self, occ: TargetOccurrence, prev: Optional[Token],
                         nxt: Optional[Token], following: Sequence[Token]) -> UsagePrediction:
        # 1) 実質名詞シグナル（漢字）を先に判定
        if self._substantive_signal(occ.lemma, prev, nxt, following):
            return UsagePrediction(Kind.KANJI, 0.7, occ.kanji_clause, self.method, occ)
        # 2) 直前が用言の連体形 → 連体修飾を名詞化 → 形式名詞 → かな（強い手掛かり）
        if prev is not None and prev.is_yougen and prev.is_rentai:
            return UsagePrediction(Kind.KANA, 0.85, occ.kana_clause, self.method, occ)
        # 3) 直前が「の」→ 多くは形式名詞（次のとおり・事故のとき）→ かな
        if prev is not None and prev.surface == "の":
            return UsagePrediction(Kind.KANA, 0.7, occ.kana_clause, self.method, occ)
        # 4) 既定: 形式名詞（かな）。規範の既定であり、実質名詞の取りこぼしは既知の限界。
        return UsagePrediction(Kind.KANA, 0.55, occ.kana_clause, self.method, occ)

    @staticmethod
    def _substantive_signal(lemma: str, prev: Optional[Token], nxt: Optional[Token],
                            following: Sequence[Token]) -> bool:
        """実質名詞（漢字が正しい）である統語・意味の手掛かり。被覆は限定的。"""
        follow_lemmas = {t.lemma for t in following}
        if lemma == "こと":
            # 「事の重大さ」等の慣用
            if nxt is not None and nxt.surface == "の":
                return bool(follow_lemmas & _KOTO_SUBSTANTIVE_HEAD)
            return False
        if lemma == "とき":
            if prev is not None and prev.pos1 == "連体詞" and prev.surface in _DEMONSTRATIVE:
                return True  # その時が来た
            if nxt is not None and nxt.surface in ("を", "が") and (follow_lemmas & _TEMPORAL_VERB):
                return True  # 時を移さず／時が来る
            return False
        if lemma == "ところ":
            if prev is not None and prev.lemma in _PLACE_VERB_PREV:
                return True  # 住む所
            return False
        if lemma == "もの":
            if prev is not None and prev.lemma in _PHYSICAL_ADJ:
                return True  # 重い物
            if nxt is not None and nxt.surface == "を" and (follow_lemmas & _PHYSICAL_VERB):
                return True  # 物を運ぶ
            return False
        # わけ・とおり・うち はほぼ常に形式名詞（かな）
        return False


class NeuralUsageClassifier:
    """HuggingFace token 分類モデルのラッパ（0.1〜0.2B, CPU 即応）。

    ラベル体系: O / TARGET-KANA / TARGET-KANJI。監視語の各出現トークンに
    『あるべき表記』を付与する token 分類として学習する（scripts/train.py 参照）。
    入力の表層は正誤どちらもあり得るため、モデルは表層ではなく文脈から判定を学ぶ。

    実証する主張: 「hard 集合の正解率」＋「速度」。ヒューリスティックが取りこぼす
    意味的な実質名詞判別をデータから学習し、量子化して CPU で即応させる。
    """

    method = "neural"
    LABELS = ["O", "TARGET-KANA", "TARGET-KANJI"]

    def __init__(self, model_dir: str, norms: Optional[Norms] = None,
                 analyzer: Optional[Analyzer] = None, device: str = "cpu"):
        # 遅延 import: torch/transformers はコア依存に含めない
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.norms = norms or load_norms()
        self.analyzer = analyzer or get_analyzer()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.hf_model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()

    def classify_text(self, text: str) -> List[UsagePrediction]:
        targets = find_targets(text, self.norms, self.analyzer)
        if not targets:
            return []
        enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt",
                             truncation=True, max_length=256)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            logits = self.hf_model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        probs = self.torch.softmax(logits, dim=-1)

        preds: List[UsagePrediction] = []
        for occ in targets:
            idx = self._subtoken_for_span(offsets, occ.span.start, occ.span.end)
            if idx is None:
                # モデルが拾えない場合はヒューリスティックに委譲
                preds.append(HeuristicUsageClassifier(self.norms, self.analyzer)
                             .classify_occurrence(text, occ))
                continue
            p = probs[idx]
            kana_p = float(p[self.LABELS.index("TARGET-KANA")])
            kanji_p = float(p[self.LABELS.index("TARGET-KANJI")])
            if kanji_p >= kana_p:
                preds.append(UsagePrediction(Kind.KANJI, kanji_p / (kana_p + kanji_p + 1e-9),
                                             occ.kanji_clause, self.method, occ))
            else:
                preds.append(UsagePrediction(Kind.KANA, kana_p / (kana_p + kanji_p + 1e-9),
                                             occ.kana_clause, self.method, occ))
        return preds

    @staticmethod
    def _subtoken_for_span(offsets, start: int, end: int) -> Optional[int]:
        for i, (a, b) in enumerate(offsets):
            if a == b:  # 特殊トークン
                continue
            if a <= start < b or (start <= a < end):
                return i
        return None


class UsageClassifier:
    """本体 facade。学習済みモデルがあればニューラル、無ければヒューリスティック。"""

    def __init__(self, model_dir: Optional[str] = None, norms: Optional[Norms] = None,
                 analyzer: Optional[Analyzer] = None):
        self.norms = norms or load_norms()
        self.analyzer = analyzer or get_analyzer()
        self._impl = None
        self.method = "heuristic"
        if model_dir:
            try:
                self._impl = NeuralUsageClassifier(model_dir, self.norms, self.analyzer)
                self.method = "neural"
            except Exception as e:  # モデル読込失敗時は縮退
                self._impl = None
                self._load_error = str(e)
        if self._impl is None:
            self._impl = HeuristicUsageClassifier(self.norms, self.analyzer)

    def classify_text(self, text: str) -> List[UsagePrediction]:
        return self._impl.classify_text(text)

    def classify_occurrence(self, text: str, occ: TargetOccurrence) -> UsagePrediction:
        if hasattr(self._impl, "classify_occurrence"):
            return self._impl.classify_occurrence(text, occ)
        # ニューラル実装は text 一括のみ → 該当 occ を取り出す
        for p in self._impl.classify_text(text):
            if p.occurrence and p.occurrence.span == occ.span:
                return p
        return HeuristicUsageClassifier(self.norms, self.analyzer).classify_occurrence(text, occ)
