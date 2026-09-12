"""collect.py — 正例の収集と、公開文書取り込みの枠組み。

実証する主張: 「hard 集合の正解率」を測るための、文脈が本物の正例づくり。

方針（制約より）:
  - 官公庁・自治体の公開文書は利用条件を必ず確認し、本文をそのまま再配布しない。
    このリポジトリに同梱するのは、著作権・再配布の問題を避けるための合成種文
    （原文の転載ではない、公用文体で自作した文）に限る。実データからの学習は
    DocumentSource 経由で各自の環境で行う前提とし、出典と利用条件はカードに明記する。
  - 種文は『同一語が両用法で出現する』ように設計してあり、hard 集合を分離できる。

種文はすべて自作（CC0）。表層と gold_kind だけを注記し、スパンは検出器から導出するので
オフセットの手作業ミスが起きない（build_seed_labeled が整合を検証する）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .analysis import Analyzer, find_targets, get_analyzer
from .datamodel import LabeledSpan
from .norms import Norms, load_norms

SEED_LICENSE = "CC0-1.0"
SEED_SOURCE = "synthetic-seed"

# (本文, [(検出されるべき表層, 正しい種類)], 文書種別)
# 文書種別: tsuchi(通知) / hourei(法令調) / kouhou(広報) — strictness の説明にも使う。
SEED_SENTENCES: List[Tuple[str, List[Tuple[str, str]], str]] = [
    # --- こと / 事 -----------------------------------------------------
    ("申請の内容について確認することが必要である。", [("こと", "kana")], "tsuchi"),
    ("提出された書類に不備がないことを確認する。", [("こと", "kana")], "tsuchi"),
    ("期限までに手続を終えることが求められる。", [("こと", "kana")], "hourei"),
    ("委員会で事の是非を論じる。", [("事", "kanji")], "hourei"),
    ("事の重大さに鑑み、速やかに対応する。", [("事", "kanji")], "tsuchi"),
    ("報告書には事の次第を記載する。", [("事", "kanji")], "tsuchi"),

    # --- とき / 時 -----------------------------------------------------
    ("事故が発生したときは直ちに連絡する。", [("とき", "kana")], "tsuchi"),
    ("申請を行うときの手数料を定める。", [("とき", "kana")], "hourei"),
    ("災害が起こったときに備える。", [("とき", "kana")], "kouhou"),
    ("その時が来たら速やかに移行する。", [("時", "kanji"), ("来", "kanji")], "kouhou"),
    ("時を移さず措置を講じる。", [("時", "kanji")], "hourei"),

    # --- ところ / 所 ---------------------------------------------------
    ("現在のところ支障は生じていない。", [("ところ", "kana")], "tsuchi"),
    ("調査を進めているところである。", [("ところ", "kana")], "tsuchi"),
    ("御指摘のところを修正した。", [("ところ", "kana")], "tsuchi"),
    ("住む所を確保する必要がある。", [("所", "kanji")], "kouhou"),
    ("勤める所を市内に移す。", [("所", "kanji")], "kouhou"),

    # --- もの / 物 -----------------------------------------------------
    ("正しいものと認められる。", [("もの", "kana")], "hourei"),
    ("必要なものを準備しておく。", [("もの", "kana"), ("おく", "kana")], "tsuchi"),
    ("該当するものは届け出ること。", [("もの", "kana"), ("こと", "kana")], "hourei"),
    ("重い物を運搬する際は注意する。", [("物", "kanji")], "kouhou"),
    ("古い物を廃棄して整理する。", [("物", "kanji")], "kouhou"),

    # --- わけ / うち / とおり / ため -----------------------------------
    ("要望を拒むわけにはいかない。", [("わけ", "kana")], "tsuchi"),
    ("検討を重ねているうちに時間を要した。", [("うち", "kana")], "tsuchi"),
    ("三日のうちに回答する。", [("うち", "kana")], "tsuchi"),
    ("次のとおり実施する。", [("とおり", "kana")], "tsuchi"),
    ("記録のとおり処理した。", [("とおり", "kana")], "tsuchi"),
    ("確認のため書類を添付する。", [("ため", "kana")], "tsuchi"),

    # --- いただく / 頂く -----------------------------------------------
    ("御確認いただくようお願いする。", [("いただく", "kana")], "tsuchi"),
    ("書類を御提出いただく。", [("いただく", "kana")], "tsuchi"),
    ("記念品を頂く。", [("頂く", "kanji")], "kouhou"),
    ("祝いの品を頂いた。", [("頂い", "kanji")], "kouhou"),

    # --- ください / くださる / 下さい / 下さる --------------------------
    ("こちらに御記入ください。", [("ください", "kana")], "tsuchi"),
    ("詳細は担当まで御連絡ください。", [("ください", "kana")], "tsuchi"),
    ("資料を下さい。", [("下さい", "kanji")], "kouhou"),
    ("御指導くださる方に感謝する。", [("くださる", "kana")], "kouhou"),
    ("賞状を下さる。", [("下さる", "kanji")], "kouhou"),

    # --- みる / 見る ---------------------------------------------------
    ("一度検討してみる。", [("みる", "kana")], "tsuchi"),
    ("運用状況を見て判断する。", [("見", "kanji")], "tsuchi"),
    ("現地の状況を見る。", [("見る", "kanji")], "tsuchi"),

    # --- おく / 置く ---------------------------------------------------
    ("資料を事前に配布しておく。", [("おく", "kana")], "tsuchi"),
    ("書類を机に置いておく。", [("置い", "kanji"), ("おく", "kana")], "tsuchi"),
    ("荷物を棚に置く。", [("置く", "kanji")], "kouhou"),

    # --- いく / 行く ---------------------------------------------------
    ("段階的に整備を進めていく。", [("いく", "kana")], "tsuchi"),
    ("担当者が現地へ行く。", [("行く", "kanji")], "tsuchi"),

    # --- くる / 来る ---------------------------------------------------
    ("寒さが増してくる。", [("くる", "kana")], "kouhou"),
    ("担当者が来る。", [("来る", "kanji")], "tsuchi"),

    # --- 意味的に難しい事例（ヒューリスティックの被覆ギャップ） -----------
    # 統語だけでは解けず、実質語かどうかは意味に依存する。規則ベース近似が取りこぼす
    # ことが想定される事例。ニューラル版が実データから学習して埋める担当領域。
    ("壊れた物を回収する。", [("物", "kanji")], "kouhou"),        # 壊れた=連体形だが物理物→漢字
    ("会場となる所を下見する。", [("所", "kanji")], "kouhou"),      # となる=連体形だが場所→漢字
    ("時を大切にする。", [("時", "kanji")], "kouhou"),            # 時刻的な実質語→漢字
]


@dataclass
class DocumentSource:
    """実データ取り込みの記述（学習用コーパスの出典と利用条件）。

    実際の収集は各自の環境で行う。ここでは出典・利用条件・再配布可否を必ず記録し、
    データセットカードへ転記できるようにする。本文の同梱・再配布はしない。
    """
    name: str
    url: str
    license: str
    redistributable: bool
    note: str = ""


# 参考: 公開文書の代表的な取り込み候補（利用条件は取得時点で必ず再確認すること）。
KNOWN_SOURCES: List[DocumentSource] = [
    DocumentSource("政府白書（e-Gov 等で公開）", "https://www.e-gov.go.jp/",
                   "各府省の利用規約（多くは出典明示で利用可）", False,
                   "本文の再配布可否は文書ごとに異なる。断片引用に留める。"),
    DocumentSource("自治体の例規集・広報（各自治体サイト）", "",
                   "自治体ごとの利用規約（CC BY 等の例あり）", False,
                   "CC BY 等であれば出典明示で利用可。要個別確認。"),
    DocumentSource("法令（e-Gov 法令検索）", "https://laws.e-gov.go.jp/",
                   "e-Gov 法令データ 利用規約", False,
                   "条文は規範に沿って書かれている前提だが、hard 集合で検証する。"),
]


def collect_from_source(source: DocumentSource, texts: List[str]) -> List[str]:
    """外部文書テキストの取り込み口（正規化のみ。ライセンス確認は呼び出し側の責務）。

    再配布不可のソースは本文をリポジトリに保存しない。ここでは軽い正規化だけ行い、
    学習パイプラインへ渡すことを想定する。
    """
    if not source.redistributable:
        # 再配布不可: 本文は保存せず、その場の学習にのみ用いる想定。
        pass
    return [_normalize(t) for t in texts]


def _normalize(text: str) -> str:
    return text.replace("　", " ").strip()


def build_seed_labeled(norms: Optional[Norms] = None,
                       analyzer: Optional[Analyzer] = None) -> List[LabeledSpan]:
    """同梱の合成種文から LabeledSpan（正例）を構築する。

    検出器で対象語を見つけ、注記(gold_kind)と突き合わせて厳密なラベルにする。
    表層の並びが注記と一致しない場合はエラー（種文の記述ミスを早期に検出する）。
    """
    norms = norms or load_norms()
    analyzer = analyzer or get_analyzer()
    out: List[LabeledSpan] = []
    for i, (text, annos, doc_type) in enumerate(SEED_SENTENCES):
        occs = find_targets(text, norms, analyzer)
        if len(occs) != len(annos):
            raise ValueError(
                f"種文 {i} の対象数不一致: text={text!r} 検出={[o.surface for o in occs]} 注記={annos}")
        for occ, (exp_surface, gold) in zip(occs, annos):
            if occ.surface != exp_surface:
                raise ValueError(
                    f"種文 {i} 表層不一致: text={text!r} 検出={occ.surface!r} 注記={exp_surface!r}")
            lex = norms.lexeme(occ.lemma)
            clause = lex.kana_clause if gold == "kana" else lex.kanji_clause
            out.append(LabeledSpan(
                text=text, span=occ.span, surface=occ.surface, lemma=occ.lemma,
                category=occ.category, gold_kind=gold, clause_id=clause,
                flipped=False, doc_id=f"seed-{i:03d}", source=SEED_SOURCE, license=SEED_LICENSE,
            ))
    return mark_hard(out)


def mark_hard(labeled: List[LabeledSpan]) -> List[LabeledSpan]:
    """hard 集合の付与: 同一語(lemma)がコーパス内で両用法(kana/kanji)で出現する事例。

    ここでの正解率が主要指標。一律置換は原理的に片方を必ず誤るため、既存手法との差が
    最も明確に出る集合。
    """
    kinds_by_lemma: Dict[str, set] = {}
    for x in labeled:
        kinds_by_lemma.setdefault(x.lemma, set()).add(x.gold_kind)
    hard_lemmas = {k for k, v in kinds_by_lemma.items() if len(v) >= 2}
    for x in labeled:
        x.is_hard = x.lemma in hard_lemmas
    return out_sorted(labeled)


def out_sorted(labeled: List[LabeledSpan]) -> List[LabeledSpan]:
    return sorted(labeled, key=lambda x: (x.doc_id, x.span.start))


def load_rule_cases() -> List[dict]:
    """規則層のテストデータ（数字・単位・括弧・句読点・送り仮名）。別途整理する層。"""
    path = Path(__file__).resolve().parent.parent / "data" / "seeds" / "rule_cases.jsonl"
    if not path.exists():
        return []
    import json
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
