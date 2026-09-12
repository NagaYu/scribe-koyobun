"""テスト共通ヘルパ。"""


def predict_kind(clf, labeled):
    """LabeledSpan の該当スパンに対する予測種類（"kana"/"kanji"）を返す。"""
    for p in clf.classify_text(labeled.text):
        if p.occurrence and p.occurrence.span == labeled.span:
            return p.kind.value
    return None
