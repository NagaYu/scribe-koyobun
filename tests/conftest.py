"""共有フィクスチャ。"""
import pytest

from scribe.collect import build_seed_labeled
from scribe.model import HeuristicUsageClassifier
from scribe.norms import load_norms


@pytest.fixture(scope="session")
def norms():
    return load_norms()


@pytest.fixture(scope="session")
def seeds():
    return build_seed_labeled()


@pytest.fixture(scope="session")
def heuristic():
    return HeuristicUsageClassifier()
