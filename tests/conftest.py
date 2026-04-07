"""
Shared pytest fixtures.

Model-dependent tests are marked with @pytest.mark.slow and skipped by default.
Run them explicitly with:  pytest -m slow
"""

from __future__ import annotations

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow (model-loading) tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def fake_vectors():
    """A small set of random orthogonal-ish vectors over the 20 core emotions."""
    from emotion_scope.config import CORE_EMOTION_NAMES
    torch.manual_seed(0)
    d = 64
    vectors = {}
    for name in CORE_EMOTION_NAMES:
        v = torch.randn(d)
        vectors[name] = v / v.norm()
    return vectors


@pytest.fixture
def gemma_tokenizer():
    """Real Gemma 2 tokenizer, without loading the model weights."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    except Exception as e:
        pytest.skip(f"gemma-2-2b-it tokenizer unavailable: {e}")
