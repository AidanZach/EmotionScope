"""Tests for utils — cosine matrices, valence separation, content token range."""

import pytest
import torch

from emotion_scope.utils import (
    average_pairwise_cosine,
    cosine_similarity_matrix,
    find_content_token_range,
    last_content_token_index,
    valence_separation,
)


def test_cosine_matrix_shape(fake_vectors):
    sim, names = cosine_similarity_matrix(fake_vectors)
    n = len(names)
    assert sim.shape == (n, n)
    # Diagonal should be ~1 (L2-normalized inputs)
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_cosine_matrix_symmetric(fake_vectors):
    sim, _ = cosine_similarity_matrix(fake_vectors)
    assert torch.allclose(sim, sim.T, atol=1e-6)


def test_valence_separation_with_random_vectors(fake_vectors):
    # Random unit vectors: separation should be near 0, not strongly negative
    sep = valence_separation(fake_vectors)
    assert -1.0 <= sep <= 1.0


def test_valence_separation_with_constructed_vectors():
    """If positive emotions all point one way and negatives the other, separation should be ~-1."""
    d = 16
    direction = torch.zeros(d)
    direction[0] = 1.0
    vectors = {}
    from emotion_scope.config import CORE_EMOTIONS
    for e in CORE_EMOTIONS:
        v = direction.clone() if e["valence"] > 0.3 else (-direction if e["valence"] < -0.3 else torch.randn(d))
        vectors[e["name"]] = v / (v.norm() + 1e-9)
    sep = valence_separation(vectors)
    assert sep < -0.5


def test_average_pairwise_cosine_identical_vectors():
    d = 8
    v = torch.randn(d)
    v = v / v.norm()
    vectors = {f"e{i}": v.clone() for i in range(5)}
    avg = average_pairwise_cosine(vectors)
    assert abs(avg - 1.0) < 1e-5


def test_find_content_token_range_no_markers():
    """With no special tokens, content range should be the whole sequence."""
    class DummyTokenizer:
        name_or_path = "some/unknown-model"
        added_tokens_decoder = {}
        all_special_ids = []
        def encode(self, text, add_special_tokens=False):
            return []
    ids = torch.tensor([1, 2, 3, 4, 5])
    start, end = find_content_token_range(ids, DummyTokenizer())
    assert (start, end) == (0, 5)


@pytest.mark.slow
def test_find_content_token_range_gemma(gemma_tokenizer):
    """With real Gemma 2 tokenizer, chat template tokens should be excluded."""
    messages = [{"role": "user", "content": "Hello world"}]
    text = gemma_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = gemma_tokenizer(text, return_tensors="pt")["input_ids"]
    start, end = find_content_token_range(ids, gemma_tokenizer)
    # The content range should exclude BOS, <start_of_turn>, role tokens, <end_of_turn>
    assert start > 0
    assert end <= ids.shape[1]
    # At least some content tokens should remain
    assert end > start
    idx = last_content_token_index(ids, gemma_tokenizer)
    assert idx == end - 1
