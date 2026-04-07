"""Tests for the color mapping — no model required."""

import re

from emotion_scope.visualize import emotion_to_emoji, scores_to_color

HEX_RE = re.compile(r"^#[0-9a-f]{6}$")


def test_scores_to_color_returns_valid_hex():
    scores = {"happy": 0.8, "sad": 0.1, "calm": 0.3}
    hex_color, hsv, valence, arousal = scores_to_color(scores)
    assert HEX_RE.match(hex_color)
    assert len(hsv) == 3
    assert -1.0 <= valence <= 1.0
    assert -1.0 <= arousal <= 1.0


def test_scores_to_color_all_zero():
    """Empty / all-zero scores should not crash and should return a valid color."""
    hex_color, _, valence, arousal = scores_to_color({})
    assert HEX_RE.match(hex_color)
    assert valence == 0.0
    assert arousal == 0.0


def test_positive_vs_negative_valence_differ():
    pos_hex, _, pos_val, _ = scores_to_color({"happy": 1.0, "loving": 1.0})
    neg_hex, _, neg_val, _ = scores_to_color({"sad": 1.0, "gloomy": 1.0})
    assert pos_val > neg_val
    assert pos_hex != neg_hex


def test_negative_scores_are_clipped():
    """Negative activations should not contribute."""
    hex_a, _, val_a, _ = scores_to_color({"happy": 1.0})
    hex_b, _, val_b, _ = scores_to_color({"happy": 1.0, "sad": -5.0})
    assert val_a == val_b


def test_emotion_to_emoji():
    assert emotion_to_emoji("happy") != "?"
    assert emotion_to_emoji("nonexistent_emotion") == "?"
