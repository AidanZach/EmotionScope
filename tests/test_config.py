"""Smoke tests for the config module — no model required."""

from emotion_scope.config import (
    CORE_EMOTIONS,
    CORE_EMOTION_NAMES,
    ExtractionConfig,
    ModelConfig,
    ProbeConfig,
    ValidationThresholds,
)


def test_core_emotions_count():
    assert len(CORE_EMOTIONS) == 20
    assert len(CORE_EMOTION_NAMES) == 20


def test_core_emotions_fields():
    for e in CORE_EMOTIONS:
        assert "name" in e and isinstance(e["name"], str)
        assert -1.0 <= e["valence"] <= 1.0
        assert -1.0 <= e["arousal"] <= 1.0


def test_core_emotions_cover_quadrants():
    pos = [e for e in CORE_EMOTIONS if e["valence"] > 0.3]
    neg = [e for e in CORE_EMOTIONS if e["valence"] < -0.3]
    high = [e for e in CORE_EMOTIONS if e["arousal"] > 0.3]
    low = [e for e in CORE_EMOTIONS if e["arousal"] < -0.3]
    assert pos and neg and high and low


def test_dataclass_defaults():
    ec = ExtractionConfig()
    assert 0 < ec.probe_layer_fraction < 1
    assert ec.use_content_range is True
    pc = ProbeConfig()
    assert pc.top_k_emotions > 0
    mc = ModelConfig()
    assert mc.default_model.startswith("google/gemma")
    vt = ValidationThresholds()
    assert vt.valence_separation_max < 0
