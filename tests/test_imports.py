"""Package imports should resolve without loading torch-heavy backends."""


def test_top_level_imports():
    import emotion_scope
    assert emotion_scope.__version__
    from emotion_scope import (
        DualEmotionState,
        EmotionExtractor,
        EmotionProbe,
        EmotionState,
        ValidationResult,
        Validator,
        emotion_to_emoji,
        load_model,
        scores_to_color,
    )
    assert callable(load_model)
    assert callable(scores_to_color)
    assert callable(emotion_to_emoji)


def test_speakers_module_loads():
    """speakers.py is now a real implementation, not a stub."""
    from emotion_scope.speakers import SpeakerSeparator, SpeakerValidationResult
    assert callable(SpeakerSeparator)


def test_steer_stub_raises():
    import pytest
    from emotion_scope.steer import steer
    with pytest.raises(NotImplementedError):
        steer()
