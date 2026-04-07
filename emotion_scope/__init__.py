"""
emotion-scope: Extract, probe, and visualize functional emotion vectors
from open-weight language models.

Replicates and extends Anthropic's "Emotion Concepts and their Function
in a Large Language Model" (April 2026) on open-weight models.
"""

__version__ = "0.1.0"

from emotion_scope.extract import EmotionExtractor
from emotion_scope.probe import EmotionProbe, EmotionState, DualEmotionState
from emotion_scope.visualize import scores_to_color, emotion_to_emoji
from emotion_scope.models import load_model
from emotion_scope.validate import Validator, ValidationResult

__all__ = [
    "__version__",
    "EmotionExtractor",
    "EmotionProbe",
    "EmotionState",
    "DualEmotionState",
    "scores_to_color",
    "emotion_to_emoji",
    "load_model",
    "Validator",
    "ValidationResult",
]
