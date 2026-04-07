"""
Central configuration for emotion-scope.

Single source of truth for all constants, emotion definitions, model defaults,
and paths. Other modules import from here.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
VECTORS_DIR = RESULTS_DIR / "vectors"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for _d in (VECTORS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclass configs
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Configuration for emotion vector extraction."""
    probe_layer_fraction: float = 22 / 26      # Default 84.6% depth (layer 22/26 for Gemma 2 2B). Run a sweep to find optimal.
    skip_tokens: int = 20                     # Skip first N tokens when averaging (legacy; content-range is preferred)
    neutral_pca_variance: float = 0.5         # Variance threshold for neutral PCA denoising
    stories_per_emotion: int = 50             # Max templates per emotion to use
    normalize_vectors: bool = True            # L2-normalize final vectors
    device: str = "auto"                      # "auto", "cuda", "cpu"
    use_content_range: bool = True            # Use find_content_token_range instead of skip_tokens


@dataclass
class ProbeConfig:
    """Configuration for real-time probing.

    IMPORTANT — extraction vs probing positions differ:
    Extraction averages over CONTENT tokens only (via find_content_token_range).
    Probing reads at the RESPONSE-PREPARATION position (last token of full
    prompt, after all template markup). This matches Anthropic's methodology
    ("the ':' token following 'Assistant'") and was validated at 83% top-3
    accuracy vs 75% for the last-content-token position.
    """
    token_position: str = "last_content"      # "last_content" = response-prep position, "last", or int
    top_k_emotions: int = 5                   # Number of top emotions to return
    include_color: bool = True                # Include color mapping in output


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    default_model: str = "google/gemma-2-2b-it"
    cloud_models: List[str] = field(default_factory=lambda: [
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ])
    validation_models: List[str] = field(default_factory=lambda: [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
    ])
    use_4bit: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "float16"


@dataclass
class ValidationThresholds:
    """
    Pass/fail thresholds for the Phase 1 validation gate.
    Decided with the architect — all four must pass for vectors to be considered valid.
    """
    tylenol_min_spearman: float = 0.7          # afraid score vs log-dose
    tylenol_calm_max_spearman: float = -0.5    # calm score vs log-dose (negative = inverse)
    confusion_top3_min_accuracy: float = 0.6   # top-3 accuracy on implicit scenarios
    valence_separation_max: float = -0.2       # cosine between mean pos/neg (more negative = better)
    richness_max_avg_cosine: float = 0.5       # average pairwise cosine between all vectors


# ---------------------------------------------------------------------------
# Core emotions (20) — valence/arousal coordinates
# ---------------------------------------------------------------------------

# Selected to cover the valence-arousal space while including the emotions
# Anthropic found most alignment-relevant (desperate, calm, angry, nervous).
#
# Valence/arousal coordinates are hand-assigned based on the Russell (1980)
# circumplex model of affect and cross-referenced with the NRC Emotion
# Intensity Lexicon (Mohammad, 2018). These are approximate placements —
# the extraction pipeline does NOT depend on these values being exact;
# they are used only for (1) visualization color mapping, (2) the valence
# separation validation metric, and (3) weighted arousal/valence readouts.
# The emotion VECTORS are derived purely from neural activations, not from
# these metadata coordinates.
CORE_EMOTIONS: List[dict] = [
    {"name": "happy",        "valence":  0.8, "arousal":  0.5},
    {"name": "sad",          "valence": -0.7, "arousal": -0.1},
    {"name": "afraid",       "valence": -0.7, "arousal":  0.8},
    {"name": "angry",        "valence": -0.6, "arousal":  0.8},
    {"name": "calm",         "valence":  0.3, "arousal": -0.5},
    {"name": "desperate",    "valence": -0.9, "arousal":  0.9},
    {"name": "hopeful",      "valence":  0.7, "arousal":  0.3},
    {"name": "frustrated",   "valence": -0.5, "arousal":  0.6},
    {"name": "curious",      "valence":  0.4, "arousal":  0.5},
    {"name": "proud",        "valence":  0.8, "arousal":  0.4},
    {"name": "guilty",       "valence": -0.6, "arousal":  0.2},
    {"name": "surprised",    "valence":  0.1, "arousal":  0.7},
    {"name": "loving",       "valence":  0.9, "arousal":  0.3},
    {"name": "hostile",      "valence": -0.8, "arousal":  0.7},
    {"name": "nervous",      "valence": -0.3, "arousal":  0.6},
    {"name": "confident",    "valence":  0.7, "arousal":  0.3},
    {"name": "brooding",     "valence": -0.3, "arousal":  0.1},
    {"name": "enthusiastic", "valence":  0.8, "arousal":  0.9},
    {"name": "reflective",   "valence":  0.0, "arousal": -0.2},
    {"name": "gloomy",       "valence": -0.6, "arousal": -0.3},
]

CORE_EMOTION_NAMES: List[str] = [e["name"] for e in CORE_EMOTIONS]


# ---------------------------------------------------------------------------
# Chat template markers — used by find_content_token_range in utils.py
# ---------------------------------------------------------------------------

# Known special tokens / template markers per model family.
# Anything matching these should NOT count as "content" when averaging activations.
CHAT_TEMPLATE_MARKERS: dict = {
    "gemma": [
        "<start_of_turn>", "<end_of_turn>",
        "<bos>", "<eos>", "<pad>",
        "user\n", "model\n",
        "user", "model",
    ],
    "llama": [
        "<|begin_of_text|>", "<|end_of_text|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<|eot_id|>",
        "user", "assistant", "system",
    ],
    "mistral": [
        "<s>", "</s>",
        "[INST]", "[/INST]",
    ],
    "phi": [
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<s>", "</s>",
    ],
    "qwen": [
        "<|im_start|>", "<|im_end|>",
        "system", "user", "assistant",
    ],
    "deepseek": [
        "<|begin▁of▁sentence|>", "<|end▁of▁sentence|>",
        "<|User|>", "<|Assistant|>",
        "User:", "Assistant:",
    ],
    "generic": [
        "<s>", "</s>", "<|im_start|>", "<|im_end|>",
        "<|endoftext|>", "<|padding|>",
    ],
}
