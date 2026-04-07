"""
Emotion state to color mapping.

Two systems:
1. scores_to_color() — legacy HSL mapping, still used by probe.py for quick hex
2. OKLCH palette + blend_emotion_colors() — the real visualization system from
   VISUALIZATION_SPEC.md, used by the orb renderer and Gradio demo

The OKLCH palette is the source of truth for all visual output. The HSL path
exists only for backward compatibility in CLI / non-visual contexts.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from emotion_scope.config import CORE_EMOTIONS


# =========================================================================
#  OKLCH Palette  (from VISUALIZATION_SPEC.md §2.2)
# =========================================================================

@dataclass(frozen=True)
class OKLCHColor:
    L: float  # Lightness 0-1
    C: float  # Chroma 0-0.4 (we stay ≤0.18)
    H: float  # Hue in degrees

# Lightness carries emotional weight:
#   Positive emotions: L 0.65-0.85 (light, lifted)
#   Negative emotions: L 0.25-0.50 (dark, heavy)
#   Neutral: L 0.55-0.65
# Chroma never exceeds 0.18 — muted and sophisticated.
OKLCH_PALETTE: Dict[str, OKLCHColor] = {
    "happy":        OKLCHColor(0.82, 0.15,  85),
    "loving":       OKLCHColor(0.80, 0.12,  60),
    "confident":    OKLCHColor(0.70, 0.13,  70),
    "proud":        OKLCHColor(0.70, 0.13,  70),
    "hopeful":      OKLCHColor(0.75, 0.10,  85),
    "calm":         OKLCHColor(0.65, 0.08, 220),
    "curious":      OKLCHColor(0.72, 0.12, 175),
    "reflective":   OKLCHColor(0.60, 0.03, 250),
    "surprised":    OKLCHColor(0.80, 0.14, 200),
    "enthusiastic": OKLCHColor(0.82, 0.15,  85),
    "nervous":      OKLCHColor(0.55, 0.09, 290),
    "frustrated":   OKLCHColor(0.50, 0.10,  35),
    "guilty":       OKLCHColor(0.38, 0.08, 310),
    "angry":        OKLCHColor(0.45, 0.16,  25),
    "sad":          OKLCHColor(0.40, 0.10, 270),
    "afraid":       OKLCHColor(0.35, 0.12, 300),
    "desperate":    OKLCHColor(0.30, 0.18,  10),
    "hostile":      OKLCHColor(0.25, 0.14,  15),
    "brooding":     OKLCHColor(0.35, 0.04, 260),
    "gloomy":       OKLCHColor(0.42, 0.03, 240),
}

NEUTRAL_COLOR = OKLCHColor(0.58, 0.03, 240)


# =========================================================================
#  OKLCH → sRGB conversion
# =========================================================================

def oklch_to_rgb(color: OKLCHColor) -> Tuple[int, int, int]:
    """Convert OKLCH to sRGB (0-255 per channel), clamped to gamut."""
    L, C, H = color.L, color.C, color.H
    h_rad = math.radians(H)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)

    # OKLab → linear sRGB via the LMS intermediate
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l3 = l_ ** 3
    m3 = m_ ** 3
    s3 = s_ ** 3

    r_lin = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    def gamma(v: float) -> float:
        if v > 0.0031308:
            return 1.055 * (v ** (1 / 2.4)) - 0.055
        return 12.92 * v

    return (
        max(0, min(255, round(gamma(r_lin) * 255))),
        max(0, min(255, round(gamma(g_lin) * 255))),
        max(0, min(255, round(gamma(b_lin) * 255))),
    )


def oklch_to_hex(color: OKLCHColor) -> str:
    r, g, b = oklch_to_rgb(color)
    return f"#{r:02x}{g:02x}{b:02x}"


# =========================================================================
#  Color blending  (VISUALIZATION_SPEC.md §9)
# =========================================================================

def blend_emotion_colors(
    scores: Dict[str, float],
    palette: Optional[Dict[str, OKLCHColor]] = None,
) -> OKLCHColor:
    """
    Blend active emotion colors weighted by their scores.
    Uses circular hue interpolation (vector average in cartesian coords).
    Only positive scores above 0.05 contribute.
    """
    pal = palette or OKLCH_PALETTE
    active = [(name, score) for name, score in scores.items()
              if score > 0.05 and name in pal]
    active.sort(key=lambda x: -x[1])

    if not active:
        return NEUTRAL_COLOR

    total = sum(s for _, s in active)
    L = sum(pal[n].L * s for n, s in active) / total
    C = sum(pal[n].C * s for n, s in active) / total

    # Circular hue interpolation via cartesian average
    hx = sum(math.cos(math.radians(pal[n].H)) * s for n, s in active) / total
    hy = sum(math.sin(math.radians(pal[n].H)) * s for n, s in active) / total
    H = math.degrees(math.atan2(hy, hx)) % 360

    return OKLCHColor(L, C, H)


def emotional_complexity(scores: Dict[str, float]) -> float:
    """
    Normalized entropy of the score distribution. 0 = one dominant,
    1 = all emotions equally active. Used to drive visual complexity
    (multi-color currents in the orb).
    """
    active = [s for s in scores.values() if s > 0.05]
    if len(active) <= 1:
        return 0.0
    total = sum(active)
    if total == 0:
        return 0.0
    entropy = 0.0
    for s in active:
        p = s / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(active))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def scores_to_orb_state(
    scores: Dict[str, float],
    emotions_metadata: Optional[List[dict]] = None,
) -> dict:
    """
    Convert emotion scores to a full orb state dict, suitable for
    JSON serialization and passing to the JS orb renderer.

    Returns dict with keys: valence, arousal, intensity, complexity,
    dominant, secondary, color_hex, color_oklch, secondary_hex,
    secondary_oklch, top_emotions.
    """
    metadata_list = emotions_metadata or CORE_EMOTIONS
    metadata = {e["name"]: e for e in metadata_list}

    # Valence / arousal (weighted average)
    valence = 0.0
    arousal = 0.0
    total_w = 0.0
    for name, score in scores.items():
        if name in metadata:
            w = max(score, 0.0)
            valence += w * metadata[name]["valence"]
            arousal += w * metadata[name]["arousal"]
            total_w += w
    if total_w > 0:
        valence /= total_w
        arousal /= total_w
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    # Dominant & secondary
    sorted_scores = sorted(
        [(n, s) for n, s in scores.items() if s > 0.05],
        key=lambda x: -x[1],
    )
    dominant = sorted_scores[0][0] if sorted_scores else "neutral"
    secondary = sorted_scores[1][0] if len(sorted_scores) > 1 else ""
    intensity = sorted_scores[0][1] if sorted_scores else 0.0

    # Colors
    blended = blend_emotion_colors(scores)
    sec_color = OKLCH_PALETTE.get(secondary, NEUTRAL_COLOR) if secondary else NEUTRAL_COLOR
    complexity = emotional_complexity(scores)

    return {
        "valence": valence,
        "arousal": arousal,
        "intensity": min(intensity, 1.0),
        "complexity": complexity,
        "dominant": dominant,
        "secondary": secondary,
        "color_hex": oklch_to_hex(blended),
        "color_oklch": {"L": blended.L, "C": blended.C, "H": blended.H},
        "secondary_hex": oklch_to_hex(sec_color),
        "secondary_oklch": {"L": sec_color.L, "C": sec_color.C, "H": sec_color.H},
        "top_emotions": sorted_scores[:5],
    }


# =========================================================================
#  Legacy HSL mapping  (kept for backward compat with probe.py / CLI)
# =========================================================================

def scores_to_color(
    scores: Dict[str, float],
    emotions_metadata: Optional[List[dict]] = None,
) -> Tuple[str, tuple, float, float]:
    """
    Legacy HSL color mapping. Still used by EmotionProbe._activation_to_state
    for quick CLI output. For visual output, use scores_to_orb_state instead.
    """
    metadata_list = emotions_metadata or CORE_EMOTIONS
    metadata = {e["name"]: e for e in metadata_list}

    valence = 0.0
    arousal = 0.0
    total_weight = 0.0

    for name, score in scores.items():
        if name in metadata:
            weight = max(score, 0.0)
            valence += weight * metadata[name]["valence"]
            arousal += weight * metadata[name]["arousal"]
            total_weight += weight

    if total_weight > 0:
        valence /= total_weight
        arousal /= total_weight

    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    hue_norm = (valence + 1.0) / 2.0 * (240.0 / 360.0)
    saturation = 0.3 + min(abs(arousal), 1.0) * 0.7
    lightness = 0.35 + (valence + 1.0) / 2.0 * 0.3

    r, g, b = colorsys.hls_to_rgb(hue_norm, lightness, saturation)
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )
    return hex_color, (hue_norm * 360.0, saturation, lightness), valence, arousal


def emotion_to_emoji(dominant: str) -> str:
    """Map dominant emotion name to a single-glyph indicator."""
    emoji_map = {
        "happy": "\U0001F60A", "sad": "\U0001F622", "afraid": "\U0001F628",
        "angry": "\U0001F620", "calm": "\U0001F60C", "desperate": "\U0001F630",
        "hopeful": "\U0001F91E", "frustrated": "\U0001F624", "curious": "\U0001F914",
        "proud": "\U0001F60E", "guilty": "\U0001F614", "surprised": "\U0001F632",
        "loving": "\U0001F970", "hostile": "\U0001F621", "nervous": "\U0001F62C",
        "confident": "\U0001F60E", "brooding": "\U0001F928",
        "enthusiastic": "\U0001F929", "reflective": "\U0001FA9E", "gloomy": "\U0001F636",
    }
    return emoji_map.get(dominant, "?")
