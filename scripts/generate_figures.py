"""
Generate all paper figures from existing data.

Produces:
  1. Tylenol intensity curve (actual scores per dose, not just correlations)
  2. Layer sweep profile (valence separation vs layer depth)
  3. Confusion matrix heatmap (12 implicit scenarios)
  4. Emotion vector similarity matrix (20 x 20 cosine)
  5. Speaker separation orthogonality chart
  6. Valence-arousal circumplex scatter (20 emotions in 2D PCA space)

All figures saved to results/figures/ as both PNG (paper) and SVG (web).

Usage:
    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from emotion_scope.config import CORE_EMOTIONS, DATA_DIR, FIGURES_DIR, METRICS_DIR
from emotion_scope.utils import cosine_similarity_matrix
from emotion_scope.visualize import oklch_to_rgb, OKLCH_PALETTE

# ── Style ──
plt.rcParams.update({
    "figure.facecolor": "#0e0e14",
    "axes.facecolor": "#12121a",
    "axes.edgecolor": "#2a2a3a",
    "axes.labelcolor": "#b0b0c8",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#808098",
    "ytick.color": "#808098",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": "#d0d0e0",
    "grid.color": "#1e1e2a",
    "grid.alpha": 0.5,
    "legend.facecolor": "#16161e",
    "legend.edgecolor": "#2a2a3a",
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0e0e14",
})


def emotion_color(name: str) -> str:
    p = OKLCH_PALETTE.get(name)
    if p is None:
        return "#808090"
    r, g, b = oklch_to_rgb(p)
    return f"#{r:02x}{g:02x}{b:02x}"


def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.svg")
    print(f"  Saved {name}.png + .svg")
    plt.close(fig)


# =========================================================================
#  Figure 1: Tylenol Intensity — Actual Scores Per Dose
#  Shows the raw cosine similarity scores, not just rank correlations.
#  This is the honest version that reveals signal magnitude.
# =========================================================================

def fig_tylenol():
    print("Figure 1: Tylenol intensity (raw scores per dose)")

    # Re-run the probe to get per-dose scores
    from emotion_scope.extract import EmotionExtractor
    from emotion_scope.models import load_model
    from emotion_scope.probe import EmotionProbe

    model, tok, backend, info = load_model("google/gemma-2-2b-it", run_smoke_test=False)
    saved = EmotionExtractor.load("results/vectors/google_gemma-2-2b-it.pt")
    info["probe_layer"] = saved.get("probe_layer_used", info["probe_layer"])
    probe = EmotionProbe(model, tok, backend, info, saved["vectors"], saved.get("emotions"))

    doses = [500, 1000, 2000, 4000, 8000, 12000, 16000]

    all_scores: dict[str, list[float]] = {}
    for dose in doses:
        # Use analyze_conversation for parity with validation and demo
        msg = f"I just took {dose} mg of Tylenol for my back pain. What should I do?"
        dual = probe.analyze_conversation(msg)
        for name, score in dual.model_state.scores.items():
            all_scores.setdefault(name, []).append(score)

    # Select emotions with meaningful signal (range > 0.012) + a few small ones for contrast
    emotions_strong = []
    emotions_weak = []
    for name in all_scores:
        vals = all_scores[name]
        rng = max(vals) - min(vals)
        if rng > 0.012:
            emotions_strong.append(name)
        else:
            emotions_weak.append(name)

    # Plot: line chart of score vs dose for strong-signal emotions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: actual score trajectories
    for name in ["afraid", "desperate", "nervous", "calm", "hopeful", "sad"]:
        if name not in all_scores:
            continue
        vals = all_scores[name]
        c = emotion_color(name)
        rng = max(vals) - min(vals)
        lw = 2.5 if rng > 0.015 else 1.2
        alpha = 1.0 if rng > 0.015 else 0.5
        ax1.plot(doses, vals, color=c, linewidth=lw, marker="o", markersize=4,
                 label=f"{name} (Δ={rng:.3f})", alpha=alpha)

    ax1.set_xscale("log")
    ax1.set_xlabel("Tylenol dosage (mg, log scale)")
    ax1.set_ylabel("Cosine similarity with emotion vector")
    ax1.set_title("Tylenol Intensity — Raw Emotion Scores vs Dosage")
    ax1.axhline(y=0, color="#404060", linewidth=0.5, linestyle="-")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y")

    # Mark safe/danger zones
    ax1.axvspan(doses[0], 4000, alpha=0.03, color="#55aa55")
    ax1.axvspan(4000, doses[-1], alpha=0.03, color="#cc5555")
    ax1.text(800, ax1.get_ylim()[1] - 0.005, "Safe range", fontsize=8,
             color="#55aa55", alpha=0.7)
    ax1.text(8000, ax1.get_ylim()[1] - 0.005, "Dangerous", fontsize=8,
             color="#cc5555", alpha=0.7)

    # Right panel: signal range comparison (why some correlations are meaningless)
    all_names = sorted(all_scores.keys(),
                       key=lambda n: max(all_scores[n]) - min(all_scores[n]),
                       reverse=True)
    ranges = [max(all_scores[n]) - min(all_scores[n]) for n in all_names]
    colors = [emotion_color(n) for n in all_names]

    bars = ax2.barh(range(len(all_names)), ranges, color=colors, height=0.65,
                    edgecolor="#2a2a3a", linewidth=0.5)

    # Mark the noise threshold
    ax2.axvline(x=0.012, color="#ffaa44", linewidth=1, linestyle="--", alpha=0.7)
    ax2.text(0.013, -0.8, "Signal threshold", fontsize=7, color="#ffaa44")

    ax2.set_yticks(range(len(all_names)))
    ax2.set_yticklabels(all_names, fontsize=8)
    ax2.set_xlabel("Score range across dosages")
    ax2.set_title("Signal Magnitude\n(range < 0.012 = noise)")
    ax2.invert_yaxis()

    for i, (name, rng) in enumerate(zip(all_names, ranges)):
        label_color = "#a0a0b8" if rng > 0.012 else "#505060"
        ax2.text(rng + 0.001, i, f"{rng:.3f}", va="center", fontsize=7,
                 color=label_color)

    fig.suptitle("Tylenol Intensity Test — Gemma 2 2B IT, Layer 22",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "tylenol_intensity")


# =========================================================================
#  Figure 2: Layer Sweep Profile
# =========================================================================

def fig_layer_sweep():
    print("Figure 2: Layer sweep profile")

    layers =     [3,      6,      9,      12,     15,     18,     21,     24,     25]
    val_seps = [-0.709, -0.701, -0.736, -0.747, -0.754, -0.765, -0.772, -0.764, -0.740]
    depths = [l / 26 * 100 for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(layers, val_seps, color="#5588cc", linewidth=2, marker="o",
            markersize=6, markerfacecolor="#5588cc", markeredgecolor="#3366aa",
            zorder=3)

    best_idx = val_seps.index(min(val_seps))
    ax.plot(layers[best_idx], val_seps[best_idx], "o", markersize=12,
            markerfacecolor="none", markeredgecolor="#ffaa44", markeredgewidth=2,
            zorder=4)
    ax.annotate(f"Layer {layers[best_idx]}\n({depths[best_idx]:.0f}% depth)\nval_sep = {val_seps[best_idx]:.3f}",
                xy=(layers[best_idx], val_seps[best_idx]),
                xytext=(layers[best_idx] + 2, val_seps[best_idx] + 0.015),
                fontsize=9, color="#ffaa44",
                arrowprops=dict(arrowstyle="->", color="#ffaa44", lw=1))

    default_layer = int(26 * 2 / 3)
    ax.axvline(x=default_layer, color="#aa5555", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(default_layer + 0.3, max(val_seps) - 0.002, f"Default 2/3\n(layer {default_layer})",
            fontsize=8, color="#aa5555", alpha=0.8)

    ax.axhline(y=-0.2, color="#55aa55", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.text(2, -0.2 + 0.005, "Validation threshold (< -0.2)", fontsize=7,
            color="#55aa55", alpha=0.6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Valence Separation (cosine, more negative = better)")
    ax.set_title("Probe Layer Sweep — Gemma 2 2B IT (26 layers)")
    ax.set_xticks(layers)
    ax.grid(True, axis="y")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"{d:.0f}%" for d in depths], fontsize=7)
    ax2.set_xlabel("Model depth (%)", fontsize=9, color="#808098")
    ax2.tick_params(colors="#606078")

    fig.tight_layout()
    save_fig(fig, "layer_sweep")


# =========================================================================
#  Figure 3: Confusion Matrix (Implicit Scenarios)
# =========================================================================

def fig_confusion():
    print("Figure 3: Confusion matrix")
    path = METRICS_DIR / "validation_results.json"
    if not path.exists():
        print("  SKIP: validation_results.json not found")
        return

    data = json.loads(path.read_text())
    scenarios = data["details"]["confusion"]["details"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, s in enumerate(scenarios):
        short = s["scenario"][:45] + "..."
        top3 = s["top3"]
        hit = s["hit"]

        y = len(scenarios) - 1 - i
        ax.text(-0.02, y, short, ha="right", va="center", fontsize=8,
                color="#b0b0c8", transform=ax.get_yaxis_transform())

        for j, emo in enumerate(top3):
            is_expected = emo in s["expected"]
            ec = emotion_color(emo)
            alpha = 0.9 if is_expected else 0.4
            ax.barh(y, 1, left=j, height=0.7, color=ec, alpha=alpha,
                    edgecolor="#2a2a3a", linewidth=0.5)
            ax.text(j + 0.5, y, emo, ha="center", va="center", fontsize=7,
                    color="#e0e0f0" if is_expected else "#808098")

        color = "#44aa66" if hit else "#cc5555"
        marker = "\u2713" if hit else "\u2717"
        ax.text(3.2, y, marker, ha="center", va="center", fontsize=12,
                color=color, fontweight="bold")

    ax.set_xlim(-0.02, 3.5)
    ax.set_ylim(-0.5, len(scenarios) - 0.5)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Top 1", "Top 2", "Top 3"])
    ax.set_yticks([])
    acc = data["details"]["confusion"]["top3_accuracy"]
    ax.set_title(f"Implicit Scenario Confusion Matrix — {acc:.1%} Top-3 Accuracy")

    fig.tight_layout()
    save_fig(fig, "confusion_matrix")


# =========================================================================
#  Figure 4: Emotion Vector Similarity Matrix (20 x 20)
# =========================================================================

def fig_similarity_matrix():
    print("Figure 4: Emotion similarity matrix")
    vectors_path = Path("results/vectors/google_gemma-2-2b-it.pt")
    if not vectors_path.exists():
        print("  SKIP: vectors not found")
        return

    saved = torch.load(str(vectors_path), weights_only=False)
    vectors = saved["vectors"]
    sim, names = cosine_similarity_matrix(vectors)
    sim_np = sim.numpy()

    meta = {e["name"]: e["valence"] for e in CORE_EMOTIONS}
    order = sorted(range(len(names)), key=lambda i: meta.get(names[i], 0))
    sorted_names = [names[i] for i in order]
    sim_sorted = sim_np[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = matplotlib.colormaps.get_cmap("RdBu_r")
    im = ax.imshow(sim_sorted, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="equal")

    ax.set_xticks(range(len(sorted_names)))
    ax.set_yticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_title("Pairwise Cosine Similarity — 20 Emotion Vectors\n"
                 "(sorted by valence, negative \u2192 positive)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", color="#b0b0c8", fontsize=9)
    cbar.ax.tick_params(colors="#808098")

    fig.tight_layout()
    save_fig(fig, "similarity_matrix")


# =========================================================================
#  Figure 5: Speaker Separation Orthogonality
# =========================================================================

def fig_speaker_orthogonality():
    print("Figure 5: Speaker separation orthogonality")
    speaker_path = Path("results/vectors/speaker_separation.pt")
    if not speaker_path.exists():
        print("  SKIP: speaker_separation.pt not found")
        return

    saved = torch.load(str(speaker_path), weights_only=False)
    current = saved.get("current_speaker", {})
    other = saved.get("other_speaker", {})

    if not current or not other:
        print("  SKIP: empty vectors")
        return

    emotions = sorted(set(current.keys()) & set(other.keys()))
    cosines = []
    for e in emotions:
        c = torch.nn.functional.normalize(current[e], dim=0)
        o = torch.nn.functional.normalize(other[e], dim=0)
        cosines.append(torch.dot(c, o).item())

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = [emotion_color(e) for e in emotions]
    ax.barh(range(len(emotions)), cosines, color=colors, height=0.65,
            edgecolor="#2a2a3a", linewidth=0.5)

    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions)
    ax.set_xlabel("Cosine similarity between current-speaker and other-speaker vectors")
    ax.set_title("Speaker Separation — Same Emotion, Different Speaker Roles")
    ax.axvline(x=0, color="#404060", linewidth=0.8)
    ax.axvline(x=0.3, color="#cc5555", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(0.32, len(emotions) - 0.5, "Fail threshold (0.3)", fontsize=7,
            color="#cc5555", alpha=0.8)

    mean_cos = np.mean(cosines)
    ax.axvline(x=mean_cos, color="#ffaa44", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.text(mean_cos - 0.02, -0.7, f"Mean = {mean_cos:.3f}", fontsize=9,
            color="#ffaa44", ha="right")

    for i, (e, c) in enumerate(zip(emotions, cosines)):
        ax.text(c + (0.02 if c >= 0 else -0.02), i,
                f"{c:.3f}", va="center",
                ha="left" if c >= 0 else "right",
                fontsize=8, color="#a0a0b8")

    ax.invert_yaxis()
    ax.set_xlim(-0.7, 0.5)
    fig.tight_layout()
    save_fig(fig, "speaker_orthogonality")


# =========================================================================
#  Figure 6: Valence-Arousal Circumplex (2D PCA)
# =========================================================================

def fig_circumplex():
    print("Figure 6: Valence-arousal circumplex")
    vectors_path = Path("results/vectors/google_gemma-2-2b-it.pt")
    if not vectors_path.exists():
        print("  SKIP: vectors not found")
        return

    saved = torch.load(str(vectors_path), weights_only=False)
    vectors = saved["vectors"]

    names = list(vectors.keys())
    matrix = torch.stack([vectors[n] for n in names]).numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(names):
        c = emotion_color(name)
        meta = next((e for e in CORE_EMOTIONS if e["name"] == name), None)
        size = 80 + (abs(meta["arousal"]) * 60 if meta else 0)

        ax.scatter(coords[i, 0], coords[i, 1], c=c, s=size, edgecolors="#2a2a3a",
                   linewidth=0.5, zorder=3, alpha=0.9)
        ax.annotate(name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=c, alpha=0.9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance) — likely Valence",
                  fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance) — likely Arousal",
                  fontsize=10)
    ax.set_title("Emotion Vector Space — PCA Projection\n(Gemma 2 2B IT, layer 21)")
    ax.axhline(y=0, color="#2a2a3a", linewidth=0.5)
    ax.axvline(x=0, color="#2a2a3a", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "circumplex_pca")


# =========================================================================
#  Main
# =========================================================================

def main():
    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)

    fig_tylenol()
    fig_layer_sweep()
    fig_confusion()
    fig_similarity_matrix()
    fig_speaker_orthogonality()
    fig_circumplex()

    print("=" * 60)
    print(f"All figures saved to {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
