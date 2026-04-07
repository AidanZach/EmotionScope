"""
Full pipeline: layer comparison → extraction → validation.

Runs the complete extraction pipeline with all bug fixes applied:
  1. Fixed find_content_token_range (anchor-based, excludes template markup)
  2. Corrected grand mean (pooled, not mean-of-means)
  3. Corrected PCA threshold (searchsorted, not reversed logic)
  4. New corpus: 1000 stories (50/emotion), 100 neutral prompts

Extracts at layers 20-23 with full 50 templates/emotion to definitively
select the optimal probe layer, then runs the validation suite.

Usage:
    uv run python scripts/run_full_pipeline.py
    uv run python scripts/run_full_pipeline.py --layers 20 21 22 23
    uv run python scripts/run_full_pipeline.py --templates-per-emotion 10  # faster test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from emotion_scope.config import ExtractionConfig, CORE_EMOTIONS, VECTORS_DIR, METRICS_DIR
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.probe import EmotionProbe
from emotion_scope.utils import valence_separation, average_pairwise_cosine
from emotion_scope.validate import Validator
from emotion_scope.visualize import scores_to_orb_state


def main():
    parser = argparse.ArgumentParser(description="Full extraction pipeline")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 21, 22, 23],
                        help="Layers to compare (full extraction at each)")
    parser.add_argument("--templates-per-emotion", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("EMOTIONSCOPE FULL PIPELINE")
    print(f"  Model: {args.model}")
    print(f"  Layers to test: {args.layers}")
    print(f"  Templates/emotion: {args.templates_per_emotion}")
    print(f"  Corpus: data/templates/emotion_stories.jsonl")
    print(f"  Neutral: data/neutral/neutral_prompts.jsonl")
    print("=" * 70)

    # Load model once
    model, tokenizer, backend, info = load_model(
        args.model, device=args.device, dtype=args.dtype, run_smoke_test=False)
    print(f"\nModel loaded: {info['n_layers']} layers, d_model={info['d_model']}, backend={backend}")

    # ── Step 1: Extract at each candidate layer ──
    print("\n" + "=" * 70)
    print("STEP 1: LAYER COMPARISON (full extraction at each layer)")
    print("=" * 70)

    layer_results = {}
    layer_extractors = {}
    config = ExtractionConfig(stories_per_emotion=args.templates_per_emotion)

    for layer in args.layers:
        print(f"\n--- Layer {layer} ({layer/info['n_layers']*100:.1f}% depth) ---")
        t0 = time.time()

        info_copy = dict(info)
        info_copy["probe_layer"] = layer

        extractor = EmotionExtractor(model, tokenizer, backend, info_copy, config=config)
        vectors = extractor.extract()

        # Metrics
        vs = valence_separation(vectors)
        avg_cos = average_pairwise_cosine(vectors)
        k = extractor.neutral_pca.shape[1] if extractor.neutral_pca is not None else 0

        # Also compute raw (pre-denoising) valence separation
        raw_normed = {n: F.normalize(v, dim=0) for n, v in extractor.raw_vectors.items()}
        vs_raw = valence_separation(raw_normed)

        elapsed = time.time() - t0

        layer_results[layer] = {
            "valence_sep_denoised": vs,
            "valence_sep_raw": vs_raw,
            "avg_pairwise_cosine": avg_cos,
            "pca_k": k,
            "n_vectors": len(vectors),
            "time_seconds": elapsed,
        }
        layer_extractors[layer] = extractor

        print(f"  raw val_sep={vs_raw:.4f}  denoised val_sep={vs:.4f}  "
              f"avg_cos={avg_cos:.4f}  k={k}  time={elapsed:.0f}s")

    # Select winner
    best_layer = min(layer_results, key=lambda l: layer_results[l]["valence_sep_denoised"])
    print(f"\n{'='*70}")
    print(f"LAYER COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Layer':<8} | {'Depth':>6} | {'Raw':>10} | {'Denoised':>10} | {'Avg cos':>10} | {'k':>4} | {'Time':>6}")
    print("-" * 70)
    for layer in sorted(layer_results):
        r = layer_results[layer]
        depth = f"{layer/info['n_layers']*100:.1f}%"
        marker = " <-- BEST" if layer == best_layer else ""
        print(f"{layer:<8} | {depth:>6} | {r['valence_sep_raw']:>10.4f} | "
              f"{r['valence_sep_denoised']:>10.4f} | {r['avg_pairwise_cosine']:>10.4f} | "
              f"{r['pca_k']:>4} | {r['time_seconds']:>5.0f}s{marker}")

    print(f"\nOptimal layer: {best_layer} (denoised val_sep = {layer_results[best_layer]['valence_sep_denoised']:.4f})")

    # ── Step 2: Save the winning vectors ──
    best_extractor = layer_extractors[best_layer]
    best_extractor.probe_layer = best_layer
    best_extractor.model_info["probe_layer"] = best_layer
    out_path = VECTORS_DIR / f"google_gemma-2-2b-it.pt"
    best_extractor.save(str(out_path))
    print(f"Saved vectors to {out_path}")

    # Save layer sweep data for figure generation
    sweep_path = METRICS_DIR / "layer_sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({
            "layers": {str(l): r for l, r in layer_results.items()},
            "best_layer": best_layer,
            "model": args.model,
            "templates_per_emotion": args.templates_per_emotion,
            "n_emotions": 20,
            "n_neutral": 100,
        }, f, indent=2)
    print(f"Saved layer sweep data to {sweep_path}")

    # ── Step 3: Validation ──
    print(f"\n{'='*70}")
    print("STEP 2: VALIDATION SUITE")
    print(f"{'='*70}")

    info["probe_layer"] = best_layer
    probe = EmotionProbe(
        model=model, tokenizer=tokenizer, backend=backend,
        model_info=info, emotion_vectors=best_extractor.emotion_vectors,
        emotions_metadata=CORE_EMOTIONS)

    validator = Validator(probe=probe, emotion_vectors=best_extractor.emotion_vectors)
    result = validator.run_all()

    # ── Step 4: Quick behavioral test ──
    print(f"\n{'='*70}")
    print("STEP 3: BEHAVIORAL SPOT-CHECK")
    print(f"{'='*70}")

    test_cases = [
        ("Anger",     "I am furious right now. My coworker lied to my face."),
        ("Fear",      "The doctor called and said they need to discuss my results urgently."),
        ("Sadness",   "My grandmother passed away last night. I cannot stop crying."),
        ("Happiness", "I just got the promotion I have been working toward for three years!"),
        ("Calm",      "I am sitting by the lake watching the sunset. Everything is peaceful."),
    ]

    for label, msg in test_cases:
        dual = probe.analyze_conversation(msg)
        orb = scores_to_orb_state(dual.model_state.scores)
        top3 = orb["top_emotions"][:3]
        top_str = ", ".join(f"{n}({s:+.3f})" for n, s in top3)
        print(f"  {label:<12} => {top_str}  v={orb['valence']:+.2f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Optimal layer: {best_layer} ({best_layer/info['n_layers']*100:.1f}% depth)")
    print(f"  Valence separation: {layer_results[best_layer]['valence_sep_denoised']:.4f}")
    print(f"  PCA components retained: {layer_results[best_layer]['pca_k']}")
    print(f"  Validation: {'ALL PASSED' if result.all_passed else 'SOME FAILED'}")
    print(f"  Vectors: {out_path}")
    print(f"  Metrics: {METRICS_DIR / 'validation_results.json'}")
    print(f"  Layer data: {sweep_path}")


if __name__ == "__main__":
    main()
