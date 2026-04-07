"""
CLI: run the Phase 1 validation suite against a set of extracted vectors.

Usage:
    uv run python scripts/validate_all.py --vectors results/vectors/google_gemma-2-2b-it.pt
"""

from __future__ import annotations

import argparse

import torch

from emotion_scope import EmotionExtractor, EmotionProbe, Validator, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate extracted emotion vectors.")
    parser.add_argument("--vectors", required=True, help="Path to saved vectors .pt file")
    parser.add_argument("--model", default=None, help="Override model name (default: from vectors file)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--use-4bit", action="store_true")

    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    info = saved["model_info"]
    emotions = saved.get("emotions")

    model_name = args.model or info["model_name"]

    model, tokenizer, backend, live_info = load_model(
        model_name=model_name,
        device=args.device,
        dtype=args.dtype,
        use_4bit=args.use_4bit,
    )

    # Use the probe layer that was used at extraction time (may differ from default)
    live_info["probe_layer"] = saved.get("probe_layer_used", info["probe_layer"])

    probe = EmotionProbe(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        model_info=live_info,
        emotion_vectors=vectors,
        emotions_metadata=emotions,
    )

    validator = Validator(probe=probe, emotion_vectors=vectors)
    result = validator.run_all()

    if result.all_passed:
        print("\nAll validation gates passed. Proceed to Phase 2.")
    else:
        print("\nOne or more validation gates failed. See details above.")


if __name__ == "__main__":
    main()
