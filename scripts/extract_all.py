"""
CLI: extract emotion vectors from a model and save them to results/vectors/.

Usage:
    uv run python scripts/extract_all.py --model google/gemma-2-2b-it
    uv run python scripts/extract_all.py --model google/gemma-2-2b-it --sweep-layers
    uv run python scripts/extract_all.py --model google/gemma-2-9b-it --use-4bit
"""

from __future__ import annotations

import argparse

from emotion_scope import EmotionExtractor, load_model
from emotion_scope.config import ExtractionConfig
from emotion_scope.utils import average_pairwise_cosine, valence_separation


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract emotion vectors from a model.")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-4bit", action="store_true", help="NF4 quantization (forces HF backend)")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformer_lens", "huggingface"])
    parser.add_argument("--sweep-layers", action="store_true", help="Search for best probe layer")
    parser.add_argument("--stories-per-emotion", type=int, default=50)
    parser.add_argument("--out", default=None, help="Override save path")

    args = parser.parse_args()

    model, tokenizer, backend, info = load_model(
        model_name=args.model,
        device=args.device,
        use_4bit=args.use_4bit,
        dtype=args.dtype,
        backend=args.backend,
    )

    config = ExtractionConfig(stories_per_emotion=args.stories_per_emotion)
    extractor = EmotionExtractor(model, tokenizer, backend, info, config=config)

    if args.sweep_layers:
        extractor.find_best_probe_layer()

    vectors = extractor.extract()

    print("\n--- Quick metrics ---")
    print(f"n_vectors: {len(vectors)}")
    print(f"valence_separation: {valence_separation(vectors):.4f} (want < -0.2)")
    print(f"avg_pairwise_cosine: {average_pairwise_cosine(vectors):.4f} (want < 0.5)")

    extractor.save(args.out)


if __name__ == "__main__":
    main()
