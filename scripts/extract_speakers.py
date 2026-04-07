"""
CLI: extract speaker-separated emotion vectors from a model.

Loads a model, runs the SpeakerSeparator extraction pipeline, validates
the resulting vectors with three tests (orthogonality, internal consistency,
thermostat), and saves the results.

Usage:
    uv run python scripts/extract_speakers.py --model google/gemma-2-2b-it
    uv run python scripts/extract_speakers.py \
        --model google/gemma-2-2b-it \
        --dialogues data/templates/two_speaker_dialogues.jsonl \
        --output results/vectors/speaker_separation.pt
"""

from __future__ import annotations

import argparse

from emotion_scope import load_model
from emotion_scope.speakers import SpeakerSeparator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract current-speaker and other-speaker emotion vectors."
    )
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-4bit", action="store_true", help="NF4 quantization (forces HF backend)")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformer_lens", "huggingface"])
    parser.add_argument("--dialogues", default=None, help="Path to two_speaker_dialogues.jsonl")
    parser.add_argument("--neutral", default=None, help="Path to neutral_prompts.jsonl")
    parser.add_argument("--output", default=None, help="Override save path for vectors")

    args = parser.parse_args()

    # Load model
    model, tokenizer, backend, info = load_model(
        model_name=args.model,
        device=args.device,
        use_4bit=args.use_4bit,
        dtype=args.dtype,
        backend=args.backend,
    )

    # Extract speaker vectors
    separator = SpeakerSeparator(model, tokenizer, backend, info)
    result = separator.extract(
        dialogues_path=args.dialogues,
        neutral_path=args.neutral,
    )

    current_vectors = result["current_speaker"]
    other_vectors = result["other_speaker"]

    print(f"\nExtracted {len(current_vectors)} current-speaker vectors")
    print(f"Extracted {len(other_vectors)} other-speaker vectors")

    # Run validation
    print("\n" + "=" * 60)
    print("SPEAKER SEPARATION VALIDATION")
    print("=" * 60)

    validation = separator.validate(current_vectors, other_vectors)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test A (Orthogonality):       {'PASS' if validation.orthogonality_passed else 'FAIL'}"
          f"  (mean cos = {validation.orthogonality_mean:.3f})")
    print(f"  Test B (Internal consistency): {'PASS' if validation.consistency_passed else 'FAIL'}"
          f"  (current = {validation.current_valence_sep:.3f}, other = {validation.other_valence_sep:.3f})")
    print(f"  Test C (Thermostat):           {'PASS' if validation.thermostat_passed else 'FAIL'}"
          f"  (mean delta = {validation.thermostat_mean_delta:.3f})")

    if validation.all_passed:
        print("\nAll speaker-separation tests PASSED.")
    else:
        print("\nOne or more speaker-separation tests FAILED.")

    # Save
    save_path = separator.save(result, path=args.output)
    print(f"\nVectors saved to: {save_path}")


if __name__ == "__main__":
    main()
