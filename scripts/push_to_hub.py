"""
Push extracted vectors and figures to HuggingFace Hub.

Usage:
    # Push everything to your repo
    uv run python scripts/push_to_hub.py --repo AidanZach/EmotionScope-vectors

    # Push vectors for a specific model only
    uv run python scripts/push_to_hub.py \
        --repo AidanZach/EmotionScope-vectors \
        --model google/gemma-2-2b-it

    # Push without figures
    uv run python scripts/push_to_hub.py \
        --repo AidanZach/EmotionScope-vectors \
        --no-figures

Requires: `hf login` or HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse

from emotion_scope.hub import upload_results


def main():
    parser = argparse.ArgumentParser(
        description="Push EmotionScope vectors and figures to HuggingFace Hub"
    )
    parser.add_argument("--repo", required=True,
                        help="HF repo ID (e.g., AidanZach/EmotionScope-vectors)")
    parser.add_argument("--model", default=None,
                        help="Specific model to upload vectors for (default: all)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Don't upload figures")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo")
    parser.add_argument("--message", default=None,
                        help="Custom commit message")
    args = parser.parse_args()

    upload_results(
        hub_repo=args.repo,
        model_name=args.model,
        include_figures=not args.no_figures,
        private=args.private,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
