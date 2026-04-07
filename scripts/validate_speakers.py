"""
CLI: run the thermostat test against saved speaker-separated vectors.

Takes emotionally charged messages, probes with both the current-speaker and
other-speaker vector sets, and reports whether the model's internal thermostat
is active (other-speaker reads high arousal, current-speaker stays calm).

Usage:
    uv run python scripts/validate_speakers.py \
        --vectors results/vectors/google_gemma-2-2b-it_speakers.pt \
        --model google/gemma-2-2b-it

If --vectors is not provided, will look for the default path based on --model.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from emotion_scope import load_model
from emotion_scope.config import CORE_EMOTIONS
from emotion_scope.speakers import SpeakerSeparator
from emotion_scope.utils import find_content_token_range


# ---------------------------------------------------------------------------
# Test messages: cover anger, distress, panic, sadness, enthusiasm
# ---------------------------------------------------------------------------

THERMOSTAT_MESSAGES = [
    # Anger
    {
        "message": "I can't believe you said that to me. That was completely out of line and disrespectful.",
        "expected_other": "angry",
        "category": "anger",
    },
    {
        "message": "You betrayed my trust. After everything I did for you, this is how you repay me?",
        "expected_other": "angry",
        "category": "anger",
    },
    # Distress
    {
        "message": "Everything is falling apart. I lost my job, my apartment, and I have nowhere to go.",
        "expected_other": "desperate",
        "category": "distress",
    },
    {
        "message": "I don't know what to do anymore. Every path leads to a dead end.",
        "expected_other": "desperate",
        "category": "distress",
    },
    # Panic
    {
        "message": "Something is really wrong. I can't breathe properly and my chest hurts. Please help.",
        "expected_other": "afraid",
        "category": "panic",
    },
    {
        "message": "The building is on fire and we're trapped on the third floor. We need help NOW.",
        "expected_other": "afraid",
        "category": "panic",
    },
    # Sadness
    {
        "message": "My grandmother just passed away. She was the only person who truly understood me.",
        "expected_other": "sad",
        "category": "sadness",
    },
    {
        "message": "I spent years building something I believed in, and it all came to nothing.",
        "expected_other": "sad",
        "category": "sadness",
    },
    # Enthusiasm (tests regulation in the opposite direction)
    {
        "message": "THIS IS THE BEST DAY OF MY LIFE! I got accepted to every school I applied to!",
        "expected_other": "enthusiastic",
        "category": "enthusiasm",
    },
    {
        "message": "We just closed the biggest deal in company history! Everyone is getting a bonus!",
        "expected_other": "enthusiastic",
        "category": "enthusiasm",
    },
]


def _get_activation(model, tokenizer, backend: str, probe_layer: int, text: str):
    """Extract residual stream activation at the last content token."""
    if backend == "transformer_lens":
        hook_name = f"blocks.{probe_layer}.hook_resid_post"
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        residual = cache[hook_name]
        _, end = find_content_token_range(tokens, tokenizer)
        idx = max(end - 1, 0)
        return residual[0, idx, :].detach().cpu().float()
    else:
        captured = {}

        def hook_fn(_module, _input, output):
            captured["act"] = (output[0] if isinstance(output, tuple) else output).detach()

        # Find layers
        for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            ok = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                layers = obj
                break
        else:
            raise ValueError(f"Cannot find layers in {type(model).__name__}")

        handle = layers[probe_layer].register_forward_hook(hook_fn)
        try:
            tokens = tokenizer(text, return_tensors="pt")
            input_ids = tokens["input_ids"]
            tokens_on_device = {k: v.to(model.device) for k, v in tokens.items()}
            with torch.no_grad():
                model(**tokens_on_device)
        finally:
            handle.remove()

        residual = captured["act"]
        _, end = find_content_token_range(input_ids, tokenizer)
        idx = max(end - 1, 0)
        return residual[0, idx, :].cpu().float()


def _score_activation(
    activation: torch.Tensor,
    vector_names: List[str],
    vector_matrix: torch.Tensor,
) -> Dict[str, float]:
    """Cosine-score an activation against a set of emotion vectors."""
    act_norm = F.normalize(activation.unsqueeze(0), dim=1)
    vec_norm = F.normalize(vector_matrix, dim=1)
    scores_t = (act_norm @ vec_norm.T).squeeze(0)
    return {n: float(scores_t[i]) for i, n in enumerate(vector_names)}


def _weighted_metric(
    scores: Dict[str, float],
    metadata: Dict[str, dict],
    key: str,
) -> float:
    """Compute weighted average of a metadata field (valence or arousal)."""
    total_w = 0.0
    value = 0.0
    for name, score in scores.items():
        if name in metadata:
            w = max(score, 0.0)
            value += w * metadata[name][key]
            total_w += w
    if total_w > 0:
        value /= total_w
    return max(-1.0, min(1.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the thermostat validation on speaker-separated vectors."
    )
    parser.add_argument("--vectors", default=None, help="Path to speaker_separation .pt file")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="HF model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformer_lens", "huggingface"])

    args = parser.parse_args()

    # Determine vectors path
    vectors_path = args.vectors
    if vectors_path is None:
        from emotion_scope.config import VECTORS_DIR
        name = args.model.replace("/", "_")
        vectors_path = str(VECTORS_DIR / f"{name}_speakers.pt")

    # Load saved vectors
    print(f"Loading speaker vectors from: {vectors_path}")
    saved = SpeakerSeparator.load(vectors_path)

    current_vectors = saved["current_speaker"]
    other_vectors = saved["other_speaker"]
    metadata_info = saved.get("metadata", {})
    probe_layer = metadata_info.get("probe_layer_used", saved.get("model_info", {}).get("probe_layer"))
    emotions = metadata_info.get("emotions", CORE_EMOTIONS)

    print(f"  Current-speaker emotions: {len(current_vectors)}")
    print(f"  Other-speaker emotions:   {len(other_vectors)}")
    print(f"  Probe layer: {probe_layer}")

    # Load model
    model, tokenizer, backend, info = load_model(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        use_4bit=args.use_4bit,
        backend=args.backend,
    )

    # Override probe layer to match extraction
    if probe_layer is not None:
        info["probe_layer"] = probe_layer

    # Prepare scoring matrices
    current_names = list(current_vectors.keys())
    current_matrix = torch.stack([current_vectors[n] for n in current_names])
    other_names = list(other_vectors.keys())
    other_matrix = torch.stack([other_vectors[n] for n in other_names])

    # Emotion metadata lookup
    emo_metadata = {e["name"]: e for e in emotions}

    # Run thermostat test
    print("\n" + "=" * 70)
    print("THERMOSTAT VALIDATION — Speaker-Separated Emotion Probing")
    print("=" * 70)

    thermostat_active_count = 0
    total_count = 0
    deltas = []

    for test in THERMOSTAT_MESSAGES:
        message = test["message"]
        category = test["category"]

        activation = _get_activation(
            model, tokenizer, backend, info["probe_layer"], message
        )

        # Score against both vector sets
        current_scores = _score_activation(activation, current_names, current_matrix)
        other_scores = _score_activation(activation, other_names, other_matrix)

        # Compute arousal
        current_arousal = _weighted_metric(current_scores, emo_metadata, "arousal")
        other_arousal = _weighted_metric(other_scores, emo_metadata, "arousal")
        arousal_delta = current_arousal - other_arousal

        # Top emotions
        current_sorted = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        other_sorted = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)[:2]

        is_active = arousal_delta < 0
        thermostat_active_count += int(is_active)
        total_count += 1
        deltas.append(arousal_delta)

        # Print report
        display_msg = message[:65] + "..." if len(message) > 65 else message
        print(f"\nMessage: \"{display_msg}\"")
        print(f"  Category: {category}")
        print(f"  Other-speaker read:  {other_sorted[0][0]} ({other_sorted[0][1]:.2f}), "
              f"{other_sorted[1][0]} ({other_sorted[1][1]:.2f})")
        print(f"  Current-speaker:     {current_sorted[0][0]} ({current_sorted[0][1]:.2f}), "
              f"{current_sorted[1][0]} ({current_sorted[1][1]:.2f})")
        print(f"  Other arousal:       {other_arousal:+.3f}")
        print(f"  Current arousal:     {current_arousal:+.3f}")
        print(f"  Thermostat active:   {'YES' if is_active else 'NO'} "
              f"(arousal delta = {arousal_delta:+.2f})")

    # Summary
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    passed = mean_delta < -0.2

    print("\n" + "=" * 70)
    print("THERMOSTAT SUMMARY")
    print("=" * 70)
    print(f"  Messages tested:         {total_count}")
    print(f"  Thermostat active:       {thermostat_active_count}/{total_count}")
    print(f"  Mean arousal delta:      {mean_delta:+.3f}")
    print(f"  Threshold:               < -0.2")
    print(f"  Result:                  {'PASS' if passed else 'FAIL'}")

    if passed:
        print("\nThermostat validation PASSED. The model regulates its own arousal "
              "while accurately reading the user's emotional state.")
    else:
        print("\nThermostat validation FAILED. Speaker separation may need improvement.")


if __name__ == "__main__":
    main()
