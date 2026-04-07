"""
Generate emotion-tagged stories using the model itself (Anthropic's approach).

For models large enough to generate coherent text (7B+), this produces
higher-quality training data than static templates because the activations
reflect how the model processes its OWN emotional content — matching
Anthropic's methodology exactly.

For small models (2B), use the LLM-generated templates in data/templates/
instead. Gemma 2 2B cannot generate coherent multi-paragraph stories.

Usage:
    # Generate 50 stories per emotion using the loaded model
    uv run python scripts/generate_stories.py --model google/gemma-2-9b-it

    # Generate more stories per emotion, with 4-bit quantization
    uv run python scripts/generate_stories.py \
        --model google/gemma-2-9b-it \
        --use-4bit \
        --stories-per-emotion 100

    # Use a specific output directory
    uv run python scripts/generate_stories.py \
        --model meta-llama/Llama-3-8b-instruct \
        --output data/story_contributions/llama3_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from emotion_scope.config import CORE_EMOTIONS, CORE_EMOTION_NAMES, DATA_DIR
from emotion_scope.models import load_model

# Prompt template for story generation.
# Designed to elicit situational narratives that evoke the emotion
# without naming it — matching the quality of the existing corpus.
GENERATION_PROMPT = """Write a brief narrative vignette (2-4 sentences) that evokes the feeling of {emotion} through a specific scene. Rules:
- Describe a concrete moment with physical details, sensory language, and specific actions
- Do NOT use the word "{emotion}" or any obvious synonym
- Do NOT name any emotion directly — the reader should FEEL it from the situation
- Make it a unique, realistic scenario — not a cliché or template
- Write in third person past tense

Example for a different emotion: "The dog's water bowl was still by the back door three weeks later, and no one could bring themselves to put it away, so it just stayed there gathering a thin film of dust."

Now write one for {emotion}:"""

# Synonyms to filter out — if the model names the emotion, regenerate
EMOTION_SYNONYMS = {
    "happy": {"happy", "happiness", "happily", "joyful", "joy", "joyous", "elated"},
    "sad": {"sad", "sadness", "sadly", "sorrowful", "sorrow", "grief", "grieving"},
    "afraid": {"afraid", "fear", "fearful", "scared", "terrified", "terror", "frightened"},
    "angry": {"angry", "anger", "angrily", "furious", "fury", "rage", "enraged"},
    "calm": {"calm", "calmly", "calmness", "serene", "serenity", "tranquil"},
    "desperate": {"desperate", "desperately", "desperation", "despair"},
    "hopeful": {"hopeful", "hopefully", "hopefulness", "hope"},
    "frustrated": {"frustrated", "frustrating", "frustration"},
    "curious": {"curious", "curiously", "curiosity"},
    "proud": {"proud", "proudly", "pride"},
    "guilty": {"guilty", "guilt", "guiltily"},
    "surprised": {"surprised", "surprising", "surprise", "surprisingly", "astonished"},
    "loving": {"loving", "lovingly", "love", "loved"},
    "hostile": {"hostile", "hostility", "hostilely"},
    "nervous": {"nervous", "nervously", "nervousness", "anxious", "anxiety"},
    "confident": {"confident", "confidently", "confidence"},
    "brooding": {"brooding", "broodingly", "brooded"},
    "enthusiastic": {"enthusiastic", "enthusiastically", "enthusiasm"},
    "reflective": {"reflective", "reflectively", "reflection"},
    "gloomy": {"gloomy", "gloomily", "gloominess", "gloom"},
}


def generate_story(model, tokenizer, emotion: str, max_retries: int = 3) -> str | None:
    """Generate a single story for the given emotion."""
    prompt = GENERATION_PROMPT.format(emotion=emotion)
    synonyms = EMOTION_SYNONYMS.get(emotion, {emotion})

    for attempt in range(max_retries):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=200,
                temperature=0.8 + attempt * 0.1,  # increase temp on retry
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = output[0, input_ids["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up: take first paragraph, strip quotes
        text = text.split("\n\n")[0].strip()
        text = text.strip('"').strip("'").strip()

        if len(text) < 40:
            continue

        # Check for emotion word leakage
        words = set(re.findall(r'\b\w+\b', text.lower()))
        if words & synonyms:
            continue  # Retry — model named the emotion

        return text

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion stories using the model itself (Anthropic's approach)"
    )
    parser.add_argument("--model", default="google/gemma-2-9b-it",
                        help="HF model id (should be 7B+ for coherent generation)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--stories-per-emotion", type=int, default=50)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: data/story_contributions/{model}_generated.jsonl)")
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="Specific emotions to generate (default: all 20)")
    args = parser.parse_args()

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_slug = args.model.replace("/", "_")
        output_path = DATA_DIR / "story_contributions" / f"{model_slug}_generated.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate] Loading {args.model}...")
    # Load with HF backend for generation (TransformerLens doesn't support generate)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    emotions = args.emotions or CORE_EMOTION_NAMES
    n = args.stories_per_emotion
    total = 0
    failed = 0

    print(f"[generate] Generating {n} stories per emotion for {len(emotions)} emotions")
    print(f"[generate] Output: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for emotion in emotions:
            successes = 0
            pbar = tqdm(range(n), desc=f"  {emotion}", leave=True)
            for _ in pbar:
                text = generate_story(model, tokenizer, emotion)
                if text:
                    entry = {"emotion": emotion, "text": text}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    successes += 1
                    total += 1
                else:
                    failed += 1
                pbar.set_postfix(ok=successes, fail=failed)

    print(f"\n[generate] Done. {total} stories written, {failed} failed.")
    print(f"[generate] Merge with: uv run python scripts/ingest_stories.py")


if __name__ == "__main__":
    main()
