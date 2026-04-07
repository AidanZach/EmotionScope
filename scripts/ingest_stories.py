"""
Ingest, validate, and merge emotion story contributions.

Reads all .jsonl/.md/.txt files from data/story_contributions/,
validates each line, deduplicates, and merges with the existing
emotion_stories.jsonl corpus.

Validation checks per line:
  1. Valid JSON
  2. Has emotion and text fields
  3. Emotion is from the approved 20-emotion list
  4. Text is non-empty and at least 40 characters
  5. (Soft) Emotion word does not appear in the text
  6. (Soft) Text is not too short (< 80 chars) or too long (> 800 chars)

Usage:
    uv run python scripts/ingest_stories.py
    uv run python scripts/ingest_stories.py --dry-run
    uv run python scripts/ingest_stories.py --no-existing   # don't merge with current corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict

APPROVED_EMOTIONS = {
    "happy", "sad", "afraid", "angry", "calm", "desperate", "hopeful",
    "frustrated", "curious", "proud", "guilty", "surprised", "loving",
    "hostile", "nervous", "confident", "brooding", "enthusiastic",
    "reflective", "gloomy",
}

# Synonyms/derivatives to soft-check for emotion leakage
EMOTION_SYNONYMS = {
    "happy": {"happy", "happiness", "happily", "happier", "happiest"},
    "sad": {"sad", "sadness", "sadly", "sadder", "saddest"},
    "afraid": {"afraid", "fear", "fearful", "feared", "fearing", "frightened"},
    "angry": {"angry", "anger", "angrily", "angrier", "angriest", "furious", "fury"},
    "calm": {"calm", "calmly", "calmer", "calmest", "calmness"},
    "desperate": {"desperate", "desperately", "desperation"},
    "hopeful": {"hopeful", "hopefully", "hopefulness"},
    "frustrated": {"frustrated", "frustrating", "frustration"},
    "curious": {"curious", "curiously", "curiosity"},
    "proud": {"proud", "proudly", "pride"},
    "guilty": {"guilty", "guilt", "guiltily"},
    "surprised": {"surprised", "surprising", "surprise", "surprisingly"},
    "loving": {"loving", "lovingly"},
    "hostile": {"hostile", "hostility", "hostilely"},
    "nervous": {"nervous", "nervously", "nervousness"},
    "confident": {"confident", "confidently", "confidence"},
    "brooding": {"brooding", "broodingly", "brooded"},
    "enthusiastic": {"enthusiastic", "enthusiastically", "enthusiasm"},
    "reflective": {"reflective", "reflectively"},
    "gloomy": {"gloomy", "gloomily", "gloominess", "gloom"},
}

CONTRIBUTIONS_DIR = Path("data/story_contributions")
EXISTING_CORPUS = Path("data/templates/emotion_stories.jsonl")
OUTPUT_PATH = Path("data/templates/emotion_stories.jsonl")


def extract_jsonl_lines(filepath: Path) -> list[str]:
    """Extract lines that look like JSON from any file type."""
    lines = []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```") or line.startswith("---"):
            continue
        if line.startswith("{") and line.endswith("}"):
            lines.append(line)
    return lines


def validate_line(raw: str, line_num: int, source: str) -> tuple[dict | None, str | None, list[str]]:
    """Validate a single JSONL line. Returns (entry, error, warnings)."""
    warnings = []

    try:
        entry = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}", []

    if "emotion" not in entry:
        return None, "Missing 'emotion' field", []
    if "text" not in entry:
        return None, "Missing 'text' field", []

    emotion = entry["emotion"].strip().lower()
    text = entry["text"].strip()

    entry["emotion"] = emotion
    entry["text"] = text

    if emotion not in APPROVED_EMOTIONS:
        return None, f"Unknown emotion: '{emotion}'", []

    if len(text) < 40:
        return None, f"Text too short ({len(text)} chars, need >= 40)", []

    # Soft checks
    if len(text) < 80:
        warnings.append(f"Short text ({len(text)} chars)")
    if len(text) > 800:
        warnings.append(f"Long text ({len(text)} chars)")

    # Check for emotion word leakage
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    synonyms = EMOTION_SYNONYMS.get(emotion, {emotion})
    leaked = text_words & synonyms
    if leaked:
        warnings.append(f"Emotion word(s) in text: {leaked}")

    return entry, None, warnings


def content_hash(text: str) -> str:
    """Hash for deduplication — normalized text content."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ingest and merge emotion story contributions")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't write")
    parser.add_argument("--no-existing", action="store_true",
                        help="Don't include existing emotion_stories.jsonl in merge")
    args = parser.parse_args()

    all_entries: list[dict] = []
    errors: list[str] = []
    total_warnings = 0

    source_files: list[Path] = []

    if not args.no_existing and EXISTING_CORPUS.exists():
        source_files.append(EXISTING_CORPUS)

    if CONTRIBUTIONS_DIR.exists():
        for f in sorted(CONTRIBUTIONS_DIR.iterdir()):
            if f.suffix in (".jsonl", ".json", ".md", ".txt"):
                source_files.append(f)

    if not source_files:
        print("[ingest-stories] No source files found.")
        print(f"  Place .jsonl files in {CONTRIBUTIONS_DIR}/")
        return

    print(f"[ingest-stories] Found {len(source_files)} source files")

    for filepath in source_files:
        lines = extract_jsonl_lines(filepath)
        file_ok = 0
        file_err = 0
        file_warn = 0

        for i, raw in enumerate(lines):
            entry, error, warns = validate_line(raw, i + 1, filepath.name)
            if error:
                errors.append(f"  {filepath.name}:{i+1}: {error}")
                file_err += 1
            else:
                entry["_source"] = filepath.name
                all_entries.append(entry)
                file_ok += 1
                file_warn += len(warns)
                total_warnings += len(warns)

        print(f"  {filepath.name}: {file_ok} valid, {file_err} errors, {file_warn} warnings")

    # Deduplicate
    seen_hashes = set()
    unique_entries = []
    dupes = 0
    for entry in all_entries:
        h = content_hash(entry["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_entries.append(entry)
        else:
            dupes += 1

    # Stats
    emotion_counts = defaultdict(int)
    for e in unique_entries:
        emotion_counts[e["emotion"]] += 1

    print()
    print(f"[ingest-stories] Summary:")
    print(f"  Total valid lines:  {len(all_entries)}")
    print(f"  Duplicates removed: {dupes}")
    print(f"  Unique stories:     {len(unique_entries)}")
    print(f"  Soft warnings:      {total_warnings}")
    print(f"  Hard errors:        {len(errors)}")

    if errors:
        print()
        print("[ingest-stories] Errors (lines dropped):")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Per-emotion breakdown
    print()
    print(f"[ingest-stories] Per-emotion counts (target: 50 each):")
    for em in sorted(APPROVED_EMOTIONS):
        count = emotion_counts.get(em, 0)
        bar = "#" * min(count, 60)
        marker = "OK" if count >= 50 else f"NEED {50 - count} MORE"
        print(f"  {em:14s}: {count:3d}  {bar}  [{marker}]")

    total_target = 50 * 20
    total_have = sum(emotion_counts.values())
    print(f"\n  Total: {total_have}/{total_target} ({total_have/total_target*100:.0f}%)")

    # Write output
    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for entry in unique_entries:
                out = {"emotion": entry["emotion"], "text": entry["text"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\n[ingest-stories] Wrote {len(unique_entries)} stories to {OUTPUT_PATH}")
    else:
        print(f"\n[ingest-stories] DRY RUN — no files written")


if __name__ == "__main__":
    main()
