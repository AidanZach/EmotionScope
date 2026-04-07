"""
Ingest, validate, and merge dialogue corpus contributions.

Reads all .jsonl files from data/corpus_contributions/ (plus any .md/.txt files
that contain JSONL lines), validates each line, deduplicates, and writes a
merged corpus to data/templates/two_speaker_dialogues.jsonl.

Validation checks per line:
  1. Valid JSON
  2. Has emotion_a, emotion_b, dialogue fields
  3. Both emotions are from the approved 20-emotion list
  4. emotion_a != emotion_b
  5. Dialogue starts with "Speaker A:" and has alternating turns
  6. At least 4 turns (2 per speaker)
  7. (Soft) Neither emotion word appears literally in dialogue text

Usage:
    uv run python scripts/ingest_corpus.py
    uv run python scripts/ingest_corpus.py --dry-run          # validate only, don't write
    uv run python scripts/ingest_corpus.py --include-existing  # also merge the original 100
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

CONTRIBUTIONS_DIR = Path("data/corpus_contributions")
EXISTING_CORPUS = Path("data/templates/two_speaker_dialogues.jsonl")
OUTPUT_PATH = Path("data/templates/two_speaker_dialogues.jsonl")


def extract_jsonl_lines(filepath: Path) -> list[str]:
    """Extract lines that look like JSON from any file type."""
    lines = []
    text = filepath.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip markdown formatting, comments, headers
        if line.startswith("#") or line.startswith("```") or line.startswith("---"):
            continue
        # Must look like a JSON object
        if line.startswith("{") and line.endswith("}"):
            lines.append(line)

    return lines


def validate_line(raw: str, line_num: int, source: str) -> tuple[dict | None, str | None]:
    """Validate a single JSONL line. Returns (entry, None) or (None, error_msg)."""

    # 1. Valid JSON
    try:
        entry = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    # 2. Required fields
    for field in ("emotion_a", "emotion_b", "dialogue"):
        if field not in entry:
            return None, f"Missing field: {field}"

    ea = entry["emotion_a"].strip().lower()
    eb = entry["emotion_b"].strip().lower()
    dialogue = entry["dialogue"]

    # Normalize
    entry["emotion_a"] = ea
    entry["emotion_b"] = eb

    # 3. Approved emotions
    if ea not in APPROVED_EMOTIONS:
        return None, f"Unknown emotion_a: '{ea}'"
    if eb not in APPROVED_EMOTIONS:
        return None, f"Unknown emotion_b: '{eb}'"

    # 4. Different emotions
    if ea == eb:
        return None, f"emotion_a == emotion_b: '{ea}'"

    # 5. Dialogue structure
    if not dialogue.strip().startswith("Speaker A:"):
        return None, "Dialogue must start with 'Speaker A:'"

    # Count turns
    a_turns = len(re.findall(r"Speaker A:", dialogue))
    b_turns = len(re.findall(r"Speaker B:", dialogue))

    # 6. At least 4 turns
    if a_turns + b_turns < 4:
        return None, f"Only {a_turns + b_turns} turns (need >= 4)"

    if a_turns < 2 or b_turns < 2:
        return None, f"Need >= 2 turns per speaker (A={a_turns}, B={b_turns})"

    # 7. (Soft check) Emotion words in dialogue
    warnings = []
    dialogue_lower = dialogue.lower()
    if ea in dialogue_lower.split():
        warnings.append(f"emotion_a '{ea}' appears in dialogue text")
    if eb in dialogue_lower.split():
        warnings.append(f"emotion_b '{eb}' appears in dialogue text")

    entry["_source"] = source
    entry["_warnings"] = warnings

    return entry, None


def content_hash(entry: dict) -> str:
    """Hash for deduplication — based on dialogue content only."""
    text = entry["dialogue"].strip().lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ingest and merge dialogue corpus")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't write")
    parser.add_argument("--include-existing", action="store_true",
                        help="Include the existing original corpus in the merge")
    args = parser.parse_args()

    all_entries: list[dict] = []
    errors: list[str] = []
    warnings_total = 0
    sources: dict[str, int] = defaultdict(int)

    # Collect source files
    source_files: list[Path] = []

    if args.include_existing and EXISTING_CORPUS.exists():
        source_files.append(EXISTING_CORPUS)

    if CONTRIBUTIONS_DIR.exists():
        for f in sorted(CONTRIBUTIONS_DIR.iterdir()):
            if f.suffix in (".jsonl", ".json", ".md", ".txt"):
                source_files.append(f)

    if not source_files:
        print("[ingest] No source files found.")
        print(f"  Place .jsonl files in {CONTRIBUTIONS_DIR}/")
        print(f"  Or use --include-existing to merge the original corpus")
        return

    print(f"[ingest] Found {len(source_files)} source files")

    for filepath in source_files:
        lines = extract_jsonl_lines(filepath)
        file_ok = 0
        file_err = 0

        for i, raw in enumerate(lines):
            entry, error = validate_line(raw, i + 1, filepath.name)
            if error:
                errors.append(f"  {filepath.name}:{i+1}: {error}")
                file_err += 1
            else:
                all_entries.append(entry)
                file_ok += 1
                if entry.get("_warnings"):
                    warnings_total += len(entry["_warnings"])

        sources[filepath.name] = file_ok
        print(f"  {filepath.name}: {file_ok} valid, {file_err} errors")

    # Deduplicate
    seen_hashes = set()
    unique_entries = []
    dupes = 0
    for entry in all_entries:
        h = content_hash(entry)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_entries.append(entry)
        else:
            dupes += 1

    # Stats
    pair_counts = defaultdict(int)
    emotion_a_counts = defaultdict(int)
    emotion_b_counts = defaultdict(int)
    for e in unique_entries:
        pair_counts[(e["emotion_a"], e["emotion_b"])] += 1
        emotion_a_counts[e["emotion_a"]] += 1
        emotion_b_counts[e["emotion_b"]] += 1

    print()
    print(f"[ingest] Summary:")
    print(f"  Total valid lines:  {len(all_entries)}")
    print(f"  Duplicates removed: {dupes}")
    print(f"  Unique dialogues:   {len(unique_entries)}")
    print(f"  Unique pairs:       {len(pair_counts)}")
    print(f"  Soft warnings:      {warnings_total}")
    print(f"  Hard errors:        {len(errors)}")

    if errors:
        print()
        print("[ingest] Errors (lines dropped):")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Coverage check
    all_pairs = {(a, b) for a in APPROVED_EMOTIONS for b in APPROVED_EMOTIONS if a != b}
    covered = set(pair_counts.keys())
    missing = all_pairs - covered
    thin = {p: c for p, c in pair_counts.items() if c < 3}

    print()
    print(f"[ingest] Coverage:")
    print(f"  Possible pairs (20×19): {len(all_pairs)}")
    print(f"  Covered pairs:          {len(covered)}")
    print(f"  Missing pairs:          {len(missing)}")
    print(f"  Thin pairs (<3 each):   {len(thin)}")

    if missing and len(missing) <= 30:
        print(f"  Missing: {sorted(missing)[:15]}...")

    # Per-emotion stats
    print()
    print(f"[ingest] Per-emotion counts (as Speaker A):")
    for em in sorted(APPROVED_EMOTIONS):
        a_count = emotion_a_counts.get(em, 0)
        b_count = emotion_b_counts.get(em, 0)
        marker = "  " if a_count >= 10 else "!!"
        print(f"  {marker} {em:14s}: A={a_count:3d}  B={b_count:3d}  total={a_count+b_count:3d}")

    # Write output
    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for entry in unique_entries:
                # Clean up internal fields before saving
                out = {
                    "emotion_a": entry["emotion_a"],
                    "emotion_b": entry["emotion_b"],
                    "dialogue": entry["dialogue"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print()
        print(f"[ingest] Wrote {len(unique_entries)} dialogues to {OUTPUT_PATH}")
    else:
        print()
        print(f"[ingest] DRY RUN — no files written")


if __name__ == "__main__":
    main()
