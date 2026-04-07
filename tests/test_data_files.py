"""Sanity checks for the data files shipped in Phase 1."""

import json

from emotion_scope.config import CORE_EMOTION_NAMES, DATA_DIR


def test_emotion_stories_jsonl_exists_and_covers_all_emotions():
    path = DATA_DIR / "templates" / "emotion_stories.jsonl"
    assert path.exists()
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            counts[entry["emotion"]] = counts.get(entry["emotion"], 0) + 1
    for name in CORE_EMOTION_NAMES:
        assert name in counts, f"missing templates for '{name}'"
        assert counts[name] >= 10, f"'{name}' has only {counts[name]} templates (need >=10)"


def test_neutral_prompts_jsonl():
    path = DATA_DIR / "neutral" / "neutral_prompts.jsonl"
    assert path.exists()
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 50
    for l in lines:
        entry = json.loads(l)
        assert "text" in entry and isinstance(entry["text"], str)


def test_intensity_scales_structure():
    path = DATA_DIR / "validation" / "intensity_scales.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "tylenol" in data
    for name, scenario in data.items():
        assert "template" in scenario
        assert "{value}" in scenario["template"]
        assert "values" in scenario and len(scenario["values"]) >= 3


def test_implicit_scenarios_structure():
    path = DATA_DIR / "validation" / "implicit_scenarios.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 12
    for scen in data:
        assert "scenario" in scen
        assert "expected_emotions" in scen and scen["expected_emotions"]
        for e in scen["expected_emotions"]:
            assert e in CORE_EMOTION_NAMES


def test_emotions_core20_json():
    path = DATA_DIR / "emotions_core20.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 20
    names = {e["name"] for e in data}
    assert names == set(CORE_EMOTION_NAMES)
