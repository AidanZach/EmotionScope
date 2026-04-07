# Implementation Instructions — Phase 1 Sprint

## Context
You've scaffolded the directory structure. Now implement the actual code.
Your 10 questions have been answered by the architect. Key decisions below.

## Answers to Your Questions (Summary)

1. **Probe layer:** Default `floor(2L/3)`, PLUS implement `find_best_probe_layer()` that sweeps all layers and picks best valence-separation. One-time cost per model.

2. **Last token position:** DO NOT use `residual[0, -1, :]` blindly. Implement `find_content_token_range(input_ids, tokenizer)` in `utils.py` that identifies actual content tokens vs chat template markup (`<start_of_turn>`, `<end_of_turn>`, role tokens). For extraction: average across content tokens only. For probing: read the last content token of the user's message (just before `<end_of_turn>`).

3. **Speaker separation:** For each emotion e, current-speaker vector = mean activation across dialogues where e_A = e (averaging over all e_B values) minus grand mean. Other-speaker vector = mean across dialogues where e_B = e (averaging over all e_A values) minus grand mean. Defer full implementation to Phase 2 — stub it out for now.

4. **Neutral PCA:** 50% is confirmed from Anthropic's paper. Keep as default, make configurable. Run both raw and denoised through Tylenol test to confirm denoising helps.

5. **Steering:** alpha=0.5 relative to avg residual norm. Apply across middle third of layers [L/3, 2L/3]. Defer to Phase 2.

6. **Dialogue corpus:** Defer to Phase 2. Stub `speakers.py`.

7. **Color mapping:** Use CLAUDE.md version. Mark as TODO for redesign.

8. **Validation thresholds:** Tylenol Spearman r > 0.7 (afraid vs log-dose), valence separation cosine < -0.2, confusion matrix top-3 accuracy > 60%, emotion richness avg pairwise cosine < 0.5.

9. **TransformerLens:** Try it first. If `run_with_cache` works on Gemma 2 2B and returns `blocks.{layer}.hook_resid_post` with correct shape, use it. If anything is broken, fall back to HF hooks immediately.

10. **Demo:** Local Gemma 2B, per-turn batch probing. Defer to Phase 2.

---

## Implementation Order — Do This Now

### Step 1: `emotion_scope/config.py`
Implement exactly as specified in CLAUDE.md. This is the source of truth.
Add `find_content_token_range` to the planned utils interface.

### Step 2: `emotion_scope/utils.py`
Implement:
- `cosine_similarity_matrix(vectors)` — pairwise cosine sim
- `valence_separation(vectors)` — metric for validation gate
- `find_content_token_range(input_ids, tokenizer)` — returns (start_idx, end_idx) of content tokens, excluding chat template tokens. For Gemma 2, these are `<start_of_turn>`, `<end_of_turn>`, `user\n`, `model\n`. For non-chat models, return (0, seq_len).
- `get_device()` — auto-detect cuda/cpu

### Step 3: `emotion_scope/models.py`
Implement exactly as in CLAUDE.md but with one addition:
After loading, run a smoke test: forward pass on "Hello world", verify the cache/hook captures a tensor of shape (1, *, d_model) at the probe layer.
Print a confirmation: "Model loaded: {name}, {n_layers} layers, d_model={d_model}, probe_layer={probe_layer}, backend={backend}"

Try TransformerLens first for Gemma 2. If it fails, fall back to HF.

### Step 4: `emotion_scope/visualize.py`
Implement as in CLAUDE.md. Add a `# TODO: full aesthetic redesign pending` comment at top.

### Step 5: `emotion_scope/extract.py`
Implement as in CLAUDE.md with these modifications:
- In `_get_activation_tl`: use `names_filter` to only cache the probe layer (saves VRAM)
- In `_get_activation_tl` and `_get_activation_hf`: use `find_content_token_range` to average only over content tokens
- Add `find_best_probe_layer(self, test_texts=None)` method that:
  1. Extracts vectors at every 3rd layer (for speed)
  2. Computes valence_separation at each
  3. Returns the layer with the most negative cosine (best separation)
  4. Updates self.probe_layer

### Step 6: `emotion_scope/probe.py`
Implement as in CLAUDE.md with the token position fix:
- For `analyze()`: use last content token, not raw last token
- For `analyze_conversation()`: tokenize with the model's chat template, find the user content range, read at the last user content token

### Step 7: `data/templates/emotion_stories.jsonl`
Create the template file. Format: one JSON object per line.
```json
{"emotion": "afraid", "text": "The character's heart raced as they realized the door was locked from the outside and there was no way to call for help"}
```
Create at MINIMUM 10 templates per core emotion (20 emotions x 10 = 200 lines).
Templates must evoke the emotion through SITUATION, not by naming it.
Bad: "They felt very afraid"
Good: "A cold wave swept through them when the lab results came back positive for the thing they'd been dreading"

### Step 8: `data/neutral/neutral_prompts.jsonl`
Create 50+ neutral prompts. Format:
```json
{"text": "The weather today is partly cloudy with temperatures around seventy degrees"}
```
Topics: weather, software installation, cooking measurements, geography facts, meeting schedules, chemical formulas, historical dates, transportation schedules.

### Step 9: `data/validation/intensity_scales.json`
Create the Tylenol test and other intensity scales (see the full JSON below).

### Step 10: `data/validation/implicit_scenarios.json`
Create 12 scenarios that evoke emotions without naming them. Covering: frustration, relief, grief, excitement, betrayal, gratitude, dread, triumph, loneliness, wonder, guilt, serenity.

### Step 11: `emotion_scope/validate.py`
Implement the validation suite with the concrete thresholds:
```python
class ValidationResult:
    tylenol_passed: bool          # Spearman r > 0.7 for afraid vs log-dose
    confusion_passed: bool        # Top-3 accuracy > 60%
    valence_passed: bool          # Cosine < -0.2
    richness_passed: bool         # Avg pairwise cosine < 0.5
    all_passed: bool              # AND of all above
    details: dict                 # Full metrics for reporting
```

### Step 12: `tests/conftest.py` + basic tests
Create fixtures and minimal tests:
- Test that `config.py` loads correctly
- Test that `visualize.scores_to_color` returns valid hex colors
- Test that `utils.cosine_similarity_matrix` produces correct shapes
- Test that `utils.find_content_token_range` correctly identifies Gemma chat template tokens

### Step 13: First real run (DO NOT EXECUTE — Gemma 2B needs to download)
Just write the script. User will run it after downloading weights.

---

## What NOT To Do Yet

- Do NOT implement `speakers.py` beyond a stub — Phase 2
- Do NOT implement `steer.py` beyond a stub — Phase 2
- Do NOT build the Gradio demo — Phase 2
- Do NOT implement streaming/token-by-token probing — Phase 3
- Do NOT optimize for speed — correctness first
- Do NOT download Gemma 9B — 2B only for now

## Success Criteria for This Sprint

```bash
cd C:\Users\AJZax\Projects\EmotionScope\emotion-scope
uv run python -c "from emotion_scope import load_model, EmotionExtractor; print('imports work')"
uv run python -c "from emotion_scope.visualize import scores_to_color; print(scores_to_color({'happy': 0.5, 'sad': 0.1}, []))"
uv run pytest tests/ -v
```
