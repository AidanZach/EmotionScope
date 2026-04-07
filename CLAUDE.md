# CLAUDE.md — EmotionScope

## Project Identity

**EmotionScope** is an open-source Python toolkit for extracting, probing, and visualizing "functional emotion" vectors from open-weight language models. It replicates and extends Anthropic's April 2026 paper ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2026/emotions/index.html) on models anyone can run.

Independent research. Not affiliated with Anthropic or Google.

---

## Quick Orientation

The project has three layers:

1. **`emotion_scope/`** — Python package. Extracts emotion direction vectors from transformer residual streams, probes them in real-time, validates them.
2. **`backend/server.py`** — FastAPI server. Loads a model + vectors, serves chat + emotion probe endpoints.
3. **`frontend/`** — React + Three.js app. Renders animated orbs that visualize emotion state in real-time.

---

## Setup

### Prerequisites
- Python 3.11+ 
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ (for frontend)
- NVIDIA GPU recommended (8GB+ VRAM for Gemma 2 2B in fp16)

### Detect your hardware
```bash
# GPU
nvidia-smi

# CPU + RAM
# Windows:
systeminfo | findstr /C:"Processor" /C:"Total Physical Memory"
# Linux/Mac:
lscpu && free -h
```

### VRAM requirements by model
| Model | fp16 | 4-bit NF4 |
|-------|------|-----------|
| Gemma 2 2B IT | ~4 GB | ~2 GB |
| Gemma 2 9B IT | ~18 GB | ~5-6 GB |
| Llama 3 8B | ~16 GB | ~5 GB |
| Gemma 2 27B IT | ~54 GB | ~16 GB |

### Install
```bash
git clone https://github.com/AidanZach/EmotionScope.git
cd EmotionScope
uv sync                        # Python dependencies
cd frontend && npm install     # Frontend dependencies
```

---

## Running the Full Stack

### 1. Extract emotion vectors (one-time per model)
```bash
uv run python scripts/extract_all.py \
  --model google/gemma-2-2b-it \
  --sweep-layers
```
Output: `results/vectors/google_gemma-2-2b-it.pt`

### 2. Validate
```bash
uv run python scripts/validate_all.py \
  --vectors results/vectors/google_gemma-2-2b-it.pt
```

### 3. Start backend
```bash
uv run uvicorn backend.server:app --port 8000
```
Or with a different model:
```bash
ES_MODEL=google/gemma-2-9b-it \
ES_VECTORS=results/vectors/google_gemma-2-9b-it.pt \
uv run uvicorn backend.server:app --port 8000
```

### 4. Start frontend
```bash
cd frontend && npm run dev
# Open http://localhost:5173
```

---

## Swapping Models

EmotionScope supports any HuggingFace-compatible transformer. The pipeline is:

```bash
# 1. Extract (always use --sweep-layers for a new model)
uv run python scripts/extract_all.py \
  --model <HF_MODEL_ID> \
  --sweep-layers \
  --use-4bit              # optional, for large models on limited VRAM

# 2. Validate
uv run python scripts/validate_all.py \
  --vectors results/vectors/<model_slug>.pt

# 3. (Optional) Generate model-specific training data
#    For 7B+ models that can generate coherent text:
uv run python scripts/generate_stories.py \
  --model <HF_MODEL_ID> \
  --use-4bit
uv run python scripts/ingest_stories.py   # merge into corpus

# 4. Run server with the new model
ES_MODEL=<HF_MODEL_ID> \
ES_VECTORS=results/vectors/<model_slug>.pt \
uv run uvicorn backend.server:app --port 8000
```

**Supported chat template families:** Gemma, Llama, Mistral, Phi, Qwen, DeepSeek, and a generic fallback. Auto-detected from the tokenizer name.

**Backend selection:** TransformerLens is preferred (faster, cleaner hooks) but only supports ~15 model families. All other models automatically fall back to HuggingFace with manual forward hooks. No code changes needed.

---

## Architecture

```
emotion_scope/
  config.py          20 core emotions, valence/arousal metadata, paths, thresholds
  models.py          load_model() — TransformerLens or HuggingFace backend
  extract.py         EmotionExtractor — contrastive mean difference + PCA denoising
  probe.py           EmotionProbe — real-time cosine similarity scoring
  validate.py        Tylenol intensity, top-3 recall, valence separation, richness
  visualize.py       OKLCH color mapping, scores_to_orb_state()
  speakers.py        Speaker separation (experimental)
  utils.py           Token range detection, cosine matrix, valence separation metric

backend/
  server.py          FastAPI: /health, /chat, /probe endpoints

frontend/src/
  components/
    EmotionOrb.jsx   R3F orb with MeshDistortMaterial, environment map, bloom
    ValenceStrip.jsx  Dual-marker valence gradient
    DetailPanel.jsx   Collapsible per-emotion score bars
    Timeline.jsx      Conversation emotion history
    Legend.jsx         Visual guide explaining each orb property
    ExplainerPanel.jsx "How this works" explainer
    ChatPanel.jsx     Chat input + message display
  utils/
    palette.js        OKLCH emotion palette, emotionToOrbProps()
    oklch.js          OKLCH to RGB conversion, circular hue blending

scripts/
  extract_all.py     CLI: extract vectors for any model
  validate_all.py    CLI: run validation suite
  generate_stories.py CLI: model-generated training data (Anthropic's approach)
  ingest_stories.py  Merge + validate story contributions
  ingest_corpus.py   Merge + validate dialogue contributions

data/
  templates/         1,000 emotion stories + 1,240 two-speaker dialogues
  neutral/           100 neutral prompts for PCA denoising
  validation/        Tylenol intensity scales, implicit scenarios
  story_contributions/   Additional story batches (merged by ingest_stories.py)
  corpus_contributions/  Additional dialogue batches (merged by ingest_corpus.py)
```

---

## Data Pipeline

**Emotion stories** (`data/templates/emotion_stories.jsonl`):
- 1,000 stories (20 emotions x 50 each)
- Format: `{"emotion": "afraid", "text": "The CT scan showed..."}`
- Used by `extract.py` to build base emotion direction vectors
- Can be expanded via `scripts/ingest_stories.py` from `data/story_contributions/`

**Two-speaker dialogues** (`data/templates/two_speaker_dialogues.jsonl`):
- 1,240 dialogues covering all 380 emotion pairs
- Format: `{"emotion_a": "afraid", "emotion_b": "calm", "dialogue": "Speaker A: ...\nSpeaker B: ..."}`
- Used by `speakers.py` for speaker separation vectors
- Can be expanded via `scripts/ingest_corpus.py` from `data/corpus_contributions/`

**Neutral prompts** (`data/neutral/neutral_prompts.jsonl`):
- 100 factual/procedural texts with zero emotional content
- Used for PCA denoising during extraction

To generate more data, use `Documentation/STORY_GENERATION_PROMPT.md` or `Documentation/CORPUS_GENERATION_PROMPT.md` — give them to any capable LLM.

---

## Key Technical Decisions

- **Probe layer:** Default is `round(n_layers * 22/26)` for Gemma-family, falls back to 2/3 for unknown models. Always use `--sweep-layers` for a new model — Gemma 2 2B's optimum is 84.6% (layer 22/26).
- **Grand mean:** Pooled across all raw activations (not mean-of-means). Correct regardless of per-emotion sample imbalance.
- **PCA denoising:** Top-k components explaining 50% of neutral variance projected out. With n=100 << d=2304, PCA is rank-constrained.
- **Token position:** Probes at the last token of the full prompt (model turn marker), matching Anthropic's described methodology.
- **Valence/arousal metadata:** Based on Russell (1980) circumplex + NRC Lexicon. Used for visualization only — vectors are purely from activations.
- **Frontend rendering:** R3F + MeshDistortMaterial + environment map + bloom. Per-property spring physics.

---

## Testing

```bash
uv run pytest tests/ -v              # Unit tests
uv run ruff check emotion_scope/     # Lint
cd frontend && npm run build         # Frontend build check
```

---

## HuggingFace Hub Integration

Pre-extracted vectors are hosted on the Hub. The server auto-downloads them if missing.

```python
# Download vectors programmatically
from emotion_scope.hub import download_vectors
path = download_vectors("google/gemma-2-2b-it")
```

```bash
# Push your results to the Hub
hf login
uv run python scripts/push_to_hub.py --repo your-username/emotion-scope-vectors

# Push vectors for a specific model
uv run python scripts/push_to_hub.py \
  --repo your-username/emotion-scope-vectors \
  --model google/gemma-2-9b-it
```

The default Hub repo is `AidanZach/EmotionScope-vectors` (configurable via `ES_HUB_REPO` env var).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_MODEL` | `google/gemma-2-2b-it` | HuggingFace model ID for the server |
| `ES_VECTORS` | `results/vectors/google_gemma-2-2b-it.pt` | Path to extracted vectors |
| `ES_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `ES_MAX_TOKENS` | `150` | Max generation tokens per response |
| `ES_HUB_REPO` | `AidanZach/EmotionScope-vectors` | HF Hub repo for auto-download |

---

## Documentation

All research docs are in `Documentation/`:
- `PaperDraft.md` — arXiv paper draft
- `ResearchPhase1.md` — detailed Phase 1 findings
- `MATHS.md` — mathematical cross-verification against Anthropic's paper
- `ANTPaperRefference.md` — Anthropic paper technical reference
- `VISUALIZATION_SPEC.md` — orb design specification
- `EmotionScope.md` — research vision document
- `STORY_GENERATION_PROMPT.md` — distributable prompt for generating emotion stories
- `CORPUS_GENERATION_PROMPT.md` — distributable prompt for generating dialogues
