"""
EmotionScope FastAPI backend.

Loads Gemma 2 2B IT once at startup, keeps it in memory, and serves:
  POST /chat  — real model.generate() + emotion probe at layer 21

Start with:
    uv run uvicorn backend.server:app --port 8000
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emotion_scope.config import CORE_EMOTIONS
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.probe import EmotionProbe
from emotion_scope.visualize import scores_to_orb_state

# ── Configuration via env vars (with sane defaults) ──
MODEL_NAME = os.environ.get("ES_MODEL", "google/gemma-2-2b-it")
VECTORS_PATH = os.environ.get(
    "ES_VECTORS", "results/vectors/google_gemma-2-2b-it.pt")
SPEAKER_VECTORS_PATH = os.environ.get(
    "ES_SPEAKER_VECTORS", "results/vectors/speaker_separation.pt")
DEVICE = os.environ.get("ES_DEVICE", "auto")
MAX_NEW_TOKENS = int(os.environ.get("ES_MAX_TOKENS", "150"))

# ── Global state ──
_model = None
_tokenizer = None
_backend: str = ""
_info: dict = {}
_probe: Optional[EmotionProbe] = None           # Phase 1 single-speaker vectors (fallback)
_probe_current: Optional[EmotionProbe] = None    # current-speaker vectors (Phase 2)
_probe_other: Optional[EmotionProbe] = None      # other-speaker vectors (Phase 2)
_has_speaker_separation: bool = False


def _init_model():
    global _model, _tokenizer, _backend, _info
    global _probe, _probe_current, _probe_other, _has_speaker_separation

    vectors_path = Path(VECTORS_PATH)
    if not vectors_path.exists():
        # Try auto-downloading from HuggingFace Hub
        print(f"[server] Vectors not found at {vectors_path}, attempting Hub download...")
        try:
            from emotion_scope.hub import download_vectors
            vectors_path = download_vectors(MODEL_NAME)
            print(f"[server] Downloaded vectors to {vectors_path}")
        except Exception:
            raise FileNotFoundError(
                f"Vectors not found at {VECTORS_PATH} and Hub download failed. "
                f"Run: uv run python scripts/extract_all.py --model {MODEL_NAME} --sweep-layers"
            )

    saved = EmotionExtractor.load(str(vectors_path))
    vectors = saved["vectors"]
    emotions = saved.get("emotions", CORE_EMOTIONS)
    saved_info = saved["model_info"]

    _model, _tokenizer, _backend, _info = load_model(
        model_name=MODEL_NAME, device=DEVICE, run_smoke_test=False)
    _info["probe_layer"] = saved.get("probe_layer_used", saved_info["probe_layer"])

    # Validate that vectors match the loaded model
    sample_vec = next(iter(vectors.values()))
    if sample_vec.shape[0] != _info["d_model"]:
        raise ValueError(
            f"Vector dimension mismatch: vectors have d={sample_vec.shape[0]} "
            f"but model {MODEL_NAME} has d_model={_info['d_model']}. "
            f"Vectors were extracted from {saved_info.get('model_name', 'unknown')}. "
            f"Re-run extraction for this model."
        )
    if _info["probe_layer"] >= _info["n_layers"]:
        raise ValueError(
            f"Probe layer {_info['probe_layer']} >= model layer count "
            f"{_info['n_layers']}. Vectors were extracted from a deeper model."
        )

    # Phase 1 probe (single-speaker vectors — fallback for all probing)
    _probe = EmotionProbe(
        model=_model, tokenizer=_tokenizer, backend=_backend,
        model_info=_info, emotion_vectors=vectors, emotions_metadata=emotions)

    # Phase 2: Try to load speaker-separated vectors if available.
    # These are experimental — geometric separation is confirmed but
    # behavioral accuracy is mixed at 2B scale. The frontend offers a
    # toggle so researchers can compare single vs dual-speaker mode.
    speaker_path = Path(SPEAKER_VECTORS_PATH)
    if speaker_path.exists():
        try:
            sp_saved = torch.load(str(speaker_path), weights_only=False)
            current_vecs = sp_saved.get("current_speaker_vectors", {})
            other_vecs = sp_saved.get("other_speaker_vectors", {})
            sp_emotions = sp_saved.get("emotions", emotions)

            if current_vecs and other_vecs:
                _probe_current = EmotionProbe(
                    model=_model, tokenizer=_tokenizer, backend=_backend,
                    model_info=_info, emotion_vectors=current_vecs,
                    emotions_metadata=sp_emotions)
                _probe_other = EmotionProbe(
                    model=_model, tokenizer=_tokenizer, backend=_backend,
                    model_info=_info, emotion_vectors=other_vecs,
                    emotions_metadata=sp_emotions)
                _has_speaker_separation = True
                print(f"[server] Speaker vectors loaded ({len(current_vecs)} emotions)")
        except Exception as e:
            print(f"[server] Speaker vectors failed to load: {e}")
            _has_speaker_separation = False
    else:
        _has_speaker_separation = False

    print(f"[server] Ready: {MODEL_NAME} @ layer {_info['probe_layer']}, "
          f"backend={_backend}, mode=single-speaker (Phase 1 vectors, "
          f"{len(vectors)} emotions)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    yield


app = FastAPI(title="EmotionScope", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ──

class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    speaker_mode: str = "single"  # "single" or "dual"

class EmotionStateResponse(BaseModel):
    valence: float
    arousal: float
    intensity: float
    complexity: float
    dominant: str
    secondary: str
    color_hex: str
    color_oklch: dict
    secondary_hex: str
    secondary_oklch: dict
    top_emotions: list

class TokenAttribution(BaseModel):
    tokens: list[str]       # decoded token strings
    scores: list[float]     # per-token cosine sim with dominant emotion vector

class ChatResponse(BaseModel):
    response: str
    user_emotion: dict
    model_emotion: dict
    token_attribution: TokenAttribution | None = None  # per-token emotion heatmap


# ── Endpoints ──

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "probe_layer": _info.get("probe_layer"),
        "n_layers": _info.get("n_layers"),
        "d_model": _info.get("d_model"),
        "backend": _backend,
        "n_emotions": len(_probe.emotion_names) if _probe else 0,
        "emotions": _probe.emotion_names if _probe else [],
        "speaker_separation": _has_speaker_separation,
        "mode": "dual-speaker" if _has_speaker_separation else "single-speaker",
    }


# ── Test/debug endpoints ──

class ProbeOnlyRequest(BaseModel):
    """Probe a message without generation — fast path for visualization testing."""
    text: str

@app.post("/probe")
async def probe_only(req: ProbeOnlyRequest):
    """
    Probe-only endpoint: runs a single forward pass on the text and returns
    emotion states from both speaker perspectives (if available).
    No generation, no response text. ~1-2 seconds.
    """
    if _probe is None:
        return {"error": "Model not loaded"}

    state = _probe.analyze(req.text)
    activation = state  # for single-speaker fallback

    result = {"single": scores_to_orb_state(state.scores)}

    if _has_speaker_separation and _probe_current and _probe_other:
        # Get the raw activation to project onto both vector sets
        dual = _probe.analyze_conversation(req.text)
        if dual.raw_activation is not None:
            current = _probe_current.analyze_activation(dual.raw_activation)
            other = _probe_other.analyze_activation(dual.raw_activation)
            result["current_speaker"] = scores_to_orb_state(current.scores)
            result["other_speaker"] = scores_to_orb_state(other.scores)

    return result


# Pre-scripted test scenarios for visualization development.
# Each scenario has known emotional content so you can verify the orb
# responds correctly without waiting for generation.
TEST_SCENARIOS = [
    {"label": "Tylenol safe",     "text": "User: I just took 500 mg of Tylenol.\n\nAssistant:"},
    {"label": "Tylenol danger",   "text": "User: I just took 16000 mg of Tylenol.\n\nAssistant:"},
    {"label": "Grief",            "text": "User: They buried my grandmother this morning. I keep picking up the phone to call her.\n\nAssistant:"},
    {"label": "Excitement",       "text": "User: I got into the program! Full scholarship! I can't believe it!\n\nAssistant:"},
    {"label": "Anger",            "text": "User: My boss took credit for my entire project in front of the whole company.\n\nAssistant:"},
    {"label": "Fear",             "text": "User: The biopsy results come in tomorrow and I haven't been able to sleep.\n\nAssistant:"},
    {"label": "Calm",             "text": "User: I'm sitting on the porch with my coffee watching the sunrise. Nowhere to be.\n\nAssistant:"},
    {"label": "Curiosity",        "text": "User: Have you ever wondered how octopuses evolved such complex intelligence?\n\nAssistant:"},
    {"label": "Guilt",            "text": "User: I yelled at my daughter this morning over something stupid and I can't stop thinking about it.\n\nAssistant:"},
    {"label": "Mixed distress",   "text": "User: Everything is falling apart. I lost my job, my relationship is ending, and I can't pay rent.\n\nAssistant:"},
    {"label": "Loving",           "text": "User: The baby grabbed my finger today and just held on. Wouldn't let go for ten minutes.\n\nAssistant:"},
    {"label": "Neutral",          "text": "User: What's the weather forecast for tomorrow?\n\nAssistant:"},
]

@app.get("/test-scenarios")
async def get_test_scenarios():
    """Return the list of pre-scripted test scenarios."""
    return {"scenarios": [s["label"] for s in TEST_SCENARIOS]}

@app.post("/test-scenario/{index}")
async def run_test_scenario(index: int):
    """
    Run a pre-scripted scenario through the probe (no generation).
    Returns the emotion state in ~1-2 seconds.
    Use this to rapidly test visualization responses.
    """
    if index < 0 or index >= len(TEST_SCENARIOS):
        return {"error": f"Index {index} out of range (0-{len(TEST_SCENARIOS)-1})"}
    if _probe is None:
        return {"error": "Model not loaded"}

    scenario = TEST_SCENARIOS[index]
    state = _probe.analyze(scenario["text"])
    orb_state = scores_to_orb_state(state.scores)
    return {
        "label": scenario["label"],
        "text": scenario["text"],
        "emotion": orb_state,
    }

@app.get("/test-all")
async def run_all_test_scenarios():
    """
    Run ALL test scenarios and return results. Takes ~15-20 seconds.
    Useful for generating a visualization test matrix.
    """
    if _probe is None:
        return {"error": "Model not loaded"}

    results = []
    for i, scenario in enumerate(TEST_SCENARIOS):
        state = _probe.analyze(scenario["text"])
        orb_state = scores_to_orb_state(state.scores)
        results.append({
            "index": i,
            "label": scenario["label"],
            "emotion": orb_state,
        })
    return {"results": results}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Real generation + emotion probing.

    Step A: Probe the user's message (model's representation while processing it)
    Step B: Generate real response, hook first token for model's emotional state
    Step C: Return clean text + both emotion states
    """
    # Build conversation for the model
    conversation = [{"role": m.role, "content": m.content} for m in req.history]
    conversation.append({"role": "user", "content": req.message})

    use_dual = (req.speaker_mode == "dual" and _has_speaker_separation
                and _probe_current and _probe_other)

    # Step A: Probe user message + token attribution
    user_emotion, attribution = _probe_user(conversation)

    # Step B: Generate + probe model state
    response_text, model_emotion = _generate_with_probe(conversation)

    # Step C (optional): Re-score with speaker-separated vectors
    if use_dual:
        # Re-probe user message with other-speaker vectors (model's read of user)
        dual_user = _probe_other.analyze_conversation(
            user_message=conversation[-1]["content"])
        user_emotion = scores_to_orb_state(dual_user.model_state.scores)

        # Re-probe model activation with current-speaker vectors (model's own state)
        # The model_emotion already has the captured activation from generate_with_probe
        # We need to get it from the captured first token — stored during generation
        # For now, re-probe the prompt with current-speaker vectors
        dual_model = _probe_current.analyze_conversation(
            user_message=conversation[-1]["content"])
        model_emotion = scores_to_orb_state(dual_model.model_state.scores)

    return ChatResponse(
        response=response_text,
        user_emotion=user_emotion,
        model_emotion=model_emotion,
        token_attribution=attribution,
    )


def _probe_user(conversation: list[dict]) -> tuple[dict, dict | None]:
    """
    Forward pass on user message, read emotion at last content token.
    Also computes per-token attribution for the dominant emotion.

    Returns: (orb_state_dict, token_attribution_dict_or_None)
    """
    dual = _probe.analyze_conversation(
        user_message=conversation[-1]["content"])
    orb_state = scores_to_orb_state(dual.model_state.scores)

    # Per-token attribution: which tokens activate the dominant emotion?
    attribution = None
    try:
        attribution = _compute_token_attribution(
            conversation[-1]["content"], orb_state.get("dominant"))
    except Exception as e:
        print(f"[server] Token attribution failed: {e}")

    return orb_state, attribution


def _generate_with_probe(conversation: list[dict]) -> tuple[str, dict]:
    """Generate real response, capture residual stream at first generated token."""

    probe_layer = _info["probe_layer"]
    captured: dict = {}

    # Build chat prompt
    apply = getattr(_tokenizer, "apply_chat_template", None)
    if apply is not None:
        prompt = apply(conversation, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in conversation
        ) + "\nAssistant:"

    if _backend == "transformer_lens":
        input_ids = _model.to_tokens(prompt)
        input_len = input_ids.shape[1]
        hook_name = f"blocks.{probe_layer}.hook_resid_post"

        def capture_hook(activation, hook):
            if "first_token" not in captured:
                captured["first_token"] = activation[0, -1, :].detach().cpu().float()
            return activation

        _model.add_hook(hook_name, capture_hook)
        try:
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=True,
            )
        finally:
            _model.reset_hooks()

        new_token_ids = output_ids[0, input_len:]
        response_text = _model.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    else:
        # HuggingFace backend
        tokens = _tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(_model.device)
        input_len = input_ids.shape[1]
        attention_mask = tokens.get(
            "attention_mask", torch.ones_like(input_ids)).to(_model.device)

        layers = _find_layers(_model)

        def hf_hook(_module, _input, output):
            if "first_token" not in captured:
                act = output[0] if isinstance(output, tuple) else output
                captured["first_token"] = act[0, -1, :].detach().cpu().float()

        handle = layers[probe_layer].register_forward_hook(hf_hook)
        try:
            output_ids = _model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7, top_p=0.9, do_sample=True)
        finally:
            handle.remove()

        new_token_ids = output_ids[0, input_len:]
        response_text = _tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Score the captured activation using Phase 1 vectors (20 emotions).
    model_state = {}
    if "first_token" in captured and _probe is not None:
        emotion = _probe.analyze_activation(captured["first_token"])
        model_state = scores_to_orb_state(emotion.scores)

    return response_text.strip(), model_state


def _compute_token_attribution(text: str, dominant_emotion: str | None) -> dict | None:
    """
    Compute per-token cosine similarity with the dominant emotion vector.

    Instead of averaging the residual stream across tokens then projecting,
    we project EACH token's activation onto the emotion direction. This shows
    which tokens in the input contribute most to the emotion reading.
    """
    if _probe is None or not dominant_emotion or dominant_emotion not in _probe.emotion_names:
        return None

    # Get the emotion vector for the dominant emotion
    idx = _probe.emotion_names.index(dominant_emotion)
    emotion_vec = _probe.vector_matrix[idx]  # (d_model,)
    emotion_vec_norm = F.normalize(emotion_vec.unsqueeze(0), dim=1)  # (1, d_model)

    probe_layer = _info["probe_layer"]

    if _backend == "transformer_lens":
        # Get full residual stream at all token positions
        messages = [{"role": "user", "content": text}]
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = _model.to_tokens(prompt)
        hook_name = f"blocks.{probe_layer}.hook_resid_post"

        _, cache = _model.run_with_cache(input_ids, names_filter=hook_name)
        residual = cache[hook_name]  # (1, seq_len, d_model)

        # Decode each token
        token_ids = input_ids[0].tolist()
        token_strings = [_tokenizer.decode([tid]) for tid in token_ids]

        # Cosine similarity per token position
        residual_norm = F.normalize(residual[0].float().cpu(), dim=1)  # (seq_len, d_model)
        scores = (residual_norm @ emotion_vec_norm.T).squeeze(1).tolist()  # (seq_len,)

        # Filter to content tokens only (skip template markup)
        from emotion_scope.utils import find_content_token_range
        start, end = find_content_token_range(input_ids, _tokenizer)

        content_tokens = token_strings[start:end]
        content_scores = scores[start:end]

        return {
            "tokens": content_tokens,
            "scores": content_scores,
        }
    else:
        # HuggingFace backend — similar but with hooks
        messages = [{"role": "user", "content": text}]
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tokens = _tokenizer(prompt, return_tensors="pt")
        input_ids_tensor = tokens["input_ids"].to(_model.device)

        captured = {}

        def hook_fn(module, inp, output):
            act = output[0] if isinstance(output, tuple) else output
            captured["residual"] = act.detach()

        layers = _find_layers(_model)
        handle = layers[probe_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            _model(**{k: v.to(_model.device) for k, v in tokens.items()})
        handle.remove()

        residual = captured["residual"]  # (1, seq_len, d_model)
        token_ids = input_ids_tensor[0].tolist()
        token_strings = [_tokenizer.decode([tid]) for tid in token_ids]

        residual_norm = F.normalize(residual[0].float().cpu(), dim=1)
        scores = (residual_norm @ emotion_vec_norm.T).squeeze(1).tolist()

        from emotion_scope.utils import find_content_token_range
        tl_ids = torch.tensor([token_ids]).to(input_ids_tensor.device)
        start, end = find_content_token_range(tl_ids, _tokenizer)

        return {
            "tokens": token_strings[start:end],
            "scores": scores[start:end],
        }


def _find_layers(model):
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
            return obj
    raise ValueError(f"Cannot find layers in {type(model).__name__}")
