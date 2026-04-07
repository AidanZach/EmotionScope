"""
Model loading utilities.

Supports two backends:
1. TransformerLens HookedTransformer (preferred — run_with_cache)
2. HuggingFace transformers with manual hooks (fallback, required for 4-bit)

The loader tries TransformerLens first for Gemma 2 (it has native support for
the local-global attention pattern and logit softcapping). If that path fails
for any reason, it falls back to HuggingFace hooks without retrying — per the
architect's direction, don't waste time debugging TL internals.

Usage:
    model, tokenizer, backend, info = load_model("google/gemma-2-2b-it")
"""

from __future__ import annotations

import warnings
from typing import Tuple, Literal

import torch

# TransformerLens 2.15 passes `torch_dtype` to transformers 4.57+ which renamed
# it to `dtype`. This is a known upstream mismatch — the warning is noise, not an
# error. Suppress it so users don't see a wall of red on every load.
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*Use.*dtype.*")

# The torch_dtype warning also comes through Python logging (logger.warning_once),
# so we also need to raise the transformers logger level for configuration_utils.
import logging as _logging
_logging.getLogger("transformers.configuration_utils").setLevel(_logging.ERROR)

# TransformerLens emits benign root-logger warnings for Gemma 2 (RMSNorm, softcap)
# via logging.warning() on the root logger. These are expected and harmless:
#   "center_unembed=True ... softcap ... Setting center_unembed=False"
#   "not using LayerNorm, so writing weights can't be centered! Skipping"
#   "With reduced precision, use from_pretrained_no_processing"
#   "float16 models may not work on CPU"
class _GemmaWarningFilter(_logging.Filter):
    _SUPPRESSED = [
        "center_unembed",
        "writing weights can't be centered",
        "reduced precision",
        "float16 models may not work on CPU",
        "final RMS normalization",
    ]
    def filter(self, record: _logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)

_logging.getLogger().addFilter(_GemmaWarningFilter())

BackendType = Literal["transformer_lens", "huggingface"]


def load_model(
    model_name: str = "google/gemma-2-2b-it",
    device: str = "auto",
    use_4bit: bool = False,
    dtype: str = "float16",
    backend: str = "auto",
    run_smoke_test: bool = True,
) -> Tuple:
    """
    Load a model for emotion probing.

    Args:
        model_name: HuggingFace model ID or TransformerLens name.
        device: "auto", "cuda", "cpu".
        use_4bit: Use bitsandbytes NF4 quantization. Forces HF backend.
        dtype: "float16", "bfloat16", "float32".
        backend: "auto", "transformer_lens", "huggingface".
        run_smoke_test: Forward "Hello world" through the model to verify
            the residual stream at the probe layer is reachable.

    Returns:
        (model, tokenizer, backend_type, model_info)
        model_info is a dict with keys: n_layers, d_model, probe_layer,
        model_name, backend, (quantized).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    actual_backend = _select_backend(backend, use_4bit)

    if actual_backend == "transformer_lens":
        try:
            result = _load_transformer_lens(model_name, device, dtype)
        except Exception as e:
            print(f"[models] TransformerLens load failed ({type(e).__name__}: {e})")
            print("[models] Falling back to HuggingFace backend.")
            result = _load_huggingface(model_name, device, use_4bit, dtype)
    else:
        result = _load_huggingface(model_name, device, use_4bit, dtype)

    model, tokenizer, backend_type, info = result

    if run_smoke_test:
        _smoke_test(model, tokenizer, backend_type, info)

    print(
        f"[models] Loaded: {info['model_name']} | "
        f"{info['n_layers']} layers | d_model={info['d_model']} | "
        f"probe_layer={info['probe_layer']} | backend={backend_type}"
    )

    return model, tokenizer, backend_type, info


def _select_backend(backend: str, use_4bit: bool) -> BackendType:
    """Choose the best backend for the given model."""
    if backend != "auto":
        return backend  # type: ignore[return-value]
    if use_4bit:
        return "huggingface"
    try:
        import transformer_lens  # noqa: F401
        return "transformer_lens"
    except ImportError:
        return "huggingface"


# ---------------------------------------------------------------------------
# TransformerLens backend
# ---------------------------------------------------------------------------

def _load_transformer_lens(model_name: str, device: str, dtype: str):
    from transformer_lens import HookedTransformer

    torch_dtype = getattr(torch, dtype)

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch_dtype,
        fold_ln=True,
        center_writing_weights=True,
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    from emotion_scope.config import ExtractionConfig
    probe_layer = round(n_layers * ExtractionConfig().probe_layer_fraction)

    info = {
        "n_layers": n_layers,
        "d_model": d_model,
        "probe_layer": probe_layer,
        "model_name": model_name,
        "backend": "transformer_lens",
    }
    return model, model.tokenizer, "transformer_lens", info


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

def _load_huggingface(model_name: str, device: str, use_4bit: bool, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"trust_remote_code": True}

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = getattr(torch, dtype)
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    d_model = getattr(config, "hidden_size", None)
    from emotion_scope.config import ExtractionConfig
    probe_layer = round(n_layers * ExtractionConfig().probe_layer_fraction) if n_layers else None

    info = {
        "n_layers": n_layers,
        "d_model": d_model,
        "probe_layer": probe_layer,
        "model_name": model_name,
        "backend": "huggingface",
        "quantized": use_4bit,
    }
    return model, tokenizer, "huggingface", info


# ---------------------------------------------------------------------------
# Smoke test — verify the probe layer is actually reachable
# ---------------------------------------------------------------------------

def _smoke_test(model, tokenizer, backend: str, info: dict) -> None:
    """Forward pass 'Hello world' and confirm we can capture the probe layer."""
    probe_layer = info["probe_layer"]
    d_model = info["d_model"]
    test_text = "Hello world"

    try:
        if backend == "transformer_lens":
            hook_name = f"blocks.{probe_layer}.hook_resid_post"
            _, cache = model.run_with_cache(test_text, names_filter=hook_name)
            act = cache[hook_name]
        else:
            captured: dict = {}

            def hook_fn(_module, _input, output):
                captured["act"] = output[0] if isinstance(output, tuple) else output

            layers = _find_layers_module(model)
            handle = layers[probe_layer].register_forward_hook(hook_fn)
            try:
                tokens = tokenizer(test_text, return_tensors="pt")
                tokens = {k: v.to(model.device) for k, v in tokens.items()}
                with torch.no_grad():
                    model(**tokens)
            finally:
                handle.remove()
            act = captured["act"]

        if act.ndim != 3 or act.shape[-1] != d_model:
            raise RuntimeError(
                f"Probe-layer activation has unexpected shape {tuple(act.shape)}; "
                f"expected (*, *, {d_model})"
            )
    except Exception as e:
        print(f"[models] WARNING: smoke test failed: {type(e).__name__}: {e}")


def _find_layers_module(model):
    """Locate the nn.ModuleList of transformer blocks inside an HF model."""
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
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")
