"""
Emotion vector extraction pipeline.

Implements Anthropic's methodology from "Emotion Concepts and their Function
in a Large Language Model" (April 2026):

    1. Load emotion-tagged text corpus
    2. Extract residual stream activations at the probe layer (default 2L/3)
    3. Average across CONTENT token positions (excluding chat template markup)
    4. Compute contrastive mean difference per emotion (minus grand mean)
    5. Denoise via neutral PCA projection (top components explaining 50% variance)
    6. L2-normalize

Optional: find_best_probe_layer() sweeps depths and selects the layer with
the strongest valence separation, per the architect's direction.

Usage:
    from emotion_scope import EmotionExtractor, load_model

    model, tokenizer, backend, info = load_model("google/gemma-2-2b-it")
    extractor = EmotionExtractor(model, tokenizer, backend, info)
    vectors = extractor.extract()
    extractor.save()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

from emotion_scope.config import (
    CORE_EMOTIONS,
    DATA_DIR,
    VECTORS_DIR,
    ExtractionConfig,
)
from emotion_scope.utils import (
    find_content_token_range,
    valence_separation,
)


class EmotionExtractor:
    """
    Extracts emotion vectors from a language model's residual stream.
    Backend-agnostic: works with both TransformerLens and raw HuggingFace.
    """

    def __init__(
        self,
        model,
        tokenizer,
        backend: str,
        model_info: dict,
        config: Optional[ExtractionConfig] = None,
        emotions: Optional[List[dict]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.model_info = model_info
        self.config = config or ExtractionConfig()
        self.emotions = emotions or CORE_EMOTIONS

        self.probe_layer: int = model_info["probe_layer"]
        self.d_model: int = model_info["d_model"]
        self.n_layers: int = model_info["n_layers"]

        # Results
        self.emotion_vectors: Optional[Dict[str, torch.Tensor]] = None
        self.raw_vectors: Optional[Dict[str, torch.Tensor]] = None
        self.neutral_pca: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def extract(
        self,
        templates_path: Optional[Path] = None,
        neutral_path: Optional[Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full extraction pipeline and return normalized vectors."""
        templates = self._load_templates(templates_path)
        neutral_texts = self._load_neutral(neutral_path)

        # Step 1: per-emotion activations
        emotion_activations: Dict[str, torch.Tensor] = {}
        for emotion_info in tqdm(self.emotions, desc="Extracting emotions"):
            name = emotion_info["name"]
            texts = templates.get(name, [])
            if not texts:
                print(f"[extract] warning: no templates for '{name}', skipping")
                continue

            acts = []
            for text in texts[: self.config.stories_per_emotion]:
                a = self._get_residual_activation(text)
                if a is not None:
                    acts.append(a)
            if acts:
                emotion_activations[name] = torch.stack(acts)

        # Step 2: contrastive mean difference
        self.raw_vectors = self._compute_contrastive_vectors(emotion_activations)

        # Step 3: neutral PCA denoising
        self.neutral_pca = self._compute_neutral_pca(neutral_texts)
        denoised = self._denoise_vectors(self.raw_vectors)

        # Step 4: normalize
        if self.config.normalize_vectors:
            self.emotion_vectors = {
                n: F.normalize(v, dim=0) for n, v in denoised.items()
            }
        else:
            self.emotion_vectors = denoised

        return self.emotion_vectors

    def find_best_probe_layer(
        self,
        templates_path: Optional[Path] = None,
        stride: int = 3,
    ) -> int:
        """
        Sweep probe layers at the given stride and pick the one with the
        most negative valence-separation cosine (best positive/negative split).

        Updates self.probe_layer in place and returns the chosen layer.
        This is a one-time cost per model.
        """
        templates = self._load_templates(templates_path)
        candidate_layers = list(range(max(1, stride), self.n_layers, stride))
        if self.n_layers - 1 not in candidate_layers:
            candidate_layers.append(self.n_layers - 1)

        original_layer = self.probe_layer
        best_layer = original_layer
        best_score = float("inf")
        results: Dict[int, float] = {}

        for layer in tqdm(candidate_layers, desc="Layer sweep"):
            self.probe_layer = layer
            # Quick extraction with few templates per emotion for speed
            act_per_emotion: Dict[str, torch.Tensor] = {}
            for emotion_info in self.emotions:
                name = emotion_info["name"]
                texts = templates.get(name, [])[:5]  # 5 per emotion for speed
                acts = [self._get_residual_activation(t) for t in texts]
                acts = [a for a in acts if a is not None]
                if acts:
                    act_per_emotion[name] = torch.stack(acts)

            raw = self._compute_contrastive_vectors(act_per_emotion)
            normed = {n: F.normalize(v, dim=0) for n, v in raw.items()}
            score = valence_separation(normed, self.emotions)
            results[layer] = score
            if score < best_score:
                best_score = score
                best_layer = layer

        self.probe_layer = best_layer
        print(f"[extract] Layer sweep results: {results}")
        print(f"[extract] Best probe layer: {best_layer} (valence_sep={best_score:.3f})")
        return best_layer

    # ------------------------------------------------------------------
    # Activation extraction (backend-specific)
    # ------------------------------------------------------------------

    def _get_residual_activation(self, text: str) -> Optional[torch.Tensor]:
        try:
            if self.backend == "transformer_lens":
                return self._get_activation_tl(text)
            return self._get_activation_hf(text)
        except Exception as e:
            print(f"[extract] activation extraction failed: {type(e).__name__}: {e}")
            return None

    def _get_activation_tl(self, text: str) -> torch.Tensor:
        """TransformerLens: run_with_cache on only the probe layer."""
        hook_name = f"blocks.{self.probe_layer}.hook_resid_post"
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=hook_name,
        )
        residual = cache[hook_name]  # (1, seq_len, d_model)
        return self._average_content_tokens(residual, tokens)

    def _get_activation_hf(self, text: str) -> torch.Tensor:
        """HuggingFace: forward hook on the probe layer."""
        captured: dict = {}

        def hook_fn(_module, _input, output):
            captured["act"] = (output[0] if isinstance(output, tuple) else output).detach()

        layers = self._get_layers_module()
        handle = layers[self.probe_layer].register_forward_hook(hook_fn)
        try:
            tokens = self.tokenizer(text, return_tensors="pt")
            input_ids = tokens["input_ids"]
            tokens_on_device = {k: v.to(self.model.device) for k, v in tokens.items()}
            with torch.no_grad():
                self.model(**tokens_on_device)
        finally:
            handle.remove()

        residual = captured["act"]  # (1, seq_len, d_model)
        return self._average_content_tokens(residual, input_ids)

    def _average_content_tokens(
        self,
        residual: torch.Tensor,
        input_ids,
    ) -> torch.Tensor:
        """
        Average the residual stream across CONTENT tokens only (skipping
        chat template markup). This is used for EXTRACTION only.

        IMPORTANT: Probing uses a DIFFERENT position (response-preparation
        token = last token of full prompt). The distinction matters:
        extraction benefits from content-only averaging to learn clean
        emotion directions, while probing benefits from the response-prep
        position where the model has compressed its full emotional assessment.
        See probe.py and MATHS.md for details.
        """
        seq_len = residual.shape[1]

        if self.config.use_content_range:
            start, end = find_content_token_range(input_ids, self.tokenizer)
            # Safety: if the range is implausibly tiny, expand
            if end - start < 2:
                start = min(self.config.skip_tokens, max(seq_len - 1, 0))
                end = seq_len
        else:
            start = min(self.config.skip_tokens, max(seq_len - 1, 0))
            end = seq_len

        slab = residual[0, start:end, :]
        if slab.shape[0] == 0:
            slab = residual[0]
        return slab.mean(dim=0).detach().cpu().float()

    def _get_layers_module(self):
        for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = self.model
            ok = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                return obj
        raise ValueError(f"Cannot find transformer layers in {type(self.model).__name__}")

    # ------------------------------------------------------------------
    # Contrastive mean difference + PCA denoising
    # ------------------------------------------------------------------

    def _compute_contrastive_vectors(
        self, emotion_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        v_e = mean(acts_e) - grand_mean(all activations).

        Grand mean is computed from ALL raw activations weighted by count,
        not from the mean-of-means (which would bias the result if emotions
        have unequal sample counts due to dropped activations).
        """
        if not emotion_activations:
            return {}
        emotion_means = {n: a.mean(dim=0) for n, a in emotion_activations.items()}
        # Correct grand mean: weighted by sample count per emotion
        all_acts = torch.cat(list(emotion_activations.values()), dim=0)  # (total_samples, d_model)
        grand_mean = all_acts.mean(dim=0)
        return {n: m - grand_mean for n, m in emotion_means.items()}

    def _compute_neutral_pca(
        self, neutral_texts: List[str]
    ) -> Optional[torch.Tensor]:
        """Top components of neutral-text activations, explaining config.neutral_pca_variance."""
        if not neutral_texts:
            return None

        neutral_acts = []
        for text in tqdm(neutral_texts, desc="Neutral PCA"):
            a = self._get_residual_activation(text)
            if a is not None:
                neutral_acts.append(a)

        if len(neutral_acts) < 3:
            print("[extract] warning: insufficient neutral data for PCA denoising")
            return None

        matrix = torch.stack(neutral_acts).numpy()
        pca = PCA()
        pca.fit(matrix)

        cumvar = pca.explained_variance_ratio_.cumsum()
        # k = first component index where cumulative variance >= threshold
        # np.searchsorted finds the insertion point for the threshold value
        k = int(np.searchsorted(cumvar, self.config.neutral_pca_variance)) + 1
        k = min(k, len(neutral_acts) - 1)
        components = torch.tensor(pca.components_[:k].T, dtype=torch.float32)  # (d_model, k)
        return components

    def _denoise_vectors(
        self, vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project neutral subspace out of each emotion vector."""
        if self.neutral_pca is None:
            return vectors
        U = self.neutral_pca  # (d_model, k)
        return {n: v - U @ (U.T @ v) for n, v in vectors.items()}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_templates(self, path: Optional[Path] = None) -> Dict[str, List[str]]:
        path = path or DATA_DIR / "templates" / "emotion_stories.jsonl"
        templates: Dict[str, List[str]] = {}
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    templates.setdefault(entry["emotion"], []).append(entry["text"])
        else:
            print(f"[extract] No templates at {path}, using built-in defaults")
            templates = self._generate_default_templates()
        return templates

    def _load_neutral(self, path: Optional[Path] = None) -> List[str]:
        path = path or DATA_DIR / "neutral" / "neutral_prompts.jsonl"
        if Path(path).exists():
            texts: List[str] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(json.loads(line)["text"])
            return texts
        print(f"[extract] No neutral corpus at {path}, using built-in defaults")
        return self._generate_default_neutral()

    def _generate_default_templates(self) -> Dict[str, List[str]]:
        """Minimal built-in templates — enough to smoke-test the pipeline."""
        out: Dict[str, List[str]] = {}
        for e in self.emotions:
            n = e["name"]
            out[n] = [
                f"The character felt deeply {n} when they discovered that",
                f"A wave of being {n} washed over them as they realized",
                f"Nothing could describe how {n} they felt in that moment when",
            ]
        return out

    def _generate_default_neutral(self) -> List[str]:
        return [
            "The weather today is partly cloudy with temperatures around",
            "To install the software, first download the installer from the",
            "The meeting was scheduled for three o'clock in the afternoon at",
            "According to the manual, the device requires a standard power",
            "The population of the country was approximately twelve million",
            "Please refer to section four of the document for additional",
            "The train departs from platform seven at half past eight every",
            "The chemical formula for water is H2O consisting of two hydrogen",
            "Turn left at the intersection and continue for approximately",
            "The fiscal year ends on December thirty-first for most companies",
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> Path:
        if self.emotion_vectors is None:
            raise ValueError("No vectors extracted yet. Call extract() first.")
        if path is None:
            name = self.model_info["model_name"].replace("/", "_")
            path = VECTORS_DIR / f"{name}.pt"
        path = Path(path)
        torch.save(
            {
                "vectors": self.emotion_vectors,
                "raw_vectors": self.raw_vectors,
                "neutral_pca": self.neutral_pca,
                "model_info": self.model_info,
                "config": vars(self.config),
                "emotions": self.emotions,
                "probe_layer_used": self.probe_layer,
            },
            path,
        )
        print(f"[extract] Saved emotion vectors to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> dict:
        return torch.load(path, weights_only=False)
