"""
Dual-speaker emotion separation.

Extracts separate "current speaker" and "other speaker" emotion vectors from
a language model, replicating Anthropic's finding that models maintain distinct
emotional representations for each participant in a conversation.

Algorithm:
    1. Load two-speaker dialogues where Speaker A expresses emotion_a and
       Speaker B expresses emotion_b (with emotion_a != emotion_b).
    2. For each dialogue, run a forward pass and extract the residual stream
       activation at the last content token of Speaker A's final turn.
    3. Compute contrastive vectors:
       current_speaker_vector[e] = mean(acts | emotion_a == e, over all emotion_b) - grand_mean
       other_speaker_vector[e]   = mean(acts | emotion_b == e, over all emotion_a) - grand_mean
    4. Denoise via neutral PCA projection.
    5. L2-normalize.

Validation:
    A. Orthogonality: cosine(current[e], other[e]) per emotion, mean < 0.3
    B. Internal consistency: valence_separation on each set independently < -0.2
    C. Thermostat: emotionally charged messages produce high arousal on other-speaker
       and low arousal on current-speaker (the model regulates its own state)

Usage:
    from emotion_scope.speakers import SpeakerSeparator
    from emotion_scope import load_model

    model, tokenizer, backend, info = load_model("google/gemma-2-2b-it")
    separator = SpeakerSeparator(model, tokenizer, backend, info)
    result = separator.extract("data/templates/two_speaker_dialogues.jsonl")
    report = separator.validate(result["current_speaker"], result["other_speaker"])
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Validation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeakerValidationResult:
    """Results from the three speaker-separation validation tests."""
    # Test A: Speaker separation (low alignment between same-emotion vectors)
    orthogonality_scores: Dict[str, float]  # per-emotion cosine(current[e], other[e])
    orthogonality_mean: float               # mean same-emotion cosine
    orthogonality_passed: bool

    # Test B: Internal consistency (valence separation)
    current_valence_sep: float
    other_valence_sep: float
    consistency_passed: bool

    # Test C: Thermostat
    thermostat_results: List[dict]  # per-message results
    thermostat_mean_delta: float
    thermostat_passed: bool

    # Baselines (Test A context)
    cross_emotion_mean: float = 0.0         # mean cross-emotion cosine (baseline)
    random_baseline_std: float = 0.0        # 1/sqrt(d_model) — expected std for random

    @property
    def all_passed(self) -> bool:
        return self.orthogonality_passed and self.consistency_passed and self.thermostat_passed


# ---------------------------------------------------------------------------
# SpeakerSeparator
# ---------------------------------------------------------------------------

class SpeakerSeparator:
    """
    Extracts current-speaker and other-speaker emotion vectors from a
    language model's residual stream using two-speaker dialogues.
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
        self.emotion_names = [e["name"] for e in self.emotions]

        self.probe_layer: int = model_info["probe_layer"]
        self.d_model: int = model_info["d_model"]

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    def extract(
        self,
        dialogues_path: Optional[str] = None,
        neutral_path: Optional[str] = None,
    ) -> dict:
        """
        Run the full speaker-separation extraction pipeline.

        Args:
            dialogues_path: Path to two_speaker_dialogues.jsonl. If None,
                uses built-in defaults.
            neutral_path: Path to neutral_prompts.jsonl for PCA denoising.

        Returns:
            Dict with keys:
                'current_speaker': Dict[str, Tensor] — emotion -> vector
                'other_speaker':   Dict[str, Tensor] — emotion -> vector
                'metadata': dict with extraction details
        """
        # Step 1: Load dialogues
        dialogues = self._load_dialogues(dialogues_path)
        neutral_texts = self._load_neutral(neutral_path)

        print(f"[speakers] Loaded {len(dialogues)} dialogues")

        # Step 2: Extract activations with (emotion_a, emotion_b) labels
        labeled_activations = self._extract_dialogue_activations(dialogues)

        print(f"[speakers] Extracted {len(labeled_activations)} activations")

        # Step 3: Compute contrastive vectors for each speaker role
        current_raw, other_raw = self._compute_speaker_vectors(labeled_activations)

        print(f"[speakers] Current-speaker emotions: {len(current_raw)}")
        print(f"[speakers] Other-speaker emotions: {len(other_raw)}")

        # Step 4: Neutral PCA denoising
        neutral_pca = self._compute_neutral_pca(neutral_texts)
        current_denoised = self._denoise_vectors(current_raw, neutral_pca)
        other_denoised = self._denoise_vectors(other_raw, neutral_pca)

        # Step 5: L2-normalize
        current_vectors = {
            n: F.normalize(v, dim=0) for n, v in current_denoised.items()
        }
        other_vectors = {
            n: F.normalize(v, dim=0) for n, v in other_denoised.items()
        }

        return {
            "current_speaker": current_vectors,
            "other_speaker": other_vectors,
            "metadata": {
                "model_info": self.model_info,
                "probe_layer_used": self.probe_layer,
                "config": vars(self.config),
                "emotions": self.emotions,
                "n_dialogues": len(dialogues),
                "n_activations": len(labeled_activations),
                "neutral_pca_components": neutral_pca.shape[1] if neutral_pca is not None else 0,
            },
        }

    # ------------------------------------------------------------------
    # Dialogue activation extraction
    # ------------------------------------------------------------------

    def _extract_dialogue_activations(
        self,
        dialogues: List[dict],
    ) -> List[dict]:
        """
        For each dialogue, extract the residual stream activation at the last
        content token of Speaker A's final turn.

        Returns list of dicts with keys: emotion_a, emotion_b, activation.
        """
        results = []
        for entry in tqdm(dialogues, desc="Extracting dialogues"):
            emotion_a = entry["emotion_a"]
            emotion_b = entry["emotion_b"]
            dialogue_text = entry["dialogue"]

            # Find the text up to and including Speaker A's final turn
            probe_text = self._get_speaker_a_final_turn_text(dialogue_text)
            if probe_text is None:
                continue

            activation = self._get_activation_at_last_content_token(probe_text)
            if activation is not None:
                results.append({
                    "emotion_a": emotion_a,
                    "emotion_b": emotion_b,
                    "activation": activation,
                })

        return results

    def _get_speaker_a_final_turn_text(self, dialogue_text: str) -> Optional[str]:
        """
        Parse dialogue to find Speaker A's final turn. Returns the full text
        up to the end of that turn.

        The dialogue format is:
            Speaker A: ...
            Speaker B: ...
            Speaker A: ...
            Speaker B: ...

        We want the text up to the end of Speaker A's last turn (before the
        final Speaker B turn, if any).
        """
        # Split on "Speaker A:" and "Speaker B:" markers
        # Find the last occurrence of "Speaker A:" in the text
        parts = dialogue_text.split("Speaker A:")
        if len(parts) < 2:
            return None

        # Reconstruct text up to and including the last Speaker A turn
        # The last "Speaker A:" segment may be followed by "Speaker B:"
        last_a_segment = parts[-1]

        # Check if there's a "Speaker B:" after the last "Speaker A:" segment
        b_pos = last_a_segment.find("Speaker B:")
        if b_pos != -1:
            # Include everything up to but not including the next Speaker B turn
            last_a_text = last_a_segment[:b_pos].rstrip()
        else:
            last_a_text = last_a_segment.rstrip()

        # Reconstruct the full text up to the end of last Speaker A turn
        # Join all parts back except the tail after last Speaker B
        prefix = "Speaker A:".join(parts[:-1])
        full_text = prefix + "Speaker A:" + last_a_text

        if not full_text.strip():
            return None

        return full_text.strip()

    def _get_activation_at_last_content_token(
        self,
        text: str,
    ) -> Optional[torch.Tensor]:
        """
        Run text through the model and return the residual stream activation
        at the last content token of the probe layer.
        """
        try:
            if self.backend == "transformer_lens":
                return self._get_activation_tl(text)
            return self._get_activation_hf(text)
        except Exception as e:
            print(f"[speakers] activation extraction failed: {type(e).__name__}: {e}")
            return None

    def _get_activation_tl(self, text: str) -> torch.Tensor:
        """TransformerLens backend: run_with_cache, read last content token."""
        hook_name = f"blocks.{self.probe_layer}.hook_resid_post"
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens, names_filter=hook_name)
        residual = cache[hook_name]  # (1, seq_len, d_model)
        idx = self._last_content_idx(tokens, residual.shape[1])
        return residual[0, idx, :].detach().cpu().float()

    def _get_activation_hf(self, text: str) -> torch.Tensor:
        """HuggingFace backend: forward hook, read last content token."""
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
        idx = self._last_content_idx(input_ids, residual.shape[1])
        return residual[0, idx, :].cpu().float()

    def _last_content_idx(self, input_ids, seq_len: int) -> int:
        """Return the index of the last content token."""
        _, end = find_content_token_range(input_ids, self.tokenizer)
        return max(end - 1, 0)

    def _get_layers_module(self):
        """Find transformer layers in HF model."""
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
    # Contrastive vector computation
    # ------------------------------------------------------------------

    def _compute_speaker_vectors(
        self,
        labeled_activations: List[dict],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute contrastive current-speaker and other-speaker vectors.

        current_speaker_vector[e] = mean(acts where emotion_a == e, over all emotion_b) - grand_mean
        other_speaker_vector[e]   = mean(acts where emotion_b == e, over all emotion_a) - grand_mean
        """
        # Group activations by emotion_a and emotion_b
        by_emotion_a: Dict[str, List[torch.Tensor]] = defaultdict(list)
        by_emotion_b: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for entry in labeled_activations:
            by_emotion_a[entry["emotion_a"]].append(entry["activation"])
            by_emotion_b[entry["emotion_b"]].append(entry["activation"])

        # Compute means per emotion for each role
        current_means: Dict[str, torch.Tensor] = {}
        for name, acts in by_emotion_a.items():
            if acts:
                current_means[name] = torch.stack(acts).mean(dim=0)

        other_means: Dict[str, torch.Tensor] = {}
        for name, acts in by_emotion_b.items():
            if acts:
                other_means[name] = torch.stack(acts).mean(dim=0)

        # Separate grand means per speaker role.
        # Current-speaker and other-speaker activations are drawn from different
        # distributions (different emotion-pair contexts), so a shared grand mean
        # would bias the contrastive difference.
        current_all = [a for acts in by_emotion_a.values() for a in acts]
        other_all = [a for acts in by_emotion_b.values() for a in acts]
        current_grand = torch.stack(current_all).mean(dim=0) if current_all else torch.zeros_like(next(iter(current_means.values())))
        other_grand = torch.stack(other_all).mean(dim=0) if other_all else torch.zeros_like(next(iter(other_means.values())))

        # Contrastive difference — each role uses its own centroid
        current_vectors = {n: m - current_grand for n, m in current_means.items()}
        other_vectors = {n: m - other_grand for n, m in other_means.items()}

        return current_vectors, other_vectors

    # ------------------------------------------------------------------
    # Neutral PCA denoising (same method as extract.py)
    # ------------------------------------------------------------------

    def _compute_neutral_pca(
        self,
        neutral_texts: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Compute PCA on neutral text activations. Returns top-k components
        explaining config.neutral_pca_variance of the variance.
        """
        if not neutral_texts:
            return None

        neutral_acts = []
        for text in tqdm(neutral_texts, desc="Neutral PCA"):
            act = self._get_activation_at_last_content_token(text)
            if act is not None:
                neutral_acts.append(act)

        if len(neutral_acts) < 3:
            print("[speakers] warning: insufficient neutral data for PCA denoising")
            return None

        matrix = torch.stack(neutral_acts).numpy()
        pca = PCA()
        pca.fit(matrix)

        cumvar = pca.explained_variance_ratio_.cumsum()
        k = int((cumvar < self.config.neutral_pca_variance).sum()) + 1
        k = min(k, len(neutral_acts) - 1)
        components = torch.tensor(pca.components_[:k].T, dtype=torch.float32)  # (d_model, k)
        return components

    def _denoise_vectors(
        self,
        vectors: Dict[str, torch.Tensor],
        neutral_pca: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project neutral subspace out of each vector."""
        if neutral_pca is None:
            return vectors
        U = neutral_pca  # (d_model, k)
        return {n: v - U @ (U.T @ v) for n, v in vectors.items()}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_dialogues(self, path: Optional[str] = None) -> List[dict]:
        """
        Load two-speaker dialogues from JSONL.

        Expected format per line:
            {"emotion_a": "angry", "emotion_b": "afraid",
             "dialogue": "Speaker A: ...\\nSpeaker B: ...\\nSpeaker A: ..."}
        """
        path = Path(path) if path else DATA_DIR / "templates" / "two_speaker_dialogues.jsonl"
        dialogues = []

        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if "emotion_a" in entry and "emotion_b" in entry and "dialogue" in entry:
                        dialogues.append(entry)
        else:
            print(f"[speakers] No dialogues at {path}, using built-in defaults")
            dialogues = self._generate_default_dialogues()

        return dialogues

    def _load_neutral(self, path: Optional[str] = None) -> List[str]:
        """Load neutral text corpus for PCA denoising."""
        path = Path(path) if path else DATA_DIR / "neutral" / "neutral_prompts.jsonl"
        if path.exists():
            texts = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(json.loads(line)["text"])
            return texts
        print(f"[speakers] No neutral corpus at {path}, using built-in defaults")
        return self._generate_default_neutral()

    def _generate_default_neutral(self) -> List[str]:
        """Built-in neutral text for PCA denoising."""
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

    def _generate_default_dialogues(self) -> List[dict]:
        """
        Built-in fallback dialogues for bootstrapping. Generates a small set
        covering key emotion pairings. A full corpus should be placed at
        data/templates/two_speaker_dialogues.jsonl.
        """
        # Emotion pairs to cover: each core emotion as Speaker A paired with
        # a contrasting emotion for Speaker B.
        pairings = [
            ("angry", "afraid"),
            ("angry", "calm"),
            ("angry", "sad"),
            ("afraid", "calm"),
            ("afraid", "confident"),
            ("happy", "sad"),
            ("happy", "angry"),
            ("happy", "nervous"),
            ("sad", "hopeful"),
            ("sad", "happy"),
            ("calm", "angry"),
            ("calm", "desperate"),
            ("desperate", "calm"),
            ("desperate", "hopeful"),
            ("hopeful", "gloomy"),
            ("frustrated", "calm"),
            ("frustrated", "happy"),
            ("curious", "nervous"),
            ("proud", "guilty"),
            ("guilty", "loving"),
            ("surprised", "calm"),
            ("loving", "hostile"),
            ("hostile", "afraid"),
            ("nervous", "confident"),
            ("confident", "nervous"),
            ("brooding", "enthusiastic"),
            ("enthusiastic", "reflective"),
            ("reflective", "enthusiastic"),
            ("gloomy", "hopeful"),
        ]

        templates = {
            "angry": [
                "I can't believe you did that. How could you be so careless? This is completely unacceptable and I'm furious.",
                "You had one job and you blew it. I've had it with your excuses.",
                "Don't you dare try to justify what happened. I'm done listening.",
            ],
            "afraid": [
                "I don't know what's going to happen and that terrifies me. What if everything falls apart?",
                "Something doesn't feel right. I'm scared we're making a huge mistake.",
                "Please don't leave me alone here. I can't handle this by myself.",
            ],
            "calm": [
                "Let's take a step back and look at this rationally. There's no need to rush.",
                "I understand how you feel, and I think we can work through this together.",
                "Everything will be fine. We just need to be patient and think clearly.",
            ],
            "happy": [
                "This is amazing! I'm so thrilled about how everything turned out!",
                "I can't stop smiling. Today has been absolutely wonderful.",
                "Life is beautiful, isn't it? I feel so grateful for everything.",
            ],
            "sad": [
                "I just feel empty inside. Nothing seems to matter anymore.",
                "I've been trying so hard, but it all feels pointless now.",
                "I miss how things used to be. I don't think they'll ever be that way again.",
            ],
            "desperate": [
                "Please, you have to help me. There's no one else I can turn to.",
                "I've tried everything and nothing works. I don't know what to do anymore.",
                "Time is running out. If we don't act now, everything is lost.",
            ],
            "hopeful": [
                "I really think things are going to get better. I can feel it.",
                "We've been through worse than this. I believe we'll find a way.",
                "There's a light at the end of this tunnel, I'm sure of it.",
            ],
            "frustrated": [
                "Why won't this work? I've been trying the same thing over and over.",
                "I'm so tired of hitting walls everywhere I turn.",
                "Nothing ever goes according to plan. It's exhausting.",
            ],
            "curious": [
                "That's really interesting. I wonder why it works that way?",
                "Tell me more about that. I want to understand every detail.",
                "What if we tried a completely different approach? I'm intrigued.",
            ],
            "proud": [
                "I worked incredibly hard for this and it paid off. I'm so proud of what we achieved.",
                "Look at what we built together. This is something to be genuinely proud of.",
                "I never thought I could do it, but I did. I proved everyone wrong.",
            ],
            "guilty": [
                "I know I hurt you and I'm so sorry. I should never have said those things.",
                "It's my fault this happened. I keep replaying it in my mind.",
                "I don't deserve your forgiveness, but I want you to know how sorry I am.",
            ],
            "surprised": [
                "Wait, what? I had no idea that was even possible!",
                "I never expected this. I'm completely caught off guard.",
                "You're kidding! That changes everything I thought I knew.",
            ],
            "loving": [
                "You mean the world to me. I can't imagine life without you.",
                "I just want you to know how deeply I care about you.",
                "Every moment with you is a gift. You make everything better.",
            ],
            "hostile": [
                "Get out of my sight. I don't want anything to do with you.",
                "You're nothing but a liar. Stay away from me.",
                "I will never forgive you for what you've done. Never.",
            ],
            "nervous": [
                "I have a bad feeling about this. My hands won't stop shaking.",
                "What if they find out? What if everything goes wrong?",
                "I keep second-guessing myself. I can't make a decision.",
            ],
            "confident": [
                "I've got this under control. Trust me, I know exactly what I'm doing.",
                "We're going to nail this. There's not a doubt in my mind.",
                "Bring it on. I'm ready for whatever comes next.",
            ],
            "brooding": [
                "I've been thinking about what happened for days. Something still doesn't add up.",
                "There's a darkness I can't shake. It follows me everywhere.",
                "I keep going over it in my mind, turning it over and over.",
            ],
            "enthusiastic": [
                "This is going to be incredible! I can't wait to get started!",
                "I'm so excited about this opportunity. Let's make it happen!",
                "Everything about this project fires me up. Let's go!",
            ],
            "reflective": [
                "Looking back, I realize things could have gone differently.",
                "I've been thinking about what really matters in the long run.",
                "Sometimes the quiet moments teach you the most about yourself.",
            ],
            "gloomy": [
                "The sky is grey and so is everything else. Nothing brightens this day.",
                "I can't remember the last time I felt anything but this heaviness.",
                "What's the point of trying? The outcome is always the same.",
            ],
        }

        dialogues = []
        for emotion_a, emotion_b in pairings:
            a_lines = templates.get(emotion_a, [f"I feel {emotion_a} right now."])
            b_lines = templates.get(emotion_b, [f"I feel {emotion_b} right now."])

            # Generate 3 dialogues per pairing using different template combos
            for i in range(min(3, len(a_lines))):
                a1 = a_lines[i % len(a_lines)]
                b1 = b_lines[i % len(b_lines)]
                a2 = a_lines[(i + 1) % len(a_lines)]
                b2 = b_lines[(i + 1) % len(b_lines)]

                dialogue = (
                    f"Speaker A: {a1}\n"
                    f"Speaker B: {b1}\n"
                    f"Speaker A: {a2}\n"
                    f"Speaker B: {b2}"
                )
                dialogues.append({
                    "emotion_a": emotion_a,
                    "emotion_b": emotion_b,
                    "dialogue": dialogue,
                })

        return dialogues

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        current_vectors: Dict[str, torch.Tensor],
        other_vectors: Dict[str, torch.Tensor],
    ) -> SpeakerValidationResult:
        """
        Run three validation tests on the extracted speaker vectors.

        Test A: Speaker separation — cosine(current[e], other[e]) per emotion.
               Measures whether the same emotion is represented differently
               depending on who is expressing it. Threshold < 0.3 (~73°)
               requires substantial directional divergence, though not full
               orthogonality (which would be < 0.1 / ~84°).
        Test B: Internal consistency — valence separation on each set
        Test C: Thermostat — emotionally charged messages produce opposite
                arousal patterns on the two vector sets
        """
        # Test A: Speaker separation (cosine alignment between same-emotion vectors)
        orth_scores = self._test_orthogonality(current_vectors, other_vectors)
        orth_mean = sum(orth_scores.values()) / len(orth_scores) if orth_scores else 1.0
        orth_pass = orth_mean < 0.3

        # Cross-role baseline: mean cos(current[e_i], other[e_j]) for i != j
        # If separation is working, same-emotion alignment should be LOWER than
        # cross-emotion alignment (the vectors are specialized, not generic).
        cross_scores = []
        common = sorted(set(current_vectors.keys()) & set(other_vectors.keys()))
        for i, n1 in enumerate(common):
            for j, n2 in enumerate(common):
                if i != j:
                    c = F.normalize(current_vectors[n1], dim=0)
                    o = F.normalize(other_vectors[n2], dim=0)
                    cross_scores.append(torch.dot(c, o).item())
        cross_mean = sum(cross_scores) / len(cross_scores) if cross_scores else 0.0

        # Random baseline for this dimensionality
        d_model = next(iter(current_vectors.values())).shape[0]
        random_std = 1.0 / (d_model ** 0.5)

        print(f"\n[speakers] Test A — Speaker Separation")
        for name, score in sorted(orth_scores.items()):
            print(f"  {name:15s} cos(current, other) = {score:+.3f}")
        print(f"  MEAN same-emotion cos  = {orth_mean:.4f}")
        print(f"  MEAN cross-emotion cos = {cross_mean:.4f}")
        print(f"  Random baseline (d={d_model}): E[cos] ~ 0 +/- {random_std:.4f}")
        print(f"  Same-emotion z-score   = {orth_mean / random_std:.1f}s above random")
        print(f"  Threshold < 0.3 (~73°) — {'PASS' if orth_pass else 'FAIL'}")

        # Test B: Internal consistency
        current_vs = valence_separation(current_vectors, self.emotions)
        other_vs = valence_separation(other_vectors, self.emotions)
        consistency_pass = current_vs < -0.2 and other_vs < -0.2

        print(f"\n[speakers] Test B — Internal Consistency")
        print(f"  Current-speaker valence separation: {current_vs:.4f} (threshold < -0.2)")
        print(f"  Other-speaker valence separation:   {other_vs:.4f} (threshold < -0.2)")
        print(f"  — {'PASS' if consistency_pass else 'FAIL'}")

        # Test C: Thermostat
        thermostat_results = self._test_thermostat(current_vectors, other_vectors)
        deltas = [r["arousal_delta"] for r in thermostat_results]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        thermostat_pass = mean_delta < -0.2  # Current-speaker should have lower arousal

        print(f"\n[speakers] Test C — Thermostat")
        for r in thermostat_results:
            status = "YES" if r["arousal_delta"] < 0 else "NO"
            print(f"  Message: \"{r['message'][:50]}...\"")
            print(f"    Other-speaker read:  {r['other_top'][0][0]} ({r['other_top'][0][1]:.2f}), "
                  f"{r['other_top'][1][0]} ({r['other_top'][1][1]:.2f})")
            print(f"    Current-speaker:     {r['current_top'][0][0]} ({r['current_top'][0][1]:.2f}), "
                  f"{r['current_top'][1][0]} ({r['current_top'][1][1]:.2f})")
            print(f"    Thermostat active:   {status} (arousal delta = {r['arousal_delta']:+.2f})")
        print(f"  MEAN arousal delta = {mean_delta:.3f} (threshold < -0.2) — "
              f"{'PASS' if thermostat_pass else 'FAIL'}")

        return SpeakerValidationResult(
            orthogonality_scores=orth_scores,
            orthogonality_mean=orth_mean,
            orthogonality_passed=orth_pass,
            cross_emotion_mean=cross_mean,
            random_baseline_std=random_std,
            current_valence_sep=current_vs,
            other_valence_sep=other_vs,
            consistency_passed=consistency_pass,
            thermostat_results=thermostat_results,
            thermostat_mean_delta=mean_delta,
            thermostat_passed=thermostat_pass,
        )

    def _test_orthogonality(
        self,
        current_vectors: Dict[str, torch.Tensor],
        other_vectors: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between current[e] and other[e] for
        each emotion present in both sets.

        Interpretation note: in d=2304 dimensions, random unit vectors have
        expected cosine ≈ 0 with std ≈ 1/√d ≈ 0.021. So a mean of 0.08
        is ~4σ above random — the vectors are slightly MORE correlated than
        chance, not orthogonal. This is expected: "afraid-as-expressed" and
        "afraid-as-perceived" share semantic content.

        The test measures whether the correlation is LOW ENOUGH to be useful
        for discrimination, not whether it's literally zero.
        """
        scores = {}
        common = set(current_vectors.keys()) & set(other_vectors.keys())
        for name in sorted(common):
            c = F.normalize(current_vectors[name], dim=0)
            o = F.normalize(other_vectors[name], dim=0)
            scores[name] = torch.dot(c, o).item()
        return scores

    def _test_thermostat(
        self,
        current_vectors: Dict[str, torch.Tensor],
        other_vectors: Dict[str, torch.Tensor],
    ) -> List[dict]:
        """
        Test the thermostat hypothesis: when processing emotionally charged
        messages, the model's OTHER-speaker read should show high arousal
        while CURRENT-speaker should show low arousal (the model regulates).

        Uses 10 messages covering: anger, distress, panic, sadness, enthusiasm.
        """
        test_messages = [
            # Anger
            ("I can't believe you said that to me. That was completely out of line.",
             "anger"),
            ("You lied to me this whole time? How could you do that?",
             "anger"),
            # Distress
            ("Everything is falling apart and I don't know how to fix any of it.",
             "distress"),
            ("I've been struggling for months and nobody seems to care.",
             "distress"),
            # Panic
            ("Oh god, I think something is seriously wrong. I need help right now.",
             "panic"),
            ("We're going to lose everything if we don't act immediately.",
             "panic"),
            # Sadness
            ("I just found out my best friend passed away. I can't process this.",
             "sadness"),
            ("I feel so alone. Nobody understands what I'm going through.",
             "sadness"),
            # Enthusiasm (tests regulation in the other direction)
            ("THIS IS THE BEST DAY OF MY LIFE! EVERYTHING IS PERFECT!",
             "enthusiasm"),
            ("I just won the lottery! I'm going to buy a mansion and a yacht!",
             "enthusiasm"),
        ]

        # Build emotion metadata lookup
        metadata = {e["name"]: e for e in self.emotions}

        # Stack vectors for batch scoring
        current_names = list(current_vectors.keys())
        current_matrix = torch.stack([current_vectors[n] for n in current_names])
        other_names = list(other_vectors.keys())
        other_matrix = torch.stack([other_vectors[n] for n in other_names])

        results = []
        for message, category in test_messages:
            activation = self._get_activation_at_last_content_token(message)
            if activation is None:
                continue

            # Score against both vector sets
            act_norm = F.normalize(activation.unsqueeze(0), dim=1)

            current_scores_t = (act_norm @ F.normalize(current_matrix, dim=1).T).squeeze(0)
            other_scores_t = (act_norm @ F.normalize(other_matrix, dim=1).T).squeeze(0)

            current_scores = {n: float(current_scores_t[i]) for i, n in enumerate(current_names)}
            other_scores = {n: float(other_scores_t[i]) for i, n in enumerate(other_names)}

            # Compute arousal for each
            current_arousal = self._weighted_arousal(current_scores, metadata)
            other_arousal = self._weighted_arousal(other_scores, metadata)

            # Top emotions for reporting
            current_sorted = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            other_sorted = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)

            results.append({
                "message": message,
                "category": category,
                "current_arousal": current_arousal,
                "other_arousal": other_arousal,
                "arousal_delta": current_arousal - other_arousal,
                "current_top": current_sorted[:2],
                "other_top": other_sorted[:2],
            })

        return results

    def _weighted_arousal(
        self,
        scores: Dict[str, float],
        metadata: Dict[str, dict],
    ) -> float:
        """Compute weighted average arousal from emotion scores."""
        total_w = 0.0
        arousal = 0.0
        for name, score in scores.items():
            if name in metadata:
                w = max(score, 0.0)
                arousal += w * metadata[name]["arousal"]
                total_w += w
        if total_w > 0:
            arousal /= total_w
        return max(-1.0, min(1.0, arousal))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, result: dict, path: Optional[str] = None) -> Path:
        """
        Save speaker separation vectors and metadata.

        Args:
            result: The dict returned by extract().
            path: Override save path.
        """
        if path is None:
            name = self.model_info["model_name"].replace("/", "_")
            path = VECTORS_DIR / f"{name}_speakers.pt"
        path = Path(path)

        torch.save(result, path)
        print(f"[speakers] Saved speaker vectors to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> dict:
        """Load previously saved speaker separation vectors."""
        return torch.load(path, weights_only=False)
