"""
Real-time emotion probing during model inference.

Hooks the forward pass, reads the residual stream at the probe layer, and
scores it against the extracted emotion vectors via cosine similarity.

Key departure from a naive implementation: the probe reads at the LAST CONTENT
TOKEN, not the raw last token. With chat-templated prompts (Gemma, Llama),
the raw last token is usually <end_of_turn> or a role marker, not user content.
`last_content_token_index` from utils identifies the correct position.

Usage:
    probe = EmotionProbe(model, tokenizer, backend, info, vectors)
    state = probe.analyze("I'm having a terrible day")
    print(state.dominant, state.color_hex)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from emotion_scope.config import CORE_EMOTIONS, ProbeConfig
from emotion_scope.utils import last_content_token_index
from emotion_scope.visualize import scores_to_color


@dataclass
class EmotionState:
    """Emotion probe result for a single analysis."""
    scores: Dict[str, float]
    top_emotions: List[Tuple[str, float]]
    valence: float
    arousal: float
    color_hex: str
    color_hsv: tuple
    dominant: str


@dataclass
class DualEmotionState:
    """Combined result with both speaker states (user_read requires speakers.py)."""
    model_state: EmotionState
    user_read: Optional[EmotionState] = None
    raw_activation: Optional[torch.Tensor] = None


class EmotionProbe:
    """Real-time emotion probing during inference."""

    def __init__(
        self,
        model,
        tokenizer,
        backend: str,
        model_info: dict,
        emotion_vectors: Dict[str, torch.Tensor],
        emotions_metadata: Optional[List[dict]] = None,
        config: Optional[ProbeConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.model_info = model_info
        self.config = config or ProbeConfig()

        self.emotion_names: List[str] = list(emotion_vectors.keys())
        self.vector_matrix = torch.stack(
            [emotion_vectors[n] for n in self.emotion_names]
        )  # (n_emotions, d_model)

        self.emotions_metadata = emotions_metadata or CORE_EMOTIONS
        self.probe_layer: int = model_info["probe_layer"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> EmotionState:
        """Analyze the emotional content of a raw text input."""
        activation = self._get_activation(text, use_chat_template=False)
        return self._activation_to_state(activation)

    def analyze_activation(self, activation: torch.Tensor) -> EmotionState:
        """Score a raw residual-stream activation against the emotion vectors.

        Use this when you've captured the activation yourself (e.g. via a
        forward hook during generation) and just need the emotion readout.
        """
        if activation.ndim > 1:
            activation = activation.squeeze()
        return self._activation_to_state(activation.cpu().float())

    def analyze_conversation(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> DualEmotionState:
        """
        Analyze a user turn. Reads the model's representation at the final
        content token of the user's message, before any assistant generation.

        Phase 1: only model_state is populated. user_read requires speaker
        separation (Phase 2).
        """
        prompt = self._format_chat(user_message, system_prompt)
        activation = self._get_activation(prompt, use_chat_template=True)
        model_state = self._activation_to_state(activation)
        return DualEmotionState(
            model_state=model_state,
            user_read=None,
            raw_activation=activation,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_chat(self, user_message: str, system_prompt: Optional[str]) -> str:
        """
        Build a chat-templated prompt. Prefers the tokenizer's apply_chat_template
        when available; otherwise falls back to a plain User/Assistant format.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if apply is not None:
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        if system_prompt:
            return f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        return f"User: {user_message}\n\nAssistant:"

    def _get_activation(self, text: str, use_chat_template: bool) -> torch.Tensor:
        """
        Run the text through the model, capture the residual stream at the
        probe layer, and return the activation at the response-preparation
        position.

        IMPORTANT — extraction vs probing position distinction:
        Vectors are EXTRACTED by averaging over content tokens only (excluding
        template markup). This gives clean emotion directions.
        But PROBING reads at the LAST token of the full prompt (the response-
        preparation position), where the model has compressed its full
        emotional assessment of the situation. Testing showed 83% top-3
        accuracy at the response-preparation position vs 75% at the last
        content token. See MATHS.md for the full analysis.
        """
        if self.backend == "transformer_lens":
            hook_name = f"blocks.{self.probe_layer}.hook_resid_post"
            tokens = self.model.to_tokens(text)
            _, cache = self.model.run_with_cache(tokens, names_filter=hook_name)
            residual = cache[hook_name]  # (1, seq_len, d_model)
            idx = self._select_probe_index(tokens, residual.shape[1], use_chat_template)
            return residual[0, idx, :].detach().cpu().float()

        # HuggingFace backend
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

        residual = captured["act"]
        idx = self._select_probe_index(input_ids, residual.shape[1], use_chat_template)
        return residual[0, idx, :].cpu().float()

    def _select_probe_index(
        self,
        input_ids,
        seq_len: int,
        use_chat_template: bool,
    ) -> int:
        """
        Select the token position for probing.

        For chat-templated prompts, uses the RESPONSE-PREPARATION position:
        the last token of the full prompt (after all template markup,
        immediately before generation begins). This is equivalent to
        Anthropic's measurement at "the ':' token following 'Assistant'."

        Testing on 12 scenarios showed:
          - Response-preparation position: 83% top-3 accuracy
          - Last content token: 75% top-3 accuracy

        The response-preparation position is where the model has compressed
        its full situational and emotional assessment into the representation
        needed for response generation. Content tokens carry the raw signal
        but the response-prep token carries the processed assessment.
        """
        mode = self.config.token_position
        if isinstance(mode, int):
            return max(0, min(mode, seq_len - 1))
        if mode == "last":
            return seq_len - 1
        # "last_content" mode — for probing, this now means the response-
        # preparation position (last token of full prompt), NOT the last
        # content token. The name is kept for backward compatibility but
        # the behavior matches Anthropic's methodology.
        return seq_len - 1

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
        raise ValueError(f"Cannot find layers in {type(self.model).__name__}")

    def _activation_to_state(self, activation: torch.Tensor) -> EmotionState:
        """Cosine-score the activation against each emotion vector."""
        activation_norm = F.normalize(activation.unsqueeze(0), dim=1)
        vectors_norm = F.normalize(self.vector_matrix, dim=1)
        scores_tensor = (activation_norm @ vectors_norm.T).squeeze(0)

        scores = {
            name: float(scores_tensor[i].item())
            for i, name in enumerate(self.emotion_names)
        }
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_emotions[: self.config.top_k_emotions]

        color_hex, color_hsv, valence, arousal = scores_to_color(
            scores, self.emotions_metadata
        )

        return EmotionState(
            scores=scores,
            top_emotions=top_k,
            valence=valence,
            arousal=arousal,
            color_hex=color_hex,
            color_hsv=color_hsv,
            dominant=top_k[0][0] if top_k else "unknown",
        )
