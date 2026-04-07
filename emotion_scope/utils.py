"""
Shared utilities for emotion-scope.

Key functions:
- cosine_similarity_matrix: pairwise cosine sim of a dict of emotion vectors
- valence_separation: scalar metric for how well vectors separate pos/neg valence
- find_content_token_range: identify content tokens vs chat template markup
- get_device: auto-detect cuda/cpu
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from emotion_scope.config import CHAT_TEMPLATE_MARKERS, CORE_EMOTIONS


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> str:
    """Return 'cuda' if available (and desired), else 'cpu'."""
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Cosine / separation metrics
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    vectors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute pairwise cosine similarity matrix for a dict of emotion vectors.

    Returns:
        (similarity_matrix, emotion_names) where similarity_matrix has shape
        (n, n) and names is the row/column ordering.
    """
    names = list(vectors.keys())
    if not names:
        return torch.empty(0, 0), []
    matrix = torch.stack([vectors[n] for n in names])
    matrix_norm = F.normalize(matrix, dim=1)
    sim = matrix_norm @ matrix_norm.T
    return sim, names


def valence_separation(
    vectors: Dict[str, torch.Tensor],
    emotions_metadata: Optional[List[dict]] = None,
) -> float:
    """
    Cosine similarity between the mean positive-valence vector and the mean
    negative-valence vector. More negative = better separation.

    Returns 0.0 if either bucket is empty.
    """
    metadata = emotions_metadata or CORE_EMOTIONS
    pos_names = [e["name"] for e in metadata if e["valence"] > 0.3]
    neg_names = [e["name"] for e in metadata if e["valence"] < -0.3]

    pos_vecs = [vectors[n] for n in pos_names if n in vectors]
    neg_vecs = [vectors[n] for n in neg_names if n in vectors]

    if not pos_vecs or not neg_vecs:
        return 0.0

    pos_mean = F.normalize(torch.stack(pos_vecs).mean(dim=0), dim=0)
    neg_mean = F.normalize(torch.stack(neg_vecs).mean(dim=0), dim=0)
    return torch.dot(pos_mean, neg_mean).item()


def average_pairwise_cosine(vectors: Dict[str, torch.Tensor]) -> float:
    """
    Average off-diagonal cosine similarity across all emotion vectors.
    Lower = more distinct representations (emotion richness).
    """
    sim, names = cosine_similarity_matrix(vectors)
    n = len(names)
    if n < 2:
        return 0.0
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


# ---------------------------------------------------------------------------
# Content token range — the critical fix for chat-templated prompts
# ---------------------------------------------------------------------------

def _detect_model_family(tokenizer) -> str:
    """Guess which chat template family a tokenizer belongs to."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    name_lower = name.lower()
    for family in ("gemma", "llama", "mistral", "phi", "qwen", "deepseek"):
        if family in name_lower:
            return family
    return "generic"


def _collect_marker_token_ids(tokenizer, family: str) -> set:
    """
    Build the set of token ids that correspond to chat-template markup
    for this tokenizer/family. Uses added_tokens_decoder plus known strings.
    """
    marker_ids: set = set()

    # 1. All special tokens the tokenizer itself declares
    try:
        for tok_id, tok in tokenizer.added_tokens_decoder.items():
            # added_tokens_decoder maps id -> AddedToken
            content = getattr(tok, "content", str(tok))
            if content and (content.startswith("<") or content.startswith("[")):
                marker_ids.add(int(tok_id))
    except Exception:
        pass

    # 2. all_special_ids as a safety net
    for sid in getattr(tokenizer, "all_special_ids", []) or []:
        marker_ids.add(int(sid))

    # 3. Family-specific marker strings — encode and add single-token matches
    markers = CHAT_TEMPLATE_MARKERS.get(family, []) + CHAT_TEMPLATE_MARKERS["generic"]
    for marker in markers:
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) == 1:
                marker_ids.add(int(ids[0]))
        except Exception:
            continue

    return marker_ids


def find_content_token_range(
    input_ids,
    tokenizer,
) -> Tuple[int, int]:
    """
    Identify the (start, end) range of USER CONTENT tokens in a tokenized
    chat-templated input, excluding all template markup.

    For Gemma 2, a chat prompt looks like:
        <bos> <start_of_turn> user \\n [CONTENT...] <end_of_turn> \\n <start_of_turn> model \\n
    This function returns (start, end) spanning only [CONTENT...].

    Strategy: find the structural anchors (<end_of_turn> after user content)
    and use them to bound the content range, rather than classifying individual
    tokens as markers (which fails on ambiguous tokens like newlines).

    Args:
        input_ids: 1-D tensor or list of token ids for a single sequence.
        tokenizer: the HuggingFace/TransformerLens tokenizer.

    Returns:
        (start_idx, end_idx) — Python slice indices. end_idx is exclusive.
        For non-chat models or when markers can't be detected, returns
        (0, len(input_ids)), matching raw-text behavior.
    """
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        ids_list = input_ids.tolist()
    else:
        ids_list = list(input_ids)

    n = len(ids_list)
    if n == 0:
        return 0, 0

    family = _detect_model_family(tokenizer)

    # -- Anchor-based approach for known families --
    # Find specific structural tokens and use them to bracket content.

    # Get anchor token IDs
    eot_id = None  # <end_of_turn> or equivalent
    sot_id = None  # <start_of_turn> or equivalent

    if family == "gemma":
        # Gemma uses token 107 = <end_of_turn>, 106 = <start_of_turn>
        try:
            eot_candidates = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
            sot_candidates = tokenizer.encode("<start_of_turn>", add_special_tokens=False)
            if len(eot_candidates) == 1:
                eot_id = eot_candidates[0]
            if len(sot_candidates) == 1:
                sot_id = sot_candidates[0]
        except Exception:
            pass
    elif family == "llama":
        try:
            eot_candidates = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            sot_candidates = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
            if len(eot_candidates) == 1:
                eot_id = eot_candidates[0]
            if len(sot_candidates) == 1:
                sot_id = sot_candidates[0]
        except Exception:
            pass

    if eot_id is not None:
        # Find the FIRST <end_of_turn> — this marks the end of user content
        # Content starts after the preamble (BOS + <start_of_turn> + role + \n)
        # Content ends at the <end_of_turn> token

        # Find first <end_of_turn>
        eot_pos = None
        for i, tid in enumerate(ids_list):
            if tid == eot_id:
                eot_pos = i
                break

        if eot_pos is not None:
            # Content ends at eot_pos (exclusive — don't include <end_of_turn> itself)
            end = eot_pos

            # Content starts after the preamble.
            # Walk forward past all special/markup tokens at the beginning.
            marker_ids = _collect_marker_token_ids(tokenizer, family)
            start = 0
            while start < end and ids_list[start] in marker_ids:
                start += 1

            # Also skip the newline right after the role token (e.g., "user\n")
            # Token 108 in Gemma is \n — skip it if it's right after the role
            if start < end:
                decoded = tokenizer.decode([ids_list[start]])
                if decoded.strip() == "" and start + 1 < end:
                    start += 1

            if end > start:
                return start, end

    # -- Fallback: marker-based stripping (original approach) --
    marker_ids = _collect_marker_token_ids(tokenizer, family)
    if not marker_ids:
        return 0, n

    start = 0
    while start < n and ids_list[start] in marker_ids:
        start += 1
    end = n
    while end > start and ids_list[end - 1] in marker_ids:
        end -= 1

    if end <= start:
        return 0, n
    return start, end


def last_content_token_index(
    input_ids,
    tokenizer,
) -> int:
    """
    Return the index of the last content token — the token at position
    `end - 1` of the content range. This is the preferred probe position
    for 'user read' extraction.
    """
    _, end = find_content_token_range(input_ids, tokenizer)
    return max(end - 1, 0)
