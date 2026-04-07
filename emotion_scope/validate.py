"""
Validation suite — the Phase 1 "go/no-go" gate.

Thresholds were decided with the architect:
    - Tylenol intensity test: Spearman r > 0.7 (afraid vs log-dose),
      and calm Spearman r < -0.5.
    - Confusion matrix on implicit scenarios: top-3 accuracy > 0.6
    - Valence separation: cosine(pos_mean, neg_mean) < -0.2
    - Emotion richness: average pairwise cosine < 0.5

If ALL four pass, proceed to Phase 2. If Tylenol fails, the vectors aren't
capturing abstract emotion — try a different probe layer or model. If only
the confusion matrix is weak, it's likely a template quality issue.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from scipy.stats import spearmanr

from emotion_scope.config import (
    CORE_EMOTIONS,
    DATA_DIR,
    METRICS_DIR,
    ValidationThresholds,
)
from emotion_scope.probe import EmotionProbe
from emotion_scope.utils import (
    average_pairwise_cosine,
    valence_separation,
)


@dataclass
class ValidationResult:
    tylenol_passed: bool = False
    confusion_passed: bool = False
    valence_passed: bool = False
    richness_passed: bool = False
    all_passed: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        def mark(ok: bool) -> str:
            return "PASS" if ok else "FAIL"
        lines = [
            f"[{mark(self.tylenol_passed)}] Tylenol intensity",
            f"[{mark(self.confusion_passed)}] Confusion matrix (top-3)",
            f"[{mark(self.valence_passed)}] Valence separation",
            f"[{mark(self.richness_passed)}] Emotion richness",
            f"OVERALL: [{mark(self.all_passed)}]",
        ]
        return "\n".join(lines)


class Validator:
    """Runs the full Phase 1 validation suite against a probe + vectors."""

    def __init__(
        self,
        probe: EmotionProbe,
        emotion_vectors: Dict[str, torch.Tensor],
        thresholds: Optional[ValidationThresholds] = None,
    ):
        self.probe = probe
        self.vectors = emotion_vectors
        self.thresholds = thresholds or ValidationThresholds()

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_tylenol(self, path: Optional[Path] = None) -> dict:
        """
        Intensity scaling test. For each scenario (e.g., tylenol dosage),
        check that expected_increasing emotions correlate positively with the
        scale variable, and expected_decreasing correlate negatively.
        """
        path = path or DATA_DIR / "validation" / "intensity_scales.json"
        if not Path(path).exists():
            return {"passed": False, "error": f"missing {path}"}

        with open(path, encoding="utf-8") as f:
            scales = json.load(f)

        results = {}
        afraid_spearman = None
        calm_spearman = None

        for scenario_name, scenario in scales.items():
            template = scenario["template"]
            values = scenario["values"]
            inc = scenario.get("expected_increasing", [])
            dec = scenario.get("expected_decreasing", [])

            # Collect scores for each value
            log_values = [math.log(v) if v > 0 else 0.0 for v in values]
            emotion_series: Dict[str, List[float]] = {}

            for v in values:
                text = template.replace("{value}", str(v))
                # Strip any raw "User: ... \nAssistant:" wrapper — we pass
                # the bare user message through analyze_conversation() which
                # applies the model's native chat template. This ensures
                # validation uses the same probe path as the live demo.
                user_msg = text
                for prefix in ("User: ", "User:\n"):
                    if user_msg.startswith(prefix):
                        user_msg = user_msg[len(prefix):]
                for suffix in ("\n\nAssistant:", "\nAssistant:"):
                    if user_msg.endswith(suffix):
                        user_msg = user_msg[: -len(suffix)]
                user_msg = user_msg.strip()
                dual = self.probe.analyze_conversation(user_msg)
                state = dual.model_state
                for name, score in state.scores.items():
                    emotion_series.setdefault(name, []).append(score)

            scenario_result: Dict[str, float] = {}
            for emotion_name, series in emotion_series.items():
                if len(series) == len(log_values) and len(series) > 1:
                    rho, _ = spearmanr(log_values, series)
                    scenario_result[emotion_name] = float(rho) if not math.isnan(rho) else 0.0

            results[scenario_name] = {
                "correlations": scenario_result,
                "expected_increasing": inc,
                "expected_decreasing": dec,
            }

            # Primary gate: the canonical Tylenol test.
            # TODO (Phase 2): add direction-aware Spearman for scales that
            # descend (e.g. deadline_hours goes 72 -> 1), so non-tylenol scales
            # can contribute to the pass/fail. Cosmetic for Phase 1 — only
            # tylenol gates the go/no-go decision.
            if scenario_name == "tylenol":
                afraid_spearman = scenario_result.get("afraid")
                calm_spearman = scenario_result.get("calm")

        passed = (
            afraid_spearman is not None
            and afraid_spearman > self.thresholds.tylenol_min_spearman
            and calm_spearman is not None
            and calm_spearman < self.thresholds.tylenol_calm_max_spearman
        )

        return {
            "passed": passed,
            "afraid_spearman": afraid_spearman,
            "calm_spearman": calm_spearman,
            "thresholds": {
                "afraid_min": self.thresholds.tylenol_min_spearman,
                "calm_max": self.thresholds.tylenol_calm_max_spearman,
            },
            "scenarios": results,
        }

    def test_confusion_matrix(self, path: Optional[Path] = None) -> dict:
        """
        Top-3 recall on implicit scenarios: for each scenario, check whether
        any of the expected emotions appears in the top-3 scoring emotions.

        Note: This is recall@3, not a full confusion matrix. The name is kept
        for backward compatibility but the metric is hit-rate classification.
        """
        path = path or DATA_DIR / "validation" / "implicit_scenarios.json"
        if not Path(path).exists():
            return {"passed": False, "error": f"missing {path}"}

        with open(path, encoding="utf-8") as f:
            scenarios = json.load(f)

        n = len(scenarios)
        hits = 0
        per_scenario = []

        for scen in scenarios:
            # Use analyze_conversation for parity with the live demo path
            dual = self.probe.analyze_conversation(scen["scenario"])
            state = dual.model_state
            top3 = [name for name, _ in state.top_emotions[:3]]
            expected = set(scen.get("expected_emotions", []))
            hit = bool(expected.intersection(top3))
            hits += int(hit)
            per_scenario.append({
                "scenario": scen["scenario"][:80],
                "expected": list(expected),
                "top3": top3,
                "hit": hit,
            })

        accuracy = hits / n if n > 0 else 0.0
        return {
            "passed": accuracy > self.thresholds.confusion_top3_min_accuracy,
            "top3_accuracy": accuracy,
            "n_scenarios": n,
            "threshold": self.thresholds.confusion_top3_min_accuracy,
            "details": per_scenario,
        }

    def test_valence_separation(self) -> dict:
        cos = valence_separation(self.vectors, CORE_EMOTIONS)
        return {
            "passed": cos < self.thresholds.valence_separation_max,
            "cosine": cos,
            "threshold": self.thresholds.valence_separation_max,
        }

    def test_richness(self) -> dict:
        avg = average_pairwise_cosine(self.vectors)
        return {
            "passed": avg < self.thresholds.richness_max_avg_cosine,
            "avg_cosine": avg,
            "threshold": self.thresholds.richness_max_avg_cosine,
        }

    # ------------------------------------------------------------------
    # Full suite
    # ------------------------------------------------------------------

    def run_all(self, save_path: Optional[Path] = None) -> ValidationResult:
        details = {
            "tylenol": self.test_tylenol(),
            "confusion": self.test_confusion_matrix(),
            "valence": self.test_valence_separation(),
            "richness": self.test_richness(),
        }
        result = ValidationResult(
            tylenol_passed=bool(details["tylenol"].get("passed", False)),
            confusion_passed=bool(details["confusion"].get("passed", False)),
            valence_passed=bool(details["valence"].get("passed", False)),
            richness_passed=bool(details["richness"].get("passed", False)),
            details=details,
        )
        result.all_passed = (
            result.tylenol_passed
            and result.confusion_passed
            and result.valence_passed
            and result.richness_passed
        )

        if save_path is None:
            save_path = METRICS_DIR / "validation_results.json"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "passed": {
                        "tylenol": result.tylenol_passed,
                        "confusion": result.confusion_passed,
                        "valence": result.valence_passed,
                        "richness": result.richness_passed,
                        "all": result.all_passed,
                    },
                    "details": _json_safe(details),
                },
                f,
                indent=2,
            )
        print(f"[validate] Results written to {save_path}")
        print(result.summary())
        return result


def _json_safe(obj):
    """Recursively convert tensors / numpy floats to plain Python."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj
