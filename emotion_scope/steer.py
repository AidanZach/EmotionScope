"""
Emotion steering experiments. Phase 2 implementation.

Planned algorithm (confirmed with architect):
Inject v_e into the residual stream at each of the middle-third layers
[L/3, 2L/3] during generation:

    x^(l)_t  <-  x^(l)_t + alpha * ||x^(l)_t||_avg * v_hat_e

where alpha = 0.5 (Anthropic's value) and ||x^(l)_t||_avg is the average
residual stream L2 norm at layer l (NOT raw addition — scaled by avg norm).
"""


def steer(*args, **kwargs):
    raise NotImplementedError(
        "Steering is Phase 2. See EmotionScope.md Section 3.4 "
        "and IMPLEMENTATION_INSTRUCTIONS.md Step 5."
    )
