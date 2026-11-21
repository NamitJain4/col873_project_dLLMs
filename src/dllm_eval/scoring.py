"""Token positions used by each masked model's prediction head.

Dream retains the causal LM's one-position prediction shift. Its official
generation code shifts logits right before looking up masked target tokens.
LLaDA predicts a masked token at its own position.
"""


def masked_prediction_positions(kind, target_positions):
    if kind not in ("dream", "llada"):
        raise ValueError(f"unsupported masked model kind: {kind}")
    positions = target_positions - 1 if kind == "dream" else target_positions
    if (positions < 0).any():
        raise ValueError("Dream requires a preceding token for every scored target")
    return positions
