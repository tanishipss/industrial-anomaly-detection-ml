import numpy as np
from src.config import T_HIGH, T_LOW

def dual_threshold_decision(probs, expected_positive_rate):
    preds = np.full(len(probs), -1)

    preds[probs >= T_HIGH] = 1
    preds[probs <= T_LOW] = 0

    ambiguous = np.where(preds == -1)[0]
    amb_probs = probs[ambiguous]

    k = int(len(ambiguous) * expected_positive_rate)
    top_idx = ambiguous[np.argsort(amb_probs)[-k:]]

    preds[top_idx] = 1
    preds[preds == -1] = 0

    return preds
