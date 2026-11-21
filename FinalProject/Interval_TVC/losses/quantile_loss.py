from typing import Sequence

import torch
from torch import Tensor


QUANTILE_LEVELS = (0.025, 0.5, 0.975)


def quantile_loss(
    predictions: Tensor,
    targets: Tensor,
    quantiles: Sequence[float] = QUANTILE_LEVELS,
) -> Tensor:
    """Mean pinball loss for predictions shaped [batch, horizon, channel, quantile]."""
    if predictions.ndim != targets.ndim + 1:
        raise ValueError("predictions must have one more dimension than targets")
    if predictions.shape[:-1] != targets.shape:
        raise ValueError("prediction and target dimensions do not align")
    if predictions.shape[-1] != len(quantiles):
        raise ValueError("the last prediction dimension must match quantiles")
    if any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    if tuple(sorted(quantiles)) != tuple(quantiles):
        raise ValueError("quantiles must be in ascending order")

    q = predictions.new_tensor(quantiles)
    error = targets.unsqueeze(-1) - predictions
    losses = torch.maximum(q * error, (q - 1.0) * error)
    return losses.mean()
