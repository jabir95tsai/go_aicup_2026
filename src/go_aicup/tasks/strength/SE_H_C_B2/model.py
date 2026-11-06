"""The requested 60 -> 32 -> 10 head, returning logits for CE."""
import numpy as np
import torch
from torch import nn

from .common import BUCKETS


class Head(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(60, 32), nn.ReLU(), nn.Linear(32, 10))


def validate_probabilities(proba):
    if (proba.ndim != 2 or proba.shape[1] != 10 or not np.isfinite(proba).all()
            or (proba < 0).any() or not np.allclose(proba.sum(1), 1, atol=1e-6)):
        raise ValueError('invalid ten-class posterior')
    return proba


@torch.inference_mode()
def probabilities(heads, values, k):
    values = np.asarray(values, np.float32)
    k = np.asarray(k)
    if values.shape != (len(k), 60) or not np.isfinite(values).all() or not np.isin(k, BUCKETS).all():
        raise ValueError('invalid question features or k')
    result = np.empty((len(k), 10), np.float32)
    for bucket in BUCKETS:
        mask = k == bucket
        if not mask.any():
            continue
        record = heads[bucket]
        model = Head(); model.load_state_dict(record['model']); model.eval()
        x = ((values[mask].astype(np.float64) - record['mean']) / record['scale']).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('invalid standardization')
        result[mask] = model(torch.from_numpy(x)).softmax(-1).numpy()
    return validate_probabilities(result)
