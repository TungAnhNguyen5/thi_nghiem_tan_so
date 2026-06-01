from __future__ import annotations

import numpy as np


def window_iq(
    iq: np.ndarray,
    window_size: int,
    hop_size: int,
) -> list[np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if iq.size < window_size:
        raise ValueError(
            "IQ capture is too short for the selected window size"
        )

    return [
        iq[start:start + window_size]
        for start in range(0, iq.size - window_size + 1, hop_size)
    ]
