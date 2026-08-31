from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelStandardizer:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        if X.ndim != 3:
            raise ValueError(f"Expected 3D EEG, got {X.shape}")

        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(
            X.std(axis=(0, 2), keepdims=True),
            self.eps,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Fit standardizer first")

        standardized = (X - self.mean_) / self.std_
        return standardized.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)