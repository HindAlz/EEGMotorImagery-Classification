from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


@dataclass(frozen=True)
class EEGDataset:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: Optional[np.ndarray]
    runs: Optional[np.ndarray]
    sampling_rate: float
    label_values: np.ndarray

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[2])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)

    values = np.unique(labels)
    mapping = {label: idx for idx, label in enumerate(values.tolist())}

    y = np.asarray(
        [mapping[label] for label in labels.tolist()],
        dtype=np.int64,
    )

    return y, values


def load_dataset(
    subjects: list[int] | None = None,
) -> EEGDataset:
    """
    Load BNCI2014-001 through MOABB.

    Returns
    -------
    X
        EEG epochs shaped (trials, channels, samples).
    y
        Integer labels 0..K-1.
    subjects
        Subject ID for each trial.
    sessions
        Session identifier for each trial.
    runs
        Run identifier for each trial, when supplied by MOABB metadata.
    """

    dataset = BNCI2014_001()

    if subjects is None:
        subjects = list(dataset.subject_list)

    paradigm = MotorImagery()

    X, labels, meta = paradigm.get_data(
        dataset=dataset,
        subjects=subjects,
    )

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected MOABB EEG shape "
            f"(trials, channels, samples), got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite EEG values.")

    y, label_values = _encode_labels(labels)

    trial_subjects = meta["subject"].to_numpy(dtype=np.int64)

    sessions = (
        meta["session"].astype(str).to_numpy()
        if "session" in meta.columns
        else None
    )

    runs = (
        meta["run"].astype(str).to_numpy()
        if "run" in meta.columns
        else None
    )

    # BNCI2014-001 is sampled at 250 Hz.
    sampling_rate = 250.0

    return EEGDataset(
        X=X,
        y=y,
        subjects=trial_subjects,
        sessions=sessions,
        runs=runs,
        sampling_rate=sampling_rate,
        label_values=label_values,
    )