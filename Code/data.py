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


def _encode_labels(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    label_values = np.unique(labels)
    label_to_index = {
        label: index
        for index, label in enumerate(label_values.tolist())
    }
    encoded_labels = np.asarray(
        [label_to_index[label] for label in labels.tolist()],
        dtype=np.int64,
    )

    return encoded_labels, label_values


def load_dataset(
    subjects: list[int] | None = None,
) -> EEGDataset:
    dataset = BNCI2014_001()

    if subjects is None:
        subjects = list(dataset.subject_list)

    paradigm = MotorImagery()
    X, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=subjects,
    )
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            "Expected MOABB EEG shape "
            f"(trials, channels, samples), got {X.shape}"
        )

    if len(X) != len(labels) or len(X) != len(metadata):
        raise ValueError(
            "Inconsistent MOABB trial, label, and metadata lengths."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains NaN or infinite EEG values."
        )

    y, label_values = _encode_labels(labels)

    if "subject" not in metadata.columns:
        raise ValueError(
            "MOABB metadata is missing the 'subject' column."
        )

    trial_subjects = metadata["subject"].to_numpy(
        dtype=np.int64
    )
    sessions = (
        metadata["session"].astype(str).to_numpy()
        if "session" in metadata.columns
        else None
    )
    runs = (
        metadata["run"].astype(str).to_numpy()
        if "run" in metadata.columns
        else None
    )

    requested_subjects = sorted(int(subject) for subject in subjects)
    returned_subjects = sorted(
        np.unique(trial_subjects).astype(int).tolist()
    )

    if returned_subjects != requested_subjects:
        raise ValueError(
            f"Requested subjects {requested_subjects}, "
            f"but MOABB returned {returned_subjects}."
        )

    return EEGDataset(
        X=X,
        y=y,
        subjects=trial_subjects,
        sessions=sessions,
        runs=runs,
        sampling_rate=250.0,
        label_values=label_values,
    )