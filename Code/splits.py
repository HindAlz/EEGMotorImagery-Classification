from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

@dataclass(frozen=True)
class SubjectSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    test_subject: int

def make_loso_splits(y, subjects, validation_strategy="next_subject"):
    y = np.asarray(y)
    subjects = np.asarray(subjects)

    if len(y) != len(subjects):
        raise ValueError("y/subjects length mismatch")

    splitter = LeaveOneGroupOut()
    dummy_features = np.zeros((len(y), 1))
    splits = []

    for fold_index, (dev_idx, test_idx) in enumerate(
        splitter.split(dummy_features, y, groups=subjects)
    ):
        test_subject = int(np.unique(subjects[test_idx])[0])
        dev_subjects = np.sort(np.unique(subjects[dev_idx]))

        if validation_strategy == "next_subject":
            val_subject = int(dev_subjects[fold_index % len(dev_subjects)])
        elif validation_strategy == "highest_subject":
            val_subject = int(dev_subjects[-1])
        else:
            raise ValueError(
                f"Unknown validation strategy: {validation_strategy}"
            )

        val_mask = subjects[dev_idx] == val_subject
        val_idx = dev_idx[val_mask]
        train_idx = dev_idx[~val_mask]

        split = SubjectSplit(
            fold_index + 1,
            train_idx,
            val_idx,
            test_idx,
            tuple(sorted(np.unique(subjects[train_idx]).astype(int).tolist())),
            tuple(sorted(np.unique(subjects[val_idx]).astype(int).tolist())),
            test_subject,
        )

        _validate(split)
        splits.append(split)

    expected_subjects = sorted(
        np.unique(subjects).astype(int).tolist()
    )
    actual_test_subjects = sorted(split.test_subject for split in splits)

    if actual_test_subjects != expected_subjects:
        raise RuntimeError("LOSO coverage mismatch")

    return splits

def _validate(split):
    train_subjects = set(split.train_subjects)
    val_subjects = set(split.val_subjects)
    test_subjects = {split.test_subject}

    has_subject_leakage = (
        train_subjects & val_subjects
        or train_subjects & test_subjects
        or val_subjects & test_subjects
    )
    if has_subject_leakage:
        raise RuntimeError(f"Subject leakage in fold {split.fold}")

    if np.intersect1d(split.train_idx, split.val_idx).size:
        raise RuntimeError("Train/validation overlap")

    if np.intersect1d(split.train_idx, split.test_idx).size:
        raise RuntimeError("Train/test overlap")

    if np.intersect1d(split.val_idx, split.test_idx).size:
        raise RuntimeError("Validation/test overlap")