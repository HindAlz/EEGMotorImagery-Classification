from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from data import load_dataset


def parse_subjects(
    value: str | None,
) -> list[int] | None:
    if value is None:
        return None

    return [
        int(subject.strip())
        for subject in value.split(",")
        if subject.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject IDs; default: all",
    )
    args = parser.parse_args()

    dataset = load_dataset(
        parse_subjects(args.subjects)
    )
    unique_subjects = np.unique(dataset.subjects)

    sessions = (
        sorted(np.unique(dataset.sessions).tolist())
        if dataset.sessions is not None
        else "not supplied"
    )
    runs = (
        sorted(np.unique(dataset.runs).tolist())
        if dataset.runs is not None
        else "not supplied"
    )

    print("Dataset: BNCI2014-001 via MOABB")
    print("=" * 60)
    print(f"Trials:          {dataset.n_trials}")
    print(
        f"Subjects:        {len(unique_subjects)} "
        f"-> {unique_subjects.tolist()}"
    )
    print(f"Classes:         {dataset.n_classes}")
    print(f"Channels:        {dataset.n_channels}")
    print(f"Time points:     {dataset.n_samples}")
    print(
        f"Sampling rate:   "
        f"{dataset.sampling_rate:g} Hz"
    )
    print(
        f"Original labels: "
        f"{dataset.label_values.tolist()}"
    )
    print(f"Sessions:        {sessions}")
    print(f"Runs:            {runs}")
    print()

    for subject in unique_subjects:
        subject_mask = dataset.subjects == subject
        class_counts = Counter(
            dataset.y[subject_mask].tolist()
        )
        counts_text = ", ".join(
            f"class {class_index}: "
            f"{class_counts.get(class_index, 0)}"
            for class_index in range(dataset.n_classes)
        )

        print(
            f"Subject {int(subject)}: "
            f"{subject_mask.sum()} trials | "
            f"{counts_text}"
        )

    assert (
        dataset.n_trials
        == len(dataset.y)
        == len(dataset.subjects)
    )

    if dataset.sessions is not None:
        assert len(dataset.sessions) == dataset.n_trials

    if dataset.runs is not None:
        assert len(dataset.runs) == dataset.n_trials

    assert np.isfinite(dataset.X).all()

    print("\nAll basic MOABB dataset checks passed.")


if __name__ == "__main__":
    main()