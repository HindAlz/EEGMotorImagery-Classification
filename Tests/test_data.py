import numpy as np

from data import load_dataset
from preprocessing import ChannelStandardizer


def test_load_single_subject():
    dataset = load_dataset(subjects=[1])

    assert dataset.X.ndim == 3
    assert dataset.y.ndim == 1
    assert dataset.subjects.ndim == 1

    assert len(dataset.X) == len(dataset.y)
    assert len(dataset.X) == len(dataset.subjects)

    assert np.all(dataset.subjects == 1)

    assert dataset.n_channels == dataset.X.shape[1]
    assert dataset.n_samples == dataset.X.shape[2]

    assert dataset.n_classes == 4

    assert np.isfinite(dataset.X).all()


def test_labels_are_zero_indexed():
    dataset = load_dataset(subjects=[1])

    unique_labels = np.unique(dataset.y)

    assert np.array_equal(
        unique_labels,
        np.arange(dataset.n_classes),
    )


def test_metadata_lengths_match_trials():
    dataset = load_dataset(subjects=[1])

    assert len(dataset.subjects) == dataset.n_trials

    if dataset.sessions is not None:
        assert len(dataset.sessions) == dataset.n_trials

    if dataset.runs is not None:
        assert len(dataset.runs) == dataset.n_trials


def test_channel_standardizer():
    rng = np.random.default_rng(42)

    X_train = rng.normal(
        loc=10,
        scale=2,
        size=(20, 22, 100),
    ).astype(np.float32)

    X_test = rng.normal(
        loc=100,
        scale=5,
        size=(5, 22, 100),
    ).astype(np.float32)

    standardizer = ChannelStandardizer()

    X_train_scaled = standardizer.fit_transform(
        X_train
    )

    X_test_scaled = standardizer.transform(
        X_test
    )

    # Training data should be approximately zero-mean per channel.
    train_channel_means = X_train_scaled.mean(
        axis=(0, 2)
    )

    assert np.allclose(
        train_channel_means,
        0,
        atol=1e-5,
    )

    # The test distribution should NOT become zero-centered,
    # because the scaler was not fitted on test data.
    assert abs(X_test_scaled.mean()) > 1