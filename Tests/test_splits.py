import numpy as np

from splits import make_loso_splits


def data():
    labels = np.tile(np.repeat(np.arange(4), 3), 9)
    subjects = np.repeat(np.arange(1, 10), 12)
    return labels, subjects


def test_nine_folds():
    labels, subjects = data()
    splits = make_loso_splits(labels, subjects)

    assert len(splits) == 9


def test_each_subject_once():
    labels, subjects = data()
    splits = make_loso_splits(labels, subjects)

    test_subjects = sorted(split.test_subject for split in splits)
    assert test_subjects == list(range(1, 10))


def test_no_leakage():
    labels, subjects = data()

    for split in make_loso_splits(labels, subjects):
        train_subjects = set(subjects[split.train_idx])
        val_subjects = set(subjects[split.val_idx])
        test_subjects = set(subjects[split.test_idx])

        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)
        