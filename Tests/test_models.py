import torch

from models import ATCNet, EEGNet, ShallowConvNet


def test_eegnet():
    model = EEGNet()
    inputs = torch.randn(2, 1, 22, 1001)

    outputs = model(inputs)

    assert outputs.shape == (2, 4)


def test_shallow_convnet():
    model = ShallowConvNet()
    inputs = torch.randn(2, 1, 22, 1001)

    outputs = model(inputs)

    assert outputs.shape == (2, 4)


def test_atcnet():
    model = ATCNet()
    inputs = torch.randn(2, 1, 22, 1001)

    outputs = model(inputs)

    assert outputs.shape == (2, 4)