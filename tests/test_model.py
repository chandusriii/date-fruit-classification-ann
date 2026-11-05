import torch
from src.model import DateFruitANN


def test_output_shape_matches_seven_classes():
    model = DateFruitANN(input_features=34, num_classes=7)
    assert model(torch.randn(8, 34)).shape == (8, 7)


def test_model_has_trainable_parameters():
    model = DateFruitANN()
    assert all(parameter.requires_grad for parameter in model.parameters())
