import torch
from PIL import Image
import numpy as np

from src.models.cnn import SimpleCNN
from src.data.transforms import get_train_transforms


def test_model_forward():
    model = SimpleCNN(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 5)


def test_transforms():
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    tfm = get_train_transforms(224)
    t = tfm(img)
    assert t.shape == (3, 224, 224)


if __name__ == "__main__":
    test_model_forward()
    test_transforms()
    print("ok")
