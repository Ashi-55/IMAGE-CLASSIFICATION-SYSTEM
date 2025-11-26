import os
from typing import Tuple, Dict

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(data_dir: str, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_ds = ImageFolder(train_dir, transform=get_train_transforms(image_size))
    val_ds = ImageFolder(val_dir, transform=get_val_transforms(image_size))
    class_to_idx = train_ds.class_to_idx
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_to_idx
