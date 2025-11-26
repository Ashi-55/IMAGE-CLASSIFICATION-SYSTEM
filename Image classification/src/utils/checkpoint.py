import os
import torch
from typing import Dict

from ..models.cnn import SimpleCNN


def save_checkpoint(model, optimizer, epoch: int, class_to_idx: Dict[str, int], checkpoint_dir: str, image_size: int, mean, std, best: bool):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "best.pt" if best else "last.pt")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "arch": "SimpleCNN",
        "num_classes": len(class_to_idx),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    num_classes = payload.get("num_classes")
    model = SimpleCNN(num_classes)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    class_to_idx = payload.get("class_to_idx")
    meta = {
        "image_size": payload.get("image_size"),
        "mean": payload.get("mean"),
        "std": payload.get("std"),
        "arch": payload.get("arch"),
    }
    return model, class_to_idx, meta
