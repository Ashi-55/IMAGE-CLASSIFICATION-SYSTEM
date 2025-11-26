import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--topk", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_to_idx, meta = load_checkpoint(args.checkpoint, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    size = meta["image_size"]
    mean = meta["mean"]
    std = meta["std"]
    tfm = transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        values, indices = probs.topk(args.topk, dim=1)
    for i in range(args.topk):
        cls = idx_to_class[indices[0, i].item()]
        p = values[0, i].item()
        print(f"{i+1}: {cls} {p:.4f}")


if __name__ == "__main__":
    main()
