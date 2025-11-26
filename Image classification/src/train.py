import argparse
import torch
from torch import nn, optim

from src.models.cnn import SimpleCNN
from src.data.loader import create_dataloaders
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.metrics import accuracy
from src.utils.checkpoint import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(outputs, targets) * bs
            total_count += bs
    return total_loss / total_count, total_acc / total_count


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = create_dataloaders(args.data_dir, args.image_size, args.batch_size, args.num_workers)
    model = SimpleCNN(num_classes=len(class_to_idx)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(outputs, targets) * bs
            total_count += bs
        train_loss = total_loss / total_count
        train_acc = total_acc / total_count
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        save_checkpoint(model, optimizer, epoch, class_to_idx, args.checkpoint_dir, args.image_size, IMAGENET_MEAN, IMAGENET_STD, best=False)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, class_to_idx, args.checkpoint_dir, args.image_size, IMAGENET_MEAN, IMAGENET_STD, best=True)
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    print(f"best_val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
