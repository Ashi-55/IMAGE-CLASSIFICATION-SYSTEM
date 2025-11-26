# Image Classification (PyTorch CNN)

Production-ready image classification project using a simple, strong CNN in PyTorch. Includes dataset layout, augmentation/normalization, training with metrics and checkpointing, and single-image inference.

## Features
- CNN model with BatchNorm, Dropout, AdaptiveAvgPool
- Folder-based dataset via `ImageFolder`
- Data augmentation (random crop/flip/jitter) and normalization
- Training with accuracy and loss, best/last checkpoints
- Single-image prediction CLI with top‑k outputs
- Minimal tests to sanity-check model and transforms

## Project Structure
```
requirements.txt
src/
  models/
    cnn.py
  data/
    transforms.py
    loader.py
  utils/
    checkpoint.py
    metrics.py
  train.py
  infer.py
  tests/
    smoke_test.py
data/
  train/  # put class subfolders here
  val/    # put class subfolders here
checkpoints/  # created during training
```

## Setup
- Python 3.11+ and a working PyTorch install (GPU optional)
- Install dependencies:
```
python -m pip install -r requirements.txt
```
- Optional quick test:
```
python -m src.tests.smoke_test
```

## Dataset Layout
Organize images by class under `train` and `val`:
```
data/
  train/
    cats/
      img_001.jpg
      ...
    dogs/
      img_002.jpg
      ...
  val/
    cats/
      img_101.jpg
    dogs/
      img_102.jpg
```
Any number of classes is supported; folder names become class labels.

## Train
Run training with defaults:
```
python src/train.py --data-dir data --epochs 10 --batch-size 64 --lr 0.001 --image-size 224 --num-workers 4 --checkpoint-dir checkpoints
```
Key flags:
- `--data-dir`: root containing `train/` and `val/`
- `--epochs`: number of training epochs
- `--batch-size`: images per batch
- `--lr`: learning rate
- `--weight-decay`: L2 regularization (default `1e-4`)
- `--image-size`: input size (default `224`)
- `--num-workers`: data loader workers (Windows users may prefer `0–4`)
- `--checkpoint-dir`: output directory for checkpoints

Outputs per epoch:
- `train_loss`, `train_acc`, `val_loss`, `val_acc`
- Checkpoints: `checkpoints/last.pt` and best model `checkpoints/best.pt`

## Predict (Single Image)
Use a saved checkpoint to classify one image:
```
python src/infer.py --image path/to/image.jpg --checkpoint checkpoints/best.pt --topk 3
```
Prints top‑k class names and probabilities.

## Model & Data
- Model: 3× Conv(3×3) → BN → ReLU → MaxPool, channels 32→64→128, then `AdaptiveAvgPool2d(1)`, `Dropout(0.5)`, `Linear(128 → num_classes)` (`src/models/cnn.py`)
- Augmentation: random resized crop, horizontal flip, color jitter for train; resize+center crop for val (`src/data/transforms.py`)
- Normalization: ImageNet mean/std
- Dataloaders: `ImageFolder(train)`, `ImageFolder(val)` with separate transforms (`src/data/loader.py`)

## Checkpoints & Metadata
- Saved with model/optimizer states and metadata: `class_to_idx`, `image_size`, `mean`, `std`, `num_classes` (`src/utils/checkpoint.py`)
- Loading rebuilds the model architecture automatically for inference

## Testing
Basic smoke tests (optional):
```
python -m src.tests.smoke_test
```

## Extending
- Increase model capacity (channels, blocks) for larger datasets
- Add mixed precision (`torch.cuda.amp`) and learning rate scheduling
- Integrate TensorBoard/Weights & Biases for richer logging
- Add reproducibility controls (seeding, deterministic flags)

## Notes
- GPU usage is automatic when available; otherwise CPU is used
- Windows users: set `--num-workers` conservatively if you see DataLoader issues
