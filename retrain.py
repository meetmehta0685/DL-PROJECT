"""
Retrain the deployment model with ImageNet weights.

- Starts ResNet18 from ImageNet weights (uses certifi CA bundle so the download works on macOS)
- Builds a patient-grouped validation split from train (the provided val split has only 16 images)
- Uses class-weighted cross-entropy for the ~3:1 Pneumonia:Normal imbalance
- Evaluates the best epoch once on the test split and writes the metrics next to the checkpoint

Usage:
    python retrain.py --data-dir ../chest_xray --epochs 8
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import certifi

os.environ.setdefault('SSL_CERT_FILE', certifi.where())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import ChestXRayDataset, get_transforms
from src.model import get_model
from src.utils import compute_metrics, set_seed


def patient_id(path: str) -> str:
    """Group images of the same patient: person123_bacteria_45 -> person123, IM-0115-0001 -> IM-0115."""
    name = Path(path).stem
    match = re.match(r'(person\d+)_', name)
    if match:
        return match.group(1)
    return re.sub(r'-\d+$', '', name)


def grouped_split(dataset: ChestXRayDataset, val_fraction: float, seed: int):
    """Split indices so no patient appears in both train and validation."""
    groups = {}
    for idx, path in enumerate(dataset.images):
        groups.setdefault(patient_id(path), []).append(idx)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_target = int(len(dataset) * val_fraction)
    val_idx, train_idx = [], []
    for key in keys:
        (val_idx if len(val_idx) < val_target else train_idx).extend(groups[key])
    return train_idx, val_idx


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, probs_all = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc='train' if training else 'eval', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.softmax(outputs, 1)[:, 1].detach().cpu().numpy())
    labels_all, probs_all = np.array(labels_all), np.array(probs_all)
    metrics = compute_metrics(labels_all, (probs_all >= 0.5).astype(int))
    metrics['loss'] = total_loss / len(labels_all)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../chest_xray')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', default='checkpoints/retrained.pt')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_tf, eval_tf = get_transforms(image_size=224)
    full_train_aug = ChestXRayDataset(args.data_dir, 'train', transform=train_tf)
    full_train_eval = ChestXRayDataset(args.data_dir, 'train', transform=eval_tf)
    test_ds = ChestXRayDataset(args.data_dir, 'test', transform=eval_tf)

    train_idx, val_idx = grouped_split(full_train_aug, val_fraction=0.1, seed=42)
    train_ds, val_ds = Subset(full_train_aug, train_idx), Subset(full_train_eval, val_idx)
    print(f'Train {len(train_ds)} | Val {len(val_ds)} (patient-grouped) | Test {len(test_ds)}')

    loader_kw = dict(batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    # Inverse-frequency class weights computed on the training subset
    train_labels = np.array(full_train_aug.labels)[train_idx]
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float32, device=device)
    print(f'Class counts {counts.tolist()} -> weights {class_weights.tolist()}')

    model = get_model('resnet18', num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss, best_epoch = float('inf'), -1
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch}: train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.4f} | "
            f"val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.4f} recall {val_m['recall']:.4f}",
            flush=True,
        )
        if val_m['loss'] < best_loss:
            best_loss, best_epoch = val_m['loss'], epoch
            torch.save({'model_state_dict': model.state_dict()}, args.out)

    model.load_state_dict(torch.load(args.out, map_location=device)['model_state_dict'])
    test_m = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    test_m = {k: float(v) for k, v in test_m.items()}
    test_m['best_epoch'] = best_epoch
    print('TEST', json.dumps(test_m))
    Path(args.out).with_suffix('.json').write_text(json.dumps(test_m, indent=2))


if __name__ == '__main__':
    main()
