import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

# Ensure project root on sys.path when running as a script (python scripts/train.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_densenet import get_model, CLASSES
from app.preprocessing import train_transforms, val_transforms


def accuracy(output, target) -> float: # NOTE: AI Generated
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


def save_checkpoint(state: Dict[str, Any], is_best: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / 'last.pt')
    if is_best:
        torch.save(state, out_dir / 'best.pt')


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, log_interval: int) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = len(loader.dataset)
    seen = 0
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        seen += batch_size
        running_loss += loss.item() * batch_size
        batch_acc = accuracy(logits.detach(), labels)
        running_acc += batch_acc * batch_size
        if log_interval and (batch_idx % log_interval == 0 or seen == total):
            avg_loss = running_loss / seen
            avg_acc = running_acc / seen
            print(f"Epoch {epoch} [{seen}/{total} ({100.0*seen/total:.1f}%)] loss={avg_loss:.4f} acc={avg_acc:.3f} (batch_acc={batch_acc:.3f})")
    return running_loss / total, running_acc / total


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_total += loss.item() * images.size(0)
            acc_total += accuracy(logits, labels) * images.size(0)
    return loss_total / len(loader.dataset), acc_total / len(loader.dataset)


def parse_args():
    ap = argparse.ArgumentParser(description="Train DenseNet121 tumor classifier")
    ap.add_argument('--data-root', type=Path, required=True, help='Root with train/ and val/ folders (ImageFolder)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--out-dir', type=Path, default=Path('models/weights'))
    ap.add_argument('--no-pretrained', action='store_true', help='Disable ImageNet pretrained weights')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--log-interval', type=int, default=50, help='Batches between progress logs (0 to disable)')
    ap.add_argument('--brain-crop', action='store_true', default=True, help='Apply heuristic brain crop before resizing')
    ap.add_argument('--corner-mask-prob', type=float, default=0.5, help='Probability to apply random corner masking (training only)')
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_dir = args.data_root / 'train'
    val_dir = args.data_root / 'val'
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit('Expected train/ and val/ directories under data-root')

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms(brain_crop=args.brain_crop, corner_mask_prob=args.corner_mask_prob))
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms(brain_crop=args.brain_crop))

    if train_ds.classes != list(CLASSES):
        print(f"[WARN] Dataset classes {train_ds.classes} differ from hardcoded {CLASSES}. Using dataset ordering for training and saving to checkpoint.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.log_interval)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'classes': train_ds.classes,
        }, is_best=is_best, out_dir=args.out_dir)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}{' *' if is_best else ''}")

    print(f"Training complete. Best val acc: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
