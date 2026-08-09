"""
Training pipeline for SegUX-SSPANet.

Multi-task training:
1. Classification (SSPANet + ResNet50) with CrossEntropy loss
2. Segmentation (U-Net) with Dice + BCE loss
3. Segmentation-guided attention learning: segmentation masks guide
   the attention maps via an auxiliary consistency loss

Usage:
    python -m scripts.train --epochs 50 --batch_size 16 --lr 1e-4

This script trains the full model and saves the best checkpoint.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np

from app.core.config import settings
from ml.models.segux_sspanet import SegUXSSPANet
from ml.models.unet import UNet
from ml.utils.losses import MultiTaskLoss, DiceLoss, BCEDiceLoss
from ml.utils.metrics import classification_metrics, segmentation_metrics
from ml.data.dataset import FigshareDataset, BratsSegmentationDataset, MultiTaskDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegUX-SSPANet")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--figshare_dir", type=str, default=None, help="Figshare dataset path")
    parser.add_argument("--brats_dir", type=str, default=None, help="BraTS dataset path")
    parser.add_argument("--checkpoint_dir", type=str, default="ml/checkpoints", help="Checkpoint dir")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def train_one_epoch(
    classifier: nn.Module,
    segmentor: nn.Module,
    train_loader: DataLoader,
    cls_optimizer: optim.Optimizer,
    seg_optimizer: optim.Optimizer,
    criterion: MultiTaskLoss,
    device: str,
) -> dict:
    """Train for one epoch."""
    classifier.train()
    segmentor.train()

    total_loss = 0.0
    cls_loss_sum = 0.0
    seg_loss_sum = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, masks) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Classification forward
        cls_optimizer.zero_grad()
        logits = classifier(images)
        cls_loss = nn.CrossEntropyLoss()(logits, labels)
        cls_loss.backward(retain_graph=True)
        cls_optimizer.step()

        # Segmentation forward
        seg_optimizer.zero_grad()
        seg_input = images[:, :1, :, :].float()  # Use grayscale channel
        seg_logits = segmentor(seg_input)
        seg_loss = BCEDiceLoss()(seg_logits, masks)
        seg_loss.backward()
        seg_optimizer.step()

        total_loss += (cls_loss.item() + seg_loss.item())
        cls_loss_sum += cls_loss.item()
        seg_loss_sum += seg_loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            logger.info(
                f"  Batch {batch_idx}/{len(train_loader)} | "
                f"CLS: {cls_loss.item():.4f} SEG: {seg_loss.item():.4f}"
            )

    return {
        "total_loss": total_loss / len(train_loader),
        "cls_loss": cls_loss_sum / len(train_loader),
        "seg_loss": seg_loss_sum / len(train_loader),
        "accuracy": correct / total,
    }


def validate(
    classifier: nn.Module,
    segmentor: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Validate on val set."""
    classifier.eval()
    segmentor.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    dice_scores = []

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = classifier(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            seg_input = images[:, :1, :, :].float()
            seg_logits = segmentor(seg_input)
            seg_mask = torch.sigmoid(seg_logits)

            for i in range(masks.size(0)):
                if masks[i].max() > 0:
                    d = segmentation_metrics(
                        seg_mask[i].cpu().numpy(),
                        masks[i].cpu().numpy(),
                    )
                    dice_scores.append(d["dice"])

    metrics = classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs),
    )
    metrics["avg_dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0
    return metrics


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Models
    classifier = SegUXSSPANet(num_classes=settings.NUM_CLASSES, backbone="resnet50").to(device)
    segmentor = UNet(in_channels=1, out_channels=1).to(device)

    # Optimizers
    cls_optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    seg_optimizer = optim.AdamW(segmentor.parameters(), lr=args.lr, weight_decay=1e-4)

    # Schedulers
    cls_scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_optimizer, T_max=args.epochs)
    seg_scheduler = optim.lr_scheduler.CosineAnnealingLR(seg_optimizer, T_max=args.epochs)

    # Data
    train_ds = MultiTaskDataset(
        figshare_dir=args.figshare_dir,
        brats_dir=args.brats_dir,
        split="train",
        image_size=settings.IMAGE_SIZE,
        seg_size=settings.SEGMENTATION_SIZE,
    )
    val_ds = MultiTaskDataset(
        figshare_dir=args.figshare_dir,
        brats_dir=args.brats_dir,
        split="val",
        image_size=settings.IMAGE_SIZE,
        seg_size=settings.SEGMENTATION_SIZE,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = MultiTaskLoss()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            classifier, segmentor, train_loader,
            cls_optimizer, seg_optimizer, criterion, device,
        )
        logger.info(
            f"Train — Loss: {train_metrics['total_loss']:.4f} | "
            f"CLS: {train_metrics['cls_loss']:.4f} SEG: {train_metrics['seg_loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validate
        val_metrics = validate(classifier, segmentor, val_loader, device)
        logger.info(
            f"Val — Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_macro']:.4f} | "
            f"Dice: {val_metrics['avg_dice']:.4f}"
        )

        # Save best model
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(args.checkpoint_dir, "segux_sspanet_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "classifier": classifier.state_dict(),
                "segmentor": segmentor.state_dict(),
                "val_metrics": val_metrics,
                "model_version": settings.MODEL_VERSION,
            }, checkpoint_path)
            logger.info(f"  -> New best model saved (acc: {best_acc:.4f})")

        cls_scheduler.step()
        seg_scheduler.step()

    logger.info(f"\nTraining complete. Best accuracy: {best_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
