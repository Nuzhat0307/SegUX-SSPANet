"""
Training pipeline for SegUX-SSPANet.

Datasets are intentionally separate.

1. BraTS
   -> trains U-Net segmentation model

2. Figshare
   -> trains SegUX-SSPANet classification model
   -> U-Net generates segmentation guidance for each Figshare image

Classes:
0 = glioma
1 = meningioma
2 = pituitary
3 = no_tumor

Important:
The Figshare and BraTS datasets are NOT paired by index.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from loguru import logger

from app.core.config import settings

from ml.models.segux_sspanet import SegUXSSPANet
from ml.models.unet import UNet

from ml.utils.losses import BCEDiceLoss

from ml.utils.metrics import (
    classification_metrics,
    segmentation_metrics,
)

from ml.data.dataset import (
    FigshareDataset,
    BratsSegmentationDataset,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train SegUX-SSPANet"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Classification training epochs",
    )

    parser.add_argument(
        "--seg_epochs",
        type=int,
        default=20,
        help="U-Net segmentation training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seg_lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--figshare_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--brats_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="ml/checkpoints",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# SEGMENTATION TRAINING
# ============================================================

def train_segmentation_epoch(
    segmentor,
    loader,
    optimizer,
    device,
    checkpoint_path="ml/checkpoints/segmentation_recovery.pth",
):
    """
    Train segmentation for one epoch.

    Saves a recovery checkpoint every 100 batches so that
    progress is not completely lost if the laptop restarts.
    """

    segmentor.train()

    criterion = BCEDiceLoss()

    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = segmentor(images)

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # ------------------------------------------------
        # Recovery checkpoint
        # ------------------------------------------------

        if (batch_idx + 1) % 100 == 0:

            torch.save(
                {
                    "batch": batch_idx + 1,
                    "model_state_dict": segmentor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )

            logger.info(
                f"Recovery checkpoint saved at "
                f"SEG Batch {batch_idx + 1}"
            )

        # ------------------------------------------------
        # Progress
        # ------------------------------------------------

        if batch_idx % 50 == 0:

            logger.info(
                f"SEG Batch "
                f"{batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return {
        "loss": total_loss / max(
            len(loader),
            1
        )
    }


# ============================================================
# SEGMENTATION VALIDATION
# ============================================================

def validate_segmentation(
    segmentor,
    loader,
    device,
):

    segmentor.eval()

    dice_scores = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = segmentor(images)

            predicted_masks = torch.sigmoid(
                logits
            )

            for i in range(
                masks.size(0)
            ):

                metric = segmentation_metrics(
                    predicted_masks[i]
                    .cpu()
                    .numpy(),

                    masks[i]
                    .cpu()
                    .numpy(),
                )

                dice_scores.append(
                    metric["dice"]
                )

    return {
        "avg_dice": (
            float(np.mean(dice_scores))
            if dice_scores
            else 0.0
        )
    }


# ============================================================
# GENERATE SEGMENTATION GUIDANCE
# ============================================================

def generate_segmentation_guidance(
    segmentor,
    images,
    device,
):
    """
    Generate segmentation masks for Figshare images.

    Figshare images:
        [B, 3, 224, 224]

    U-Net requires:
        [B, 1, 256, 256]

    Therefore:
        1. Convert RGB-like 3-channel image to grayscale
        2. Resize to 256x256
        3. Run U-Net
        4. Resize prediction back to 224x224
    """

    # Convert 3-channel image to grayscale

    grayscale = images.mean(
        dim=1,
        keepdim=True,
    )

    # U-Net input size

    grayscale = torch.nn.functional.interpolate(
        grayscale,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():

        segmentation_logits = segmentor(
            grayscale
        )

        segmentation_mask = torch.sigmoid(
            segmentation_logits
        )

        # Resize mask to classifier image size

        segmentation_mask = (
            torch.nn.functional.interpolate(
                segmentation_mask,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

    return segmentation_mask.clamp(
        0.0,
        1.0,
    )


# ============================================================
# CLASSIFICATION TRAINING
# ============================================================

def train_classification_epoch(
    classifier,
    segmentor,
    loader,
    optimizer,
    device,
):

    classifier.train()

    # U-Net is already trained.
    # It is used only to generate guidance.

    segmentor.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    correct = 0

    total = 0

    for batch_idx, (
        images,
        labels,
    ) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # Generate segmentation guidance
        # ----------------------------------------------------

        segmentation_guidance = (
            generate_segmentation_guidance(
                segmentor,
                images,
                device,
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        optimizer.zero_grad()

        logits = classifier(
            images,
            segmentation_guidance,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if batch_idx % 50 == 0:

            logger.info(
                f"CLS Batch "
                f"{batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return {
        "loss": total_loss / max(
            len(loader),
            1,
        ),

        "accuracy": correct / max(
            total,
            1,
        ),
    }


# ============================================================
# CLASSIFICATION VALIDATION
# ============================================================

def validate_classification(
    classifier,
    segmentor,
    loader,
    device,
):

    classifier.eval()

    segmentor.eval()

    all_preds = []

    all_labels = []

    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Generate segmentation guidance

            segmentation_guidance = (
                generate_segmentation_guidance(
                    segmentor,
                    images,
                    device,
                )
            )

            # Classification

            logits = classifier(
                images,
                segmentation_guidance,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = probs.argmax(
                dim=1,
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    metrics = classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    logger.info(
        f"Training on device: {device}"
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    # ========================================================
    # MODELS
    # ========================================================

    classifier = SegUXSSPANet(
        num_classes=settings.NUM_CLASSES,
        backbone="resnet50",
        pretrained=True,
    ).to(device)

    segmentor = UNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    # ========================================================
    # OPTIMIZERS
    # ========================================================

    cls_optimizer = optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    seg_optimizer = optim.AdamW(
        segmentor.parameters(),
        lr=args.seg_lr,
        weight_decay=1e-4,
    )

    # ========================================================
    # DATASETS
    # ========================================================

    # --------------------------------------------------------
    # FIGSHARE
    # --------------------------------------------------------

    train_cls_ds = FigshareDataset(
        data_dir=args.figshare_dir,
        split="train",
        image_size=settings.IMAGE_SIZE,
        train_samples=1000,
        val_samples=200,
    )

    val_cls_ds = FigshareDataset(
        data_dir=args.figshare_dir,
        split="val",
        image_size=settings.IMAGE_SIZE,
        train_samples=1000,
        val_samples=200,
    )

    train_cls_loader = DataLoader(
        train_cls_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_cls_loader = DataLoader(
        val_cls_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # BRATS
    # --------------------------------------------------------

    train_seg_ds = BratsSegmentationDataset(
        data_dir=args.brats_dir,
        split="train",
        image_size=256,
        train_samples=500,
        val_samples=200,
    )

    val_seg_ds = BratsSegmentationDataset(
        data_dir=args.brats_dir,
        split="val",
        image_size=256,
        train_samples=500,
        val_samples=200,
    )

    train_seg_loader = DataLoader(
        train_seg_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_seg_loader = DataLoader(
        val_seg_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    logger.info(
        f"Figshare train: "
        f"{len(train_cls_ds)}"
    )

    logger.info(
        f"Figshare val: "
        f"{len(val_cls_ds)}"
    )

    logger.info(
        f"BraTS train: "
        f"{len(train_seg_ds)}"
    )

    logger.info(
        f"BraTS val: "
        f"{len(val_seg_ds)}"
    )

    # ========================================================
    # STAGE 1
    # TRAIN U-NET
    # ========================================================

    logger.info(
        "\n"
        + "=" * 60
    )

    logger.info(
        "STAGE 1: TRAINING U-NET"
    )

    logger.info(
        "=" * 60
    )

    best_dice = 0.0

    best_segmentor_path = os.path.join(
        args.checkpoint_dir,
        "segux_unet_best.pth",
    )

    seg_scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            seg_optimizer,
            T_max=args.seg_epochs,
        )
    )

    for epoch in range(
        args.seg_epochs
    ):

        logger.info(
            f"\nSegmentation Epoch "
            f"{epoch + 1}/"
            f"{args.seg_epochs}"
        )

        seg_train = train_segmentation_epoch(
            segmentor,
            train_seg_loader,
            seg_optimizer,
            device,
            checkpoint_path=os.path.join(
                args.checkpoint_dir,
                "segmentation_recovery.pth",
            ),
        )

        seg_val = validate_segmentation(
            segmentor,
            val_seg_loader,
            device,
        )

        logger.info(
            f"SEG Train Loss: "
            f"{seg_train['loss']:.4f}"
        )

        logger.info(
            f"SEG Validation Dice: "
            f"{seg_val['avg_dice']:.4f}"
        )

        if (
            seg_val["avg_dice"]
            > best_dice
        ):

            best_dice = (
                seg_val["avg_dice"]
            )

            torch.save(
                segmentor.state_dict(),
                best_segmentor_path,
            )

            logger.info(
                "Best U-Net saved."
            )

        seg_scheduler.step()

    # --------------------------------------------------------
    # Load best U-Net
    # --------------------------------------------------------

    if os.path.exists(
        best_segmentor_path
    ):

        segmentor.load_state_dict(
            torch.load(
                best_segmentor_path,
                map_location=device,
                weights_only=True,
            )
        )

        logger.info(
            f"Loaded best U-Net: "
            f"{best_segmentor_path}"
        )

    # Freeze U-Net during classification

    for parameter in (
        segmentor.parameters()
    ):

        parameter.requires_grad = False

    segmentor.eval()

    # ========================================================
    # STAGE 2
    # TRAIN CLASSIFIER
    # ========================================================

    logger.info(
        "\n"
        + "=" * 60
    )

    logger.info(
        "STAGE 2: TRAINING SEGUX-SSPANET"
    )

    logger.info(
        "=" * 60
    )

    cls_scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            cls_optimizer,
            T_max=args.epochs,
        )
    )

    best_accuracy = 0.0

    best_epoch = 0

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "segux_sspanet_best.pth",
    )

    # ========================================================
    # CLASSIFICATION LOOP
    # ========================================================

    for epoch in range(
        args.epochs
    ):

        logger.info(
            "\n"
            + "=" * 60
        )

        logger.info(
            f"Classification Epoch "
            f"{epoch + 1}/"
            f"{args.epochs}"
        )

        logger.info(
            "=" * 60
        )

        # ----------------------------------------------------
        # Train classifier
        # ----------------------------------------------------

        cls_train = (
            train_classification_epoch(
                classifier,
                segmentor,
                train_cls_loader,
                cls_optimizer,
                device,
            )
        )

        logger.info(
            f"CLS Train | "
            f"Loss: "
            f"{cls_train['loss']:.4f} | "
            f"Accuracy: "
            f"{cls_train['accuracy']:.4f}"
        )

        # ----------------------------------------------------
        # Validate classifier
        # ----------------------------------------------------

        cls_val = validate_classification(
            classifier,
            segmentor,
            val_cls_loader,
            device,
        )

        logger.info(
            f"CLS Validation | "
            f"Accuracy: "
            f"{cls_val['accuracy']:.4f} | "
            f"F1: "
            f"{cls_val['f1_macro']:.4f}"
        )

        # ----------------------------------------------------
        # Save best complete model
        # ----------------------------------------------------

        if (
            cls_val["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                cls_val["accuracy"]
            )

            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,

                    "classifier":
                        classifier.state_dict(),

                    "segmentor":
                        segmentor.state_dict(),

                    "val_metrics": {
                        **cls_val,
                        "seg_dice": best_dice,
                    },

                    "model_version":
                        settings.MODEL_VERSION,
                },
                checkpoint_path,
            )

            logger.info(
                f"New best complete model saved: "
                f"{checkpoint_path}"
            )

        cls_scheduler.step()

    # ========================================================
    # COMPLETE
    # ========================================================

    logger.info(
        "\n"
        + "=" * 60
    )

    logger.info(
        "TRAINING COMPLETE"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Best segmentation Dice: "
        f"{best_dice:.4f}"
    )

    logger.info(
        f"Best classification accuracy: "
        f"{best_accuracy:.4f}"
    )

    logger.info(
        f"Best classification epoch: "
        f"{best_epoch}"
    )

    logger.info(
        f"Final checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()