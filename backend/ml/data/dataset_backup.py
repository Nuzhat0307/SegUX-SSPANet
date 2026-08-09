"""
Dataset module for brain tumor MRI data.

Supports:
1. Figshare Brain Tumor Dataset (3 classes + no tumor)
2. BraTS (Brain Tumor Segmentation) — for segmentation-guided learning

Assumptions:
- Figshare images are organized in class-named folders
- BraTS provides NIfTI volumes with segmentation masks
- For BraTS, we extract 2D slices and convert to PNG for the framework
"""
import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Tuple, Callable
from loguru import logger

from app.core.config import settings


TUMOR_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3,
}
IDX_TO_TUMOR_CLASS = {v: k for k, v in TUMOR_CLASS_TO_IDX.items()}


class FigshareDataset(Dataset):
    """
    Figshare Brain Tumor Dataset loader.
    Expected structure:
        data_dir/
            glioma/
                img1.jpg
                ...
            meningioma/
                ...
            pituitary/
                ...
            no_tumor/
                ...
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir or os.path.join(settings.DATASET_DIR, "figshare")
        self.split = split
        self.transform = transform
        self.image_size = image_size

        self.samples = []
        for class_name, class_idx in TUMOR_CLASS_TO_IDX.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")) +
                          glob.glob(os.path.join(class_dir, "*.png")))
            for f in files:
                self.samples.append((f, class_idx))

        # Split
        np.random.seed(seed)
        np.random.shuffle(self.samples)
        n = len(self.samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        elif split == "test":
            self.samples = self.samples[val_end:]

        logger.info(f"FigshareDataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        image = image.resize((self.image_size, self.image_size))
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Convert to 3-channel for pretrained backbones
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).repeat(3, 1, 1)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class BratsSegmentationDataset(Dataset):
    """
    BraTS segmentation dataset (2D slices).
    Expected structure (after preprocessing):
        data_dir/
            images/
                case_001_slice_050.png
                ...
            masks/
                case_001_slice_050_mask.png
                ...
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.data_dir = data_dir or os.path.join(settings.DATASET_DIR, "brats_2d")
        self.split = split
        self.transform = transform
        self.image_size = image_size

        image_dir = os.path.join(self.data_dir, "images")
        mask_dir = os.path.join(self.data_dir, "masks")

        self.samples = []
        if os.path.isdir(image_dir):
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                                glob.glob(os.path.join(image_dir, "*.jpg")))
            for img_path in image_files:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mask_dir, f"{basename}_mask.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, f"{basename}.png")
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        np.random.seed(seed)
        np.random.shuffle(self.samples)
        n = len(self.samples)
        split_idx = int(n * train_ratio)

        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        logger.info(f"BratsSegmentationDataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("L").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))

        img_array = np.array(image, dtype=np.float32) / 255.0
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor


class MultiTaskDataset(Dataset):
    """
    Combined dataset for multi-task learning (classification + segmentation).
    Uses Figshare for classification labels and BraTS for segmentation masks.
    When segmentation masks are unavailable, returns zero masks (classification-only).
    """

    def __init__(
        self,
        figshare_dir: str = None,
        brats_dir: str = None,
        split: str = "train",
        image_size: int = 224,
        seg_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.classification_dataset = FigshareDataset(
            data_dir=figshare_dir, split=split, transform=transform, image_size=image_size,
        )
        self.segmentation_dataset = BratsSegmentationDataset(
            data_dir=brats_dir, split=split, transform=transform, image_size=seg_size,
        )
        self.seg_size = seg_size

    def __len__(self) -> int:
        return len(self.classification_dataset)

    def __getitem__(self, idx: int):
        img, label = self.classification_dataset[idx]
        # If we have segmentation data, use a random sample
        if len(self.segmentation_dataset) > 0:
            seg_idx = idx % len(self.segmentation_dataset)
            _, mask = self.segmentation_dataset[seg_idx]
        else:
            mask = torch.zeros(1, self.seg_size, self.seg_size)
        return img, label, mask


def get_dataloaders(
    batch_size: int = 16,
    image_size: int = 224,
    figshare_dir: str = None,
    brats_dir: str = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    train_ds = FigshareDataset(data_dir=figshare_dir, split="train", image_size=image_size)
    val_ds = FigshareDataset(data_dir=figshare_dir, split="val", image_size=image_size)
    test_ds = FigshareDataset(data_dir=figshare_dir, split="test", image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
