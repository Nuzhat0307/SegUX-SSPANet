"""
Dataset loaders for SegUX-SSPANet.

Datasets are intentionally separate.

1. Figshare
   -> classification

2. BraTS 2D
   -> segmentation

The original dataset files are NOT deleted.
Only a limited number of samples are loaded for faster laptop training.
"""

import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple, Callable
from loguru import logger

from app.core.config import settings


# ============================================================
# CLASS LABELS
# ============================================================

TUMOR_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3,
}

IDX_TO_TUMOR_CLASS = {
    v: k for k, v in TUMOR_CLASS_TO_IDX.items()
}


# ============================================================
# FIGSHARE CLASSIFICATION DATASET
# ============================================================

class FigshareDataset(Dataset):
    """
    Figshare Brain Tumor Dataset.

    Used ONLY for classification.

    Classes:
        0 = glioma
        1 = meningioma
        2 = pituitary
        3 = no_tumor

    Laptop-friendly limits:
        Train = 1000 images
        Validation = 200 images

    The original images are NOT deleted.
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        train_samples: int = 1000,
        val_samples: int = 200,
        seed: int = 42,
    ):
        self.data_dir = (
            data_dir
            or os.path.join(
                settings.DATASET_DIR,
                "figshare"
            )
        )

        self.split = split
        self.transform = transform
        self.image_size = image_size

        all_samples = []

        # ----------------------------------------------------
        # Collect images from all classes
        # ----------------------------------------------------

        for class_name, class_idx in TUMOR_CLASS_TO_IDX.items():

            class_dir = os.path.join(
                self.data_dir,
                class_name
            )

            if not os.path.isdir(class_dir):
                continue

            files = sorted(
                glob.glob(
                    os.path.join(
                        class_dir,
                        "*.jpg"
                    )
                )
                +
                glob.glob(
                    os.path.join(
                        class_dir,
                        "*.jpeg"
                    )
                )
                +
                glob.glob(
                    os.path.join(
                        class_dir,
                        "*.png"
                    )
                )
            )

            for file_path in files:

                all_samples.append(
                    (
                        file_path,
                        class_idx
                    )
                )

        # ----------------------------------------------------
        # Shuffle deterministically
        # ----------------------------------------------------

        rng = np.random.default_rng(seed)

        rng.shuffle(all_samples)

        # ----------------------------------------------------
        # Select limited dataset
        # ----------------------------------------------------

        if split == "train":

            self.samples = all_samples[
                :min(
                    train_samples,
                    len(all_samples)
                )
            ]

        elif split == "val":

            val_start = min(
                train_samples,
                len(all_samples)
            )

            val_end = min(
                val_start + val_samples,
                len(all_samples)
            )

            self.samples = all_samples[
                val_start:val_end
            ]

        elif split == "test":

            test_start = min(
                train_samples + val_samples,
                len(all_samples)
            )

            self.samples = all_samples[
                test_start:
            ]

        else:

            raise ValueError(
                f"Invalid split: {split}"
            )

        logger.info(
            f"FigshareDataset [{split}]: "
            f"{len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, int]:

        img_path, label = self.samples[idx]

        image = Image.open(
            img_path
        ).convert("L")

        image = image.resize(
            (
                self.image_size,
                self.image_size
            )
        )

        image_array = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        # Grayscale -> 3 channels
        image_tensor = (
            torch.from_numpy(
                image_array
            )
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )

        if self.transform:

            image_tensor = self.transform(
                image_tensor
            )

        return image_tensor, label


# ============================================================
# BRATS SEGMENTATION DATASET
# ============================================================

class BratsSegmentationDataset(Dataset):
    """
    BraTS 2D segmentation dataset.

    Used ONLY for U-Net segmentation.

    Laptop-friendly limits:
        Train = 500 image-mask pairs
        Validation = 200 image-mask pairs

    The original dataset files are NOT deleted.
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        train_samples: int = 500,
        val_samples: int = 200,
        seed: int = 42,
    ):

        self.data_dir = (
            data_dir
            or os.path.join(
                settings.DATASET_DIR,
                "brats_2d"
            )
        )

        self.split = split
        self.transform = transform
        self.image_size = image_size

        image_dir = os.path.join(
            self.data_dir,
            "images"
        )

        mask_dir = os.path.join(
            self.data_dir,
            "masks"
        )

        all_samples = []

        # ----------------------------------------------------
        # Find image-mask pairs
        # ----------------------------------------------------

        if os.path.isdir(image_dir):

            image_files = sorted(
                glob.glob(
                    os.path.join(
                        image_dir,
                        "*.png"
                    )
                )
                +
                glob.glob(
                    os.path.join(
                        image_dir,
                        "*.jpg"
                    )
                )
            )

            for image_path in image_files:

                basename = os.path.splitext(
                    os.path.basename(image_path)
                )[0]

                # First possible mask name
                mask_path = os.path.join(
                    mask_dir,
                    f"{basename}_mask.png"
                )

                # Second possible mask name
                if not os.path.exists(mask_path):

                    mask_path = os.path.join(
                        mask_dir,
                        f"{basename}.png"
                    )

                if os.path.exists(mask_path):

                    all_samples.append(
                        (
                            image_path,
                            mask_path
                        )
                    )

        # ----------------------------------------------------
        # Shuffle deterministically
        # ----------------------------------------------------

        rng = np.random.default_rng(seed)

        rng.shuffle(all_samples)

        # ----------------------------------------------------
        # Select limited dataset
        # ----------------------------------------------------

        if split == "train":

            self.samples = all_samples[
                :min(
                    train_samples,
                    len(all_samples)
                )
            ]

        elif split == "val":

            val_start = min(
                train_samples,
                len(all_samples)
            )

            val_end = min(
                val_start + val_samples,
                len(all_samples)
            )

            self.samples = all_samples[
                val_start:val_end
            ]

        elif split == "test":

            test_start = min(
                train_samples + val_samples,
                len(all_samples)
            )

            self.samples = all_samples[
                test_start:
            ]

        else:

            raise ValueError(
                f"Invalid split: {split}"
            )

        logger.info(
            f"BratsSegmentationDataset "
            f"[{split}]: "
            f"{len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        image_path, mask_path = self.samples[idx]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("L")

        image = image.resize(
            (
                self.image_size,
                self.image_size
            )
        )

        image_array = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        image_tensor = (
            torch.from_numpy(
                image_array
            )
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = mask.resize(
            (
                self.image_size,
                self.image_size
            )
        )

        mask_array = np.array(
            mask,
            dtype=np.float32
        ) / 255.0

        # Binary mask
        mask_array = (
            mask_array > 0.5
        ).astype(np.float32)

        mask_tensor = (
            torch.from_numpy(
                mask_array
            )
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        if self.transform:

            image_tensor = self.transform(
                image_tensor
            )

        return image_tensor, mask_tensor