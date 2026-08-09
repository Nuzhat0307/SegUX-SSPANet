"""
Dataset download and preprocessing scripts.

Figshare Brain Tumor Dataset:
- Contains 3 tumor classes: glioma, meningioma, pituitary
- 3064 .mat files total, split across 4 zip archives
- URL: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- .mat files contain a struct `cjdata` with:
    cjdata.label: 1=meningioma, 2=glioma, 3=pituitary
    cjdata.image: image data
    cjdata.tumorBorder: tumor border coordinates
    cjdata.PID: patient ID

BraTS (Brain Tumor Segmentation):
- Requires registration at https://www.synapse.org/brats
- Provides multimodal MRI volumes with pixel-level segmentation masks
- We extract 2D slices for the framework

Usage:
    python -m scripts.download_dataset --dataset figshare --output ml/data/figshare
    python -m scripts.download_dataset --dataset brats --output ml/data/brats_2d
"""
import os
import argparse
import zipfile
import shutil
import numpy as np
import cv2
from PIL import Image
from loguru import logger
from typing import List, Optional


# Figshare dataset — 4 zip archives of .mat files (version 8, Dec 2024)
FIGSHARE_FILES = [
    {"id": 3381290, "name": "brainTumorDataPublic_1-766.zip",
     "url": "https://ndownloader.figshare.com/files/3381290"},
    {"id": 3381293, "name": "brainTumorDataPublic_1533-2298.zip",
     "url": "https://ndownloader.figshare.com/files/3381293"},
    {"id": 3381296, "name": "brainTumorDataPublic_767-1532.zip",
     "url": "https://ndownloader.figshare.com/files/3381296"},
    {"id": 3381302, "name": "brainTumorDataPublic_2299-3064.zip",
     "url": "https://ndownloader.figshare.com/files/3381302"},
]

# Label mapping from .mat cjdata.label
MAT_LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess brain tumor datasets")
    parser.add_argument(
        "--dataset", type=str, default="figshare",
        choices=["figshare", "brats", "all"],
        help="Dataset to download",
    )
    parser.add_argument("--output", type=str, default="ml/data", help="Output directory")
    parser.add_argument("--image_size", type=int, default=224, help="Resized image dimension")
    parser.add_argument("--seg_size", type=int, default=256, help="Segmentation image size")
    return parser.parse_args()


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL using urllib with progress feedback."""
    import urllib.request

    try:
        logger.info(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Downloaded to {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(
            "If the automatic download fails, please download the Figshare dataset manually "
            "from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 "
            "and extract it to the output directory."
        )
        return False


def extract_mat_file(mat_path: str, output_dir: str, image_size: int) -> int:
    """
    Extract image and label from a Figshare MATLAB .mat file.

    Supports:
    - MATLAB v5/v6 files using scipy
    - MATLAB v7.3 files using h5py

    Figshare labels:
        1 = meningioma
        2 = glioma
        3 = pituitary
    """

    count = 0

    try:
        # ---------------------------------------------------------
        # First try normal MATLAB files using scipy
        # ---------------------------------------------------------
        try:
            import scipy.io as sio

            mat = sio.loadmat(
                mat_path,
                struct_as_record=False,
                squeeze_me=True
            )

            if "cjdata" not in mat:
                logger.warning(f"No cjdata in {mat_path}, skipping")
                return 0

            cjdata = mat["cjdata"]

            label = int(cjdata.label)
            image = cjdata.image

        # ---------------------------------------------------------
        # MATLAB v7.3 files require h5py
        # ---------------------------------------------------------
        except (NotImplementedError, ValueError, OSError):

            import h5py

            logger.debug(
                f"Reading MATLAB v7.3 file using h5py: {mat_path}"
            )

            with h5py.File(mat_path, "r") as f:

                if "cjdata" not in f:
                    logger.warning(
                        f"No cjdata group in {mat_path}, skipping"
                    )
                    return 0

                cjdata = f["cjdata"]

                # Read label
                label_data = cjdata["label"][()]
                label = int(np.asarray(label_data).squeeze())

                # Read MRI image
                image = cjdata["image"][()]

        # ---------------------------------------------------------
        # Convert label to class name
        # ---------------------------------------------------------
        tumor_class = MAT_LABEL_MAP.get(label, "unknown")

        if tumor_class == "unknown":
            logger.warning(
                f"Unknown label {label} in {mat_path}, skipping"
            )
            return 0

        # ---------------------------------------------------------
        # Create class directory
        # ---------------------------------------------------------
        class_dir = os.path.join(output_dir, tumor_class)
        os.makedirs(class_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Normalize image to 0-255
        # ---------------------------------------------------------
        img_array = np.array(image, dtype=np.float32)

        if img_array.size == 0:
            logger.warning(
                f"Empty image in {mat_path}, skipping"
            )
            return 0

        img_min = img_array.min()
        img_max = img_array.max()

        if img_max > img_min:
            img_array = (
                (img_array - img_min)
                / (img_max - img_min)
            )

        img_array = (img_array * 255).astype(np.uint8)

        # ---------------------------------------------------------
        # Convert to PIL grayscale image
        # ---------------------------------------------------------
        img_pil = Image.fromarray(img_array).convert("L")

        # Resize to model input size
        img_pil = img_pil.resize(
            (image_size, image_size)
        )

        # ---------------------------------------------------------
        # Save PNG
        # ---------------------------------------------------------
        base_name = os.path.splitext(
            os.path.basename(mat_path)
        )[0]

        out_path = os.path.join(
            class_dir,
            f"{base_name}.png"
        )

        img_pil.save(out_path)

        count += 1

        return count

    except Exception as e:
        logger.warning(
            f"Failed to process {mat_path}: {e}"
        )
        return 0

def process_figshare_zips(zip_dir: str, output_dir: str, image_size: int):
    """Download and extract all 4 Figshare zip archives, then convert .mat to PNG."""
    total = 0

    for file_info in FIGSHARE_FILES:
        zip_name = file_info["name"]
        zip_path = os.path.join(zip_dir, zip_name)

        # Download
        if not os.path.exists(zip_path):
            success = download_file(file_info["url"], zip_path)
            if not success:
                logger.error(f"Failed to download {zip_name}, skipping")
                continue
        else:
            logger.info(f"{zip_name} already exists, skipping download")

        # Extract
        extract_dir = os.path.join(zip_dir, f"extracted_{file_info['id']}")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            logger.info(f"Extracted {zip_name}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_name}: {e}")
            continue

        # Process .mat files
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".mat") and file.lower() != "cvind.mat":
                    mat_path = os.path.join(root, file)
                    total += extract_mat_file(mat_path, output_dir, image_size)

        # Clean up extracted dir to save space
        shutil.rmtree(extract_dir, ignore_errors=True)

        # Optionally remove zip to save disk space
        os.remove(zip_path)
        logger.info(f"Removed {zip_name} to save disk space")

    logger.info(f"Total images extracted and organized: {total}")
    return total


def preprocess_brats(data_dir: str, output_dir: str, seg_size: int):
    """
    Preprocess BraTS NIfTI volumes into 2D PNG slices.
    Extracts middle slices with tumor present.

    Assumption: BraTS data is downloaded and extracted as NIfTI files.
    Requires nibabel: pip install nibabel

    BraTS file naming convention:
        BraTS2021_00001/
            BraTS2021_00001_flair.nii.gz
            BraTS2021_00001_t1.nii.gz
            BraTS2021_00001_t1ce.nii.gz
            BraTS2021_00001_t2.nii.gz
            BraTS2021_00001_seg.nii.gz
    """
    try:
        import nibabel as nib
    except ImportError:
        logger.error("nibabel is required for BraTS preprocessing. Install with: pip install nibabel")
        logger.info(
            "BraTS data must be downloaded manually from https://www.synapse.org/brats "
            "(requires free registration). Place extracted NIfTI files in the data directory."
        )
        return

    image_out = os.path.join(output_dir, "images")
    mask_out = os.path.join(output_dir, "masks")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    patient_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    count = 0
    for patient_id in patient_dirs:
        patient_path = os.path.join(data_dir, patient_id)

        # Find T1ce (contrast-enhanced T1) — best for tumor visibility
        t1ce_path = None
        seg_path = None
        for f in os.listdir(patient_path):
            if ("t1ce" in f.lower() or "-t1c" in f.lower()) and f.endswith(".nii.gz"):
                t1ce_path = os.path.join(patient_path, f)
            if "seg" in f.lower() and f.endswith(".nii.gz"):
                seg_path = os.path.join(patient_path, f)

        if not t1ce_path or not seg_path:
            continue

        try:
            t1ce = nib.load(t1ce_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()

            # Extract slices with tumor present (middle 60% of slices)
            n_slices = t1ce.shape[2]
            start = int(n_slices * 0.2)
            end = int(n_slices * 0.8)

            # Find all slices containing tumor
            tumor_slices = []

            for slice_idx in range(start, end):
                seg_slice = seg[:, :, slice_idx]

                if seg_slice.max() > 0:
                    tumor_slices.append(slice_idx)

            # Keep only a maximum of 5 tumor-containing slices per patient
            max_slices_per_patient = 5

            if len(tumor_slices) > max_slices_per_patient:
                selected_indices = np.linspace(
                    0,
                    len(tumor_slices) - 1,
                    max_slices_per_patient,
                    dtype=int
                )

                tumor_slices = [tumor_slices[i] for i in selected_indices]

            # Process selected slices
            for slice_idx in tumor_slices:
                seg_slice = seg[:, :, slice_idx]

                img_slice = t1ce[:, :, slice_idx]

                # Normalize
                img_slice = (
                        (img_slice - img_slice.min())
                        / (img_slice.max() - img_slice.min() + 1e-8)
                )

                img_slice = (img_slice * 255).astype(np.uint8)

                mask_slice = (seg_slice > 0).astype(np.uint8) * 255

                img_pil = Image.fromarray(img_slice).resize(
                    (seg_size, seg_size)
                )

                mask_pil = Image.fromarray(mask_slice).resize(
                    (seg_size, seg_size)
                )

                name = f"{patient_id}_slice_{slice_idx:03d}"

                img_pil.save(
                    os.path.join(image_out, f"{name}.png")
                )

                mask_pil.save(
                    os.path.join(mask_out, f"{name}_mask.png")
                )

                count += 1

        except Exception as e:
            logger.warning(f"Failed to process {patient_id}: {e}")

    logger.info(f"Extracted {count} 2D slices from BraTS to {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.dataset in ("figshare", "all"):
        figshare_dir = os.path.join(args.output, "figshare")
        os.makedirs(figshare_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Figshare Brain Tumor Dataset")
        logger.info("3064 .mat files across 4 zip archives")
        logger.info("=" * 60)

        # Also create a no_tumor folder placeholder (Figshare has no healthy class)
        no_tumor_dir = os.path.join(figshare_dir, "no_tumor")
        os.makedirs(no_tumor_dir, exist_ok=True)
        logger.info("Note: Figshare dataset has 3 classes only (glioma, meningioma, pituitary).")
        logger.info("The no_tumor folder is created but will be empty.")
        logger.info("For a 4-class dataset with no_tumor, consider the BraTS or Mendeley dataset.")

        # Download and process all 4 zips
        zip_dir = os.path.join(args.output, "figshare_zips")
        os.makedirs(zip_dir, exist_ok=True)
        total = process_figshare_zips(zip_dir, figshare_dir, args.image_size)

        if total > 0:
            logger.info(f"Successfully organized {total} images into class folders at {figshare_dir}")
            logger.info(f"  glioma/, meningioma/, pituitary/, no_tumor/ (empty)")
        else:
            logger.error(
                "No images were extracted. Please download manually from "
                "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"
            )

    if args.dataset in ("brats", "all"):
        brats_dir = os.path.join(args.output, "brats_2d")
        os.makedirs(brats_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("BraTS Brain Tumor Segmentation Dataset")
        logger.info("=" * 60)

        raw_dir = os.path.join(args.output, "brats_raw")
        if os.path.isdir(raw_dir):
            preprocess_brats(raw_dir, brats_dir, args.seg_size)
        else:
            logger.info(
                "\nBraTS requires manual download:\n"
                "1. Register at https://www.synapse.org/brats\n"
                "2. Download the BraTS 2021/2023 training data\n"
                "3. Extract NIfTI files to: " + raw_dir + "\n"
                "4. Re-run: python -m scripts.download_dataset --dataset brats\n"
                "   (requires: pip install nibabel)"
            )


if __name__ == "__main__":
    main()
