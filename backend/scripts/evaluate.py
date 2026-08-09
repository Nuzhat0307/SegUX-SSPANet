"""
Evaluation script for SegUX-SSPANet.

Evaluates the trained model on a SMALL TEST SUBSET of:

1. Figshare
   -> classification

2. BraTS 2D
   -> segmentation

Training configuration used:

Figshare:
    1000 training
    200 validation
    remaining images = test pool

BraTS:
    500 training
    200 validation
    remaining images = test pool

For CPU/laptop evaluation:
    default evaluation samples = 300
    default batch size = 2
    default MC Dropout samples = 30

Important:
During classification evaluation, the trained U-Net generates
segmentation guidance before the classifier prediction.
This matches the SegUX-SSPANet pipeline.

Usage in PowerShell:

python -m scripts.evaluate `
    --checkpoint ml/checkpoints/segux_sspanet_best.pth `
    --figshare_dir ml/data/figshare `
    --brats_dir ml/data/brats_2d `
    --output ml/eval_results.json
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from loguru import logger

from app.core.config import settings

from ml.models.segux_sspanet import SegUXSSPANet
from ml.models.unet import UNet

from ml.utils.metrics import (
    classification_metrics,
    segmentation_metrics,
    uncertainty_metrics,
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
        description="Evaluate SegUX-SSPANet"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/checkpoints/segux_sspanet_best.pth",
        help="Path to trained SegUX-SSPANet checkpoint",
    )

    parser.add_argument(
        "--figshare_dir",
        type=str,
        default=None,
        help="Figshare dataset directory",
    )

    parser.add_argument(
        "--brats_dir",
        type=str,
        default=None,
        help="BraTS 2D dataset directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ml/eval_results.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--eval_samples",
        type=int,
        default=300,
        help="Maximum number of test samples from EACH dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Evaluation batch size for CPU",
    )

    parser.add_argument(
        "--mc_samples",
        type=int,
        default=30,
        help="Number of Monte Carlo Dropout forward passes",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda",
    )

    return parser.parse_args()


# ============================================================
# SEGMENTATION GUIDANCE
# ============================================================

def generate_segmentation_guidance(
    segmentor,
    images,
    device,
):
    """
    Generate U-Net segmentation guidance for Figshare images.

    Figshare input:
        [B, 3, 224, 224]

    U-Net input:
        [B, 1, 256, 256]

    Process:

        3-channel image
              ↓
        grayscale
              ↓
        resize 256x256
              ↓
        U-Net
              ↓
        sigmoid mask
              ↓
        resize back to classifier size
    """

    # --------------------------------------------------------
    # Convert 3-channel image to grayscale
    # --------------------------------------------------------

    grayscale = images.mean(
        dim=1,
        keepdim=True,
    )

    # --------------------------------------------------------
    # Resize for U-Net
    # --------------------------------------------------------

    grayscale = F.interpolate(
        grayscale,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    # --------------------------------------------------------
    # U-Net prediction
    # --------------------------------------------------------

    segmentation_logits = segmentor(
        grayscale
    )

    segmentation_mask = torch.sigmoid(
        segmentation_logits
    )

    # --------------------------------------------------------
    # Resize mask back to classifier image size
    # --------------------------------------------------------

    segmentation_mask = F.interpolate(
        segmentation_mask,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return segmentation_mask.clamp(
        0.0,
        1.0,
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(
    checkpoint_path,
    device,
):
    """
    Create and load:

    1. SegUX-SSPANet classifier
    2. U-Net segmentation model
    """

    logger.info(
        "Creating models..."
    )

    # --------------------------------------------------------
    # Create classifier
    # --------------------------------------------------------

    classifier = SegUXSSPANet(
        num_classes=settings.NUM_CLASSES,
        backbone="resnet50",
        pretrained=False,
    ).to(device)

    # --------------------------------------------------------
    # Create U-Net
    # --------------------------------------------------------

    segmentor = UNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    logger.info(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # Load complete checkpoint
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

    except TypeError:

        # Compatibility with older PyTorch versions

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and "classifier" in checkpoint:

        classifier.load_state_dict(
            checkpoint["classifier"]
        )

    else:

        classifier.load_state_dict(
            checkpoint
        )

    logger.info(
        "Classifier weights loaded."
    )

    # --------------------------------------------------------
    # Load U-Net
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "segmentor" in checkpoint
    ):

        segmentor.load_state_dict(
            checkpoint["segmentor"]
        )

        logger.info(
            "Loaded U-Net from complete checkpoint."
        )

    else:

        # ----------------------------------------------------
        # Try separate U-Net checkpoint
        # ----------------------------------------------------

        unet_path = os.path.join(
            os.path.dirname(
                checkpoint_path
            ),
            "segux_unet_best.pth",
        )

        if not os.path.exists(
            unet_path
        ):

            raise FileNotFoundError(
                "U-Net weights were not found.\n"
                "Expected either:\n"
                f"1. Inside {checkpoint_path}\n"
                "OR\n"
                f"2. {unet_path}"
            )

        try:

            unet_checkpoint = torch.load(
                unet_path,
                map_location=device,
                weights_only=True,
            )

        except TypeError:

            unet_checkpoint = torch.load(
                unet_path,
                map_location=device,
            )

        segmentor.load_state_dict(
            unet_checkpoint
        )

        logger.info(
            f"Loaded U-Net from: "
            f"{unet_path}"
        )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    classifier.eval()
    segmentor.eval()

    logger.info(
        "Models loaded successfully."
    )

    return classifier, segmentor


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification(
    classifier,
    segmentor,
    loader,
    device,
):
    """
    Evaluate SegUX-SSPANet classification.

    U-Net segmentation guidance is generated before
    classification.
    """

    classifier.eval()
    segmentor.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    all_correct = []

    logger.info(
        "Running classification evaluation..."
    )

    with torch.no_grad():

        for batch_idx, (
            images,
            labels,
        ) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            # ------------------------------------------------
            # Generate U-Net segmentation guidance
            # ------------------------------------------------

            segmentation_guidance = (
                generate_segmentation_guidance(
                    segmentor,
                    images,
                    device,
                )
            )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            logits = classifier(
                images,
                segmentation_guidance,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = probs.argmax(
                dim=1
            )

            confidence = probs.max(
                dim=1
            )[0]

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_confidences.extend(
                confidence.cpu().numpy()
            )

            all_correct.extend(
                (
                    preds.cpu()
                    == labels.cpu()
                ).numpy()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx % 50 == 0
                or batch_idx == len(loader) - 1
            ):

                logger.info(
                    f"CLS Test Batch "
                    f"{batch_idx + 1}/{len(loader)}"
                )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    all_labels = np.array(
        all_labels
    )

    all_preds = np.array(
        all_preds
    )

    all_probs = np.array(
        all_probs
    )

    all_confidences = np.array(
        all_confidences
    )

    all_correct = np.array(
        all_correct
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    cls_metrics = classification_metrics(
        all_labels,
        all_preds,
        all_probs,
    )

    # --------------------------------------------------------
    # Uncertainty metrics
    # --------------------------------------------------------

    unc_metrics = uncertainty_metrics(
        all_labels,
        all_probs,
    )

    logger.info(
        f"Classification Accuracy: "
        f"{cls_metrics['accuracy']:.4f}"
    )

    logger.info(
        f"Classification F1: "
        f"{cls_metrics['f1_macro']:.4f}"
    )

    logger.info(
        f"Uncertainty Brier: "
        f"{unc_metrics['brier_score']:.4f}"
    )

    logger.info(
        f"Uncertainty ECE: "
        f"{unc_metrics['expected_calibration_error']:.4f}"
    )

    return (
        cls_metrics,
        unc_metrics,
        len(all_labels),
    )


# ============================================================
# SEGMENTATION EVALUATION
# ============================================================

def evaluate_segmentation(
    segmentor,
    loader,
    device,
):
    """
    Evaluate U-Net on the selected BraTS test subset.
    """

    segmentor.eval()

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []

    logger.info(
        "Running segmentation evaluation..."
    )

    with torch.no_grad():

        for batch_idx, (
            images,
            masks,
        ) in enumerate(loader):

            images = images.to(device)
            masks = masks.to(device)

            # ------------------------------------------------
            # U-Net prediction
            # ------------------------------------------------

            logits = segmentor(
                images
            )

            predicted_masks = torch.sigmoid(
                logits
            )

            # ------------------------------------------------
            # Metrics per image
            # ------------------------------------------------

            for i in range(
                images.size(0)
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
                    metric.get(
                        "dice",
                        0.0,
                    )
                )

                iou_scores.append(
                    metric.get(
                        "iou",
                        0.0,
                    )
                )

                if "sensitivity" in metric:

                    sensitivity_scores.append(
                        metric["sensitivity"]
                    )

                if "specificity" in metric:

                    specificity_scores.append(
                        metric["specificity"]
                    )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx % 50 == 0
                or batch_idx == len(loader) - 1
            ):

                logger.info(
                    f"SEG Test Batch "
                    f"{batch_idx + 1}/{len(loader)}"
                )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    seg_metrics = {

        "avg_dice":
            float(
                np.mean(dice_scores)
            )
            if dice_scores
            else 0.0,

        "avg_iou":
            float(
                np.mean(iou_scores)
            )
            if iou_scores
            else 0.0,

        "avg_sensitivity":
            float(
                np.mean(
                    sensitivity_scores
                )
            )
            if sensitivity_scores
            else 0.0,

        "avg_specificity":
            float(
                np.mean(
                    specificity_scores
                )
            )
            if specificity_scores
            else 0.0,
    }

    logger.info(
        f"Segmentation Dice: "
        f"{seg_metrics['avg_dice']:.4f}"
    )

    logger.info(
        f"Segmentation IoU: "
        f"{seg_metrics['avg_iou']:.4f}"
    )

    return seg_metrics


# ============================================================
# MC DROPOUT EVALUATION
# ============================================================

def evaluate_mc_dropout(
    classifier,
    segmentor,
    loader,
    device,
    mc_samples=30,
):
    """
    Estimate predictive uncertainty using MC Dropout.

    Confidence is defined consistently with live inference:

        confidence = maximum predicted-class probability

    Entropy and mutual information are reported separately.

    A case is considered uncertain when:

        1. predicted-class probability < UNCERTAINTY_THRESHOLD

    OR

        2. mutual information > epistemic threshold
    """

    logger.info(
        f"Running MC Dropout "
        f"({mc_samples} samples)..."
    )

    mc_confidences = []
    mc_entropies = []
    mc_mutual_infos = []

    confidence_threshold = float(
        getattr(
            settings,
            "UNCERTAINTY_THRESHOLD",
            0.75,
        )
    )

    # Keep this consistent with inference.py
    mutual_information_threshold = 0.30

    # --------------------------------------------------------
    # Enable dropout
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Enable ONLY dropout layers.
    #
    # Keep the complete classifier in eval mode so that
    # BatchNorm and other evaluation-time layers remain stable.
    # This matches inference.py.
    # --------------------------------------------------------

    classifier.eval()

    for module in classifier.modules():

        if isinstance(
                module,
                (
                        torch.nn.Dropout,
                        torch.nn.Dropout1d,
                        torch.nn.Dropout2d,
                        torch.nn.Dropout3d,
                        torch.nn.AlphaDropout,
                ),
        ):
            module.train()

    try:

        # --------------------------------------------------------
        # Evaluate only the first 10 batches
        # --------------------------------------------------------

        max_mc_batches = min(
            10,
            len(loader),
        )

        with torch.no_grad():

            for batch_idx, (
                    images,
                    labels,
            ) in enumerate(loader):

                if batch_idx >= max_mc_batches:
                    break

                images = images.to(device)

                # ------------------------------------------------
                # Generate segmentation guidance
                # ------------------------------------------------

                segmentation_guidance = (
                    generate_segmentation_guidance(
                        segmentor,
                        images,
                        device,
                    )
                )

                all_probs_mc = []

                # ------------------------------------------------
                # MC forward passes
                # ------------------------------------------------

                for _ in range(mc_samples):

                    logits = classifier(
                        images,
                        segmentation_guidance,
                    )

                    probs = torch.softmax(
                        logits,
                        dim=1,
                    )

                    all_probs_mc.append(
                        probs.cpu().numpy()
                    )

                all_probs_mc = np.asarray(
                    all_probs_mc,
                    dtype=np.float64,
                )

                # Shape:
                # [MC samples, batch, classes]

                mean_probs = (
                    all_probs_mc.mean(
                        axis=0
                    )
                )

                # Numerical safety
                mean_probs = np.clip(
                    mean_probs,
                    1e-10,
                    1.0,
                )

                mean_probs = (
                    mean_probs
                    / mean_probs.sum(
                        axis=1,
                        keepdims=True,
                    )
                )

                # ------------------------------------------------
                # Actual predicted-class confidence
                # ------------------------------------------------

                top_probabilities = (
                    np.max(
                        mean_probs,
                        axis=1,
                    )
                )

                # ------------------------------------------------
                # Predictive entropy
                # ------------------------------------------------

                predictive_entropy = (
                    -np.sum(
                        mean_probs
                        * np.log2(
                            mean_probs
                        ),
                        axis=1,
                    )
                )

                # ------------------------------------------------
                # Expected entropy
                # ------------------------------------------------

                expected_entropy = (
                    -np.sum(
                        all_probs_mc
                        * np.log2(
                            np.clip(
                                all_probs_mc,
                                1e-10,
                                1.0,
                            )
                        ),
                        axis=2,
                    ).mean(axis=0)
                )

                # ------------------------------------------------
                # Mutual information
                # ------------------------------------------------

                mutual_information = np.maximum(
                    predictive_entropy
                    - expected_entropy,
                    0.0,
                )

                # ------------------------------------------------
                # Store metrics
                # ------------------------------------------------

                mc_confidences.extend(
                    top_probabilities.tolist()
                )

                mc_entropies.extend(
                    predictive_entropy.tolist()
                )

                mc_mutual_infos.extend(
                    mutual_information.tolist()
                )

                if (
                        batch_idx % 2 == 0
                        or batch_idx == max_mc_batches - 1
                ):
                    logger.info(
                        f"MC Batch "
                        f"{batch_idx + 1}/{max_mc_batches}"
                    )

    finally:

        # Always restore normal inference mode
        classifier.eval()

    # --------------------------------------------------------
    # Calculate final metrics
    # --------------------------------------------------------

    if mc_confidences:

        mc_confidences = np.asarray(
            mc_confidences,
            dtype=np.float64,
        )

        mc_entropies = np.asarray(
            mc_entropies,
            dtype=np.float64,
        )

        mc_mutual_infos = np.asarray(
            mc_mutual_infos,
            dtype=np.float64,
        )

        uncertain_cases = (
            (
                mc_confidences
                < confidence_threshold
            )
            |
            (
                mc_mutual_infos
                > mutual_information_threshold
            )
        )

        mc_metrics = {

            "avg_mc_confidence":
                float(
                    np.mean(
                        mc_confidences
                    )
                ),

            "avg_predictive_entropy":
                float(
                    np.mean(
                        mc_entropies
                    )
                ),

            "avg_mutual_information":
                float(
                    np.mean(
                        mc_mutual_infos
                    )
                ),

            "uncertain_cases_ratio":
                float(
                    np.mean(
                        uncertain_cases
                    )
                ),

            "confidence_threshold":
                confidence_threshold,

            "mutual_information_threshold":
                mutual_information_threshold,
        }

    else:

        mc_metrics = {

            "avg_mc_confidence": 0.0,

            "avg_predictive_entropy": 0.0,

            "avg_mutual_information": 0.0,

            "uncertain_cases_ratio": 0.0,

            "confidence_threshold":
                confidence_threshold,

            "mutual_information_threshold":
                mutual_information_threshold,
        }

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        f"MC Confidence: "
        f"{mc_metrics['avg_mc_confidence']:.4f}"
    )

    logger.info(
        f"MC Predictive Entropy: "
        f"{mc_metrics['avg_predictive_entropy']:.4f}"
    )

    logger.info(
        f"MC Mutual Information: "
        f"{mc_metrics['avg_mutual_information']:.4f}"
    )

    logger.info(
        f"MC Uncertain Ratio: "
        f"{mc_metrics['uncertain_cases_ratio']:.2%}"
    )

    return mc_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = (
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    logger.info(
        f"Evaluating on device: {device}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    classifier, segmentor = load_models(
        args.checkpoint,
        device,
    )

    # ========================================================
    # FIGSHARE TEST DATA
    # ========================================================

    logger.info(
        "\n"
        + "=" * 60
    )

    logger.info(
        "FIGSHARE TEST DATA"
    )

    logger.info(
        "=" * 60
    )

    full_figshare_test_ds = FigshareDataset(
        data_dir=args.figshare_dir,
        split="test",
        image_size=settings.IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Take only requested number of test samples
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Stratified evaluation subset
    # --------------------------------------------------------
    # Keep the evaluation subset balanced across tumor classes.
    # This makes Accuracy, Macro-F1 and AUC more representative.

    figshare_eval_count = min(
        args.eval_samples,
        len(full_figshare_test_ds),
    )

    # Collect labels from the full test pool
    labels = np.array([
        label
        for _, label in full_figshare_test_ds.samples
    ])

    num_classes = settings.NUM_CLASSES

    rng = np.random.default_rng(42)

    selected_indices = []

    # Allocate approximately equal samples to each class
    samples_per_class = figshare_eval_count // num_classes
    remainder = figshare_eval_count % num_classes

    for class_idx in range(num_classes):

        class_indices = np.where(
            labels == class_idx
        )[0]

        rng.shuffle(class_indices)

        take = samples_per_class

        if class_idx < remainder:
            take += 1

        take = min(
            take,
            len(class_indices)
        )

        selected_indices.extend(
            class_indices[:take].tolist()
        )

    # Shuffle final evaluation set
    rng.shuffle(selected_indices)

    test_cls_ds = Subset(
        full_figshare_test_ds,
        selected_indices,
    )

    logger.info(
        "Using stratified Figshare evaluation subset: "
        f"{len(test_cls_ds)} samples"
    )

    for class_idx in range(num_classes):
        count = sum(
            labels[i] == class_idx
            for i in selected_indices
        )

        logger.info(
            f"Class {class_idx} "
            f"({settings.TUMOR_CLASSES[class_idx]}): "
            f"{count} samples"
        )

    logger.info(
        f"Full Figshare test pool: "
        f"{len(full_figshare_test_ds)}"
    )

    logger.info(
        f"Figshare evaluation samples: "
        f"{len(test_cls_ds)}"
    )

    test_cls_loader = DataLoader(
        test_cls_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # BRATS TEST DATA
    # ========================================================

    logger.info(
        "\n"
        + "=" * 60
    )

    logger.info(
        "BRATS TEST DATA"
    )

    logger.info(
        "=" * 60
    )

    full_brats_test_ds = BratsSegmentationDataset(
        data_dir=args.brats_dir,
        split="test",
    )

    # --------------------------------------------------------
    # Take only requested number of test samples
    # --------------------------------------------------------

    brats_eval_count = min(
        args.eval_samples,
        len(full_brats_test_ds),
    )

    test_seg_ds = Subset(
        full_brats_test_ds,
        range(brats_eval_count),
    )

    logger.info(
        f"Full BraTS test pool: "
        f"{len(full_brats_test_ds)}"
    )

    logger.info(
        f"BraTS evaluation samples: "
        f"{len(test_seg_ds)}"
    )

    test_seg_loader = DataLoader(
        test_seg_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # CLASSIFICATION EVALUATION
    # ========================================================

    (
        cls_metrics,
        unc_metrics,
        num_test_samples,
    ) = evaluate_classification(
        classifier,
        segmentor,
        test_cls_loader,
        device,
    )

    # ========================================================
    # SEGMENTATION EVALUATION
    # ========================================================

    seg_metrics = evaluate_segmentation(
        segmentor,
        test_seg_loader,
        device,
    )

    # ========================================================
    # MC DROPOUT
    # ========================================================

    mc_metrics = evaluate_mc_dropout(
        classifier,
        segmentor,
        test_cls_loader,
        device,
        mc_samples=args.mc_samples,
    )

    # ========================================================
    # COMPILE RESULTS
    # ========================================================

    results = {

        "model_version":
            settings.MODEL_VERSION,

        "checkpoint":
            args.checkpoint,

        "device":
            device,

        "evaluation_configuration": {

            "eval_samples_per_dataset":
                args.eval_samples,

            "batch_size":
                args.batch_size,

            "mc_samples":
                args.mc_samples,
        },

        "training_configuration": {

            "figshare_train":
                1000,

            "figshare_validation":
                200,

            "brats_train":
                500,

            "brats_validation":
                200,
        },

        "test_pool": {

            "figshare":
                len(full_figshare_test_ds),

            "brats":
                len(full_brats_test_ds),
        },

        "evaluated_samples": {

            "figshare":
                len(test_cls_ds),

            "brats":
                len(test_seg_ds),
        },

        "classification":
            cls_metrics,

        "segmentation":
            seg_metrics,

        "uncertainty":
            unc_metrics,

        "mc_dropout":
            mc_metrics,

        "num_test_samples":
            num_test_samples,
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        args.output,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    logger.info(
        f"Results saved to: "
        f"{args.output}"
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SEGUX-SSPANET EVALUATION SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: "
        f"{settings.MODEL_VERSION}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print(
        "\nTraining configuration:"
    )

    print(
        "  Figshare train: 1000"
    )

    print(
        "  Figshare validation: 200"
    )

    print(
        "  BraTS train: 500"
    )

    print(
        "  BraTS validation: 200"
    )

    print(
        "\nEvaluation:"
    )

    print(
        f"  Figshare test pool: "
        f"{len(full_figshare_test_ds)}"
    )

    print(
        f"  Figshare evaluated: "
        f"{len(test_cls_ds)}"
    )

    print(
        f"  BraTS test pool: "
        f"{len(full_brats_test_ds)}"
    )

    print(
        f"  BraTS evaluated: "
        f"{len(test_seg_ds)}"
    )

    print(
        "\nClassification:"
    )

    print(
        f"  Accuracy: "
        f"{cls_metrics['accuracy']:.4f}"
    )

    print(
        f"  F1 (macro): "
        f"{cls_metrics['f1_macro']:.4f}"
    )

    print(
        f"  AUC-ROC: "
        f"{cls_metrics.get('auc_roc', 'N/A')}"
    )

    print(
        "\nSegmentation:"
    )

    print(
        f"  Dice: "
        f"{seg_metrics['avg_dice']:.4f}"
    )

    print(
        f"  IoU: "
        f"{seg_metrics['avg_iou']:.4f}"
    )

    print(
        f"  Sensitivity: "
        f"{seg_metrics['avg_sensitivity']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{seg_metrics['avg_specificity']:.4f}"
    )

    print(
        "\nUncertainty:"
    )

    print(
        f"  Brier: "
        f"{unc_metrics['brier_score']:.4f}"
    )

    print(
        f"  ECE: "
        f"{unc_metrics['expected_calibration_error']:.4f}"
    )

    print(
        f"  MC Confidence: "
        f"{mc_metrics['avg_mc_confidence']:.4f}"
    )

    print(
        f"  MC Entropy: "
        f"{mc_metrics['avg_predictive_entropy']:.4f}"
    )

    print(
        f"  MC Uncertain Ratio: "
        f"{mc_metrics['uncertain_cases_ratio']:.2%}"
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Results saved to: "
        f"{args.output}"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()