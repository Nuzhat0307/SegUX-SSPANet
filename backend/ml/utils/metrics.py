"""
Evaluation metrics for SegUX-SSPANet.

Classification:
    - Accuracy
    - Macro precision
    - Macro recall
    - Macro F1
    - Per-class precision / recall / F1
    - Confusion matrix
    - Multiclass ROC-AUC when mathematically available

Segmentation:
    - Dice
    - IoU
    - Sensitivity
    - Specificity

Uncertainty:
    - Brier score
    - Expected Calibration Error (ECE)
"""

from typing import Dict, List, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict:
    """
    Compute classification metrics.

    y_true:
        Ground-truth integer class labels.

    y_pred:
        Predicted integer class labels.

    y_probs:
        Probability matrix with shape:
            [N, NUM_CLASSES]
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(y_true) == 0:
        raise ValueError(
            "classification_metrics received an empty dataset."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number "
            "of samples."
        )

    if y_probs.ndim != 2:
        raise ValueError(
            f"y_probs must be 2-dimensional. "
            f"Received shape: {y_probs.shape}"
        )

    if y_probs.shape[0] != len(y_true):
        raise ValueError(
            "y_probs and y_true must contain the same "
            "number of samples."
        )

    num_classes = y_probs.shape[1]

    # --------------------------------------------------------
    # All classes represented by the model
    # --------------------------------------------------------

    labels = np.arange(num_classes)

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    metrics = {
        "num_samples": int(len(y_true)),

        "num_classes": int(num_classes),

        "classes_present": sorted(
            [int(x) for x in np.unique(y_true)]
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    for class_idx in labels:

        metrics[
            f"precision_class_{int(class_idx)}"
        ] = float(
            precision_per_class[class_idx]
        )

        metrics[
            f"recall_class_{int(class_idx)}"
        ] = float(
            recall_per_class[class_idx]
        )

        metrics[
            f"f1_class_{int(class_idx)}"
        ] = float(
            f1_per_class[class_idx]
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    metrics["confusion_matrix"] = cm.tolist()

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    auc_roc: Optional[float] = None
    auc_reason: Optional[str] = None

    unique_true_classes = np.unique(y_true)

    if len(unique_true_classes) < 2:

        auc_reason = (
            "ROC-AUC cannot be calculated because "
            "the evaluation set contains fewer than "
            "two ground-truth classes."
        )

    else:

        try:

            # Normalize probabilities defensively.
            y_probs_safe = np.clip(
                y_probs,
                1e-12,
                1.0,
            )

            row_sums = y_probs_safe.sum(
                axis=1,
                keepdims=True,
            )

            y_probs_safe = (
                y_probs_safe
                / np.maximum(
                    row_sums,
                    1e-12,
                )
            )

            # ------------------------------------------------
            # Standard multiclass OVR AUC.
            #
            # If the test set contains all model classes,
            # this is directly applicable.
            # ------------------------------------------------

            if len(unique_true_classes) == num_classes:

                auc_roc = float(
                    roc_auc_score(
                        y_true,
                        y_probs_safe,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                )

            else:

                # ------------------------------------------------
                # Partial-class evaluation.
                #
                # Example:
                # model has 4 outputs but test set contains
                # only glioma / meningioma / pituitary.
                #
                # Calculate macro OVR AUC only over classes
                # actually represented in y_true.
                # ------------------------------------------------

                per_class_auc = []

                for class_idx in unique_true_classes:

                    class_idx = int(class_idx)

                    binary_true = (
                        y_true == class_idx
                    ).astype(np.int64)

                    # One-vs-rest requires both positive
                    # and negative samples.
                    if (
                        np.unique(binary_true).size
                        < 2
                    ):
                        continue

                    class_auc = roc_auc_score(
                        binary_true,
                        y_probs_safe[:, class_idx],
                    )

                    per_class_auc.append(
                        float(class_auc)
                    )

                if per_class_auc:

                    auc_roc = float(
                        np.mean(
                            per_class_auc
                        )
                    )

                    auc_reason = (
                        "Macro one-vs-rest ROC-AUC "
                        "calculated only for classes "
                        "present in the evaluation set."
                    )

                else:

                    auc_reason = (
                        "ROC-AUC could not be calculated "
                        "for the represented classes."
                    )

        except Exception as exc:

            auc_roc = None

            auc_reason = (
                "ROC-AUC calculation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    metrics["auc_roc"] = auc_roc
    metrics["auc_roc_note"] = auc_reason

    return metrics


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1.0,
) -> float:
    """
    Compute Dice Similarity Coefficient.
    """

    pred_flat = np.asarray(
        pred
    ).flatten()

    target_flat = np.asarray(
        target
    ).flatten()

    intersection = (
        pred_flat * target_flat
    ).sum()

    return float(
        (
            2.0 * intersection
            + smooth
        )
        /
        (
            pred_flat.sum()
            + target_flat.sum()
            + smooth
        )
    )


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1.0,
) -> float:
    """
    Compute Intersection over Union.
    """

    pred_flat = np.asarray(
        pred
    ).flatten()

    target_flat = np.asarray(
        target
    ).flatten()

    intersection = (
        pred_flat * target_flat
    ).sum()

    union = (
        pred_flat.sum()
        + target_flat.sum()
        - intersection
    )

    return float(
        (
            intersection
            + smooth
        )
        /
        (
            union
            + smooth
        )
    )


def segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> Dict:
    """
    Compute binary segmentation metrics.
    """

    binary_pred = (
        np.asarray(pred_mask) > 0.5
    ).astype(np.uint8)

    binary_true = (
        np.asarray(true_mask) > 0.5
    ).astype(np.uint8)

    true_positive = float(
        (
            binary_pred
            * binary_true
        ).sum()
    )

    false_positive = float(
        (
            binary_pred
            * (1 - binary_true)
        ).sum()
    )

    false_negative = float(
        (
            (1 - binary_pred)
            * binary_true
        ).sum()
    )

    true_negative = float(
        (
            (1 - binary_pred)
            * (1 - binary_true)
        ).sum()
    )

    sensitivity = (
        true_positive
        /
        max(
            true_positive
            + false_negative,
            1.0,
        )
    )

    specificity = (
        true_negative
        /
        max(
            true_negative
            + false_positive,
            1.0,
        )
    )

    return {
        "dice": dice_score(
            binary_pred,
            binary_true,
        ),

        "iou": iou_score(
            binary_pred,
            binary_true,
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),
    }


# ============================================================
# UNCERTAINTY / CALIBRATION METRICS
# ============================================================

def uncertainty_metrics(
    confidences: List[float],
    correct: List[bool],
) -> Dict:
    """
    Compute confidence calibration metrics.

    `confidence` must be the actual probability assigned
    to the predicted class.

    `correct` indicates whether the prediction was correct.
    """

    confidences = np.asarray(
        confidences,
        dtype=np.float64,
    )

    correct = np.asarray(
        correct,
        dtype=np.float64,
    )

    if len(confidences) == 0:
        return {
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
        }

    if len(confidences) != len(correct):
        raise ValueError(
            "confidences and correct must "
            "have the same length."
        )

    confidences = np.clip(
        confidences,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Brier score for confidence of predicted class
    # --------------------------------------------------------

    brier = float(
        np.mean(
            (
                confidences
                - correct
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Expected Calibration Error
    # --------------------------------------------------------

    n_bins = 10

    bin_boundaries = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(
        n_bins
    ):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:

            mask = (
                confidences >= lower
            ) & (
                confidences <= upper
            )

        else:

            mask = (
                confidences > lower
            ) & (
                confidences <= upper
            )

        if not np.any(mask):
            continue

        avg_confidence = float(
            confidences[mask].mean()
        )

        avg_accuracy = float(
            correct[mask].mean()
        )

        ece += (
            abs(
                avg_confidence
                - avg_accuracy
            )
            * mask.sum()
            / len(confidences)
        )

    return {
        "brier_score": brier,

        "expected_calibration_error":
            float(ece),
    }