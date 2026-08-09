"""
Evaluation metrics for brain tumor classification and segmentation.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from typing import Dict, List


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
    except Exception:
        metrics["auc_roc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    for i in range(len(np.unique(y_true))):
        metrics[f"precision_class_{i}"] = float(
            precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        )
        metrics[f"recall_class_{i}"] = float(
            recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        )

    return metrics


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Compute Dice similarity coefficient."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Compute Intersection over Union (Jaccard)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict:
    """Compute segmentation metrics."""
    binary_pred = (pred_mask > 0.5).astype(np.uint8)
    binary_true = (true_mask > 0.5).astype(np.uint8)

    return {
        "dice": dice_score(binary_pred, binary_true),
        "iou": iou_score(binary_pred, binary_true),
        "sensitivity": float(
            (binary_pred * binary_true).sum() / max(binary_true.sum(), 1)
        ),
        "specificity": float(
            ((1 - binary_pred) * (1 - binary_true)).sum() / max((1 - binary_true).sum(), 1)
        ),
    }


def uncertainty_metrics(confidences: List[float], correct: List[bool]) -> Dict:
    """
    Compute uncertainty quality metrics:
    - Brier score
    - Expected Calibration Error (ECE)
    """
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)

    # Brier score
    brier = float(np.mean((confidences - correct) ** 2))

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += abs(avg_conf - avg_acc) * mask.sum() / len(confidences)

    return {
        "brier_score": brier,
        "expected_calibration_error": float(ece),
    }
