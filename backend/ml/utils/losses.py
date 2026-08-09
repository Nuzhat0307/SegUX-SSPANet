"""
Loss functions for multi-task training.
- DiceLoss for segmentation
- Combined loss for joint classification + segmentation optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for more stable segmentation training."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    L = λ_cls * L_classification + λ_seg * L_segmentation
    """

    def __init__(self, cls_weight: float = 1.0, seg_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_loss = BCEDiceLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_logits: torch.Tensor = None,
        seg_targets: torch.Tensor = None,
    ) -> dict:
        cls_loss = self.cls_loss(cls_logits, cls_targets)
        total = self.cls_weight * cls_loss
        losses = {"classification": cls_loss.item()}

        if seg_logits is not None and seg_targets is not None:
            seg_loss = self.seg_loss(seg_logits, seg_targets)
            total += self.seg_weight * seg_loss
            losses["segmentation"] = seg_loss.item()

        losses["total"] = total.item()
        return {"loss": total, "metrics": losses}
