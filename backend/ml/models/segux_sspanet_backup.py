"""
SegUX-SSPANet — The full multi-task model combining:
1. ResNet50 backbone (pretrained on ImageNet)
2. SSPANet attention modules at each residual stage
3. Classification head
4. Segmentation-guided attention learning (U-Net decoder)

This is the main model architecture for the brain tumor diagnosis system.
Multi-task learning: simultaneous classification + segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

from ml.models.sspanet import SSPANet


class SegUXSSPANet(nn.Module):
    """
    SegUX-SSPANet: Segmentation-guided, Uncertainty-aware,
    Strip-Style Pooling Attention Network.

    Args:
        num_classes: Number of tumor classes (4: glioma, meningioma, pituitary, no_tumor)
        backbone: Backbone architecture ("resnet50" or "vgg16")
        pretrained: Whether to use ImageNet-pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # --- Backbone ---
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            )
            self.layer1 = resnet.layer1  # 256 channels
            self.layer2 = resnet.layer2  # 512 channels
            self.layer3 = resnet.layer3  # 1024 channels
            self.layer4 = resnet.layer4  # 2048 channels
            self.feature_channels = [256, 512, 1024, 2048]
        elif backbone == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            vgg = models.vgg16(weights=weights)
            self.features = vgg.features
            self.feature_channels = [512, 512, 512, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- SSPANet attention modules ---
        if backbone == "resnet50":
            self.sspa1 = SSPANet(256)
            self.sspa2 = SSPANet(512)
            self.sspa3 = SSPANet(1024)
            self.sspa4 = SSPANet(2048)

        # --- Classification head ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 if backbone == "resnet50" else 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # --- Dropout for MC Dropout uncertainty estimation ---
        self.mc_dropout = nn.Dropout(0.25)

    def forward_features(
            self,
            x: torch.Tensor,
            segmentation_guidance=None
    ):
        """Extract multi-scale features with segmentation-guided SSPANet."""

        if self.backbone_name == "resnet50":

            x = self.stem(x)

            f1 = self.sspa1(
                self.layer1(x),
                segmentation_guidance
            )

            f2 = self.sspa2(
                self.layer2(f1),
                segmentation_guidance
            )

            f3 = self.sspa3(
                self.layer3(f2),
                segmentation_guidance
            )

            f4 = self.sspa4(
                self.layer4(f3),
                segmentation_guidance
            )

            return [f1, f2, f3, f4]

        else:
            features = []
            current = x

            for i, layer in enumerate(self.features):
                current = layer(current)

                if i in [3, 8, 15, 22]:
                    features.append(current)

            return features

    def forward(
            self,
            x: torch.Tensor,
            segmentation_guidance=None
    ):
        """Forward pass with optional segmentation guidance."""

        features = self.forward_features(
            x,
            segmentation_guidance
        )

        final_feat = features[-1]

        pooled = self.global_pool(final_feat)

        pooled = self.mc_dropout(pooled)

        logits = self.classifier(pooled)

        return logits

    def forward_with_features(
            self,
            x: torch.Tensor,
            segmentation_guidance=None
    ):
        """Return logits and multi-scale features."""

        features = self.forward_features(
            x,
            segmentation_guidance
        )

        pooled = self.global_pool(features[-1])

        logits = self.classifier(
            self.mc_dropout(pooled)
        )

        return logits, features

    def get_target_layer(self):
        """Return the target layer for GradCAM (last convolutional block)."""
        if self.backbone_name == "resnet50":
            return self.sspa4.block.fuse_conv[0]
        else:
            return self.features[-4]

    def enable_mc_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        self.mc_dropout.p = 0.25
        self.mc_dropout.train()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        """Disable dropout layers for standard inference."""
        self.mc_dropout.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
