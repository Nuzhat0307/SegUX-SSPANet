"""
SegUX-SSPANet

Segmentation-guided, Uncertainty-aware,
Strip-Style Pooling Attention Network.

Components:
1. ResNet50 backbone
2. SSPANet attention at each residual stage
3. Segmentation-guided attention
4. Classification head
5. MC Dropout uncertainty estimation

Classes:
0 = glioma
1 = meningioma
2 = pituitary
3 = no_tumor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ml.models.sspanet import SSPANet


class SegUXSSPANet(nn.Module):

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # ====================================================
        # RESNET50 BACKBONE
        # ====================================================

        if backbone == "resnet50":

            weights = (
                models.ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )

            resnet = models.resnet50(
                weights=weights
            )

            self.stem = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
            )

            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            self.feature_channels = [
                256,
                512,
                1024,
                2048,
            ]

        elif backbone == "vgg16":

            weights = (
                models.VGG16_Weights.DEFAULT
                if pretrained
                else None
            )

            vgg = models.vgg16(
                weights=weights
            )

            self.features = vgg.features

            self.feature_channels = [
                512,
                512,
                512,
                512,
            ]

        else:

            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        # ====================================================
        # SSPANET MODULES
        # ====================================================

        if backbone == "resnet50":

            self.sspa1 = SSPANet(256)
            self.sspa2 = SSPANet(512)
            self.sspa3 = SSPANet(1024)
            self.sspa4 = SSPANet(2048)

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        final_channels = (
            2048
            if backbone == "resnet50"
            else 512
        )

        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                final_channels,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            ),
        )

        # ====================================================
        # MC DROPOUT
        # ====================================================

        self.mc_dropout = nn.Dropout(
            0.25
        )

    # ========================================================
    # SEGMENTATION GUIDANCE PREPARATION
    # ========================================================

    def prepare_segmentation_guidance(
        self,
        segmentation_guidance,
        target_size,
    ):
        """
        Resize segmentation guidance to the
        spatial dimensions of an SSPANet feature map.

        Input:
            segmentation_guidance:
                [B, 1, H, W]

        Output:
            [B, 1, target_H, target_W]
        """

        if segmentation_guidance is None:
            return None

        if segmentation_guidance.dim() == 3:

            segmentation_guidance = (
                segmentation_guidance
                .unsqueeze(1)
            )

        guidance = F.interpolate(
            segmentation_guidance.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return guidance.clamp(
            0.0,
            1.0
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def forward_features(
        self,
        x,
        segmentation_guidance=None,
    ):
        """
        Extract multi-scale features.

        Segmentation guidance is passed to every
        SSPANet stage.
        """

        if self.backbone_name == "resnet50":

            x = self.stem(x)

            # ----------------------------------------------
            # Stage 1
            # ----------------------------------------------

            f1 = self.layer1(x)

            g1 = self.prepare_segmentation_guidance(
                segmentation_guidance,
                f1.shape[-2:],
            )

            f1 = self.sspa1(
                f1,
                g1
            )

            # ----------------------------------------------
            # Stage 2
            # ----------------------------------------------

            f2 = self.layer2(f1)

            g2 = self.prepare_segmentation_guidance(
                segmentation_guidance,
                f2.shape[-2:],
            )

            f2 = self.sspa2(
                f2,
                g2
            )

            # ----------------------------------------------
            # Stage 3
            # ----------------------------------------------

            f3 = self.layer3(f2)

            g3 = self.prepare_segmentation_guidance(
                segmentation_guidance,
                f3.shape[-2:],
            )

            f3 = self.sspa3(
                f3,
                g3
            )

            # ----------------------------------------------
            # Stage 4
            # ----------------------------------------------

            f4 = self.layer4(f3)

            g4 = self.prepare_segmentation_guidance(
                segmentation_guidance,
                f4.shape[-2:],
            )

            f4 = self.sspa4(
                f4,
                g4
            )

            return [
                f1,
                f2,
                f3,
                f4,
            ]

        else:

            features = []

            current = x

            for i, layer in enumerate(
                self.features
            ):

                current = layer(
                    current
                )

                if i in [
                    3,
                    8,
                    15,
                    22,
                ]:

                    features.append(
                        current
                    )

            return features

    # ========================================================
    # CLASSIFICATION FORWARD
    # ========================================================

    def forward(
        self,
        x,
        segmentation_guidance=None,
    ):
        """
        Classification forward pass.

        segmentation_guidance is optional.

        If no segmentation guidance is supplied,
        the model behaves like the original classifier.
        """

        features = self.forward_features(
            x,
            segmentation_guidance,
        )

        final_feat = features[-1]

        pooled = self.global_pool(
            final_feat
        )

        pooled = self.mc_dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits

    # ========================================================
    # FEATURES + LOGITS
    # ========================================================

    def forward_with_features(
        self,
        x,
        segmentation_guidance=None,
    ):
        """
        Return both classification logits
        and multi-scale features.
        """

        features = self.forward_features(
            x,
            segmentation_guidance,
        )

        pooled = self.global_pool(
            features[-1]
        )

        pooled = self.mc_dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits, features

    # ========================================================
    # GRADCAM
    # ========================================================

    def get_target_layer(self):

        if self.backbone_name == "resnet50":

            return (
                self.sspa4
                .block
                .fuse_conv[0]
            )

        return self.features[-4]

    # ========================================================
    # MC DROPOUT
    # ========================================================

    def enable_mc_dropout(self):

        self.mc_dropout.p = 0.25

        self.mc_dropout.train()

        for module in self.modules():

            if isinstance(
                module,
                nn.Dropout
            ):

                module.train()

    def disable_mc_dropout(self):

        self.mc_dropout.eval()

        for module in self.modules():

            if isinstance(
                module,
                nn.Dropout
            ):

                module.eval()