"""
SSPANet — Strip-Style Pooling Attention Network.

Includes:
1. Channel attention
2. Strip pooling
3. Style pooling
4. Segmentation-guided attention
5. Residual fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        hidden_channels = max(
            in_channels // reduction,
            1
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        self.fc = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                1,
                bias=False,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                in_channels,
                1,
                bias=False,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_out + max_out
        )

        return attention * x


# ============================================================
# STRIP POOLING
# ============================================================

class StripPooling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1d_h = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=False,
        )

        self.conv1d_v = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=False,
        )

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.shape

        # Horizontal pooling
        h_pool = F.adaptive_avg_pool2d(
            x,
            (1, w)
        )

        h_pool = h_pool.squeeze(2)

        h_pool = self.conv1d_h(
            h_pool
        )

        h_pool = h_pool.unsqueeze(2)

        # Vertical pooling
        v_pool = F.adaptive_avg_pool2d(
            x,
            (h, 1)
        )

        v_pool = v_pool.squeeze(3)

        v_pool = self.conv1d_v(
            v_pool
        )

        v_pool = v_pool.unsqueeze(3)

        # Expand
        h_pool = h_pool.expand(
            b,
            -1,
            h,
            w
        )

        v_pool = v_pool.expand(
            b,
            -1,
            h,
            w
        )

        combined = self.bn(
            self.conv2d(x)
            + h_pool
            + v_pool
        )

        attention = self.sigmoid(
            combined
        )

        return attention * x


# ============================================================
# STYLE POOLING
# ============================================================

class StylePooling(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels * 2,
            out_channels,
            1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        mean = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        std = x.std(
            dim=(2, 3),
            keepdim=True
        )

        style = torch.cat(
            [mean, std],
            dim=1
        )

        style = self.conv(
            style
        )

        style = self.bn(
            style
        )

        attention = self.sigmoid(
            style
        )

        return attention * x


# ============================================================
# SSPANET BLOCK
# ============================================================

class SSPANetBlock(nn.Module):

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.channel_attn = (
            ChannelAttention(channels)
        )

        self.strip_pool = (
            StripPooling(
                channels,
                channels
            )
        )

        self.style_pool = (
            StylePooling(
                channels,
                channels
            )
        )

        # Segmentation guidance projection
        self.guidance_conv = nn.Sequential(

            nn.Conv2d(
                1,
                channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.Sigmoid(),
        )

        self.fuse_conv = nn.Sequential(

            nn.Conv2d(
                channels * 3,
                channels,
                1,
                bias=False,
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x,
        segmentation_guidance=None,
    ):

        # ----------------------------------------------------
        # Standard SSPANet branches
        # ----------------------------------------------------

        ca = self.channel_attn(x)

        sp = self.strip_pool(x)

        stp = self.style_pool(x)

        # ----------------------------------------------------
        # Fuse SSPANet branches
        # ----------------------------------------------------

        fused = self.fuse_conv(
            torch.cat(
                [ca, sp, stp],
                dim=1,
            )
        )

        # ----------------------------------------------------
        # Segmentation guidance
        # ----------------------------------------------------

        if segmentation_guidance is not None:

            # Resize mask to feature-map size
            guidance = F.interpolate(
                segmentation_guidance.float(),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            guidance = guidance.clamp(
                0.0,
                1.0
            )

            # Convert 1-channel mask
            # into channel-wise attention
            guidance_attention = (
                self.guidance_conv(
                    guidance
                )
            )

            # Emphasize tumor regions
            # while retaining background features
            guided = x * (
                1.0
                + guidance_attention
            )

            fused = fused + guided

        # ----------------------------------------------------
        # Residual output
        # ----------------------------------------------------

        return x + fused


# ============================================================
# SSPANET
# ============================================================

class SSPANet(nn.Module):

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.block = SSPANetBlock(
            channels
        )

    def forward(
        self,
        x,
        segmentation_guidance=None,
    ):

        return self.block(
            x,
            segmentation_guidance,
        )