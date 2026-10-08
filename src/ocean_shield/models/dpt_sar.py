"""
DPT-SAR: Dense Prediction Transformer Head for SAR Oil Spill Segmentation

Architecture adapted from Depth-Anything-V2 (8.2k stars)
    - Multi-scale feature fusion with progressive refinement
    - DINOv2-compatible encoder interface
    - Optimized for single-channel SAR imagery

Reference: github.com/DepthAnything/Depth-Anything-V2/depth_anything_v2/dpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionBlock(nn.Module):
    """Multi-scale feature fusion with residual refinement.
    Adapted from Depth-Anything-V2's DPT architecture."""

    def __init__(self, features: int, use_bn: bool = True):
        super().__init__()
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None,
                target_size: tuple = None) -> torch.Tensor:
        if skip is not None:
            x = x + skip
        x = self.res_conv1(x)
        x = self.res_conv2(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        return x


class DPTHead(nn.Module):
    """Dense Prediction Transformer Head — adapted from Depth-Anything-V2.

    Takes multi-scale encoder features and progressively fuses them
    through refinement blocks to produce a dense prediction at the
    input resolution.

    Differences from the original:
    - Output channels = 2 (oil vs non-oil) instead of 1 (depth)
    - No CLS token readout (SAR encoders don't use [CLS])
    - Uses BatchNorm by default (radar imagery benefits from BN)
    """

    def __init__(self, in_channels: int = 64, features: int = 128,
                 out_channels: list = None, num_classes: int = 2):
        super().__init__()
        if out_channels is None:
            out_channels = [64, 128, 256, 256]

        # 1×1 projections from encoder features to fusion features
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=oc,
                      kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        # Spatial resize layers (ConvTranspose for upscale, Conv for downscale)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Scratch layers for channel alignment
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, padding=1, bias=False)

        # Progressive fusion (coarse → fine)
        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        # Output head → segmentation logits
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, features: list, h: int, w: int) -> torch.Tensor:
        """
        Args:
            features: List of 4 multi-scale feature tensors [B, C, H_i, W_i]
            h, w: Target spatial resolution
        """
        out = []
        for i, x in enumerate(features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # Progressive refinement (coarse → fine)
        l4 = self.layer4_rn(layer_4)
        l3 = self.layer3_rn(layer_3)
        l2 = self.layer2_rn(layer_2)
        l1 = self.layer1_rn(layer_1)

        path_4 = self.refinenet4(l4, target_size=l3.shape[2:])
        path_3 = self.refinenet3(path_4, l3, target_size=l2.shape[2:])
        path_2 = self.refinenet2(path_3, l2, target_size=l1.shape[2:])
        path_1 = self.refinenet1(path_2, l1)

        out = self.output_conv(path_1)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out


class MultiScaleEncoder(nn.Module):
    """Lightweight multi-scale CNN encoder for SAR imagery.
    Produces 4-level feature hierarchy compatible with DPTHead.
    Designed for single-channel SAR input (VV polarization)."""

    def __init__(self, in_channels: int = 1, base_features: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(base_features, base_features, 2)
        self.stage2 = self._make_stage(base_features, base_features, 2, stride=2)
        self.stage3 = self._make_stage(base_features, base_features, 2, stride=2)
        self.stage4 = self._make_stage(base_features, base_features, 2, stride=2)

    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class DPT_SAR(nn.Module):
    """DPT-SAR: Complete Dense Prediction Transformer for SAR Oil Spill Segmentation.

    Combines a multi-scale CNN encoder with the DPT head architecture
    from Depth-Anything-V2. Produces pixel-level oil/non-oil segmentation.

    Architecture reference: Depth-Anything-V2 (github.com/DepthAnything/Depth-Anything-V2)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_features: int = 64, head_features: int = 128):
        super().__init__()
        self.encoder = MultiScaleEncoder(in_channels, base_features)
        self.head = DPTHead(
            in_channels=base_features,
            features=head_features,
            out_channels=[base_features] * 4,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        features = self.encoder(x)
        logits = self.head(features, h, w)
        return logits

    @torch.no_grad()
    def predict(self, sar_image_np, device: str = "cpu") -> "np.ndarray":
        """Run inference on a numpy SAR image and return segmentation mask."""
        import numpy as np
        self.eval()
        if sar_image_np.ndim == 2:
            sar_image_np = sar_image_np[None, None, :, :]  # Add batch + channel
        elif sar_image_np.ndim == 3:
            sar_image_np = sar_image_np[None, :, :, :]
        tensor = torch.from_numpy(sar_image_np.astype(np.float32) / 255.0).to(device)
        logits = self.forward(tensor)
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        return mask
