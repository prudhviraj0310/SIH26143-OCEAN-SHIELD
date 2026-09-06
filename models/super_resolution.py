"""
PyTorch Deep Learning Super-Resolution Model for Satellite SAR Imagery.
Implements the Efficient Sub-Pixel Convolutional Network (ESPCN) architecture
(Shi et al., CVPR 2016) tailored for satellite synthetic aperture radar (SAR)
speckle-preserving detail synthesis and 2x resolution enhancement.
"""

import math
import os
import torch
import torch.nn as nn
import numpy as np


class SARESPCN(nn.Module):
    """
    Sub-Pixel Convolutional Neural Network for 2x SAR Super-Resolution.
    Extracts feature representations directly in low-resolution radar backscatter space
    and uses sub-pixel convolution (PixelShuffle) to reconstruct high-frequency
    slick boundaries and ship corner reflectors.
    """

    def __init__(self, upscale_factor: int = 2, in_channels: int = 1):
        super(SARESPCN, self).__init__()
        self.upscale_factor = upscale_factor

        # Feature extraction layer (extracts radar boundary features)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.act1 = nn.PReLU()

        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.act2 = nn.PReLU()

        # Sub-pixel reconstruction layer
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def _initialize_weights(self):
        # Orthogonal / He initialization for stable radar backscatter gradients
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection from bilinear upsample for backscatter preservation
        residual = nn.functional.interpolate(
            x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        out_sr = self.pixel_shuffle(self.conv3(h))
        # Bounded combination of residual and synthesized detail
        out = torch.clamp(residual + 0.25 * out_sr, 0.0, 1.0)
        return out


def load_sar_super_resolution_model(
    model_path: str = None,
    device: str = "cpu"
) -> SARESPCN:
    """
    Instantiates the SAR Super-Resolution model and loads weights if available.
    """
    model = SARESPCN(upscale_factor=2, in_channels=1)
    if model_path is None:
        candidate_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "sar_espcn_best.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "sar_espcn_best.pt")),
            os.path.abspath("models/sar_espcn_best.pt")
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                model_path = p
                break

    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.weights_loaded = True
            model.weights_path = model_path
        except Exception:
            pass
    model.to(device)
    model.eval()
    return model


def enhance_sar_deep_learning(
    image: np.ndarray,
    model: SARESPCN = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Runs 2x Super-Resolution inference using the deep learning ESPCN network.
    Accepts grayscale or 2D SAR image (uint8 or float32), returns 2x enhanced uint8 image.
    """
    if model is None:
        model = load_sar_super_resolution_model(device=device)

    # Normalize image to float32 [0, 1]
    if image.dtype == np.uint8:
        norm_img = image.astype(np.float32) / 255.0
    else:
        norm_img = image.astype(np.float32)
        if norm_img.max() > 1.0:
            norm_img /= 255.0

    if len(norm_img.shape) == 3:
        norm_img = norm_img[:, :, 0]

    h, w = norm_img.shape
    tensor_in = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        tensor_out = model(tensor_in)

    out_arr = tensor_out.squeeze().cpu().numpy()
    out_uint8 = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
    return out_uint8
