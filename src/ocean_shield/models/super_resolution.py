"""
SAR display previews via a loaded ESPCN checkpoint or deterministic interpolation.
Neither display scaling nor generated neural detail preserves native radiometry
or establishes a measured improvement in sensor resolution.
"""

import logging
import os
from typing import Optional

import cv2
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class SARESPCN(nn.Module):
    """
    Sub-pixel neural display architecture with 2x PixelShuffle output.
    Generated detail is not authenticated slick boundaries or ship reflectors.
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
        # Display-space bilinear residual, not native radiometric preservation.
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
) -> Optional[SARESPCN]:
    """
    Instantiates the SAR Super-Resolution model and loads weights if available.
    """
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

    if not model_path or not os.path.exists(model_path):
        return None
    try:
        model = SARESPCN(upscale_factor=2, in_channels=1)
        state = torch.load(model_path, map_location=device, weights_only=True)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        model.weights_loaded = True
        model.weights_path = model_path
        return model
    except Exception as exc:
        logger.warning("SAR display checkpoint unavailable (%s); using interpolation.", type(exc).__name__)
        return None


def _display_input(image: np.ndarray):
    if image is None or getattr(image, "size", 0) == 0:
        return np.zeros((0, 0), dtype=np.float32), "empty"
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("SAR preview requires a 2D image or a supported channel image.")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] in (3, 4):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported SAR display channel count.")
    values = image.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("SAR display input contains nonfinite pixels.")
    if image.dtype == np.uint8:
        return values / 255.0, "uint8_display_scaling"
    low, high = float(np.min(values)), float(np.max(values))
    if low >= 0.0 and high <= 1.0:
        return values, "unit_interval_display_scaling"
    return ((values - low) / (high - low) if high > low else np.zeros_like(values)), "min_max_display_scaling"


def interpolate_sar_display(image: np.ndarray, scale_factor: int = 2,
                            fallback_reason: str = "trained_weights_unavailable"):
    """Shared direct/outer fallback: identical bicubic, quantized display pixels."""
    if type(scale_factor) is not int or not 1 <= scale_factor <= 8:
        raise ValueError("SAR display scale_factor must be an integer in [1, 8].")
    normalized, normalization = _display_input(image)
    if normalized.size:
        h, w = normalized.shape
        display = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        output = cv2.resize(display, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
    else:
        output = np.zeros((0, 0), dtype=np.uint8)
    return output, {
        "status": "INTERPOLATED_DISPLAY_PREVIEW" if normalized.size else "NOT_ASSESSED",
        "interpolation": "bicubic", "normalization": normalization,
        "scale_factor": scale_factor, "output_dtype": "uint8",
        "trained_weights_used": False, "generated_detail": False,
        "native_radiometry_preserved": False, "sensor_resolution_improvement_verified": False,
        "fallback_reason": fallback_reason,
        "notice": "Display interpolation/quantization only; no new measured detail or native radiometric fidelity.",
    }


def enhance_sar_deep_learning(
    image: np.ndarray,
    model: SARESPCN = None,
    device: str = "cpu",
    *,
    scale_factor: int = 2,
    return_metadata: bool = False,
):
    """
    Return a uint8 display preview, optionally paired with truthful processing metadata.
    Missing/unloadable weights and inference failures use the same deterministic
    bicubic display fallback. Randomly initialized weights are never used.
    """
    fallback, metadata = interpolate_sar_display(image, scale_factor)
    if not fallback.size or scale_factor != 2:
        if scale_factor != 2:
            metadata["fallback_reason"] = "requested_scale_not_supported_by_neural_model"
        return (fallback, metadata) if return_metadata else fallback
    try:
        if model is None:
            model = load_sar_super_resolution_model(device=device)
        if getattr(model, "weights_loaded", False) is not True:
            return (fallback, metadata) if return_metadata else fallback
        normalized, normalization = _display_input(image)
        tensor_in = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out_arr = model(tensor_in).squeeze(0).squeeze(0).cpu().numpy()
        if out_arr.shape != fallback.shape or not np.isfinite(out_arr).all():
            raise ValueError("Invalid neural display output.")
        output = np.clip(out_arr * 255, 0, 255).astype(np.uint8)
        metadata.update(status="NEURAL_DISPLAY_PREVIEW", trained_weights_used=True,
                        generated_detail=True, interpolation="bilinear_residual_plus_neural_detail",
                        normalization=normalization, fallback_reason=None,
                        notice="Generated neural detail and uint8 display scaling; native radiometry/resolution gain is not verified.")
        return (output, metadata) if return_metadata else output
    except Exception as exc:
        logger.warning("SAR neural display inference unavailable (%s); using interpolation.", type(exc).__name__)
        metadata["fallback_reason"] = "model_or_inference_unavailable"
        return (fallback, metadata) if return_metadata else fallback
