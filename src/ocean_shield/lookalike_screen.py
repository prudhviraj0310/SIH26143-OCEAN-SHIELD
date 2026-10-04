"""
Ocean Shield: Scale-Free Dark-Patch Look-Alike Screening Module
==============================================================
Experimental analyst triage module that extracts 7 scale-free morphological
and statistical contrast ratios for dark radar patches to assist human screening.

IMPORTANT SCIENTIFIC CAVEAT:
This module provides uncalibrated feature engineering for analyst triage.
It does NOT output calibrated probabilities and cannot confirm mineral oil or
discharge mechanisms from a single SAR acquisition without ancillary in-situ,
optical, or verified AIS/source evidence.

References:
- Brekke, C. & Solberg, A.H.S. (2005). "Oil spill detection by satellite remote sensing."
  Remote Sensing of Environment, 95(1), 1-13.
- Solberg, A.H.S. et al. (2007). "Oil spill detection in SAR images using a composite
  neural network and look-alike discrimination." IEEE TGRS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LookalikeFeature:
    """A dimensionless ratio quantifying a physical property of the dark patch."""
    name: str
    value: float
    oil_direction: int  # +1 if higher argues for mineral oil, -1 if lower argues for oil
    description: str


@dataclass
class LookalikeScreenResult:
    """Result of look-alike screening evaluation for analyst triage."""
    verdict: str  # Includes NOT_ASSESSED when usable evidence is absent
    screening_index: Optional[float]  # Uncalibrated [0, 1] index; null when not assessed
    rejection_reasons: List[str]
    supporting_reasons: List[str]
    features: Dict[str, Optional[float]]
    feature_details: List[Dict[str, Any]]
    status: str
    reason: Optional[str] = None
    support: Dict[str, Any] = field(default_factory=dict)
    calibration_status: str = "UNCALIBRATED_EXPERIMENTAL_HEURISTIC"
    disclaimer: str = (
        "This score is an uncalibrated morphological heuristic for analyst triage and "
        "does not constitute physical or chemical oil confirmation."
    )


FEATURE_NAMES = (
    "darkness_z", "darkness_p10_z", "texture_ratio", "edge_sharpness",
    "compactness", "solidity", "elongation",
)


def _unavailable_features(reason: str, status: str = "UNAVAILABLE", support=None) -> Dict[str, Any]:
    return {**dict.fromkeys(FEATURE_NAMES), "status": status, "reason": reason,
            "support": support or {}}


def _unavailable_result(reason: str, status: str = "UNAVAILABLE", support=None) -> LookalikeScreenResult:
    return LookalikeScreenResult(
        verdict="NOT_ASSESSED", screening_index=None, status=status, reason=reason,
        rejection_reasons=[], supporting_reasons=[], features=dict.fromkeys(FEATURE_NAMES),
        feature_details=[], support=support or {},
    )


def compute_scale_free_ratios(
    image_gray: np.ndarray,
    contour: np.ndarray,
    annulus_px: int = 20,
    guard_px: int = 3
) -> Dict[str, Any]:
    """Extract seven uncalibrated ratios with explicit support/status metadata.

    Missing pixels are never imputed for screening. The patch, local annulus and
    every Sobel stencil used by the edge ratio require finite, unmasked support.
    Truncated/flat backgrounds abstain rather than becoming low-risk evidence.
    Missing pixels elsewhere in the scene do not invalidate a supported patch.
    """
    if (not isinstance(image_gray, np.ndarray) or image_gray.ndim != 2
            or image_gray.dtype.kind not in "iuf"):
        return _unavailable_features("A real numeric 2D image is required.", "INVALID_INPUT")
    h, w = image_gray.shape
    if h < 5 or w < 5:
        return _unavailable_features("Image is too small for local support.", "INVALID_INPUT")
    if (isinstance(annulus_px, (bool, np.bool_)) or isinstance(guard_px, (bool, np.bool_))
            or not isinstance(annulus_px, (int, np.integer))
            or not isinstance(guard_px, (int, np.integer))
            or annulus_px <= 0 or guard_px < 0
            or annulus_px + guard_px > max(h, w)):
        return _unavailable_features("Invalid annulus/guard window.", "INVALID_INPUT")
    if (not isinstance(contour, np.ndarray) or contour.ndim not in (2, 3)
            or contour.shape[0] < 3 or contour.shape[1:] not in ((2,), (1, 2))
            or contour.dtype.kind not in "iuf" or np.any(np.ma.getmaskarray(contour))):
        return _unavailable_features("A supported contour with at least three 2D vertices is required.", "INVALID_INPUT")
    points = np.ma.getdata(contour).reshape(-1, 2)
    if (not np.all(np.isfinite(points)) or np.any(points < 0)
            or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1)):
        return _unavailable_features("Contour vertices must be finite and inside the image.", "INVALID_INPUT")
    geometry = points.astype(np.float32).reshape(-1, 1, 2)
    area_px = float(cv2.contourArea(geometry))
    perimeter_px = float(cv2.arcLength(geometry, closed=True))
    hull_area = float(cv2.contourArea(cv2.convexHull(geometry)))
    dim1, dim2 = cv2.minAreaRect(geometry)[1]
    if (not all(math.isfinite(v) for v in (area_px, perimeter_px, hull_area, dim1, dim2))
            or area_px < 10 or perimeter_px < 4 or hull_area <= 0 or min(dim1, dim2) <= 0):
        return _unavailable_features("Degenerate contour has no usable shape ratios.", "INVALID_INPUT")

    # Keep the full local annulus and its one-pixel derivative halo in the scene.
    margin = annulus_px + guard_px + 1
    if (np.any(points.min(axis=0) < margin)
            or points[:, 0].max() + margin > w - 1 or points[:, 1].max() + margin > h - 1):
        return _unavailable_features("Local background/gradient support is truncated at the image boundary.")
    patch_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(patch_mask, [np.rint(geometry).astype(np.int32)], -1, 255, -1)
    kernel_guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
    radius = guard_px + annulus_px
    kernel_total = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    annulus_mask = (cv2.dilate(patch_mask, kernel_total) > 0) & (cv2.dilate(patch_mask, kernel_guard) == 0)
    stencil = np.ones((3, 3), np.uint8)
    boundary_mask = cv2.morphologyEx(patch_mask, cv2.MORPH_GRADIENT, stencil) > 0
    required_mask = (patch_mask > 0) | (cv2.dilate((boundary_mask | annulus_mask).astype(np.uint8), stencil) > 0)

    with np.errstate(over="ignore", invalid="ignore"):
        gray_f = np.asarray(np.ma.getdata(image_gray), dtype=np.float64)
    valid_mask = ~np.ma.getmaskarray(image_gray) & np.isfinite(gray_f)
    support = {
        "patch_pixels": int(np.count_nonzero(patch_mask)),
        "background_pixels": int(np.count_nonzero(annulus_mask)),
        "required_pixels": int(np.count_nonzero(required_mask)),
        "missing_required_pixels": int(np.count_nonzero(required_mask & ~valid_mask)),
        "policy": "full finite unmasked local patch, annulus and gradient-stencil support",
    }
    if support["patch_pixels"] < 10 or support["background_pixels"] < 30 or not np.any(boundary_mask):
        return _unavailable_features("Insufficient local patch/background/edge samples.", support=support)
    if support["missing_required_pixels"]:
        return _unavailable_features("Masked or non-finite pixels in required screening support.", support=support)
    patch_pixels = gray_f[patch_mask > 0]
    bg_pixels = gray_f[annulus_mask]
    with np.errstate(over="ignore", invalid="ignore"):
        mu_bg, sigma_bg = float(np.mean(bg_pixels)), float(np.std(bg_pixels))
        mu_patch, sigma_patch = float(np.mean(patch_pixels)), float(np.std(patch_pixels))
    if (not all(math.isfinite(v) for v in (mu_bg, sigma_bg, mu_patch, sigma_patch))
            or sigma_bg <= 0):
        return _unavailable_features("Local background variance is absent or numerically unusable.", support=support)

    # Zero-fill only unused pixels after support validation. No sampled Sobel
    # stencil intersects these pixels, so backing data cannot affect a ratio.
    gradient_image = np.where(valid_mask, gray_f, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        sobel_x = cv2.Sobel(gradient_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gradient_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobel_x, sobel_y)
        mean_edge_grad = float(np.mean(grad_mag[boundary_mask]))
        mean_bg_grad = float(np.mean(grad_mag[annulus_mask]))
    if not math.isfinite(mean_edge_grad) or not math.isfinite(mean_bg_grad) or mean_bg_grad <= 0:
        return _unavailable_features("Background gradient support is numerically unusable.", support=support)
    values = {
        "darkness_z": (mu_bg - mu_patch) / sigma_bg,
        "darkness_p10_z": (mu_bg - float(np.percentile(patch_pixels, 10))) / sigma_bg,
        "texture_ratio": sigma_patch / sigma_bg,
        "edge_sharpness": mean_edge_grad / mean_bg_grad,
        "compactness": min(4.0 * math.pi * area_px / perimeter_px ** 2, 1.0),
        "solidity": min(area_px / hull_area, 1.0),
        "elongation": max(dim1, dim2) / min(dim1, dim2),
    }
    if not all(math.isfinite(value) for value in values.values()):
        return _unavailable_features("Feature computation is non-finite; ratios are withheld.", support=support)
    return {**{key: round(value, 3) for key, value in values.items()},
            "status": "ASSESSED", "reason": None, "support": support}


def evaluate_lookalike_screening(features: Dict[str, Any]) -> LookalikeScreenResult:
    """
    Evaluates scale-free features to provide an uncalibrated triage ranking for human review.
    
    DISCLAIMER:
    This function computes a heuristic triage index. It is NOT a calibrated probability
    and does NOT prove or confirm the presence of mineral oil.
    """
    rejection_reasons = []
    supporting_reasons = []

    if not isinstance(features, dict) or not features:
        return _unavailable_result("Screening features are not supplied.")
    support = features.get("support") if isinstance(features.get("support"), dict) else {}
    if "status" in features:
        extraction_status = features["status"]
        if not isinstance(extraction_status, str):
            return _unavailable_result("Malformed feature-extraction status.", "INVALID_INPUT")
        if extraction_status != "ASSESSED":
            status = extraction_status if extraction_status in ("UNAVAILABLE", "INVALID_INPUT") else "INVALID_INPUT"
            reason = features.get("reason")
            return _unavailable_result(reason if isinstance(reason, str) and reason else "Feature extraction was not assessed.", status, support)
    if "support" in features:
        count_keys = ("patch_pixels", "background_pixels", "required_pixels", "missing_required_pixels")
        if (not isinstance(features["support"], dict)
                or any(isinstance(support.get(key), (bool, np.bool_))
                       or not isinstance(support.get(key), (int, np.integer))
                       or support[key] < 0 for key in count_keys)):
            return _unavailable_result("Malformed screening-support counts.", "INVALID_INPUT")
        if (support["required_pixels"] < support["patch_pixels"] + support["background_pixels"]
                or support["missing_required_pixels"] > support["required_pixels"]):
            return _unavailable_result("Inconsistent screening-support counts.", "INVALID_INPUT", support)
        if (support["missing_required_pixels"] or support["patch_pixels"] < 10
                or support["background_pixels"] < 30):
            return _unavailable_result("Screening-support counts do not meet the complete-local-support policy.", support=support)
    missing = [name for name in FEATURE_NAMES if name not in features]

    # Validate every required feature before any triage rule runs. Signed contrast
    # is permitted; variances/gradients are non-negative, shape ratios are bounded.
    cleaned_features = {}
    for k in FEATURE_NAMES:
        if k not in features:
            continue
        val = features[k]
        if (isinstance(val, (bool, np.bool_)) or np.ma.is_masked(val)
                or not isinstance(val, (Real, np.integer, np.floating))):
            return _unavailable_result(f"{k} must be an unmasked finite real number.", "INVALID_INPUT", support)
        try:
            numeric = float(val)
        except (TypeError, ValueError, OverflowError):
            return _unavailable_result(f"{k} is not numerically usable.", "INVALID_INPUT", support)
        if not math.isfinite(numeric):
            return _unavailable_result(f"{k} must be finite.", "INVALID_INPUT", support)
        if ((k in ("texture_ratio", "edge_sharpness") and numeric < 0)
                or (k in ("compactness", "solidity") and not 0 < numeric <= 1)
                or (k == "elongation" and numeric < 1)):
            return _unavailable_result(f"{k} is outside its feature domain.", "INVALID_INPUT", support)
        cleaned_features[k] = numeric
    if missing:
        return _unavailable_result("Required features are missing: " + ", ".join(missing), support=support)

    # 1. Capillary wave damping check
    if cleaned_features["darkness_z"] < 0.8:
        rejection_reasons.append(
            f"Low backscatter contrast ({cleaned_features['darkness_z']:.2f} sigma < 0.80 sigma); "
            "consistent with shallow low-wind speed depression rather than viscous damping."
        )
    else:
        supporting_reasons.append(
            f"Pronounced backscatter contrast ({cleaned_features['darkness_z']:.2f} sigma below ambient water)."
        )

    # 2. Edge gradient step check
    if cleaned_features["edge_sharpness"] < 1.15:
        rejection_reasons.append(
            f"Fading diffuse boundary ({cleaned_features['edge_sharpness']:.2f}x background gradient); "
            "consistent with natural wind shadow gradient without sharp interfacial tension."
        )
    else:
        supporting_reasons.append(
            f"Distinct boundary transition ({cleaned_features['edge_sharpness']:.2f}x ambient gradient)."
        )

    # 3. Shape compactness vs elongation check
    if cleaned_features["compactness"] > 0.65 and cleaned_features["elongation"] < 1.8:
        rejection_reasons.append(
            f"High compactness ({cleaned_features['compactness']:.2f}) and low elongation ({cleaned_features['elongation']:.2f}); "
            "typical of symmetric algal bloom patches or rain downdraft cells rather than ship discharges."
        )
    elif cleaned_features["elongation"] >= 2.5:
        supporting_reasons.append(
            f"Linear elongation ({cleaned_features['elongation']:.2f}:1 axis ratio); "
            "consistent with maritime advection or moving vessel wake discharge."
        )

    # 4. Interior texture homogeneity
    if cleaned_features["texture_ratio"] > 1.25:
        rejection_reasons.append(
            f"High interior speckle variance ({cleaned_features['texture_ratio']:.2f}x ambient); "
            "surface shows no short-wave smoothing."
        )
    elif cleaned_features["texture_ratio"] < 0.85:
        supporting_reasons.append(
            f"Smooth damped interior texture ({cleaned_features['texture_ratio']:.2f}x ambient speckle)."
        )

    # Uncalibrated linear logit aggregation for triage priority
    logit = (
        -1.2
        + 0.85 * cleaned_features["darkness_z"]
        + 0.45 * cleaned_features["darkness_p10_z"]
        - 1.10 * cleaned_features["texture_ratio"]
        + 0.70 * cleaned_features["edge_sharpness"]
        - 1.50 * cleaned_features["compactness"]
        - 0.80 * cleaned_features["solidity"]
        + 0.35 * min(cleaned_features["elongation"], 8.0)
    )
    if not math.isfinite(logit):
        return _unavailable_result("Feature aggregation is non-finite.", "INVALID_INPUT", support)
    score = 1.0 / (1.0 + math.exp(-max(min(logit, 10.0), -10.0)))
    score = round(float(score), 3)

    # Triage priority tiers (explicitly avoids claiming confirmation or calibrated probability)
    if score >= 0.65:
        verdict = "CANDIDATE_HIGH_CONTRAST"
    elif score <= 0.35:
        verdict = "LOOKALIKE_PRIORITY_LOW"
    else:
        verdict = "AMBIGUOUS_INTERMEDIATE"

    specs = [
        {"name": "darkness_z", "value": cleaned_features["darkness_z"], "unit": "sigma_bg", "oil_direction": "+1"},
        {"name": "darkness_p10_z", "value": cleaned_features["darkness_p10_z"], "unit": "sigma_bg", "oil_direction": "+1"},
        {"name": "texture_ratio", "value": cleaned_features["texture_ratio"], "unit": "ratio", "oil_direction": "-1"},
        {"name": "edge_sharpness", "value": cleaned_features["edge_sharpness"], "unit": "ratio", "oil_direction": "+1"},
        {"name": "compactness", "value": cleaned_features["compactness"], "unit": "isoperimetric", "oil_direction": "-1"},
        {"name": "solidity", "value": cleaned_features["solidity"], "unit": "ratio", "oil_direction": "-1"},
        {"name": "elongation", "value": cleaned_features["elongation"], "unit": "ratio", "oil_direction": "+1"},
    ]

    return LookalikeScreenResult(
        verdict=verdict,
        screening_index=score,
        status="ASSESSED",
        support=support,
        rejection_reasons=rejection_reasons,
        supporting_reasons=supporting_reasons,
        features=cleaned_features,
        feature_details=specs,
    )
