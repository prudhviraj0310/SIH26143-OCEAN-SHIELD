"""
OCEAN-SHIELD Deep Learning Neural Network Models

Model Zoo:
- SAR_UNet: Encoder-decoder segmentation (original)
- DPT_SAR: Dense Prediction Transformer for SAR (adapted from Depth-Anything-V2)
- SARTrainer: Production training pipeline (adapted from Depth-Anything-V2)
- export_to_onnx: ONNX model export (adapted from Metric3D)
"""
from .unet import SAR_UNet, DiceBCELoss
from .dpt_sar import DPT_SAR, DPTHead, MultiScaleEncoder

__all__ = [
    "SAR_UNet",
    "DiceBCELoss",
    "DPT_SAR",
    "DPTHead",
    "MultiScaleEncoder",
]

