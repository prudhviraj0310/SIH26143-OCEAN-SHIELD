"""
ONNX Model Export & Deployment Utilities

Adapted from Metric3D (github.com/YvanYin/Metric3D/onnx/)
Provides model export, optimization, and inference for production deployment.

Supports:
- PyTorch → ONNX conversion for U-Net and DPT-SAR models
- ONNX Runtime inference for CPU/GPU deployment
- Model quantization (INT8) for edge deployment
- TorchScript tracing for mobile deployment
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 512, 512),
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    simplify: bool = True,
) -> Dict[str, Any]:
    """Export a PyTorch model to ONNX format.

    Adapted from Metric3D's ONNX export pipeline.

    Args:
        model: PyTorch model to export
        output_path: Path to save the .onnx file
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version (17 recommended)
        dynamic_axes: Dynamic axis specification for variable batch/resolution
        simplify: Whether to run onnx-simplifier

    Returns:
        Export metadata dict
    """
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Default dynamic axes for variable batch size and resolution
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Export
    start_time = time.time()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    export_time = time.time() - start_time

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Simplify if requested
    simplified = False
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                simplified = True
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info("ONNX model simplified successfully")
        except ImportError:
            logger.warning("onnx-simplifier not installed; skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    metadata = {
        "status": "SUCCESS",
        "output_path": output_path,
        "file_size_mb": round(file_size_mb, 2),
        "opset_version": opset_version,
        "input_shape": list(input_shape),
        "export_time_seconds": round(export_time, 2),
        "simplified": simplified,
        "dynamic_axes": bool(dynamic_axes),
    }

    logger.info(f"ONNX export complete: {output_path} ({file_size_mb:.1f} MB)")
    return metadata


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    quantization_type: str = "dynamic",
) -> Dict[str, Any]:
    """Quantize an ONNX model for faster inference.

    Args:
        input_path: Path to the input ONNX model
        output_path: Path to save the quantized model
        quantization_type: "dynamic" (INT8 dynamic), "static", or "float16"

    Returns:
        Quantization metadata
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        return {"status": "FAILED", "error": "onnxruntime.quantization not available"}

    start_time = time.time()

    if quantization_type == "dynamic":
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
    else:
        return {"status": "FAILED", "error": f"Unsupported quantization type: {quantization_type}"}

    quant_time = time.time() - start_time
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "status": "SUCCESS",
        "output_path": output_path,
        "quantization_type": quantization_type,
        "original_size_mb": round(original_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "compression_ratio": round(original_size / max(quantized_size, 0.01), 2),
        "quantization_time_seconds": round(quant_time, 2),
    }


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for production deployment.

    Adapted from Metric3D's test_onnx.py for SAR model inference.
    """

    def __init__(self, model_path: str, providers: Optional[list] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required: pip install onnxruntime")

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"ONNX model loaded: {model_path}, input shape: {input_shape}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            Segmentation mask as numpy array
        """
        # Normalize and reshape
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]  # (1, C, H, W)

        image = image.astype(np.float32) / 255.0

        # Run inference
        start = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: image})
        inference_time = time.time() - start

        result = outputs[0]
        if result.shape[1] > 1:  # Multi-class logits
            mask = np.argmax(result, axis=1).squeeze(0).astype(np.uint8) * 255
        else:
            mask = (result.squeeze() > 0.5).astype(np.uint8) * 255

        logger.debug(f"ONNX inference: {inference_time*1000:.1f}ms")
        return mask

    def benchmark(self, input_shape: Tuple[int, ...] = (1, 1, 512, 512),
                  num_runs: int = 50) -> Dict[str, float]:
        """Benchmark inference speed."""
        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(5):
            self.session.run([self.output_name], {self.input_name: dummy})

        times = []
        for _ in range(num_runs):
            start = time.time()
            self.session.run([self.output_name], {self.input_name: dummy})
            times.append((time.time() - start) * 1000)

        return {
            "mean_ms": round(np.mean(times), 2),
            "median_ms": round(np.median(times), 2),
            "min_ms": round(np.min(times), 2),
            "max_ms": round(np.max(times), 2),
            "std_ms": round(np.std(times), 2),
            "fps": round(1000.0 / np.mean(times), 1),
            "num_runs": num_runs,
        }


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 512, 512),
) -> Dict[str, Any]:
    """Export model to TorchScript for mobile/edge deployment."""
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    dummy_input = torch.randn(*input_shape, device=device)

    start_time = time.time()
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)
    export_time = time.time() - start_time

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "status": "SUCCESS",
        "output_path": output_path,
        "file_size_mb": round(file_size_mb, 2),
        "export_time_seconds": round(export_time, 2),
        "format": "TorchScript",
    }
