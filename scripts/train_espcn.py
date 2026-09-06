"""
Train the SAR ESPCN (Efficient Sub-Pixel Convolutional Network) Super-Resolution Model
on real Sentinel-1 SAR imagery (Zenodo repository) to reconstruct high-frequency
radar backscatter transitions, slick boundaries, and vessel corner reflectors.

Addressing NTRO specification: 'Enhancing Satellite Imagery Readability with Super-Resolution'
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ocean_shield.models.super_resolution import SARESPCN

DEFAULT_TIFF_PATH = os.path.join(PROJECT_ROOT, "datasets", "real_sar", "2018_09_26.tif")
DEFAULT_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "sar_espcn_best.pt")


def extract_patches(scene: np.ndarray, num_patches: int = 1500, patch_size: int = 64, scale: int = 2):
    """
    Extracts paired LR and HR patches from calibrated real SAR scene.
    HR = original high-resolution radar patch (patch_size * scale, patch_size * scale)
    LR = bicubic downsampled patch (patch_size, patch_size)
    """
    import cv2

    hr_size = patch_size * scale
    h, w = scene.shape[:2]
    if len(scene.shape) == 3:
        scene = scene[:, :, 0]

    hr_patches = []
    lr_patches = []
    np.random.seed(42)

    for _ in range(num_patches):
        y = np.random.randint(0, h - hr_size)
        x = np.random.randint(0, w - hr_size)
        hr = scene[y:y + hr_size, x:x + hr_size]

        # Downsample with bicubic filter to simulate lower-resolution SAR sensor
        lr = cv2.resize(hr, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

        hr_patches.append(hr)
        lr_patches.append(lr)

    hr_arr = np.array(hr_patches, dtype=np.float32)
    lr_arr = np.array(lr_patches, dtype=np.float32)

    # Tensor format: (N, 1, H, W)
    hr_tensor = torch.from_numpy(hr_arr).unsqueeze(1)
    lr_tensor = torch.from_numpy(lr_arr).unsqueeze(1)
    return lr_tensor, hr_tensor


def train(
    tiff_path: str = DEFAULT_TIFF_PATH,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_patches: int = 1500,
    device: str = "cpu"
):
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"Real SAR Sentinel-1 TIFF not found at: {tiff_path}")

    print(f"📡 Loading real Sentinel-1 SAR scene: {tiff_path}")
    try:
        import tifffile
        scene = tifffile.imread(tiff_path).astype(np.float32)
    except ImportError:
        import cv2
        scene = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if scene is None:
        raise ValueError("Failed to read SAR TIFF file.")

    # Normalize backscatter to [0.0, 1.0]
    s_min, s_max = float(scene.min()), float(scene.max())
    if s_max > 1.0:
        scene = np.clip((scene - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0)

    print(f"   Scene dimensions: {scene.shape}, dynamic range: [{scene.min():.3f}, {scene.max():.3f}]")

    print(f"✂️ Extracting {num_patches} paired (LR, HR) patches for 2x super-resolution...")
    lr_tensor, hr_tensor = extract_patches(scene, num_patches=num_patches, patch_size=64, scale=2)
    print(f"   LR input: {lr_tensor.shape}, HR target: {hr_tensor.shape}")

    model = SARESPCN(upscale_factor=2, in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = nn.L1Loss()

    best_loss = float("inf")
    n_samples = lr_tensor.shape[0]

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    print(f"\n🚀 Training SAR ESPCN on {device.upper()} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            lr_b = lr_tensor[idx].to(device)
            hr_b = hr_tensor[idx].to(device)

            optimizer.zero_grad()
            pred = model(lr_b)
            loss = criterion(pred, hr_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch + 1:2d}/{epochs:2d} | L1 Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\n✅ Training complete! Best L1 Loss: {best_loss:.6f}")
    print(f"💾 Saved weights to: {weights_path} ({os.path.getsize(weights_path)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAR ESPCN Super-Resolution Model")
    parser.add_argument("--tiff", type=str, default=DEFAULT_TIFF_PATH, help="Path to real SAR TIFF")
    parser.add_argument("--output", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to save .pt weights")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patches", type=int, default=1500, help="Number of patches to extract")
    args = parser.parse_args()

    train(
        tiff_path=args.tiff,
        weights_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_patches=args.patches
    )
