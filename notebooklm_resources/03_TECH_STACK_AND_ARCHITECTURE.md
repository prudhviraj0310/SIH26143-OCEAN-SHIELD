# OCEAN-SHIELD: Tech Stack, Architecture & ML Models

## 1. System Architecture Overview
OCEAN-SHIELD is architected as an asynchronous, defense-grade Command & Control (C2) Geospatial Information System (GIS) pipeline:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        OCEAN-SHIELD ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [ SATELLITE INPUTS ]          [ MET-OCEAN FORCING ]        [ AIS FEEDS ]  │
│  Sentinel-1 SAR GeoTIFF        HYCOM 4D NetCDF (u,v)       MarineCadastre  │
│  Sentinel-2 Optical EO         NOAA GFS Wind (U10,V10)     DGLL / CSV Feed │
│           │                             │                         │        │
│           ▼                             │                         │        │
│  ┌─────────────────────────┐            │                         │        │
│  │    SAR / EO ENGINE      │            │                         │        │
│  │ • PyTorch U-Net (Dice)  │            │                         │        │
│  │ • ESPCN 2X Super-Res    │            │                         │        │
│  │ • Adaptive CFAR Blips   │            │                         │        │
│  └───────────┬─────────────┘            │                         │        │
│              │ Slicks, Centroids, Age   │                         │        │
│              ▼                          ▼                         │        │
│  ┌──────────────────────────────────────────────┐                 │        │
│  │           HYDRODYNAMIC DRIFT ENGINE          │                 │        │
│  │  • 4th-Order Runge-Kutta (RK4) Advection     │                 │        │
│  │  • Reverse Hindcasting: Release (x0, y0, t0) │                 │        │
│  │  • Forward Forecasting: Coastal Beaching     │                 │        │
│  │  • Mackay ADIOS Kinetic Weathering Model     │                 │        │
│  └──────────────────────┬───────────────────────┘                 │        │
│                         │ Origin Point (x0, y0, t0)               │        │
│                         ▼                                         ▼        │
│             ┌────────────────────────────────────────────────────────┐     │
│             │                  AIS CORRELATION ENGINE                │     │
│             │ • Spatio-Temporal Corridor Filter (Radius + Window)    │     │
│             │ • Closest Point of Approach (CPA) Calculation          │     │
│             │ • Kinematic Speed Anomaly & Loitering Profiler         │     │
│             │ • Multi-Factor Forensic Composite Suspect Scoring      │     │
│             │ • Non-Cooperative Radar Target (Dark Ship) Matching    │     │
│             └───────────────────────────┬────────────────────────────┘     │
│                                         │ Ranked Dossier & Evidence        │
│                                         ▼                                  │
│             ┌────────────────────────────────────────────────────────┐     │
│             │               DELIVERY & VISUALIZATION                 │     │
│             │ • Real-time Glassmorphic Leaflet GIS Cockpit (60 FPS)  │     │
│             │ • Automated Coast Guard Legal Dossier Generator (PDF) │     │
│             └────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack Breakdown

| Layer | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Backend Runtime** | Python 3.11.x, Uvicorn (ASGI) | High-concurrency async REST server and numerical execution. |
| **API Framework** | FastAPI $\ge 0.110$ | Strict Pydantic v2 data validation, OpenAPI autodocs, CORS handling. |
| **Deep Learning** | PyTorch $\ge 2.1$ | Tensor computation, neural network forward inference on CPU/GPU. |
| **Image Processing** | OpenCV (`opencv-python-headless`), NumPy | Speckle filtering (Enhanced Lee), morphological closing, polygon contours. |
| **Scientific Numerical** | SciPy, NetCDF4 | Coordinate transformations, spatial indexing, multidimensional array slicing. |
| **PDF Reporting** | ReportLab $\ge 4.0$ | Automated vector PDF generation for Coast Guard violation notices. |
| **Frontend GIS** | Leaflet.js 1.9.4, HTML5 | Edge-to-edge GIS canvas, tile rendering, polygon overlays, particle swarm. |
| **Frontend Styling** | Modern Vanilla CSS3 (Custom Design System) | Aerospace/defense glassmorphism (`backdrop-filter: blur(24px)`), no heavy UI dependencies (no React/Node build step), instant 60fps response. |
| **Cloud Hosting** | Render.com, GitHub Actions | Continuous deployment with automated zero-downtime background self-pinger. |

---

## 3. Deep Learning Models

### A. PyTorch SAR U-Net (`models/unet.py`)
* **Architecture**: Fully convolutional encoder-decoder neural network with skip connections.
* **Weights File**: `models/sar_unet_best.pt` (~50.1 MB, ~13.4 million parameters).
* **Layers**:
  * Encoder: 4 downsampling blocks (`DoubleConv` with $3\times3$ kernels, `BatchNorm2d`, `ReLU`, followed by $2\times2$ `MaxPool2d`). Channel progression: $1 \to 64 \to 128 \to 256 \to 512$.
  * Bottleneck: $512 \to 1024$ channels with Dropout ($p=0.5$).
  * Decoder: 4 upsampling blocks (`ConvTranspose2d` or bilinear interpolation with skip concatenation).
  * Classifier: $1\times1$ convolution producing single-channel probability map passed through Sigmoid.
* **Tiled Sliding Window Inference**: To process giant $10,000 \times 10,000$ Sentinel-1 GeoTIFFs without running out of GPU/RAM, `predict_unet_tiled()` splits scenes into overlapping $512\times512$ chips using a 2D Hann-window weighting function to prevent edge artifacts at seam boundaries.

### B. ESPCN Super-Resolution Network (`models/super_resolution.py`)
* **Full Name**: Efficient Sub-Pixel Convolutional Network (Shi et al.).
* **Weights File**: `models/sar_espcn_best.pt` (~87 KB, ultra-lightweight).
* **Purpose**: Performs $2\times$ spatial super-resolution directly in feature space before upsampling via `PixelShuffle`. Increases the effective ground resolution of blurry 20m SAR pixels to 10m, enabling precise delineation of narrow tail trails and small bilge dumps.

### C. Adaptive CFAR Edge Engine (`sar_engine.py`)
* **Purpose**: Fast tactical detection mode (under 50ms) for low-power edge deployment or when PyTorch is not available. Uses a sliding window Constant False Alarm Rate (CFAR) threshold over Enhanced Lee filtered backscatter.
