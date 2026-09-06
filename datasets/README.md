# SIH 2026 Problem Statement SIH26143: Official Datasets & References

This folder contains the datasets and schemas specified in the official NTRO / Smart India Hackathon 2026 problem statement.

---

## 1. AIS Vessel Tracking Data
- **Official Specification Link**: [MarineCadastre AIS Portal](https://marinecadastre.gov/accessais/)
- **Standard Schema (NOAA/BOEM)**:
  `MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,VesselType,Status,Length,Width,Draft,Cargo`
- **Sample File**: [`marinecadastre_sample_ais.csv`](./marinecadastre_sample_ais.csv)
  Contains vessel transponder telemetry for the Gulf of Kachchh corridor, including candidate commercial vessels (`MT NEPTUNE GLORY`, `MT BHARAT RATNA`, `APL BUSAN`, `MV PACIFIC FORTUNE`, `OCEAN SUPREME`, `FV JAI MATSYA 9`).
- **Kinematic Anomaly Markers**:
  - `MT NEPTUNE GLORY` (IMO: 9384721, MMSI: 636019482) drops from cruising speed of **15.4 knots** down to **5.1 knots** between `07:40:00` and `08:40:00` (T - 10.5h) over coordinates `(22.384° N, 69.116° E)` to discharge dirty bilge slop water covertly, then accelerates back to 14.8 knots.

---

## 2. Satellite SAR Oil Spill Imagery
- **Official Specification**: `Zenodo - Sentinel-1 SAR Oil Spill Dataset P`
- **Official Records**:
  - **Part I**: [Zenodo 8346860](https://zenodo.org/records/8346860) — 1,200 Sentinel-1 SAR Oil Spill Scenes & Ground Truth Masks (40.7 GB / 6.2 MB)
  - **Part II**: [Zenodo 8253899](https://zenodo.org/records/8253899) — 686 Lookalike & No-Oil Scenes & Masks (43.8 GB / 0.8 MB)
  - **Part III**: [Zenodo 13761290](https://zenodo.org/records/13761290) — 908 Test Bench Images & Masks (9.4 GB)
- **Downloaded Ground Truth Data** (in `datasets/zenodo_sentinel1_sar/`):
  - `part1_oil_spill_masks/`: 20 extracted real 2048x2048 TIFF oil spill ground truth masks (plus full 1,201 mask archive)
  - `part2_lookalike_masks/`: 20 extracted real 2048x2048 TIFF lookalike ground truth masks (plus full 686 mask archive)
  - `part2_no_oil_masks/`: 20 extracted real 2048x2048 TIFF clean water ground truth masks (plus full 686 mask archive)
- **Format**: C-Band SAR Ground Range Detected (GRD) in Interferometric Wide (IW) swath mode, VV polarization in Sigma0 (dB).
- **Dataset Management Script**: Run `python scripts/download_zenodo_dataset.py` to inspect disk quota, download, and extract further batches safely.

---

## 3. Super-Resolution Satellite Readability (YouTube Reference)
- **Official Video Link**: [Enhancing Satellite Imagery Readability with Super resolution Machine Learning Models](https://www.youtube.com/watch?v=cQoHSStTEdM)
- **Engine Implementation**: Implemented in `src/ocean_shield/sar_engine.py` (`enhance_sar_super_resolution`) using sub-pixel interpolation with edge-preserving bilateral filtering and Laplacian high-frequency detail synthesis to double ground resolution (10m $\to$ 5m equivalent). Real-time interactive toggle in the Maritime Command Center web dashboard (`#srToggle`).
