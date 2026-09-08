# 🌊 OCEAN-SHIELD (SIH 2026 - Problem Statement SIH26143)
## Autonomous SAR Oil Spill Detection & Forensic Provenance Platform

Welcome to the team! This repository contains the complete working source code, trained neural network models, maritime datasets, and modern C2 Command & Control dashboard.

---

### 🚀 60-Second Quickstart (How to Run Locally)

1. **Open your Terminal** and navigate into this unzipped folder:
   ```bash
   cd OCEAN_SHIELD_PROJECT
   ```

2. **Create and activate a Python Virtual Environment**:
   ```bash
   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Server**:
   ```bash
   python server.py
   ```

5. **Open your Browser**:
   Visit **`http://127.0.0.1:8090/`** (or `http://localhost:8090/`).

---

### 🎮 How to Demo to Evaluators & Judges:

1. **Simple View (Default)**:
   - High-resolution satellite map fills the screen.
   - Look at the bottom **Executive Mission Dock**:
     - 🌊 **Footprint**: Slick area ($0.62\text{ km}^2$) and estimated mass ($14\text{ Tonnes}$).
     - ⏱️ **Landfall ETA**: Hours until beaching ($24.0+\text{ Hours}$) and vulnerable sanctuary name.
     - 🚢 **Primary Suspect**: Guilty vessel name, match score %, and closest approach distance.

2. **1-Click Auto Run**:
   - Click the green **`⚡ AUTO-RUN INVESTIGATION`** button.
   - Watch the pipeline automatically run:
     1. Sentinel-1 SAR Oil Slick Segmentation (U-Net).
     2. Runge-Kutta 4 Hydrodynamic Ocean Drift Simulation.
     3. AIS Vessel Track Kinematic Anomaly & Suspect Ranking.

3. **1-Click Police Report (PDF)**:
   - Click the red **`📄 EXPORT ICG REPORT (PDF)`** button to download the official legal evidence dossier.

4. **Deep Technical Details (Expert View)**:
   - If a judge asks for the math, physics equations, or satellite overlays, click **`🔬 SHOW FULL FORENSICS`** (or toggle `Expert View` in the top right).
   - Sidebars slide out showing the particle physics controls, ADIOS weathering rates, and AIS anomaly charts.

---

### 📁 Project Structure:

- `src/ocean_shield/`:
  - `server.py` & `api.py`: FastAPI backend REST API.
  - `templates/index.html`: Interactive C2 Dashboard UI.
  - `static/css/`: Glassmorphic tactical styling (`dashboard.css`).
  - `static/js/`: Interactive mapping and controller logic (`app.js`).
  - `models/`: PyTorch U-Net segmentation and ESPCN 2X super-resolution neural nets.
  - `drift/`: Runge-Kutta 4 hydrodynamic drift simulator.
  - `weathering/`: NOAA/ADIOS oil weathering and emulsification physics.
  - `ais/`: Vessel trajectory kinematics and anomaly lead ranker.
  - `reporting/`: Forensic PDF dossier generation.
- `models/`: Trained model weights (`sar_unet_best.pt`, `sar_espcn_best.pt`).
- `datasets/`: Maritime AIS breadcrumb CSVs, real Sentinel-1 SAR radar imagery, and ocean current files.
- `notebooklm_resources/`: Complete study notes, judge question cheat sheets, and Indian port spill histories.
- `PITCH_AND_JUDGE_GUIDE.md`: 10 tough judge questions with bulletproof winning answers.

---
*Built for Smart India Hackathon (SIH 2026) | NTRO / Indian Coast Guard Maritime Environmental Security.*
