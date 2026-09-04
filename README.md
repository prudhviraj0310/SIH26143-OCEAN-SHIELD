# 🛡️ OCEAN-SHIELD — Satellite SAR Oil Spill Detection & AIS Rogue Vessel Attribution

> **SIH Problem Statement:** SIH26143  
> **Ministry Sponsor:** National Technical Research Organisation (NTRO) / Indian Coast Guard  
> **Theme:** Disaster Management / Maritime Security  
> **Track:** SOFTWARE  
> **Smart India Hackathon 2026**

---

## 🎯 Problem Statement

India's 7,516 km coastline and 2.37 million km² Exclusive Economic Zone (EEZ) faces constant threats from illegal oil dumping, vessel-source pollution, and rogue maritime activities. Current satellite SAR imagery analysis is manual and slow, taking hours to days for attribution — by which time polluters have fled.

## 🔬 Our Solution

OCEAN-SHIELD is an **AI-powered maritime surveillance system** that fuses **Sentinel-1 SAR imagery** with **live AIS vessel tracking** to detect oil spills in near real-time and automatically attribute them to specific vessels through spatiotemporal correlation.

### Core Capabilities

- **SAR Spill Detection**: Automated dark-spot detection on Sentinel-1 C-band SAR imagery using adaptive thresholding and texture analysis
- **AIS Vessel Attribution**: Cross-correlates spill polygons with historical AIS vessel tracks to identify likely polluters
- **Drift Modeling**: Lagrangian particle drift simulation using real ocean current and wind data to back-trace spill origin
- **Search & Rescue**: Automated SAR zone computation for Coast Guard dispatch
- **Scenario Simulation**: Pre-built scenarios for Mumbai Port, Andaman Sea, and Gulf of Kutch

## 🏗️ Architecture

```
┌──────────────────────┐    ┌──────────────────────────┐
│  Sentinel-1 SAR      │───▶│  Dark-Spot Detection     │
│  C-Band Imagery      │    │  Adaptive Threshold      │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Live AIS Stream     │───▶│  Vessel Attribution      │
│  MMSI/IMO Tracking   │    │  Spatiotemporal Corr.    │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  INCOIS Currents     │───▶│  Lagrangian Drift Model  │
│  Wind / Wave Data    │    │  Back-Trace to Source    │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
                            ┌──────────────────────────┐
                            │  Coast Guard Alert &     │
                            │  SAR Zone Dispatch       │
                            └──────────────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install fastapi uvicorn jinja2 aiohttp reportlab numpy

# 3. Run the server
uvicorn server:app --host 127.0.0.1 --port 8090 --reload
```

Open `http://127.0.0.1:8090/` in your browser.

## 📁 Project Structure

```
├── server.py              # FastAPI backend
├── ais_engine.py          # AIS vessel tracking & attribution
├── drift_engine.py        # Lagrangian particle drift simulation
├── sar_engine.py          # SAR dark-spot oil spill detection
├── scenarios.py           # Pre-built incident scenarios
├── report_generator.py    # PDF Coast Guard incident report
├── templates/
│   └── index.html         # Maritime Command Center UI
├── static/
│   ├── css/               # Dark maritime theme
│   └── js/                # Interactive map & controls
└── tests/                 # Unit tests
```

## 🧪 Key Features

| Feature | Description |
|:---|:---|
| **SAR Spill Detection** | Automated C-band SAR dark-spot extraction |
| **AIS Attribution** | Cross-reference spill location with vessel AIS tracks |
| **Drift Simulation** | Lagrangian back-trace using INCOIS ocean currents |
| **Threat Scoring** | Per-vessel threat score based on proximity, heading, history |
| **Coast Guard Report** | Auto-generated PDF with coordinates, evidence, response plan |

## 📜 License

MIT License — developed for Smart India Hackathon 2026

## 👥 Team

Built for SIH 2026 | Problem Statement SIH26143
