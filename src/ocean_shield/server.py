"""
FastAPI Server: OCEAN-SHIELD API & Operational Command Center
Serves REST endpoints for SAR image analysis, Lagrangian hydrodynamic drift simulations,
AIS vessel correlation & anomaly scoring, and automated Coast Guard PDF violation dossier generation.
"""

import os
import io
import csv
import json
import gc
import hashlib
import math
import asyncio
import urllib.request
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    _pkg_root = str(Path(__file__).resolve().parent.parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from src.ocean_shield.sar_engine import SAREngine
    from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
    from src.ocean_shield.ocean_data import DataCoverageError
    from src.ocean_shield.ais_engine import AISEngine
    from src.ocean_shield.eo_engine import EOEngine
    from src.ocean_shield.scenarios import (
        get_all_scenarios, get_scenario_sar_and_currents, get_scenario_eo_data,
        image_to_base64_png
    )
    from src.ocean_shield.report_generator import DossierReportGenerator
    from src.ocean_shield.ais_ingestion import MAX_AIS_UPLOAD_BYTES, parse_marinecadastre_csv
    from src.ocean_shield.live_fetcher import (
        fetch_live_satellite_passes, fetch_live_ocean_weather, fetch_live_ais_traffic,
        fetch_world_port_index, fetch_live_oil_spill_incidents, live_provider_status
    )
else:
    from .sar_engine import SAREngine
    from .drift_engine import DriftEngine, OceanCurrentField
    from .ocean_data import DataCoverageError
    from .ais_engine import AISEngine
    from .eo_engine import EOEngine
    from .scenarios import (
        get_all_scenarios, get_scenario_sar_and_currents, get_scenario_eo_data,
        image_to_base64_png
    )
    from .report_generator import DossierReportGenerator
    from .ais_ingestion import MAX_AIS_UPLOAD_BYTES, parse_marinecadastre_csv
    from .live_fetcher import (
        fetch_live_satellite_passes, fetch_live_ocean_weather, fetch_live_ais_traffic,
        fetch_world_port_index, fetch_live_oil_spill_incidents, live_provider_status
    )

logger = logging.getLogger("ocean_shield.keep_alive")

app = FastAPI(
    title="OCEAN-SHIELD Maritime Intelligence API",
    description="Satellite SAR & Optical EO Oil Spill Detection, Hydrodynamic Hindcast & AIS Rogue Vessel Attribution",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compress responses > 500 bytes (app.js 122KB → ~25KB, HTML 70KB → ~15KB)
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "reports")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount static and datasets folders
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "datasets")
if os.path.exists(DATASETS_DIR):
    app.mount("/datasets", StaticFiles(directory=DATASETS_DIR), name="datasets")

# Shared Engine Singletons
sar_engine = SAREngine()
eo_engine = EOEngine(ground_resolution_m=10.0)
drift_engine = DriftEngine()
ais_engine = AISEngine()
report_gen = DossierReportGenerator(output_dir=REPORTS_DIR)

MAX_SAR_UPLOAD_BYTES = 25 * 1024 * 1024

# --- Render Free-Tier Memory Optimization ---
# Bounded LRU cache: keeps at most 2 scenarios in memory (~2 MB each)
# to avoid unbounded growth on the 512 MB Render free tier.
_SCENARIO_CACHE_MAX = 2
_scenario_cache: Dict[str, Tuple[np.ndarray, Any, Dict[str, Any]]] = {}
_scenario_cache_order: List[str] = []  # insertion order for LRU eviction

def resolve_scenario_sar_and_currents(scenario_id: str) -> Tuple[np.ndarray, OceanCurrentField, Dict[str, Any]]:
    """Resolve a benchmark scenario or a user-uploaded source scene.
    Results are cached in-memory with LRU eviction (max 2 entries)
    to avoid repeated NetCDF/AIS parsing while staying within Render memory limits."""
    if scenario_id in _scenario_cache:
        return _scenario_cache[scenario_id]
    result = get_scenario_sar_and_currents(scenario_id)
    # LRU eviction: drop oldest entry when cache exceeds max size
    if len(_scenario_cache) >= _SCENARIO_CACHE_MAX:
        evict_id = _scenario_cache_order.pop(0)
        _scenario_cache.pop(evict_id, None)
    _scenario_cache[scenario_id] = result
    _scenario_cache_order.append(scenario_id)
    return result


def _validated_coordinate(value: float, low: float, high: float, name: str) -> float:
    if not math.isfinite(value) or not low <= value <= high:
        raise HTTPException(status_code=422, detail=f"{name} must be between {low} and {high}.")
    return value


def _decode_uploaded_sar(content: bytes) -> np.ndarray:
    if not content:
        raise HTTPException(status_code=422, detail="SAR image is empty.")
    if len(content) > MAX_SAR_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="SAR image exceeds the 25 MB upload limit.")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=422, detail="Unsupported SAR image. Upload a PNG, JPEG, or single-band TIFF.")
    height, width = image.shape[:2]
    if height < 64 or width < 64 or image.size > 25_000_000:
        raise HTTPException(status_code=422, detail="SAR image must be 64 px–25 MP after decoding.")
    return image


# --- Request Models ---
class LiveMissionRequest(BaseModel):
    lat: float = Field(default=22.585, ge=-90.0, le=90.0)
    lon: float = Field(default=69.185, ge=-180.0, le=180.0)
    title: str = "Live Operational AOI"
    region: str = "Live Satellite & Ocean Stream"


class AnalyzeSARRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    use_super_resolution: bool = True
    model_type: str = "unet"  # "unet" (PyTorch Deep Learning) or "cfar_edge" (Fast Tactical)
    threshold_offset: float = Field(default=22.0, ge=1.0, le=100.0)


class AnalyzeEORequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"


class SimulateDriftRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    demo_mode: bool = False
    slick_lat: float
    slick_lon: float
    scene_acquired_at_utc: Optional[str] = None
    slick_age_hours: Optional[float] = Field(default=None, gt=0.0, le=168.0)
    max_lookback_hours: float = Field(default=24.0, ge=1.0, le=168.0)
    forecast_hours: float = Field(default=48.0, ge=1.0, le=168.0)
    current_u_ms: Optional[float] = Field(default=None, ge=-5.0, le=5.0)
    current_v_ms: Optional[float] = Field(default=None, ge=-5.0, le=5.0)
    wind_u_ms: Optional[float] = Field(default=None, ge=-60.0, le=60.0)
    wind_v_ms: Optional[float] = Field(default=None, ge=-60.0, le=60.0)
    met_ocean_reference: Optional[str] = Field(default=None, max_length=300)
    initial_mass_tonnes: Optional[float] = Field(default=None, gt=0.0, le=1_000_000.0)
    oil_profile: Optional[Dict[str, float]] = None


class CorrelateAISRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    demo_mode: bool = False
    origin_lat: float
    origin_lon: float
    origin_time_rel_h: float
    spatial_radius_nm: float = Field(default=25.0, gt=0.0, le=200.0)
    temporal_window_h: float = Field(default=5.0, gt=0.0, le=168.0)
    hindcast_trajectory: Optional[List[Dict[str, Any]]] = None
    radar_targets: Optional[List[Dict[str, Any]]] = None
    vessels: Optional[List[Dict[str, Any]]] = None
    ais_provenance: Optional[Dict[str, Any]] = None


class CaseSummaryRequest(BaseModel):
    """The browser's current screening outputs, sent only to render a PDF."""
    scenario_id: str = "gulf_of_kachchh"
    sar_results: Dict[str, Any]
    drift_results: Dict[str, Any]
    ais_results: Dict[str, Any]
    evidence_provenance: Optional[Dict[str, Any]] = None


class CounterfactualRequest(BaseModel):
    """Parameters for Stage 4 Forward Counterfactual Verification (Physical Re-Simulation)."""
    scenario_id: str = "gulf_of_kachchh"
    vessel_mmsi: Optional[Any] = None
    vessel_name: Optional[str] = "Suspect Vessel"
    release_lat: float
    release_lon: float
    release_time_rel_h: float
    observed_slick_lat: float
    observed_slick_lon: float
    observed_slick_polygon: Optional[List[Any]] = None
    observed_slick_area_km2: Optional[float] = None
    demo_mode: bool = True


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main operations room tactical dashboard."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>OCEAN-SHIELD UI loading...</h1>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.api_route("/techstack", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/architecture", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_techstack():
    """Serves the complete 5-layer engineering pipeline & tech stack breakdown page."""
    techstack_path = os.path.join(TEMPLATES_DIR, "techstack.html")
    if not os.path.exists(techstack_path):
        return HTMLResponse("<h1>OCEAN-SHIELD Tech Stack page not found</h1>", status_code=404)
    with open(techstack_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    # Report memory usage for Render free-tier monitoring
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes, Linux: KB
    # On Linux (Render), ru_maxrss is in KB, so adjust:
    import platform
    if platform.system() == "Linux":
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "status": "operational",
        "system": "OCEAN-SHIELD NTRO / Indian Coast Guard Pipeline",
        "version": "1.0.0",
        "sar_engine": "online",
        "drift_engine": "online",
        "ais_engine": "online",
        "memory_mb": round(rss_mb, 1),
        "models_loaded": {
            "unet": sar_engine.model_loaded,
            "super_resolution": sar_engine._sr_loaded,
        },
        "scenario_cache_entries": len(_scenario_cache),
    }


async def _keep_alive_pinger():
    """
    Automatic keep-alive loop to prevent cloud hosts (like Render free tier)
    from spinning down after 15 minutes of idle time.
    Pings every 10 minutes (600 seconds) via public HTTP.
    """
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("SELF_PING_URL")
    if not url:
        logger.info("[KeepAlive] RENDER_EXTERNAL_URL / SELF_PING_URL not configured. Self-pinger idle.")
        return

    health_url = f"{url.rstrip('/')}/api/health"
    logger.info(f"[KeepAlive] Zero-downtime self-pinger initialized for: {health_url}")

    # Wait 30 seconds for the web server to fully bind
    await asyncio.sleep(30)

    while True:
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "OCEAN-SHIELD-ZeroDowntime/1.0"}
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=15).read())
            logger.info(f"[KeepAlive] Keep-alive ping dispatched successfully -> {health_url}")
        except Exception as err:
            logger.warning(f"[KeepAlive] Keep-alive ping attempt notice: {err}")

        # Ping every 10 minutes (600s) - well below Render's 15-min idle spin-down threshold
        await asyncio.sleep(600)


@app.on_event("startup")
async def start_background_keepalive():
    asyncio.create_task(_keep_alive_pinger())


@app.get("/api/datasets/inspect")
async def inspect_datasets():
    """Returns structured inspection metadata and sample rows from all 4 authoritative datasets."""
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "datasets")

    # 1. AIS Sample
    ais_path = os.path.join(datasets_dir, "marinecadastre_real_ais.csv")
    ais_rows = []
    ais_size = 0
    if os.path.exists(ais_path):
        ais_size = os.path.getsize(ais_path)
        try:
            with open(ais_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ais_rows = [r for _, r in zip(range(12), reader)]
        except Exception as e:
            logger.warning(f"Error reading AIS sample: {e}")

    # 2. SAR Image
    sar_path = os.path.join(datasets_dir, "real_sar", "real_sentinel1_crop_512.png")
    sar_size = os.path.getsize(sar_path) if os.path.exists(sar_path) else 0

    # 3. HYCOM NetCDF
    hycom_path = os.path.join(datasets_dir, "ocean_met", "hycom_real_gulf_kachchh.nc")
    hycom_size = os.path.getsize(hycom_path) if os.path.exists(hycom_path) else 0

    # 4. Open-Meteo Wind
    wind_path = os.path.join(datasets_dir, "ocean_met", "openmeteo_wind_kachchh.json")
    wind_size = os.path.getsize(wind_path) if os.path.exists(wind_path) else 0
    wind_sample = []
    if os.path.exists(wind_path):
        try:
            with open(wind_path, "r", encoding="utf-8") as f:
                wdata = json.load(f)
                h = wdata.get("hourly", {})
                times = h.get("time", [])[:8]
                speeds = h.get("wind_speed_10m", [])[:8]
                dirs = h.get("wind_direction_10m", [])[:8]
                wind_sample = [
                    {"time": times[i], "speed_kmh": speeds[i], "direction_deg": dirs[i]}
                    for i in range(min(len(times), len(speeds), len(dirs)))
                ]
        except Exception as e:
            logger.warning(f"Error reading wind sample: {e}")

    return {
        "status": "success",
        "datasets": {
            "sar": {
                "name": "ESA Sentinel-1 C-SAR Oil Spill Imagery",
                "authority": "European Space Agency (ESA) & CERN Zenodo",
                "doi": "10.5281/zenodo.8346860",
                "portal_url": "https://zenodo.org/records/8346860",
                "copernicus_url": "https://browser.dataspace.copernicus.eu/",
                "file_path": "datasets/real_sar/real_sentinel1_crop_512.png",
                "format": "C-Band SAR GRD (Ground Range Detected), 10m Pixel Spacing",
                "polarization": "VV (Co-polarization optimal for sea surface roughness)",
                "size_bytes": sar_size,
                "preview_url": "/datasets/real_sar/real_sentinel1_crop_512.png",
                "resolution": "512 x 512 pixels (5.12 km x 5.12 km geographic swath)",
                "why_used": "Microwave radar penetrates monsoon cloud cover and operates in total darkness (24/7 all-weather maritime surveillance)."
            },
            "ais": {
                "name": "US Coast Guard & NOAA MarineCadastre Real Vessel Tracking",
                "authority": "NOAA Office for Coastal Management & BOEM",
                "portal_url": "https://marinecadastre.gov/accessais/",
                "file_path": "datasets/marinecadastre_real_ais.csv",
                "format": "NOAA/BOEM Standard CSV (15 attributes, WGS84)",
                "total_records": 1996,
                "size_bytes": ais_size,
                "columns": ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselName", "IMO", "CallSign", "VesselType", "Status", "Length", "Width", "Draft"],
                "sample_rows": ais_rows,
                "why_used": "Provides source navigation records for time-aligned corridor screening; it cannot independently establish a discharge or responsibility."
            },
            "ocean_currents": {
                "name": "NOAA HYCOM GOFS 3.1 4D Hydrodynamic Ocean Currents",
                "authority": "NOAA / National Centers for Environmental Prediction & Naval Research Lab",
                "portal_url": "https://www.hycom.org/data/gofs-3-1",
                "thredds_url": "https://tds.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/sur",
                "file_path": "datasets/ocean_met/hycom_real_gulf_kachchh.nc",
                "format": "NetCDF-4 (CF-1.8 compliant multidimensional binary array)",
                "variables": ["time", "lat", "lon", "water_u", "water_v", "wind_u", "wind_v"],
                "spatial_grid": "1/12 degree physical resolution (~9.2 km)",
                "temporal_interval": "3-hourly continuous reanalysis grids",
                "size_bytes": hycom_size,
                "why_used": "Supplies physically validated eastward and northward sea-surface current velocities (u, v) for 4th-Order Runge-Kutta Lagrangian particle drift advection."
            },
            "wind": {
                "name": "ECMWF ERA5 / Open-Meteo Marine Atmospheric Wind Reanalysis",
                "authority": "European Centre for Medium-Range Weather Forecasts (ECMWF)",
                "portal_url": "https://open-meteo.com/en/docs/marine-weather-api",
                "file_path": "datasets/ocean_met/openmeteo_wind_kachchh.json",
                "format": "JSON Hourly Time Series (10-meter surface wind vectors)",
                "hourly_records_count": 72,
                "sample_records": wind_sample,
                "size_bytes": wind_size,
                "why_used": "Calculates direct atmospheric wind leeway drag (3.2% rule with 0-15 degree Coriolis deflection angle) and Mackay oil evaporation weathering."
            }
        }
    }



@app.get("/api/scenarios")
async def list_scenarios():
    """Returns list of available pre-configured operational maritime scenarios."""
    scenarios = get_all_scenarios()
    summary_list = []
    
    for sid, sc in scenarios.items():
        is_real = sc.get("is_real_zenodo_dataset", False)
        sat_origin = sc.get("satellite_metadata", {}).get("data_origin", "Provenance resolved when the scenario scene is loaded")
        summary_list.append({
            "id": sc["id"],
            "title": sc["title"],
            "region": sc["region"],
            "center": sc["center"],
            "mission": sc["satellite_metadata"]["mission"],
            "acquisition_ist": sc["satellite_metadata"].get("acquisition_time_ist", sc["satellite_metadata"].get("acquisition_time_utc", "N/A")),
            "acquisition_utc": sc["satellite_metadata"].get("acquisition_time_utc", "N/A"),
            "sea_state": sc["ocean_conditions"]["sea_state"],
            "vessels_count": len(sc.get("ais_vessels", [])),
            "data_origin": sat_origin,
            "is_real_dataset": is_real or "Authentic" in sat_origin
        })
    return {"scenarios": summary_list}


@app.get("/api/scenario/{scenario_id}")
async def get_scenario_details(scenario_id: str):
    """Fetches comprehensive scenario parameters, SAR imagery, and AIS tracks."""
    sar_img, current_field, scenario_data = resolve_scenario_sar_and_currents(scenario_id)

    return {
        "scenario": scenario_data,
        "sar_preview_url": f"/api/scenario/{scenario_id}/sar-preview.png",
        "sr_preview_url": f"/api/scenario/{scenario_id}/sr-preview.png",
        "dimensions": {"width": sar_img.shape[1], "height": sar_img.shape[0]}
    }


@app.get("/api/scenario/{scenario_id}/sar-preview.png")
async def get_sar_preview(scenario_id: str):
    """Serve raw SAR preview as a PNG image (not base64-in-JSON)."""
    from fastapi.responses import Response
    sar_img, _, _ = resolve_scenario_sar_and_currents(scenario_id)
    _, buf = cv2.imencode('.png', sar_img)
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/scenario/{scenario_id}/sr-preview.png")
async def get_sr_preview(scenario_id: str):
    """Serve super-resolution SAR preview as a PNG image (computed on demand)."""
    from fastapi.responses import Response
    sar_img, _, _ = resolve_scenario_sar_and_currents(scenario_id)
    sr_img = sar_engine.enhance_sar_super_resolution(sar_img, scale_factor=2)
    _, buf = cv2.imencode('.png', sr_img)
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/analyze-sar")
async def analyze_sar(req: AnalyzeSARRequest):
    """Executes SAR radar speckle suppression, PyTorch U-Net or CFAR segmentation, and geometric extraction."""
    sar_img, current_field, scenario_data = resolve_scenario_sar_and_currents(req.scenario_id)

    pixel_size = float(scenario_data.get("satellite_metadata", {}).get("pixel_spacing_m", 10.0))
    center_lat = scenario_data["center"]["lat"]
    center_lon = scenario_data["center"]["lon"]

    # Process SAR scene using chosen model (PyTorch U-Net or CFAR Edge)
    results = sar_engine.process_sar_scene(
        sar_img, center_lat, center_lon, pixel_size_m=pixel_size, model_type=req.model_type
    )

    # Super-resolution enhanced image for inspection
    sr_img = sar_engine.enhance_sar_super_resolution(sar_img, scale_factor=2)
    sr_b64 = image_to_base64_png(sr_img)

    # Segmentation overlay visualization
    if req.model_type == "unet" and sar_engine.model_loaded:
        clean_mask, _ = sar_engine.predict_unet(sar_img)
    else:
        clean_mask, _ = sar_engine.segment_oil_slick(sar_img, threshold_offset=req.threshold_offset)

    colored_mask = cv2.applyColorMap(clean_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR), 0.65,
        colored_mask, 0.35, 0
    )
    overlay_b64 = image_to_base64_png(overlay)

    # --- Render Free-Tier: reclaim inference memory immediately ---
    gc.collect()

    return {
        "sar_results": results,
        "segmentation_overlay_base64": overlay_b64,
        "super_resolution_base64": sr_b64,
        "radar_detected_ships": results.get("radar_detected_ships", []),
        "active_engine": results.get("active_engine", "PyTorch U-Net"),
        "metadata": scenario_data["satellite_metadata"]
    }


@app.post("/api/analyze-sar-upload")
async def analyze_sar_upload(request: Request):
    """Analyse a user-supplied SAR raster and return a provenance-labelled result.

    The uploader supplies scene centre and pixel size because common image exports do
    not preserve GeoTIFF transforms through browser upload. Full georeferencing is a
    planned input adapter, not silently inferred here.
    """
    try:
        center_lat = float(request.headers["X-Center-Lat"])
        center_lon = float(request.headers["X-Center-Lon"])
        pixel_size_m = float(request.headers.get("X-Pixel-Size-M", "10.0"))
        threshold_offset = float(request.headers.get("X-Threshold-Offset", "22.0"))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Valid X-Center-Lat and X-Center-Lon headers are required.") from exc
    model_type = request.headers.get("X-Model-Type", "unet")
    filename = request.headers.get("X-File-Name", "sar-raster")
    acquisition_time_utc = request.headers.get("X-Acquisition-Time-UTC", "").strip()
    if acquisition_time_utc:
        try:
            datetime.fromisoformat(acquisition_time_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="X-Acquisition-Time-UTC must be an ISO-8601 timestamp.") from exc

    _validated_coordinate(center_lat, -90.0, 90.0, "center_lat")
    _validated_coordinate(center_lon, -180.0, 180.0, "center_lon")
    if not 1.0 <= pixel_size_m <= 1_000.0:
        raise HTTPException(status_code=422, detail="pixel_size_m must be between 1 and 1000.")
    if not 1.0 <= threshold_offset <= 100.0:
        raise HTTPException(status_code=422, detail="threshold_offset must be between 1 and 100.")
    if model_type not in {"unet", "cfar_edge"}:
        raise HTTPException(status_code=422, detail="model_type must be 'unet' or 'cfar_edge'.")

    content = await request.body()
    sar_image = _decode_uploaded_sar(content)
    results = sar_engine.process_sar_scene(
        sar_image, center_lat, center_lon, pixel_size_m=pixel_size_m, model_type=model_type
    )
    if model_type == "unet" and sar_engine.model_loaded:
        mask, _ = sar_engine.predict_unet(sar_image)
    else:
        mask, _ = sar_engine.segment_oil_slick(sar_image, threshold_offset=threshold_offset)
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(sar_image, cv2.COLOR_GRAY2BGR), 0.65, colored_mask, 0.35, 0)

    response = {
        "sar_results": results,
        "segmentation_overlay_base64": image_to_base64_png(overlay),
        "super_resolution_base64": image_to_base64_png(sar_engine.enhance_sar_super_resolution(sar_image, scale_factor=2)),
        "radar_detected_ships": results.get("radar_detected_ships", []),
        "active_engine": results.get("active_engine", "CFAR"),
        "provenance": {
            "source_kind": "user_uploaded_sar_raster",
            "source_filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "image_shape_px": {"width": int(sar_image.shape[1]), "height": int(sar_image.shape[0])},
            "scene_center": {"lat": center_lat, "lon": center_lon},
            "pixel_size_m": pixel_size_m,
            "acquisition_time_utc": acquisition_time_utc or None,
            "georeferencing": "user-supplied centre and pixel size; original GeoTIFF transform not retained",
        },
        "screening_notice": "Automated screening result. Analyst review and validated sensor calibration are required before operational or legal use.",
    }

    # --- Render Free-Tier: reclaim upload inference memory ---
    gc.collect()

    return response


@app.post("/api/analyze-eo")
async def analyze_eo(req: AnalyzeEORequest):
    """
    Processes Sentinel-2 MSI Multi-Spectral Optical (EO) Imagery:
    Computes Normalized Difference Oil Index (NDOI) and Floating Algae Index (FAI)
    to detect sunglint oil anomalies and reject natural lookalike algal blooms.
    """
    rgb_img, nir_band, swir_band, scenario_data = get_scenario_eo_data(req.scenario_id)
    if "raw_multispectral_bands" in scenario_data:
        clean_mask, diagnostics = eo_engine.process_multispectral_scene(
            scenario_data["raw_multispectral_bands"],
            sunglint_present=True
        )
    else:
        clean_mask, diagnostics = eo_engine.segment_optical_slick(rgb_img, nir_band=nir_band, swir_band=swir_band)

    # Compute NDOI matrix for visualization
    r = rgb_img[:, :, 2]
    ndoi_matrix = eo_engine.compute_ndoi(r, nir_band)
    # Map NDOI [-0.3, 0.6] to [0, 255] color heatmap
    ndoi_norm = np.clip((ndoi_matrix + 0.2) / 0.8, 0.0, 1.0)
    ndoi_u8 = (ndoi_norm * 255).astype(np.uint8)
    ndoi_heatmap = cv2.applyColorMap(ndoi_u8, cv2.COLORMAP_INFERNO)

    # Optical overlay
    colored_mask = cv2.applyColorMap(clean_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_img, 0.65, colored_mask, 0.35, 0)

    return {
        "sensor_type": "Sentinel-2 MSI Multi-Spectral Optical (EO)",
        "diagnostics": diagnostics,
        "rgb_image_base64": image_to_base64_png(rgb_img),
        "ndoi_heatmap_base64": image_to_base64_png(ndoi_heatmap),
        "overlay_base64": image_to_base64_png(overlay),
        "metadata": {
            "mission": "Sentinel-2B MSI Multi-Spectral Optical",
            "spectral_bands": "B4 (Red 665nm), B8 (NIR 842nm), B11 (SWIR 1610nm)",
            "ground_sampling_distance_m": 10.0,
            "lookalike_discrimination": "NDOI > 0.05 & FAI < 45 (Algal Bloom Discarded)",
            "data_origin": scenario_data.get("eo_data_origin", "Authentic Sentinel-2 MSI Level-2A BOA Reflectance (Copernicus)")
        }
    }


@app.post("/api/simulate-drift")
async def simulate_drift(req: SimulateDriftRequest):
    """
    Runs a conditional transport scenario. It refuses stale, unbounded, synthetic, or
    unaligned source inputs rather than returning a polished but invalid forecast.
    """
    _, current_field, scenario_data = resolve_scenario_sar_and_currents(req.scenario_id)

    if req.slick_age_hours is None:
        raise HTTPException(status_code=422, detail="An analyst-supported slick_age_hours hypothesis is required; a single SAR scene cannot supply it.")
    if not req.demo_mode:
        if not req.scene_acquired_at_utc:
            raise HTTPException(status_code=422, detail="scene_acquired_at_utc is required to bind met-ocean data to the source scene.")
        try:
            datetime.fromisoformat(req.scene_acquired_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="scene_acquired_at_utc must be ISO-8601 UTC.") from exc

    # Dynamic provenance label reflecting actual hydrodynamic & atmospheric sources
    if req.demo_mode:
        condition = scenario_data["ocean_conditions"]
        current_field = OceanCurrentField(
            base_current_u=condition["base_current_u"],
            base_current_v=condition["base_current_v"],
            base_wind_u=condition["base_wind_u"],
            base_wind_v=condition["base_wind_v"],
            constant_vectors=True,
        )
        met_ocean_source = "Benchmark demonstration vectors (simulated; not live or observed)"
    elif current_field.data_provider and current_field.data_provider.is_loaded:
        dp_meta = current_field.data_provider.metadata
        base_source = dp_meta.get("data_origin") or dp_meta.get("source") or dp_meta.get("title") or "CF-1.8 NetCDF hydrodynamic grid"
        if getattr(current_field.data_provider, "_has_real_wind", False):
            met_ocean_source = f"{base_source} + time-aligned Open-Meteo hourly wind"
        else:
            met_ocean_source = base_source
    else:
        met_ocean_source = "scenario demonstration analytical field"

    overrides = (req.current_u_ms, req.current_v_ms, req.wind_u_ms, req.wind_v_ms)
    if any(value is not None for value in overrides) and not all(value is not None for value in overrides):
        raise HTTPException(status_code=422, detail="Provide all four current/wind vectors or none; partial overrides mix incompatible fields.")
    if req.demo_mode and any(value is not None for value in overrides):
        raise HTTPException(status_code=422, detail="Benchmark mode uses its isolated fixture vectors; use Live Ingestion for operator-supplied data.")
    if all(value is not None for value in overrides):
        if not req.met_ocean_reference:
            raise HTTPException(status_code=422, detail="met_ocean_reference is required with operator-supplied vectors.")
        current_field = OceanCurrentField(
            base_current_u=req.current_u_ms if req.current_u_ms is not None else current_field.base_current_u,
            base_current_v=req.current_v_ms if req.current_v_ms is not None else current_field.base_current_v,
            base_wind_u=req.wind_u_ms if req.wind_u_ms is not None else current_field.base_wind_u,
            base_wind_v=req.wind_v_ms if req.wind_v_ms is not None else current_field.base_wind_v,
            tidal_amplitude=current_field.tidal_amplitude,
            tidal_period_h=current_field.tidal_period_h,
            constant_vectors=True,
        )
        met_ocean_source = f"operator-supplied vectors — {req.met_ocean_reference}"
    elif not req.demo_mode:
        if current_field.data_provider is None:
            raise HTTPException(status_code=422, detail="No source-backed hydrodynamic provider is available. Supply validated vectors with a reference.")
        try:
            current_field.data_provider.bind_detection_time(
                req.scene_acquired_at_utc, req.max_lookback_hours, req.forecast_hours
            )
            # Fail before integration if either currents or wind is unavailable at T0.
            current_field.get_velocity_at(req.slick_lat, req.slick_lon, 0.0)
        except (DataCoverageError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Met-ocean validation failed: {exc}") from exc

    target_age = float(req.slick_age_hours)

    # Backward Hindcast
    hindcast_res = drift_engine.run_hindcast(
        req.slick_lat, req.slick_lon, current_field,
        max_lookback_hours=req.max_lookback_hours,
        target_slick_age_hours=target_age
    )

    # Forward Forecast + ADIOS Weathering
    forecast_res = drift_engine.run_forecast(
        req.slick_lat, req.slick_lon, current_field,
        forecast_hours=req.forecast_hours,
        initial_mass_tonnes=req.initial_mass_tonnes,
        oil_profile=req.oil_profile,
    )

    return {
        "origin_release_point": hindcast_res["origin_release_point"],
        "hindcast_trajectory": hindcast_res["hindcast_trajectory"],
        "total_drift_distance_km": hindcast_res["total_drift_distance_km"],
        "kde_origin_contours": hindcast_res.get("kde_origin_contours", {}),
        "forecast_trajectory": forecast_res["forecast_trajectory"],
        "weathering_summary": forecast_res.get("weathering_summary", {}),
        "beaching_warning": forecast_res["beaching_warning"],
        "provenance": {
            "met_ocean_source": met_ocean_source,
            "current_u_ms": current_field.base_current_u,
            "current_v_ms": current_field.base_current_v,
            "wind_u_ms": current_field.base_wind_u,
            "wind_v_ms": current_field.base_wind_v,
            "scene_acquired_at_utc": req.scene_acquired_at_utc,
            "coverage_validated": not req.demo_mode,
            "mode": "benchmark_demo" if req.demo_mode else "source_input",
        },
        "screening_notice": (
            "BENCHMARK DEMONSTRATION: simulated inputs and outputs; not live data or an operational forecast."
            if req.demo_mode else
            "Conditional transport scenario only. It is not a release-origin inference, shoreline-impact assessment, or responsibility finding."
        ),
    }


@app.post("/api/correlate-ais")
async def correlate_ais(req: CorrelateAISRequest):
    """Ranks time-aligned AIS leads and emits radar/AIS review cues."""
    _, current_field, scenario_data = resolve_scenario_sar_and_currents(req.scenario_id)
    if req.demo_mode:
        vessels = scenario_data.get("ais_vessels", [])
        if not vessels:
            raise HTTPException(status_code=422, detail="This benchmark has no fixture AIS trajectories.")
    elif not req.vessels or not req.ais_provenance:
        raise HTTPException(status_code=422, detail="Time-aligned AIS source records and provenance are required; benchmark trajectories are not eligible for attribution.")
    else:
        vessels = req.vessels

    hindcast_traj = req.hindcast_trajectory
    if not hindcast_traj:
        raise HTTPException(status_code=422, detail="A validated conditional hindcast trajectory is required for AIS correlation.")

    # Radar/AIS comparison is optional; never synthesize contacts.
    radar_tgts = req.radar_targets or []
    radar_tgts = [dict(target, relative_time_hours=target.get("relative_time_hours", 0.0)) for target in radar_tgts]

    results = ais_engine.attribute_oil_spill(
        vessels, req.origin_lat, req.origin_lon, req.origin_time_rel_h,
        spatial_radius_nm=req.spatial_radius_nm,
        temporal_window_h=req.temporal_window_h,
        hindcast_trajectory=hindcast_traj,
        radar_targets=radar_tgts
    )

    # Stage 4: Forward Counterfactual Verification (Physical Re-Simulation)
    primary_lead = results.get("primary_review_lead")
    if primary_lead and primary_lead.get("candidate_release_point"):
        crp = primary_lead["candidate_release_point"]
        try:
            cf_res = drift_engine.run_forward_counterfactual(
                release_lat=float(crp["lat"]),
                release_lon=float(crp["lon"]),
                release_time_rel_h=float(crp["relative_time_hours"]),
                current_field=current_field,
                observed_slick_lat=float(req.origin_lat),
                observed_slick_lon=float(req.origin_lon),
                vessel_info={
                    "mmsi": primary_lead.get("mmsi"),
                    "vessel_name": primary_lead.get("vessel_name", "Primary Lead")
                }
            )
            primary_lead["counterfactual_verification"] = cf_res
            results["counterfactual_verification"] = cf_res
            results["counterfactual_ready"] = True
        except Exception as e:
            logger.warning(f"Counterfactual verification calculation notice: {e}")
            results["counterfactual_ready"] = False

    results["ais_provenance"] = req.ais_provenance or {
        "source_kind": "benchmark AIS trajectories (simulated; not live AIS)",
        "scenario_id": req.scenario_id,
        "total_vessels_in_region": len(vessels)
    }
    results["demo_mode"] = req.demo_mode
    results["screening_notice"] = (
        ("BENCHMARK DEMONSTRATION: simulated AIS lead ranking; not live data. " if req.demo_mode else "") +
        "AIS ranking is an investigative lead, not a finding of responsibility. "
        "Corroborate with calibrated imagery, chain-of-custody records, and human review."
    )
    return results


@app.post("/api/verify-counterfactual")
async def verify_counterfactual(req: CounterfactualRequest):
    """
    Stage 4: Forward Counterfactual Verification (Physical Re-Simulation).
    Re-seeds particles at the vessel's reported AIS coordinates and candidate release time t_release,
    integrates forward in time through active hydrodynamic current fields & wind leeway up to T0,
    and computes Centroid Error (km), Containment (%), Jaccard Index (IoU), and Physics Verdict.
    """
    _, current_field, scenario_data = resolve_scenario_sar_and_currents(req.scenario_id)

    slick_area = req.observed_slick_area_km2
    slick_polygon = req.observed_slick_polygon
    if slick_area is None:
        slick_area = scenario_data.get("primary_slick", {}).get("area_km2", 12.0)

    try:
        cf_res = drift_engine.run_forward_counterfactual(
            release_lat=req.release_lat,
            release_lon=req.release_lon,
            release_time_rel_h=req.release_time_rel_h,
            current_field=current_field,
            observed_slick_lat=req.observed_slick_lat,
            observed_slick_lon=req.observed_slick_lon,
            observed_slick_polygon=slick_polygon,
            observed_slick_area_km2=slick_area,
            vessel_info={
                "mmsi": req.vessel_mmsi,
                "vessel_name": req.vessel_name
            }
        )
        return cf_res
    except Exception as e:
        logger.error(f"Failed to run forward counterfactual: {e}")
        raise HTTPException(status_code=500, detail=f"Forward counterfactual verification failed: {e}")


@app.get("/api/export-dossier/{scenario_id}")
async def export_dossier(scenario_id: str):
    raise HTTPException(
        status_code=410,
        detail="Benchmark dossier export is disabled. Export only a browser case summary assembled from documented source inputs."
    )


@app.post("/api/export-case-summary")
async def export_case_summary(req: CaseSummaryRequest):
    """Render the browser's current screening state as an analyst-review PDF."""
    scenario_data = get_all_scenarios().get(req.scenario_id)
    if scenario_data is None:
        raise HTTPException(status_code=422, detail="Unknown scenario_id.")
    if not req.sar_results.get("primary_slick"):
        raise HTTPException(status_code=422, detail="Run SAR screening before exporting a case summary.")

    provenance = req.evidence_provenance or {}
    sar_provenance = provenance.get("sar") or {}
    if sar_provenance:
        scenario_data = dict(scenario_data)
        scenario_data["satellite_metadata"] = {
            **scenario_data.get("satellite_metadata", {}),
            "mission": "User-supplied SAR raster",
            "acquisition_time_utc": sar_provenance.get("acquisition_time_utc") or "Not supplied",
            "pixel_spacing_m": sar_provenance.get("pixel_size_m", "Not supplied"),
            "data_origin": "User-uploaded raster; georeferencing supplied by operator",
        }

    drift_results = dict(req.drift_results)
    if "forecast_warning" not in drift_results:
        drift_results["forecast_warning"] = drift_results.get("beaching_warning", {})
    pdf_path = report_gen.generate_pdf_dossier(
        scenario_data,
        req.sar_results,
        drift_results,
        req.ais_results,
        evidence_provenance=provenance,
    )
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate case summary PDF")
    filename = os.path.basename(pdf_path)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/upload-ais-csv")
async def upload_ais_csv(request: Request):
    """
    Directly ingests NOAA / MarineCadastre formatted CSV files:
    Headers: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft
    """
    filename = request.headers.get("X-File-Name", "ais.csv")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a .csv AIS export.")
    content = await request.body()
    if len(content) > MAX_AIS_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="AIS CSV exceeds the 25 MB upload limit.")
    try:
        parsed = parse_marinecadastre_csv(content, filename, request.headers.get("X-Reference-Time-UTC"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "status": "success",
        "vessels_parsed_count": len(parsed["vessels"]),
        "vessels": parsed["vessels"],
        "provenance": parsed["provenance"],
        "screening_notice": "AIS is used only for this browser session. Time alignment is recorded in provenance and must be checked before ranking.",
    }


# ============================================================================
# LIVE AUTOMATED DATA INGESTION ENDPOINTS (NO MANUAL UPLOADS REQUIRED)
# ============================================================================

@app.get("/api/live/status")
async def get_live_provider_status():
    """Provider readiness only; credentials are never returned to the browser."""
    return live_provider_status()

@app.get("/api/live/satellite-passes")
async def get_live_satellite_passes(lat: float, lon: float, days_back: int = 14):
    """Query public ASF catalogue metadata. This does not create or download a SAR raster."""
    _validated_coordinate(lat, -90.0, 90.0, "lat")
    _validated_coordinate(lon, -180.0, 180.0, "lon")
    return await asyncio.to_thread(fetch_live_satellite_passes, lat, lon, days_back)


@app.get("/api/live/ocean-weather")
async def get_live_ocean_weather(lat: float, lon: float):
    """Fetch current modelled marine and wind conditions without manufactured values."""
    _validated_coordinate(lat, -90.0, 90.0, "lat")
    _validated_coordinate(lon, -180.0, 180.0, "lon")
    return await asyncio.to_thread(fetch_live_ocean_weather, lat, lon)


@app.get("/api/live/ais-traffic")
async def get_live_ais_traffic(lat: float, lon: float, radius_nm: float = 25.0, collection_seconds: float = 8.0):
    """Collect received AIS position reports; an unconfigured provider returns no fake vessels."""
    _validated_coordinate(lat, -90.0, 90.0, "lat")
    _validated_coordinate(lon, -180.0, 180.0, "lon")
    if not 1.0 <= radius_nm <= 120.0:
        raise HTTPException(status_code=422, detail="radius_nm must be between 1 and 120.")
    if not 1.0 <= collection_seconds <= 20.0:
        raise HTTPException(status_code=422, detail="collection_seconds must be between 1 and 20.")
    return await fetch_live_ais_traffic(lat, lon, radius_nm=radius_nm, duration_s=collection_seconds)


@app.get("/api/live/ports")
async def get_reference_ports(
    min_lon: float = 66.0,
    min_lat: float = 5.0,
    max_lon: float = 100.0,
    max_lat: float = 38.0,
):
    """Query the current NGA WPI reference catalogue for a bounded geographic envelope."""
    _validated_coordinate(min_lon, -180.0, 180.0, "min_lon")
    _validated_coordinate(max_lon, -180.0, 180.0, "max_lon")
    _validated_coordinate(min_lat, -90.0, 90.0, "min_lat")
    _validated_coordinate(max_lat, -90.0, 90.0, "max_lat")
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(status_code=422, detail="The port query bounds must describe a non-empty envelope.")
    return await asyncio.to_thread(fetch_world_port_index, min_lon, min_lat, max_lon, max_lat)


@app.get("/api/live/incidents")
async def get_live_incidents():
    """Return only incidents received from the configured authority feed."""
    return await asyncio.to_thread(fetch_live_oil_spill_incidents)


@app.post("/api/live/create-mission")
async def create_live_mission(req: LiveMissionRequest):
    """
    Snapshot source-backed AOI context. No synthetic scene is made and no detection
    pipeline runs until a genuine SAR raster has been supplied.
    """
    _validated_coordinate(req.lat, -90.0, 90.0, "lat")
    _validated_coordinate(req.lon, -180.0, 180.0, "lon")
    satellite, met_ocean, ais = await asyncio.gather(
        asyncio.to_thread(fetch_live_satellite_passes, req.lat, req.lon),
        asyncio.to_thread(fetch_live_ocean_weather, req.lat, req.lon),
        fetch_live_ais_traffic(req.lat, req.lon),
    )
    return {
        "status": "ready_for_source_scene",
        "processing_ready": False,
        "aoi": {"lat": req.lat, "lon": req.lon, "title": req.title, "region": req.region},
        "satellite": satellite,
        "met_ocean": met_ocean,
        "ais": ais,
        "providers": live_provider_status()["providers"],
        "message": "AOI sources refreshed. Upload an authentic SAR raster with scene metadata before running spill detection, drift, or AIS attribution."
    }


@app.post("/api/live/ingest-and-run")
async def ingest_and_run_live(req: LiveMissionRequest):
    """
    This route intentionally refuses to fabricate a SAR scene. Use the source
    snapshot route followed by the authenticated SAR-upload workflow.
    """
    _validated_coordinate(req.lat, -90.0, 90.0, "lat")
    _validated_coordinate(req.lon, -180.0, 180.0, "lon")
    raise HTTPException(
        status_code=409,
        detail="Live feeds alone are not a SAR scene. Upload an authentic source raster and metadata before analysis; no synthetic scene is substituted."
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8090))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting OCEAN-SHIELD on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
