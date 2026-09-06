"""
FastAPI Server: OCEAN-SHIELD API & Operational Command Center
Serves REST endpoints for SAR image analysis, Lagrangian hydrodynamic drift simulations,
AIS vessel correlation & anomaly scoring, and automated Coast Guard PDF violation dossier generation.
"""

import os
import io
import hashlib
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .sar_engine import SAREngine
from .drift_engine import DriftEngine, OceanCurrentField
from .ais_engine import AISEngine
from .eo_engine import EOEngine
from .scenarios import (
    get_all_scenarios, get_scenario_sar_and_currents, get_scenario_eo_data,
    image_to_base64_png, generate_synthetic_sar_image
)
from .report_generator import DossierReportGenerator
from .ais_ingestion import MAX_AIS_UPLOAD_BYTES, parse_marinecadastre_csv


app = FastAPI(
    title="OCEAN-SHIELD Maritime Intelligence API",
    description="Satellite SAR & Optical EO Oil Spill Detection, Hydrodynamic Hindcast & AIS Rogue Vessel Attribution",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8090", "http://localhost:8090"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-File-Name", "X-Center-Lat", "X-Center-Lon", "X-Pixel-Size-M", "X-Model-Type", "X-Threshold-Offset", "X-Reference-Time-UTC", "X-Acquisition-Time-UTC"],
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "reports")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Shared Engine Singletons
sar_engine = SAREngine()
eo_engine = EOEngine(ground_resolution_m=10.0)
drift_engine = DriftEngine()
ais_engine = AISEngine()
report_gen = DossierReportGenerator(output_dir=REPORTS_DIR)

MAX_SAR_UPLOAD_BYTES = 25 * 1024 * 1024


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
class AnalyzeSARRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    use_super_resolution: bool = True
    model_type: str = "unet"  # "unet" (PyTorch Deep Learning) or "cfar_edge" (Fast Tactical)
    threshold_offset: float = Field(default=22.0, ge=1.0, le=100.0)


class AnalyzeEORequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"


class SimulateDriftRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    slick_lat: float
    slick_lon: float
    slick_age_hours: Optional[float] = None
    max_lookback_hours: float = Field(default=24.0, ge=1.0, le=168.0)
    forecast_hours: float = Field(default=48.0, ge=1.0, le=168.0)
    current_u_ms: Optional[float] = Field(default=None, ge=-5.0, le=5.0)
    current_v_ms: Optional[float] = Field(default=None, ge=-5.0, le=5.0)
    wind_u_ms: Optional[float] = Field(default=None, ge=-60.0, le=60.0)
    wind_v_ms: Optional[float] = Field(default=None, ge=-60.0, le=60.0)


class CorrelateAISRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
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


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main operations room tactical dashboard."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>OCEAN-SHIELD UI loading...</h1>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    return {
        "status": "operational",
        "system": "OCEAN-SHIELD NTRO / Indian Coast Guard Pipeline",
        "version": "1.0.0",
        "sar_engine": "online",
        "drift_engine": "online",
        "ais_engine": "online"
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
    sar_img, current_field, scenario_data = get_scenario_sar_and_currents(scenario_id)

    # Base64 image encoding for frontend preview
    raw_sar_b64 = image_to_base64_png(sar_img)

    # Super-Resolution enhanced SAR (Addressing NTRO YouTube reference)
    sr_sar_img = sar_engine.enhance_sar_super_resolution(sar_img, scale_factor=2)
    sr_sar_b64 = image_to_base64_png(sr_sar_img)

    return {
        "scenario": scenario_data,
        "sar_image_base64": raw_sar_b64,
        "sr_sar_image_base64": sr_sar_b64,
        "dimensions": {"width": sar_img.shape[1], "height": sar_img.shape[0]}
    }


@app.post("/api/analyze-sar")
async def analyze_sar(req: AnalyzeSARRequest):
    """Executes SAR radar speckle suppression, PyTorch U-Net or CFAR segmentation, and geometric extraction."""
    sar_img, current_field, scenario_data = get_scenario_sar_and_currents(req.scenario_id)

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

    return {
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
    Executes backward Lagrangian hindcasting to find spill release origin (x0, y0, t0),
    forward forecasting to estimate coastal collision, and Mackay ADIOS physical oil weathering.
    """
    _, current_field, scenario_data = get_scenario_sar_and_currents(req.scenario_id)

    # Dynamic provenance label reflecting actual hydrodynamic & atmospheric sources
    if current_field.data_provider and current_field.data_provider.is_loaded:
        dp_meta = current_field.data_provider.metadata
        base_source = dp_meta.get("data_origin") or dp_meta.get("source") or dp_meta.get("title") or "CF-1.8 NetCDF hydrodynamic grid"
        if getattr(current_field.data_provider, "_has_real_wind", False):
            met_ocean_source = f"{base_source} + Open-Meteo real hourly wind observations"
        else:
            met_ocean_source = base_source
    else:
        met_ocean_source = "scenario demonstration analytical field"

    overrides = (req.current_u_ms, req.current_v_ms, req.wind_u_ms, req.wind_v_ms)
    if any(value is not None for value in overrides):
        current_field = OceanCurrentField(
            base_current_u=req.current_u_ms if req.current_u_ms is not None else current_field.base_current_u,
            base_current_v=req.current_v_ms if req.current_v_ms is not None else current_field.base_current_v,
            base_wind_u=req.wind_u_ms if req.wind_u_ms is not None else current_field.base_wind_u,
            base_wind_v=req.wind_v_ms if req.wind_v_ms is not None else current_field.base_wind_v,
            tidal_amplitude=current_field.tidal_amplitude,
            tidal_period_h=current_field.tidal_period_h,
        )
        met_ocean_source = "operator-supplied current and wind vectors"

    # Determine target release age: from operator request, or physical age estimate from SAR detection
    target_age = req.slick_age_hours
    if target_age is None:
        sar_img, _, _ = get_scenario_sar_and_currents(req.scenario_id)
        sar_res = sar_engine.process_sar_scene(
            sar_img, req.slick_lat, req.slick_lon,
            pixel_size_m=float(scenario_data.get("satellite_metadata", {}).get("pixel_spacing_m", 10.0))
        )
        target_age = sar_res.get("primary_slick", {}).get("estimated_age_hours", 12.0)
    target_age = float(np.clip(target_age, 2.0, 36.0))

    # Backward Hindcast
    hindcast_res = drift_engine.run_hindcast(
        req.slick_lat, req.slick_lon, current_field,
        max_lookback_hours=req.max_lookback_hours,
        target_slick_age_hours=target_age
    )

    # Forward Forecast + ADIOS Weathering
    coast_thresh = scenario_data.get("coastline_hazard", {}).get("coastline_lat_threshold")
    forecast_res = drift_engine.run_forecast(
        req.slick_lat, req.slick_lon, current_field,
        forecast_hours=req.forecast_hours,
        coastline_lat_threshold=coast_thresh
    )

    return {
        "origin_release_point": hindcast_res["origin_release_point"],
        "hindcast_trajectory": hindcast_res["hindcast_trajectory"],
        "total_drift_distance_km": hindcast_res["total_drift_distance_km"],
        "forecast_trajectory": forecast_res["forecast_trajectory"],
        "weathering_summary": forecast_res.get("weathering_summary", {}),
        "beaching_warning": forecast_res["beaching_warning"],
        "provenance": {
            "met_ocean_source": met_ocean_source,
            "current_u_ms": current_field.base_current_u,
            "current_v_ms": current_field.base_current_v,
            "wind_u_ms": current_field.base_wind_u,
            "wind_v_ms": current_field.base_wind_v,
        },
        "screening_notice": "Drift output is scenario-grade unless supplied vectors are validated against authoritative met-ocean observations.",
    }


@app.post("/api/correlate-ais")
async def correlate_ais(req: CorrelateAISRequest):
    """Ranks time-aligned AIS leads and emits radar/AIS review cues."""
    _, current_field, scenario_data = get_scenario_sar_and_currents(req.scenario_id)
    vessels = req.vessels if req.vessels is not None else scenario_data.get("ais_vessels", [])

    hindcast_traj = req.hindcast_trajectory
    if not hindcast_traj:
        drift_res = drift_engine.run_hindcast(
            scenario_data["center"]["lat"], scenario_data["center"]["lon"],
            current_field, target_slick_age_hours=abs(req.origin_time_rel_h)
        )
        hindcast_traj = drift_res["hindcast_trajectory"]

    # Ingest radar targets if provided, else simulate realistic radar ship hull detection
    radar_tgts = req.radar_targets
    if radar_tgts is None:
        sar_img, _, _ = get_scenario_sar_and_currents(req.scenario_id)
        radar_tgts = sar_engine.detect_radar_ship_targets(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"],
            pixel_size_m=float(scenario_data.get("satellite_metadata", {}).get("pixel_spacing_m", 10.0))
        )
    radar_tgts = [dict(target, relative_time_hours=target.get("relative_time_hours", 0.0)) for target in radar_tgts]

    results = ais_engine.attribute_oil_spill(
        vessels, req.origin_lat, req.origin_lon, req.origin_time_rel_h,
        spatial_radius_nm=req.spatial_radius_nm,
        temporal_window_h=req.temporal_window_h,
        hindcast_trajectory=hindcast_traj,
        radar_targets=radar_tgts
    )

    results["ais_provenance"] = req.ais_provenance or {
        "source_kind": scenario_data.get("ais_data_origin", "embedded demonstration scenario"),
        "scenario_id": req.scenario_id,
        "total_vessels_in_region": len(vessels)
    }
    results["screening_notice"] = (
        "AIS ranking is an investigative lead, not a finding of responsibility. "
        "Corroborate with calibrated imagery, chain-of-custody records, and human review."
    )
    return results


@app.get("/api/export-dossier/{scenario_id}")
async def export_dossier(scenario_id: str):
    """Generates a demo case summary for the selected preconfigured scenario."""
    sar_img, current_field, scenario_data = get_scenario_sar_and_currents(scenario_id)

    # Run full pipeline to compile all legal facts
    sar_res = sar_engine.process_sar_scene(
        sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
    )
    slick = sar_res["primary_slick"]
    target_age = float(np.clip(slick.get("estimated_age_hours", 12.0), 2.0, 36.0))

    drift_hindcast = drift_engine.run_hindcast(
        slick["centroid"]["lat"], slick["centroid"]["lon"],
        current_field, target_slick_age_hours=target_age
    )
    origin = drift_hindcast["origin_release_point"]

    coast_thresh = scenario_data.get("coastline_hazard", {}).get("coastline_lat_threshold")
    drift_forecast = drift_engine.run_forecast(
        slick["centroid"]["lat"], slick["centroid"]["lon"],
        current_field, coastline_lat_threshold=coast_thresh
    )

    ais_res = ais_engine.attribute_oil_spill(
        scenario_data.get("ais_vessels", []),
        origin["lat"], origin["lon"], origin["estimated_t0_hours_relative"],
        radar_targets=sar_res.get("radar_detected_ships", [])
    )

    # Generate PDF
    pdf_path = report_gen.generate_pdf_dossier(
        scenario_data, sar_res,
        {
            "origin_release_point": origin,
            "total_drift_distance_km": drift_hindcast["total_drift_distance_km"],
            "forecast_warning": drift_forecast["beaching_warning"],
            "weathering_summary": drift_forecast.get("weathering_summary", {})
        },
        ais_res
    )

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate case summary PDF")

    filename = os.path.basename(pdf_path)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
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
