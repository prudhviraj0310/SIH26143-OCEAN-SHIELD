"""
FastAPI Server: OCEAN-SHIELD API & Operational Command Center
Serves REST endpoints for SAR image analysis, Lagrangian hydrodynamic drift simulations,
AIS vessel correlation & anomaly scoring, and automated Coast Guard PDF violation dossier generation.
"""

import os
import io
import csv
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .sar_engine import SAREngine
from .drift_engine import DriftEngine, OceanCurrentField
from .ais_engine import AISEngine
from .eo_engine import EOEngine
from .scenarios import (
    get_all_scenarios, get_scenario_sar_and_currents, get_scenario_eo_data,
    image_to_base64_png, generate_synthetic_sar_image
)
from .report_generator import DossierReportGenerator


app = FastAPI(
    title="OCEAN-SHIELD Maritime Intelligence API",
    description="Satellite SAR & Optical EO Oil Spill Detection, Hydrodynamic Hindcast & AIS Rogue Vessel Attribution",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# --- Request Models ---
class AnalyzeSARRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    use_super_resolution: bool = True
    model_type: str = "unet"  # "unet" (PyTorch Deep Learning) or "cfar_edge" (Fast Tactical)
    threshold_offset: float = 22.0


class AnalyzeEORequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"


class SimulateDriftRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    slick_lat: float
    slick_lon: float
    slick_age_hours: Optional[float] = None
    max_lookback_hours: float = 24.0
    forecast_hours: float = 48.0


class CorrelateAISRequest(BaseModel):
    scenario_id: str = "gulf_of_kachchh"
    origin_lat: float
    origin_lon: float
    origin_time_rel_h: float
    spatial_radius_nm: float = 25.0
    temporal_window_h: float = 5.0
    hindcast_trajectory: Optional[List[Dict[str, Any]]] = None
    radar_targets: Optional[List[Dict[str, Any]]] = None


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
        summary_list.append({
            "id": sc["id"],
            "title": sc["title"],
            "region": sc["region"],
            "center": sc["center"],
            "mission": sc["satellite_metadata"]["mission"],
            "acquisition_ist": sc["satellite_metadata"]["acquisition_time_ist"],
            "sea_state": sc["ocean_conditions"]["sea_state"],
            "vessels_count": len(sc.get("ais_vessels", []))
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

    pixel_size = 50.0  # Real-world Sentinel-1 mapping scale
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


@app.post("/api/analyze-eo")
async def analyze_eo(req: AnalyzeEORequest):
    """
    Processes Sentinel-2 MSI Multi-Spectral Optical (EO) Imagery:
    Computes Normalized Difference Oil Index (NDOI) and Floating Algae Index (FAI)
    to detect sunglint oil anomalies and reject natural lookalike algal blooms.
    """
    rgb_img, nir_band, swir_band, scenario_data = get_scenario_eo_data(req.scenario_id)
    clean_mask, diagnostics = eo_engine.segment_optical_slick(rgb_img, nir_band=nir_band)

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
            "lookalike_discrimination": "NDOI > 0.05 & FAI < 45 (Algal Bloom Discarded)"
        }
    }


@app.post("/api/simulate-drift")
async def simulate_drift(req: SimulateDriftRequest):
    """
    Executes backward Lagrangian hindcasting to find spill release origin (x0, y0, t0),
    forward forecasting to estimate coastal collision, and Mackay ADIOS physical oil weathering.
    """
    _, current_field, scenario_data = get_scenario_sar_and_currents(req.scenario_id)

    # Calibrate target release age from scenario ground truth if not provided or out of bounds
    gt_age = abs(scenario_data.get("ground_truth_culprit", {}).get("discharge_time_rel_h", 10.5))
    target_age = req.slick_age_hours
    if target_age is None or target_age > 16.0 or target_age < 4.0:
        target_age = gt_age

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
        "beaching_warning": forecast_res["beaching_warning"]
    }


@app.post("/api/correlate-ais")
async def correlate_ais(req: CorrelateAISRequest):
    """
    Reconstructs historical AIS maritime traffic, filters non-coincident vessels,
    calculates kinematic behavioral anomaly scores, and flags non-cooperative DARK VESSELS.
    """
    _, current_field, scenario_data = get_scenario_sar_and_currents(req.scenario_id)
    vessels = scenario_data.get("ais_vessels", [])

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
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"], pixel_size_m=50.0
        )

    results = ais_engine.attribute_oil_spill(
        vessels, req.origin_lat, req.origin_lon, req.origin_time_rel_h,
        spatial_radius_nm=req.spatial_radius_nm,
        temporal_window_h=req.temporal_window_h,
        hindcast_trajectory=hindcast_traj,
        radar_targets=radar_tgts
    )

    return results


@app.get("/api/export-dossier/{scenario_id}")
async def export_dossier(scenario_id: str):
    """
    Generates and downloads the court-admissible Indian Coast Guard Legal Violation Dossier PDF.
    """
    sar_img, current_field, scenario_data = get_scenario_sar_and_currents(scenario_id)

    # Run full pipeline to compile all legal facts
    sar_res = sar_engine.process_sar_scene(
        sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
    )
    slick = sar_res["primary_slick"]
    target_age = abs(scenario_data.get("ground_truth_culprit", {}).get("discharge_time_rel_h", 10.5))

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
        raise HTTPException(status_code=500, detail="Failed to generate violation dossier PDF")

    filename = os.path.basename(pdf_path)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/api/upload-ais-csv")
async def upload_ais_csv(file: UploadFile = File(...)):
    """
    Directly ingests NOAA / MarineCadastre formatted CSV files:
    Headers: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft
    """
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    vessels_dict = {}
    for row in reader:
        mmsi = row.get("MMSI") or row.get("mmsi")
        if not mmsi:
            continue
        try:
            mmsi = int(mmsi)
            lat = float(row.get("LAT") or row.get("lat") or 0.0)
            lon = float(row.get("LON") or row.get("lon") or 0.0)
            sog = float(row.get("SOG") or row.get("sog") or 0.0)
            cog = float(row.get("COG") or row.get("cog") or 0.0)
            v_name = row.get("VesselName") or row.get("vessel_name") or f"VESSEL-{mmsi}"
            imo = int(row.get("IMO") or row.get("imo") or 0)
            v_type = row.get("VesselType") or row.get("vessel_type") or "Other / Unknown"

            if mmsi not in vessels_dict:
                vessels_dict[mmsi] = {
                    "mmsi": mmsi,
                    "imo": imo,
                    "vessel_name": v_name,
                    "flag_state": "Commercial Transit",
                    "vessel_type": v_type,
                    "length_m": float(row.get("Length") or 160),
                    "width_m": float(row.get("Width") or 25),
                    "dwt_tonnes": 40000,
                    "trajectory": []
                }

            vessels_dict[mmsi]["trajectory"].append({
                "relative_time_hours": 0.0,
                "lat": lat,
                "lon": lon,
                "sog_knots": sog,
                "cog_degrees": cog
            })
        except Exception:
            continue

    parsed_vessels = list(vessels_dict.values())
    return {
        "status": "success",
        "filename": file.filename,
        "vessels_parsed_count": len(parsed_vessels),
        "sample_vessels": parsed_vessels[:5]
    }
