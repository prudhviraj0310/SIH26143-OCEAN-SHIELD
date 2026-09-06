"""
Maritime Scenarios Database: Synthetic & Realistic Radar/AIS Scenarios
Features high-fidelity SAR backscatter imagery, dynamic ocean current fields,
and realistic AIS maritime traffic based on high-risk Indian maritime zones:
1. Gulf of Kachchh (Gujarat SPM Crude Tanker Corridor)
2. Mumbai High Offshore Sector (Arabian Sea Petroleum Basin)
3. Great Nicobar Channel (Malacca Chokepoint Transit Fairway)
"""

import base64
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

from .drift_engine import OceanCurrentField


def generate_synthetic_sar_image(
    width: int = 512,
    height: int = 512,
    slick_center_px: Tuple[int, int] = (256, 260),
    slick_length: int = 140,
    slick_width: int = 40,
    angle_deg: float = 38.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a realistic Synthetic Aperture Radar (SAR) backscatter scene:
    - Sea clutter with Rayleigh-distributed speckle noise (mean DN ~125)
    - Oil slick anomaly with Marangoni capillary wave damping (-10 dB backscatter drop -> DN ~45-65)
    - Natural plume feathering, edge gradients, and dispersion filament tail.
    """
    np.random.seed(seed)

    # 1. Background sea clutter (Gamma/Rayleigh distributed speckle)
    shape_k, scale_theta = 4.0, 31.0
    speckle = np.random.gamma(shape_k, scale_theta, (height, width))
    base_sea = np.clip(speckle, 60, 210).astype(np.float32)

    # Add gentle sea swell waves (spatial periodicity)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    wave_pattern = 12.0 * np.sin(x_coords * 0.05 + y_coords * 0.03)
    base_sea += wave_pattern

    # 2. Construct oil slick damping mask
    cx, cy = slick_center_px
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # Rotate coordinates relative to slick axis
    dx = x_coords - cx
    dy = y_coords - cy
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    # Elongated trailing plume with tapering tail (wider at older end, narrower at release point)
    taper = 1.0 + 0.3 * (x_rot / (slick_length / 2.0))
    local_w = np.maximum(slick_width * taper, 8.0)

    # Normalized ellipse metric
    ellipse_dist = (x_rot / (slick_length / 2.0)) ** 2 + (y_rot / (local_w / 2.0)) ** 2

    # Smooth damping factor (1.0 inside slick, transitioning to 0 outside)
    damping = np.clip(1.0 - ellipse_dist, 0.0, 1.0)
    damping = cv2.GaussianBlur(damping.astype(np.float32), (15, 15), 4.0)

    # Add secondary trailing filaments
    filament_dist = ((x_rot - 40) / 70.0) ** 2 + ((y_rot + 15) / 12.0) ** 2
    damping_filament = np.clip(1.0 - filament_dist, 0.0, 1.0) * 0.7
    damping = np.maximum(damping, damping_filament)

    # 3. Apply backscatter suppression where oil suppresses capillary waves
    # Oil dampens Bragg scattering by up to 60-70%
    attenuation_factor = 1.0 - (0.62 * damping)
    sar_scene = base_sea * attenuation_factor

    # Add localized sensor thermal noise
    sar_scene += np.random.normal(0, 3.0, (height, width))
    sar_uint8 = np.clip(sar_scene, 0, 255).astype(np.uint8)

    return sar_uint8


def image_to_base64_png(img: np.ndarray) -> str:
    """Converts a numpy grayscale or color image to a Base64 PNG data URL."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def get_all_scenarios() -> Dict[str, Dict[str, Any]]:
    """Returns the dictionary of predefined, validated operational maritime scenarios."""
    scenarios = {
        "gulf_of_kachchh": {
            "id": "gulf_of_kachchh",
            "title": "Gulf of Kachchh — Vadinar Crude SPM Deepwater Route",
            "region": "Gujarat Offshore / Northern Arabian Sea",
            "center": {"lat": 22.465, "lon": 69.215},
            "satellite_metadata": {
                "mission": "Sentinel-1A C-Band SAR",
                "acquisition_time_utc": "2026-09-04 13:10:00 UTC",
                "acquisition_time_ist": "2026-09-04 18:40:00 IST",
                "sensor_mode": "Interferometric Wide (IW) Swath",
                "polarization": "VV (Co-polarization optimal for sea surface roughness)",
                "pixel_spacing_m": 10.0,
                "incident_angle_deg": 36.2
            },
            "ocean_conditions": {
                "base_current_u": 0.28,
                "base_current_v": 0.18,
                "base_wind_u": 4.8,
                "base_wind_v": 3.2,
                "tidal_amplitude": 0.38,
                "tidal_period_h": 12.42,
                "sea_surface_temp_c": 28.5,
                "sea_state": "Beaufort 3 (Gentle Breeze)"
            },
            "coastline_hazard": {
                "coastline_lat_threshold": 22.58,
                "coastal_zone_name": "Marine National Park & Coral Reef Sanctuary, Jamnagar",
                "distance_to_shore_km": 16.5
            },
            "ais_vessels": [
                {
                    "mmsi": 636019482,
                    "imo": 9384721,
                    "vessel_name": "MT NEPTUNE GLORY",
                    "flag_state": "Liberia",
                    "vessel_type": "Crude Oil Tanker",
                    "call_sign": "D5XY8",
                    "length_m": 274,
                    "width_m": 48,
                    "dwt_tonnes": 158000,
                    "trajectory": [
                        {"relative_time_hours": -14.0, "lat": 22.280, "lon": 68.950, "sog_knots": 15.4, "cog_degrees": 52.0},
                        {"relative_time_hours": -12.0, "lat": 22.330, "lon": 69.030, "sog_knots": 15.1, "cog_degrees": 50.0},
                        {"relative_time_hours": -11.0, "lat": 22.365, "lon": 69.085, "sog_knots": 9.2, "cog_degrees": 48.0},
                        {"relative_time_hours": -10.5, "lat": 22.384, "lon": 69.116, "sog_knots": 5.1, "cog_degrees": 44.0},
                        {"relative_time_hours": -10.0, "lat": 22.398, "lon": 69.135, "sog_knots": 5.4, "cog_degrees": 46.0},
                        {"relative_time_hours": -9.0, "lat": 22.420, "lon": 69.170, "sog_knots": 11.5, "cog_degrees": 55.0},
                        {"relative_time_hours": -7.0, "lat": 22.460, "lon": 69.240, "sog_knots": 14.8, "cog_degrees": 54.0},
                        {"relative_time_hours": -4.0, "lat": 22.520, "lon": 69.340, "sog_knots": 14.5, "cog_degrees": 55.0},
                        {"relative_time_hours": 0.0, "lat": 22.580, "lon": 69.450, "sog_knots": 4.2, "cog_degrees": 70.0}   # Vadinar Anchorage
                    ]
                },
                {
                    "mmsi": 419001234,
                    "imo": 9422100,
                    "vessel_name": "MT BHARAT RATNA",
                    "flag_state": "India",
                    "vessel_type": "Product Tanker",
                    "call_sign": "AVBK",
                    "length_m": 182,
                    "width_m": 32,
                    "dwt_tonnes": 46000,
                    "trajectory": [
                        {"relative_time_hours": -14.0, "lat": 22.410, "lon": 68.980, "sog_knots": 13.8, "cog_degrees": 65.0},
                        {"relative_time_hours": -11.0, "lat": 22.445, "lon": 69.075, "sog_knots": 13.9, "cog_degrees": 66.0},
                        {"relative_time_hours": -8.0, "lat": 22.480, "lon": 69.170, "sog_knots": 13.7, "cog_degrees": 65.0},
                        {"relative_time_hours": -5.0, "lat": 22.515, "lon": 69.270, "sog_knots": 13.6, "cog_degrees": 64.0},
                        {"relative_time_hours": 0.0, "lat": 22.560, "lon": 69.410, "sog_knots": 13.5, "cog_degrees": 65.0}
                    ]
                },
                {
                    "mmsi": 563028000,
                    "imo": 9218541,
                    "vessel_name": "APL BUSAN",
                    "flag_state": "Singapore",
                    "vessel_type": "Container Ship",
                    "call_sign": "9V8812",
                    "length_m": 294,
                    "width_m": 40,
                    "dwt_tonnes": 68000,
                    "trajectory": [
                        {"relative_time_hours": -12.0, "lat": 22.550, "lon": 68.900, "sog_knots": 19.2, "cog_degrees": 85.0},
                        {"relative_time_hours": -8.0, "lat": 22.565, "lon": 69.080, "sog_knots": 18.8, "cog_degrees": 86.0},
                        {"relative_time_hours": -4.0, "lat": 22.575, "lon": 69.260, "sog_knots": 18.5, "cog_degrees": 84.0},
                        {"relative_time_hours": 0.0, "lat": 22.585, "lon": 69.440, "sog_knots": 17.9, "cog_degrees": 85.0}
                    ]
                },
                {
                    "mmsi": 352001456,
                    "imo": 9554321,
                    "vessel_name": "MV PACIFIC FORTUNE",
                    "flag_state": "Panama",
                    "vessel_type": "Bulk Carrier",
                    "call_sign": "3EKK2",
                    "length_m": 225,
                    "width_m": 32,
                    "dwt_tonnes": 75000,
                    "trajectory": [
                        {"relative_time_hours": -13.0, "lat": 22.290, "lon": 69.180, "sog_knots": 14.1, "cog_degrees": 230.0},
                        {"relative_time_hours": -9.0, "lat": 22.220, "lon": 69.070, "sog_knots": 14.2, "cog_degrees": 232.0},
                        {"relative_time_hours": -5.0, "lat": 22.150, "lon": 68.960, "sog_knots": 14.0, "cog_degrees": 228.0},
                        {"relative_time_hours": 0.0, "lat": 22.060, "lon": 68.820, "sog_knots": 14.2, "cog_degrees": 230.0}
                    ]
                },
                {
                    "mmsi": 419098765,
                    "imo": 9123456,
                    "vessel_name": "OCEAN SUPREME",
                    "flag_state": "India",
                    "vessel_type": "Offshore Supply Vessel",
                    "call_sign": "AVOS",
                    "length_m": 78,
                    "width_m": 18,
                    "dwt_tonnes": 3200,
                    "trajectory": [
                        {"relative_time_hours": -10.0, "lat": 22.250, "lon": 69.250, "sog_knots": 8.5, "cog_degrees": 310.0},
                        {"relative_time_hours": -6.0, "lat": 22.300, "lon": 69.200, "sog_knots": 8.2, "cog_degrees": 312.0},
                        {"relative_time_hours": 0.0, "lat": 22.350, "lon": 69.150, "sog_knots": 8.4, "cog_degrees": 315.0}
                    ]
                },
                {
                    "mmsi": 419992341,
                    "imo": 0,
                    "vessel_name": "FV JAI MATSYA 9",
                    "flag_state": "India",
                    "vessel_type": "Fishing Vessel",
                    "call_sign": "IND-F1",
                    "length_m": 22,
                    "width_m": 6,
                    "dwt_tonnes": 120,
                    "trajectory": [
                        {"relative_time_hours": -12.0, "lat": 22.490, "lon": 69.150, "sog_knots": 4.5, "cog_degrees": 120.0},
                        {"relative_time_hours": -6.0, "lat": 22.480, "lon": 69.180, "sog_knots": 3.8, "cog_degrees": 140.0},
                        {"relative_time_hours": 0.0, "lat": 22.470, "lon": 69.220, "sog_knots": 4.1, "cog_degrees": 110.0}
                    ]
                }
            ]
        },
        "mumbai_high": {
            "id": "mumbai_high",
            "title": "Mumbai High Offshore Sector — Arabian Sea Commercial Fairway",
            "region": "Offshore Maharashtra / Western EEZ",
            "center": {"lat": 19.412, "lon": 71.325},
            "satellite_metadata": {
                "mission": "Sentinel-1B C-Band SAR",
                "acquisition_time_utc": "2026-09-04 01:25:00 UTC",
                "acquisition_time_ist": "2026-09-04 06:55:00 IST",
                "sensor_mode": "Interferometric Wide (IW) Swath",
                "polarization": "VV",
                "pixel_spacing_m": 10.0,
                "incident_angle_deg": 38.8
            },
            "ocean_conditions": {
                "base_current_u": 0.22,
                "base_current_v": -0.12,
                "base_wind_u": 5.2,
                "base_wind_v": -2.1,
                "tidal_amplitude": 0.25,
                "tidal_period_h": 12.42,
                "sea_surface_temp_c": 29.1,
                "sea_state": "Beaufort 3"
            },
            "coastline_hazard": {
                "coastline_lat_threshold": None,
                "coastal_zone_name": "Open Ocean Basin (High-Density Petroleum Rigs Zone)",
                "distance_to_shore_km": 145.0
            },
            "ais_vessels": [
                {
                    "mmsi": 354921000,
                    "imo": 9248102,
                    "vessel_name": "MV ORIENTAL MARINER",
                    "flag_state": "Panama",
                    "vessel_type": "Bulk Carrier",
                    "call_sign": "HO3041",
                    "length_m": 229,
                    "width_m": 32,
                    "dwt_tonnes": 82000,
                    "trajectory": [
                        {"relative_time_hours": -12.0, "lat": 19.260, "lon": 71.050, "sog_knots": 14.6, "cog_degrees": 58.0},
                        {"relative_time_hours": -9.0, "lat": 19.330, "lon": 71.180, "sog_knots": 8.0, "cog_degrees": 55.0},
                        {"relative_time_hours": -8.5, "lat": 19.344, "lon": 71.208, "sog_knots": 4.8, "cog_degrees": 52.0},  # CULPRIT SLOWDOWN & DUMP
                        {"relative_time_hours": -7.5, "lat": 19.365, "lon": 71.245, "sog_knots": 6.2, "cog_degrees": 56.0},
                        {"relative_time_hours": -6.0, "lat": 19.395, "lon": 71.300, "sog_knots": 13.9, "cog_degrees": 58.0},
                        {"relative_time_hours": 0.0, "lat": 19.520, "lon": 71.530, "sog_knots": 14.4, "cog_degrees": 58.0}
                    ]
                },
                {
                    "mmsi": 636014499,
                    "imo": 9321456,
                    "vessel_name": "STENA PROSPERITY",
                    "flag_state": "Liberia",
                    "vessel_type": "Product Tanker",
                    "call_sign": "A8TM2",
                    "length_m": 183,
                    "width_m": 32,
                    "dwt_tonnes": 49000,
                    "trajectory": [
                        {"relative_time_hours": -12.0, "lat": 19.450, "lon": 71.080, "sog_knots": 13.5, "cog_degrees": 72.0},
                        {"relative_time_hours": -6.0, "lat": 19.485, "lon": 71.260, "sog_knots": 13.4, "cog_degrees": 73.0},
                        {"relative_time_hours": 0.0, "lat": 19.520, "lon": 71.440, "sog_knots": 13.6, "cog_degrees": 72.0}
                    ]
                },
                {
                    "mmsi": 419088112,
                    "imo": 9187766,
                    "vessel_name": "SAMUDRA SEVAK",
                    "flag_state": "India",
                    "vessel_type": "Offshore Supply Vessel",
                    "call_sign": "AVSK",
                    "length_m": 85,
                    "width_m": 20,
                    "dwt_tonnes": 4100,
                    "trajectory": [
                        {"relative_time_hours": -10.0, "lat": 19.280, "lon": 71.380, "sog_knots": 7.5, "cog_degrees": 340.0},
                        {"relative_time_hours": -5.0, "lat": 19.340, "lon": 71.360, "sog_knots": 7.4, "cog_degrees": 342.0},
                        {"relative_time_hours": 0.0, "lat": 19.400, "lon": 71.340, "sog_knots": 7.6, "cog_degrees": 340.0}
                    ]
                }
            ]
        },
        "great_nicobar": {
            "id": "great_nicobar",
            "title": "Great Nicobar Channel — Six Degree Channel Chokepoint",
            "region": "Andaman & Nicobar Islands / Malacca Strait Transit",
            "center": {"lat": 6.845, "lon": 93.420},
            "satellite_metadata": {
                "mission": "Sentinel-1A C-Band SAR",
                "acquisition_time_utc": "2026-09-04 11:45:00 UTC",
                "acquisition_time_ist": "2026-09-04 17:15:00 IST",
                "sensor_mode": "Interferometric Wide (IW) Swath",
                "polarization": "VV",
                "pixel_spacing_m": 10.0,
                "incident_angle_deg": 35.5
            },
            "ocean_conditions": {
                "base_current_u": -0.32,
                "base_current_v": 0.08,
                "base_wind_u": -6.0,
                "base_wind_v": 2.5,
                "tidal_amplitude": 0.20,
                "tidal_period_h": 12.42,
                "sea_surface_temp_c": 29.8,
                "sea_state": "Beaufort 4 (Moderate Breeze)"
            },
            "coastline_hazard": {
                "coastline_lat_threshold": 6.95,
                "coastal_zone_name": "Great Nicobar Biosphere Reserve & Coral Reefs",
                "distance_to_shore_km": 28.0
            },
            "ais_vessels": [
                {
                    "mmsi": 357890123,
                    "imo": 9411234,
                    "vessel_name": "CHEM STAR III",
                    "flag_state": "Panama",
                    "vessel_type": "Chemical Tanker",
                    "call_sign": "3EVV7",
                    "length_m": 185,
                    "width_m": 32,
                    "dwt_tonnes": 52000,
                    "trajectory": [
                        {"relative_time_hours": -15.0, "lat": 6.750, "lon": 93.850, "sog_knots": 14.5, "cog_degrees": 285.0},
                        {"relative_time_hours": -12.0, "lat": 6.790, "lon": 93.660, "sog_knots": 14.1, "cog_degrees": 284.0},
                        {"relative_time_hours": -11.0, "lat": 6.809, "lon": 93.578, "sog_knots": 5.2, "cog_degrees": 280.0},  # CULPRIT DISCHARGE
                        {"relative_time_hours": -10.0, "lat": 6.825, "lon": 93.510, "sog_knots": 6.8, "cog_degrees": 286.0},
                        {"relative_time_hours": -8.0, "lat": 6.850, "lon": 93.380, "sog_knots": 14.2, "cog_degrees": 288.0},
                        {"relative_time_hours": 0.0, "lat": 6.940, "lon": 92.950, "sog_knots": 14.4, "cog_degrees": 286.0}
                    ]
                },
                {
                    "mmsi": 477123456,
                    "imo": 9654123,
                    "vessel_name": "EVER GLOBE",
                    "flag_state": "Hong Kong",
                    "vessel_type": "Container Ship",
                    "call_sign": "VRAB8",
                    "length_m": 366,
                    "width_m": 51,
                    "dwt_tonnes": 145000,
                    "trajectory": [
                        {"relative_time_hours": -14.0, "lat": 6.680, "lon": 93.900, "sog_knots": 19.5, "cog_degrees": 285.0},
                        {"relative_time_hours": -7.0, "lat": 6.780, "lon": 93.450, "sog_knots": 19.4, "cog_degrees": 286.0},
                        {"relative_time_hours": 0.0, "lat": 6.880, "lon": 93.000, "sog_knots": 19.2, "cog_degrees": 285.0}
                    ]
                }
            ]
        }
    }
    return scenarios


def get_scenario_sar_and_currents(scenario_id: str) -> Tuple[np.ndarray, OceanCurrentField, Dict[str, Any]]:
    """
    Returns (sar_image_uint8, current_field, scenario_data) for a scenario.
    """
    scenarios = get_all_scenarios()
    if scenario_id not in scenarios:
        scenario_id = "gulf_of_kachchh"

    data = scenarios[scenario_id]
    cond = data["ocean_conditions"]

    current_field = OceanCurrentField(
        base_current_u=cond["base_current_u"],
        base_current_v=cond["base_current_v"],
        base_wind_u=cond["base_wind_u"],
        base_wind_v=cond["base_wind_v"],
        tidal_amplitude=cond["tidal_amplitude"],
        tidal_period_h=cond["tidal_period_h"]
    )

    # Specific geometric shape for each scenario's slick
    if scenario_id == "gulf_of_kachchh":
        sar_img = generate_synthetic_sar_image(
            width=512, height=512,
            slick_center_px=(256, 256),
            slick_length=155, slick_width=42,
            angle_deg=42.0, seed=42
        )
    elif scenario_id == "mumbai_high":
        sar_img = generate_synthetic_sar_image(
            width=512, height=512,
            slick_center_px=(260, 250),
            slick_length=135, slick_width=38,
            angle_deg=56.0, seed=84
        )
    else:
        sar_img = generate_synthetic_sar_image(
            width=512, height=512,
            slick_center_px=(250, 265),
            slick_length=145, slick_width=36,
            angle_deg=282.0, seed=128
        )

    return sar_img, current_field, data


def generate_synthetic_eo_image(
    width: int = 512,
    height: int = 512,
    slick_center_px: Tuple[int, int] = (256, 260),
    slick_length: int = 140,
    slick_width: int = 40,
    angle_deg: float = 38.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates realistic Sentinel-2 MSI Multi-Spectral Optical imagery:
    - Deep blue oceanic baseline (B=110, G=70, R=40, NIR=35)
    - Specular sunglint oil slick with high refractive index (n ~ 1.50 vs water 1.34)
      producing elevated NIR (115-135) and elevated visible reflectance
    - Returns (rgb_image_uint8, nir_band_uint8, swir_band_uint8)
    """
    np.random.seed(seed + 101)
    # 1. Base oceanic background reflectance
    b = np.clip(np.random.normal(110, 8, (height, width)), 40, 200).astype(np.float32)
    g = np.clip(np.random.normal(70, 6, (height, width)), 20, 160).astype(np.float32)
    r = np.clip(np.random.normal(40, 5, (height, width)), 10, 120).astype(np.float32)
    nir = np.clip(np.random.normal(32, 4, (height, width)), 10, 80).astype(np.float32)
    swir = np.clip(np.random.normal(25, 4, (height, width)), 8, 60).astype(np.float32)

    # 2. Oil slick footprint
    cx, cy = slick_center_px
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    dx = x_coords - cx
    dy = y_coords - cy
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    ellipse_dist = (x_rot / (slick_length / 2.0)) ** 2 + (y_rot / (slick_width / 2.0)) ** 2
    damping = np.clip(1.0 - ellipse_dist, 0.0, 1.0)
    damping = cv2.GaussianBlur(damping.astype(np.float32), (15, 15), 4.0)

    # 3. Apply sunglint optical signature
    # Crude oil in sunglint has positive contrast in NIR and visible
    nir += damping * 85.0
    r += damping * 55.0
    g += damping * 40.0
    b += damping * 25.0
    swir += damping * 35.0

    rgb_uint8 = cv2.merge([
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8)
    ])
    nir_uint8 = np.clip(nir, 0, 255).astype(np.uint8)
    swir_uint8 = np.clip(swir, 0, 255).astype(np.uint8)

    return rgb_uint8, nir_uint8, swir_uint8


def get_scenario_eo_data(scenario_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (rgb_uint8, nir_band_uint8, swir_band_uint8, scenario_data) for multi-spectral EO processing.
    """
    scenarios = get_all_scenarios()
    if scenario_id not in scenarios:
        scenario_id = "gulf_of_kachchh"

    data = scenarios[scenario_id]
    if scenario_id == "gulf_of_kachchh":
        rgb, nir, swir = generate_synthetic_eo_image(
            width=512, height=512, slick_center_px=(256, 256),
            slick_length=155, slick_width=42, angle_deg=42.0, seed=42
        )
    elif scenario_id == "mumbai_high":
        rgb, nir, swir = generate_synthetic_eo_image(
            width=512, height=512, slick_center_px=(260, 250),
            slick_length=135, slick_width=38, angle_deg=56.0, seed=84
        )
    else:
        rgb, nir, swir = generate_synthetic_eo_image(
            width=512, height=512, slick_center_px=(250, 265),
            slick_length=145, slick_width=36, angle_deg=282.0, seed=128
        )
    return rgb, nir, swir, data
