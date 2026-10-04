#!/usr/bin/env python3
"""
OCEAN-SHIELD: Command-Line Intelligence Terminal
National Technical Research Organisation (NTRO) / Indian Coast Guard
Problem Statement SIH26143: Satellite SAR/EO Oil Spill Detection & AIS Attribution

Enables headless, scriptable, and automated execution of the entire forensic chain:
1. Multi-modal satellite remote sensing (PyTorch SAR U-Net & Optical Sentinel-2 NDOI)
2. 4th-order Runge-Kutta Lagrangian hydrodynamic drift hindcasting
3. Conditional transport; no inferred mass, age or confidence
4. AIS spatiotemporal correlation, kinematic review scoring & radar/AIS mismatch cues
5. Automated generation of analyst-review case-summary PDFs
"""

import argparse
import os
import sys
import time

import numpy as np

from src.ocean_shield.sar_engine import SAREngine
from src.ocean_shield.eo_engine import EOEngine
from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
from src.ocean_shield.ais_engine import AISEngine
from src.ocean_shield.scenarios import (
    get_scenario_sar_and_currents, get_scenario_eo_data, get_all_scenarios
)
from src.ocean_shield.report_generator import DossierReportGenerator
from src.ocean_shield.ais_ingestion import parse_marinecadastre_csv


def print_banner():
    print("""
================================================================================
  🌊 OCEAN-SHIELD : SATELLITE MARITIME SURVEILLANCE & ATTRIBUTION ENGINE
  Problem Statement: SIH26143 | NTRO / Indian Coast Guard / DG Shipping
  Dual-Mode SAR U-Net (PyTorch) + Sentinel-2 Multi-Spectral EO (NDOI)
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="OCEAN-SHIELD: Bundled research screening demo, not autonomous attribution"
    )
    parser.add_argument(
        "--scenario",
        choices=["gulf_of_kachchh", "mumbai_high", "great_nicobar", "zenodo_sentinel1_real"],
        default="gulf_of_kachchh",
        help="Strategic Indian maritime incident scenario to analyze."
    )
    parser.add_argument(
        "--engine",
        choices=["unet", "cfar_edge"],
        default="unet",
        help="SAR detection pipeline: unet (PyTorch Deep Learning) or cfar_edge (Adaptive CFAR)."
    )
    parser.add_argument(
        "--sensor",
        choices=["sar", "eo", "both"],
        default="both",
        help="Satellite sensor mode: sar, eo, or both."
    )
    parser.add_argument(
        "--ais-csv",
        type=str,
        default=None,
        help="Optional custom NOAA / MarineCadastre AIS CSV dataset to ingest."
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Compile and export an analyst-review case-summary PDF."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for generated case summaries."
    )
    parser.add_argument("--demo", action="store_true", help="Explicitly select isolated scenario forcing and synthetic AIS.")
    parser.add_argument("--age-hours", type=float, help="User-supplied conditional slick-age hypothesis, not inferred from imagery.")

    args = parser.parse_args()
    if not args.demo:
        parser.error("Bundled CLI scenes require --demo; use the dated source-ingestion API for source-backed work.")
    if args.age_hours is None or not np.isfinite(args.age_hours) or not 0 < args.age_hours <= 168:
        parser.error("--age-hours must be a finite positive analyst hypothesis no greater than 168 hours.")
    if args.ais_csv:
        parser.error("Do not mix dated source AIS with a demonstration transport field; use source ingestion instead.")
    print_banner()
    print("DEMONSTRATION ONLY — synthetic forcing and scenario trajectories; no operational or legal certification.")

    custom_ais = None

    t_start = time.time()
    print(f"[*] Initializing forensic pipeline for scenario: '{args.scenario.upper()}'")
    print(f"[*] Engine mode: {args.engine.upper()} | Satellite sensors: {args.sensor.upper()}")

    # Initialize engines
    sar_engine = SAREngine()
    eo_engine = EOEngine(ground_resolution_m=10.0)
    drift_engine = DriftEngine()
    ais_engine = AISEngine()

    sar_img, current_field, scenario_data = get_scenario_sar_and_currents(args.scenario)
    conditions = scenario_data["ocean_conditions"]
    current_field = OceanCurrentField(**{key: conditions[key] for key in
        ("base_current_u", "base_current_v", "base_wind_u", "base_wind_v")}, constant_vectors=True)
    scenario_data["ocean_data_source"] = "Explicit isolated demonstration vectors; archived sources are not used"
    scenario_data["cli_mode"] = "benchmark_demo"
    center_lat = scenario_data["center"]["lat"]
    center_lon = scenario_data["center"]["lon"]

    if args.ais_csv:
        try:
            reference_time = scenario_data["satellite_metadata"].get("acquisition_time_utc", "").replace(" UTC", "Z").replace(" ", "T")
            with open(args.ais_csv, "rb") as handle:
                parsed_ais = parse_marinecadastre_csv(handle.read(), os.path.basename(args.ais_csv), reference_time)
            custom_ais = parsed_ais["vessels"]
            provenance = parsed_ais["provenance"]
            print(
                f"[*] AIS source loaded: {provenance['vessels']} vessels / {provenance['valid_pings']} pings "
                f"(aligned to SAR acquisition: {provenance['reference_time_utc']})"
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Could not ingest --ais-csv: {exc}")

    # 1. SAR Processing
    sar_res = None
    if args.sensor in ["sar", "both"]:
        print(f"\n[1/4] Processing Sentinel-1 SAR imagery ({scenario_data['satellite_metadata']['mission']})...")
        sar_res = sar_engine.process_sar_scene(
            sar_img, center_lat, center_lon, pixel_size_m=50.0, model_type=args.engine
        )
        slick = sar_res.get("primary_slick")
        if slick:
            print(f"      - Engine Active: {sar_res['active_engine']}")
            print(f"      - Spill Observed Area: {slick['area_km2']:.2f} km²")
            print("      - Mass / thickness / identity: NOT INFERRED")
            print(f"      - Elongation Ratio: {slick['elongation']:.2f}:1")
            print(f"      - Morphology screening score: {slick['screening_score']:.1f}/100 (uncalibrated)")
            print("      - Oil confidence / age estimate: NOT INFERRED")
        else:
            print("      - No significant slick detected.")

        radar_ships = sar_res.get("radar_detected_ships", [])
        print(f"      - Radar Metallic Corner-Reflectors Detected: {len(radar_ships)} vessels")

    # 2. EO Multi-Spectral Processing
    if args.sensor in ["eo", "both"]:
        print("\n[2/4] Processing Sentinel-2 MSI Multi-Spectral Optical (EO) Imagery...")
        rgb_img, nir_band, swir_band, _ = get_scenario_eo_data(args.scenario)
        clean_mask, eo_diag = eo_engine.segment_optical_slick(rgb_img, nir_band=nir_band, swir_band=swir_band)
        print(f"      - Normalized Difference Oil Index (NDOI): +{eo_diag['mean_ndoi']:.4f}")
        print(f"      - Optical Classification: {eo_diag['classification']}")
        print("      - Optical confidence / biological identity: NOT CALIBRATED / NOT DETERMINED")

    # 3. Conditional transport, with no imagery-derived age or mass.
    print("\n[3/4] Running conditional per-particle RK4 transport...")
    target_age = args.age_hours

    hindcast_res = drift_engine.run_hindcast(
        center_lat, center_lon, current_field,
        max_lookback_hours=target_age, target_slick_age_hours=target_age
    )
    origin = hindcast_res["origin_release_point"]
    print(f"      - Conditional terminal-cloud centre: {origin['lat']:.4f}°, {origin['lon']:.4f}°")
    print(f"      - Actual integrated clock: {origin['estimated_t0_hours_relative']}h; origin confidence NOT ESTIMATED")
    print(f"      - Reverse Drift Distance: {hindcast_res['total_drift_distance_km']:.1f} km")

    # Forward forecast & Weathering
    coast_thresh = scenario_data.get("coastline_hazard", {}).get("coastline_lat_threshold")
    forecast_res = drift_engine.run_forecast(
        center_lat, center_lon, current_field,
        forecast_hours=24.0, coastline_lat_threshold=coast_thresh
    )
    w = forecast_res["weathering_summary"]
    print("      - Weathering NOT_ASSESSED: measured mass and oil profile not supplied")
    print("      - Shoreline / asset risk NOT_ASSESSED: validated geometry not supplied")

    # 4. AIS Maritime Correlation & Lead Ranking
    print("\n[4/4] Correlating Maritime AIS Traffic & Cross-Referencing Radar Contacts...")
    vessels = custom_ais if custom_ais is not None else scenario_data.get("ais_vessels", [])
    radar_ships = sar_res.get("radar_detected_ships", []) if sar_res else []

    ais_res = ais_engine.attribute_oil_spill(
        vessels, origin["lat"], origin["lon"], origin["estimated_t0_hours_relative"],
        spatial_radius_nm=30.0, temporal_window_h=5.0,
        hindcast_trajectory=hindcast_res["hindcast_trajectory"],
        radar_targets=radar_ships
    )

    culprit = ais_res.get("primary_review_lead")
    if culprit:
        print("\n" + "=" * 80)
        print(f"⚠️  HIGHEST-RANKED INVESTIGATIVE LEAD: {culprit['vessel_name']}")
        print("=" * 80)
        print(f"   - IMO: {culprit['imo']} | MMSI: {culprit['mmsi']} | Flag: {culprit['flag_state']}")
        dwt = culprit.get("dwt_tonnes")
        print(f"   - Vessel Class: {culprit['vessel_type']} ({dwt} DWT)")
        print(f"   - Heuristic lead score: {culprit['composite_suspect_score']}/100 (not confidence)")
        dt_val = culprit['closest_approach'].get('time_diff_h')
        print(f"   - Closest Approach (CPA): {culprit['closest_approach']['distance_nm']:.2f} NM to epicenter")
        print(f"   - Temporal difference: {dt_val} hours under the supplied hypothesis")
        gate = culprit['abstention_verdict']
        print(f"   - Evidence gate: {'LEAD WITHHELD' if gate['is_abstention'] else 'SCREENING ONLY'} / {gate['status']} — {gate['reason']}")
    else:
        print("[-] No vessel lead met the configured corridor threshold.")

    # Radar/AIS correlation review cues
    dark_vessels = ais_res.get("dark_vessels_detected", [])
    print(f"\n[*] Radar/AIS review cues: {len(dark_vessels)} contacts require AIS coverage verification.")
    for dv in dark_vessels:
        print(f"   - ⚡ [{dv['target_id']}] Lat: {dv['lat']:.4f}, Lon: {dv['lon']:.4f} | RCS: {dv['radar_rcs_mean_db']} dB | Distance to Spill: {dv.get('distance_to_spill_origin_nm', 'N/A')} NM | {dv.get('threat_classification', 'UNREGISTERED')}")

    # 5. Export analyst-review case summary
    if args.export_pdf:
        print("\n[*] Compiling analyst-review case summary (PDF)...")
        os.makedirs(args.output_dir, exist_ok=True)
        report_gen = DossierReportGenerator(output_dir=args.output_dir)
        pdf_path = report_gen.generate_pdf_dossier(
            scenario_data, sar_res or {},
            {
                "origin_release_point": origin,
                "total_drift_distance_km": hindcast_res["total_drift_distance_km"],
                "forecast_warning": forecast_res["beaching_warning"],
                "weathering_summary": w
            },
            ais_res,
            filename=f"Analyst_Case_Summary_{args.scenario.upper()}.pdf"
        )
        print(f"   - Dossier compiled successfully: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    elapsed = time.time() - t_start
    print(f"\n[+] Complete screening analysis finished in {elapsed:.2f} seconds.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
