"""
Unit and Integration Test Suite for OCEAN-SHIELD
Smart India Hackathon 2026 - Problem Statement SIH26143
"""

import os
import tempfile
import unittest
import cv2
import numpy as np

from src.ocean_shield.sar_engine import SAREngine
from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
from src.ocean_shield.ais_engine import AISEngine
from src.ocean_shield.scenarios import get_scenario_sar_and_currents, get_all_scenarios
from src.ocean_shield.report_generator import DossierReportGenerator
from src.ocean_shield.ais_ingestion import parse_marinecadastre_csv


class TestOceanShieldPipeline(unittest.TestCase):

    def setUp(self):
        self.sar_engine = SAREngine()
        self.drift_engine = DriftEngine()
        self.ais_engine = AISEngine()
        pdfs = tempfile.TemporaryDirectory(prefix="ocean-pipeline-pdfs-")
        self.addCleanup(pdfs.cleanup)
        self.report_gen = DossierReportGenerator(output_dir=pdfs.name)

    def test_sar_engine_segmentation_and_super_resolution(self):
        """Validates radar speckle filtering, CFAR segmentation, and Super-Resolution."""
        sar_img, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        self.assertEqual(sar_img.shape, (512, 512))

        # Test Super-Resolution
        sr_img = self.sar_engine.enhance_sar_super_resolution(sar_img, scale_factor=2)
        self.assertEqual(sr_img.shape, (1024, 1024))

        # Test Lee Filter
        filtered = self.sar_engine.enhanced_lee_filter(sar_img)
        self.assertEqual(filtered.shape, sar_img.shape)

        # Test Segmentation
        mask, contours = self.sar_engine.segment_oil_slick(sar_img)
        self.assertGreater(len(contours), 0, "Should detect at least one slick contour")

        # Test Full Scene Processing
        results = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
        )
        self.assertGreater(results["total_slicks_detected"], 0)
        slick = results["primary_slick"]
        self.assertIsNotNone(slick)
        self.assertGreater(slick["area_km2"], 0.0)
        self.assertIsNone(slick["estimated_mass_tonnes"])
        self.assertIsNone(slick["estimated_age_hours"])
        self.assertIn("polygon_geojson", slick)
        self.assertIsNone(slick["confidence_score"], "Morphology is not calibrated oil confidence")
        self.assertEqual(slick["confidence_status"], "UNCALIBRATED_SCREENING_SCORE")
        self.assertIn("screening_score", slick)
        self.assertTrue(0 <= slick["screening_score"] <= 100)
        self.assertIn("limitations", slick)
        self.assertEqual(slick["fay_spreading_age"]["status"], "NOT_ASSESSED")
        self.assertIsNone(slick["fay_spreading_age"]["estimated_age_hours"])
        if "lookalike_screening" in slick:
            self.assertNotIn("oil_probability", slick["lookalike_screening"])

    def test_drift_engine_hindcast_and_forecast(self):
        """Validates reverse Lagrangian particle hindcasting and forward forecasting."""
        _, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        current_field = OceanCurrentField(
            base_current_u=0.25, base_current_v=0.15,
            base_wind_u=4.5, base_wind_v=3.0,
            constant_vectors=True,
        )
        center_lat = scenario_data["center"]["lat"]
        center_lon = scenario_data["center"]["lon"]

        # Run Backward Hindcast
        hindcast = self.drift_engine.run_hindcast(
            center_lat, center_lon, current_field,
            max_lookback_hours=18.0, target_slick_age_hours=10.5
        )
        origin = hindcast["origin_release_point"]
        self.assertEqual(origin["assumed_slick_age_hours"], 10.5)
        self.assertIsNone(origin["confidence_percent"])
        self.assertIn("not an inferred spill origin", origin["inference_status"])
        self.assertEqual(hindcast["hindcast_trajectory"][-1]["relative_time_hours"], -10.5)
        self.assertGreater(hindcast["total_drift_distance_km"], 0.0)
        self.assertGreater(len(hindcast["hindcast_trajectory"]), 10)

        # Run Forward Forecast
        forecast = self.drift_engine.run_forecast(
            center_lat, center_lon, current_field,
            forecast_hours=24.0,
        )
        self.assertIn("forecast_trajectory", forecast)
        self.assertIn("beaching_warning", forecast)
        # A real met-ocean field may route the slick offshore; verify the model
        # reports the condition rather than asserting a scenario-specific outcome.
        self.assertIsNone(forecast["beaching_warning"]["will_beach"])
        self.assertEqual(forecast["beaching_warning"]["status"], "NOT_ASSESSED")
        self.assertIsNone(forecast["weathering_summary"])

    def test_ais_correlation_produces_review_lead_not_culprit(self):
        """Validates corridor screening without asserting a responsibility finding."""
        _, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")

        # Test candidate release coordinates derived from drift hindcast
        release_lat = 22.4995
        release_lon = 69.0754
        release_time_h = -10.5

        results = self.ais_engine.attribute_oil_spill(
            scenario_data["ais_vessels"],
            release_lat,
            release_lon,
            release_time_h
        )

        lead = results["primary_review_lead"]
        self.assertIsNotNone(lead)
        self.assertIsNone(results["primary_culprit"])
        self.assertIn("lead_priority_score", lead)
        self.assertIn("not a probability", results["screening_notice"])
        self.assertLess(lead["closest_approach"]["distance_nm"], 1.0)

    def test_all_scenarios_and_dossier_pdf(self):
        """Validates all Indian and benchmark maritime sectors and generates an analyst-review PDF."""
        scenarios = get_all_scenarios()
        self.assertGreaterEqual(len(scenarios), 3)

        for sid in scenarios:
            sar_img, current_field, data = get_scenario_sar_and_currents(sid)
            self.assertEqual(sar_img.shape, (512, 512))

        # Test Dossier PDF generation
        sar_img, current_field, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        sar_res = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
        )
        origin = {"lat": 22.385, "lon": 69.115, "slick_age_hours": 10.5, "estimated_t0_hours_relative": -10.5}
        ais_res = self.ais_engine.attribute_oil_spill(
            scenario_data["ais_vessels"], origin["lat"], origin["lon"], -10.5
        )

        pdf_path = self.report_gen.generate_pdf_dossier(
            scenario_data, sar_res,
            {
                "origin_release_point": origin,
                "total_drift_distance_km": 14.8,
                "forecast_warning": {"will_beach": True, "estimated_time_to_beach_hours": 11.2, "beaching_location": {"lat": 22.58, "lon": 69.45}}
            },
            ais_res,
            filename="Test_Verification_Dossier.pdf"
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000, "PDF should be non-empty and well-structured")

    def test_eo_engine_optical_multispectral_processing(self):
        """Validates Sentinel-2 / Optical EO NDOI index calculation, slick segmentation, and lookalike rejection."""
        from src.ocean_shield.eo_engine import EOEngine
        eo_engine = EOEngine(ground_resolution_m=10.0)

        # Create synthetic multi-spectral optical scene with sunglint oil slick
        h, w = 256, 256
        red_band = np.full((h, w), 80, dtype=np.uint8)
        nir_band = np.full((h, w), 75, dtype=np.uint8)  # Clean sea water has lower NIR

        # Add oil patch with positive NIR contrast (refractive index disparity)
        cv2.circle(red_band, (128, 128), 35, 70, -1)
        cv2.circle(nir_band, (128, 128), 35, 120, -1)

        rgb = cv2.merge([red_band, red_band, red_band])
        clean_mask, diagnostics = eo_engine.segment_optical_slick(rgb, nir_band=nir_band)

        self.assertGreater(diagnostics["total_pixels"], 0)
        self.assertGreater(diagnostics["area_km2"], 0.0)
        self.assertIn("mean_ndoi", diagnostics)
        self.assertIsNone(diagnostics["confidence"])
        self.assertEqual(diagnostics["confidence_status"], "NOT_CALIBRATED")
        self.assertNotIn("Bonn", diagnostics["classification"])
        self.assertFalse(diagnostics["has_calibrated_nir"])

    def test_pytorch_unet_deep_learning_pipeline(self):
        """Validates real PyTorch U-Net neural network architecture, weights, and inference."""
        import torch
        from src.ocean_shield.models.unet import SAR_UNet

        # 1. Architecture tensor dimensions
        model = SAR_UNet(n_channels=1, n_classes=1, bilinear=True)
        model.eval()
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 1, 256, 256))

        # 2. Checkpoint availability and SAREngine U-Net inference
        self.assertTrue(self.sar_engine.model_loaded, "Trained U-Net weights should be loaded")
        sar_img, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        mask, prob = self.sar_engine.predict_unet(sar_img)
        self.assertEqual(mask.shape, sar_img.shape)
        self.assertEqual(prob.shape, sar_img.shape)

        # 3. Dual-engine switcher check
        unet_res = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"], model_type="unet"
        )
        self.assertIn("PyTorch U-Net", unet_res["active_engine"])

        cfar_res = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"], model_type="cfar_edge"
        )
        self.assertIn("CFAR", cfar_res["active_engine"])

    def test_radar_ship_detection_and_ais_review_cues(self):
        """Validates CFAR ship extraction and time-aligned radar/AIS review cues."""
        sar_img, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        center_lat = scenario_data["center"]["lat"]
        center_lon = scenario_data["center"]["lon"]

        # 1. Radar ship detection
        radar_ships = self.sar_engine.detect_radar_ship_targets(sar_img, center_lat, center_lon, pixel_size_m=50.0)
        self.assertIsInstance(radar_ships, list)
        self.assertGreater(len(radar_ships), 0, "Should detect at least 1 radar ship target in scene")
        first_ship = radar_ships[0]
        self.assertIn("radar_rcs_mean_db", first_ship)
        self.assertIn("lat", first_ship)
        self.assertIn("lon", first_ship)

        # 2. Radar/AIS review cue. A spatial no-match is not proof of a disabled
        # transponder, even where time-aligned positions exist.
        dark_target = {
            "target_id": "RADAR-TGT-DARK-01",
            "lat": 22.42,
            "lon": 69.15,
            "radar_rcs_mean_db": 22.5,
            "area_pixels": 25
        }
        matched, dark = self.ais_engine.correlate_radar_targets_with_ais(
            [dark_target], scenario_data["ais_vessels"], center_lat, center_lon, match_threshold_nm=0.5
        )
        self.assertEqual(len(dark), 1)
        self.assertEqual(dark[0]["status"], "RADAR_AIS_SPATIAL_MISMATCH_REVIEW")

    def test_generic_weathering_sensitivity(self):
        """Checks generic weathering trends, not ADIOS equivalence or field accuracy."""
        w1 = self.drift_engine.compute_oil_weathering(elapsed_hours=4.0, wind_speed_ms=5.0, water_temp_c=28.0)
        w2 = self.drift_engine.compute_oil_weathering(elapsed_hours=18.0, wind_speed_ms=5.0, water_temp_c=28.0)

        # As time increases, evaporation and water uptake must increase
        self.assertGreater(w2["evaporated_fraction_pct"], w1["evaporated_fraction_pct"])
        self.assertGreater(w2["water_content_mousse_pct"], w1["water_content_mousse_pct"])
        self.assertGreater(w2["dynamic_viscosity_cP"], w1["dynamic_viscosity_cP"])
        self.assertGreater(w2["volume_expansion_factor"], 1.0)
        self.assertIn("Mousse", w2["weathering_classification"])

    def test_tiled_sliding_window_unet_inference(self):
        """Validates that overlapping Hann-window tiled inference scales to large scenes without memory exhaustion."""
        sar_img, _, sc = get_scenario_sar_and_currents("gulf_of_kachchh")
        # Run tiled inference on 512x512 image
        clean_mask, prob_map = self.sar_engine.predict_unet_tiled(sar_img, tile_size=256, stride=192)
        self.assertEqual(clean_mask.shape, sar_img.shape)
        self.assertEqual(prob_map.shape, sar_img.shape)
        self.assertGreater(np.sum(clean_mask > 0), 0, "Tiled inference should detect the oil slick")

    def test_marinecadastre_ais_csv_ingestion(self):
        """Validates time-preserving parsing of standard MarineCadastre CSV exports."""
        import csv
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "marinecadastre_sample_ais.csv")
        self.assertTrue(os.path.exists(csv_path), "MarineCadastre sample CSV must exist")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertGreater(len(reader), 10, "Should have multiple vessel pings")
            mmsi_list = [r["MMSI"] for r in reader]
            self.assertIn("636019482", mmsi_list, "MT NEPTUNE GLORY MMSI should be in sample")

        with open(csv_path, "rb") as f:
            parsed = parse_marinecadastre_csv(f.read(), "marinecadastre_sample_ais.csv")
        self.assertEqual(parsed["provenance"]["valid_pings"], len(reader))
        self.assertEqual(len(parsed["provenance"]["sha256"]), 64)
        first_track = parsed["vessels"][0]["trajectory"]
        self.assertEqual(first_track[-1]["relative_time_hours"], 0.0)
        self.assertLess(first_track[0]["relative_time_hours"], 0.0)

    def test_real_hycom_netcdf_ingestion_and_provenance(self):
        """Validates ingestion of genuine NOAA/Fleet Numerical 4D HYCOM NetCDF archives."""
        real_nc = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "ocean_met", "hycom_real_gulf_kachchh.nc")
        )
        if not os.path.exists(real_nc):
            self.skipTest("Real HYCOM NetCDF not found on disk")

        from src.ocean_shield.ocean_data import OceanDataProvider
        provider = OceanDataProvider(real_nc, center_lat=22.465, center_lon=69.215)
        self.assertTrue(provider.is_loaded)
        self.assertTrue(provider.metadata.get("is_model_or_reanalysis"))
        self.assertFalse(provider.metadata.get("is_synthetic"))
        self.assertIn("Fleet Numerical", provider.metadata.get("institution", ""))

        # Unbound absolute grids cannot be treated as an incident-relative field.
        from src.ocean_shield.ocean_data import DataCoverageError
        with self.assertRaises(DataCoverageError):
            provider.get_velocity_at(22.48, 69.20, t_hours_relative=0.0)

        # Bind an acquisition time within the archived HYCOM coverage. Its wind
        # archive is intentionally from another date, so operational validation
        # must reject the mixed source window.
        with self.assertRaises(DataCoverageError):
            provider.bind_detection_time("2024-01-02T00:00:00Z", 6.0, 6.0)

        # The grid itself retains explicit bounds for source validation.
        self.assertIsNotNone(provider.metadata.get("source_start_utc"))
        self.assertIsNotNone(provider.metadata.get("source_end_utc"))
        self.assertLess(provider.metadata["time_min"], 0.0)
        self.assertGreater(provider.metadata["time_max"], 0.0)

        # No unbounded velocity extraction is permitted.
        self.assertFalse(provider.has_coverage(6.0, 6.0))

    def test_forward_counterfactual_verification(self):
        """Validates Stage 4 Forward Counterfactual Verification (Physical Re-Simulation)."""
        _, _, scenario_data = get_scenario_sar_and_currents("mumbai_high")
        current_field = OceanCurrentField(base_current_u=0.25, base_current_v=0.15,
                                         base_wind_u=4.5, base_wind_v=3.0, constant_vectors=True)
        slick_center = scenario_data["center"]
        obs_lat = slick_center["lat"]
        obs_lon = slick_center["lon"]

        # Run hindcast to get origin
        hindcast = self.drift_engine.run_hindcast(
            obs_lat, obs_lon, current_field,
            max_lookback_hours=12.0, target_slick_age_hours=8.5
        )
        origin = hindcast["origin_release_point"]

        # A manufactured constant-field round trip, not real-origin validation.
        cf_match = self.drift_engine.run_forward_counterfactual(
            release_lat=origin["lat"],
            release_lon=origin["lon"],
            release_time_rel_h=origin["estimated_t0_hours_relative"],
            current_field=current_field,
            observed_slick_lat=obs_lat,
            observed_slick_lon=obs_lon,
            observed_slick_area_km2=18.5,
            vessel_info={"mmsi": 354921000, "vessel_name": "MV ORIENTAL MARINER"}
        )

        self.assertIn("verification_metrics", cf_match)
        metrics = cf_match["verification_metrics"]
        self.assertLess(metrics["centroid_distance_km"], 2.5, "Centroid error should be small for true release point")
        self.assertGreaterEqual(metrics["predicted_containment_percent"], 40.0)
        # The old >0.25 assertion tested a fabricated containment/IoU blend.
        # Circle-proxy IoU is now tested against its actual geometric formula.
        predicted_radius = cf_match["predicted_at_t0"]["spread_radius_km"]
        observed_radius = cf_match["observed_slick"]["effective_radius_km"]
        self.assertLess(predicted_radius + metrics["centroid_distance_km"], observed_radius)
        self.assertAlmostEqual(metrics["jaccard_index"], (predicted_radius / observed_radius) ** 2, delta=0.002)
        self.assertEqual(metrics["jaccard_status"], "CIRCLE_PROXY_IOU_ONLY")
        self.assertIsNone(metrics["physical_causality_score"])
        self.assertEqual(cf_match["verdict"], "CONDITIONAL_SPATIAL_AGREEMENT")
        self.assertIn("forward_trajectory", cf_match)
        self.assertGreater(len(cf_match["forward_trajectory"]), 5)

        # 2. Forward re-simulation from an unrelated distant position (Counterfactual Refutation)
        cf_refute = self.drift_engine.run_forward_counterfactual(
            release_lat=obs_lat + 0.8,
            release_lon=obs_lon - 0.8,
            release_time_rel_h=-8.5,
            current_field=current_field,
            observed_slick_lat=obs_lat,
            observed_slick_lon=obs_lon,
            observed_slick_area_km2=18.5,
            vessel_info={"mmsi": 999999999, "vessel_name": "INNOCENT VESSEL"}
        )
        self.assertEqual(cf_refute["verdict"], "CONDITIONAL_SPATIAL_MISMATCH")
        self.assertGreater(cf_refute["verification_metrics"]["centroid_distance_km"], 10.0)
        self.assertEqual(cf_refute["verification_metrics"]["predicted_containment_percent"], 0.0)

    def test_kde_highest_density_region_contours(self):
        """Validates Gaussian KDE 95%/75%/50% Highest Density Region (HDR) contour extraction."""
        _, _, scenario_data = get_scenario_sar_and_currents("mumbai_high")
        current_field = OceanCurrentField(base_current_u=0.25, base_current_v=0.15,
                                         base_wind_u=4.5, base_wind_v=3.0, constant_vectors=True)
        slick_center = scenario_data["center"]
        obs_lat = slick_center["lat"]
        obs_lon = slick_center["lon"]

        hindcast = self.drift_engine.run_hindcast(
            obs_lat, obs_lon, current_field,
            max_lookback_hours=10.0, target_slick_age_hours=7.0
        )

        self.assertIn("kde_origin_contours", hindcast)
        kde = hindcast["kde_origin_contours"]
        self.assertIn("contours", kde)
        self.assertIn("method", kde)
        self.assertIn("peak_density_lat", kde)
        self.assertIn("peak_density_lon", kde)

        contours = kde["contours"]
        self.assertGreaterEqual(len(contours), 3)

        levels_found = [round(c["level"], 2) for c in contours]
        self.assertIn(0.95, levels_found)
        self.assertIn(0.75, levels_found)
        self.assertIn(0.50, levels_found)

        c95 = next(c for c in contours if round(c["level"], 2) == 0.95)
        c50 = next(c for c in contours if round(c["level"], 2) == 0.50)

        # 95% HDR must enclose a larger area than 50% HDR
        if c95.get("approximate_area_km2") and c50.get("approximate_area_km2"):
            self.assertGreater(c95["approximate_area_km2"], c50["approximate_area_km2"])

        # Check coordinates format
        self.assertTrue(len(c95["polygon_coords"]) >= 3 or len(c95["all_polygons"]) >= 1)
        sample_pt = c95["polygon_coords"][0] if c95["polygon_coords"] else c95["all_polygons"][0][0]
        self.assertEqual(len(sample_pt), 2)  # [lat, lon]

    def test_ais_spoofing_detection(self):
        """Validates Forensic AIS Integrity & Spoofing Detector (gaps, speed jumps, MMSI MID, draught)."""
        # 1. Continuous clean trajectory
        clean_track = [
            {"relative_time_hours": -2.0, "sog_knots": 14.0, "draught_m": 12.0},
            {"relative_time_hours": -1.8, "sog_knots": 13.8, "draught_m": 12.0},
            {"relative_time_hours": -1.6, "sog_knots": 13.5, "draught_m": 12.0},
            {"relative_time_hours": -1.4, "sog_knots": 13.2, "draught_m": 12.0},
        ]
        # Speed-only messages cannot validate positional continuity.
        incomplete = self.ais_engine.detect_ais_spoofing_and_gaps(clean_track, mmsi=419001234, cpa_time_relative_h=-1.6)
        self.assertEqual(incomplete["integrity_rating"], "NOT_ASSESSED")
        for i, point in enumerate(clean_track):
            point.update(lat=22.0 + i * 0.001, lon=69.0)
        audit_clean = self.ais_engine.detect_ais_spoofing_and_gaps(clean_track, mmsi=419001234, cpa_time_relative_h=-1.6)
        self.assertFalse(audit_clean["has_anomalies"])
        self.assertEqual(audit_clean["integrity_rating"], "VERIFIED_CONTINUOUS")
        self.assertTrue(audit_clean["mmsi_valid"])

        # 2. Corrupted trajectory with 45-min transponder blackout near origin
        gap_track = [
            {"relative_time_hours": -3.0, "sog_knots": 14.0},
            {"relative_time_hours": -2.25, "sog_knots": 14.0},  # 45 min gap
            {"relative_time_hours": -2.1, "sog_knots": 14.0}
        ]
        audit_gap = self.ais_engine.detect_ais_spoofing_and_gaps(gap_track, mmsi=419001234, cpa_time_relative_h=-2.5)
        self.assertTrue(audit_gap["has_anomalies"])
        self.assertTrue(audit_gap["corridor_blackout"])
        self.assertGreaterEqual(audit_gap["max_gap_minutes"], 45.0)

        # 3. Physically impossible acceleration jump (> 3 kts/min)
        accel_track = [
            {"relative_time_hours": -1.0, "sog_knots": 6.0},
            {"relative_time_hours": -0.98, "sog_knots": 18.0}  # 12 kts in 1.2 min = 10 kts/min
        ]
        audit_accel = self.ais_engine.detect_ais_spoofing_and_gaps(accel_track, mmsi=419001234)
        self.assertTrue(audit_accel["has_anomalies"])
        self.assertGreater(audit_accel["max_acceleration_kts_min"], 3.0)
        self.assertTrue(any("KINEMATIC_SPEED_JUMP" in a for a in audit_accel["anomalies_detected"]))

        # 4. Unallocated ITU MID validation
        audit_mid = self.ais_engine.detect_ais_spoofing_and_gaps(clean_track, mmsi=999123456)
        self.assertFalse(audit_mid["mmsi_valid"])
        self.assertTrue(any("UNALLOCATED_MID" in a for a in audit_mid["anomalies_detected"]))

    def test_topsis_mcda_ranking(self):
        """Validates TOPSIS Multi-Criteria Decision Analysis & Borda Count ranking."""
        candidates = [
            {
                "mmsi": 419001111,
                "vessel_name": "PRIME SUSPECT TANKER",
                "vessel_type": "Tanker",
                "closest_approach": {"distance_nm": 0.3, "time_diff_h": 0.2},
                "kinematics": {"speed_drop_knots": 6.5, "course_variance_deg": 18.0}
            },
            {
                "mmsi": 419002222,
                "vessel_name": "DISTANT CARGO SHIP",
                "vessel_type": "Cargo",
                "closest_approach": {"distance_nm": 8.5, "time_diff_h": 3.8},
                "kinematics": {"speed_drop_knots": 0.5, "course_variance_deg": 2.0}
            },
            {
                "mmsi": 419003333,
                "vessel_name": "MEDIAN TANKER",
                "vessel_type": "Tanker",
                "closest_approach": {"distance_nm": 2.5, "time_diff_h": 1.2},
                "kinematics": {"speed_drop_knots": 3.0, "course_variance_deg": 5.0}
            }
        ]

        ranked = self.ais_engine.compute_topsis_rankings(candidates)
        self.assertEqual(len(ranked), 3)

        # Candidate with smallest CPA, tightest time coincidence, and large deceleration must win TOPSIS Rank #1
        top_lead = next(v for v in ranked if v["topsis_rank"] == 1)
        self.assertEqual(top_lead["mmsi"], 419001111)
        self.assertGreater(top_lead["topsis_closeness_score"], 60.0)
        self.assertEqual(top_lead["borda_points"], 3)

        # Distant cargo ship must rank lowest
        lowest = next(v for v in ranked if v["topsis_rank"] == 3)
        self.assertEqual(lowest["mmsi"], 419002222)
        self.assertLess(lowest["topsis_closeness_score"], top_lead["topsis_closeness_score"])
        self.assertEqual(lowest["borda_points"], 1)

    def test_cryptographic_merkle_evidence_hash(self):
        """Validates Merkle Tree Root generation and cryptographic immutability."""
        import hashlib
        from src.ocean_shield.report_generator import DossierReportGenerator
        
        leaf_hashes = [
            hashlib.sha256(b"SAR_RASTER_ASSET").hexdigest(),
            hashlib.sha256(b"AIS_TELEMETRY_ASSET").hexdigest(),
            hashlib.sha256(b"HYCOM_METOCEAN_ASSET").hexdigest(),
            hashlib.sha256(b"PYTORCH_UNET_WEIGHTS_ASSET").hexdigest(),
            hashlib.sha256(b"INCIDENT_MANIFEST_JSON").hexdigest()
        ]

        root1 = DossierReportGenerator.compute_merkle_root(leaf_hashes)
        self.assertEqual(len(root1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in root1))

        # Deterministic check
        root2 = DossierReportGenerator.compute_merkle_root(list(leaf_hashes))
        self.assertEqual(root1, root2)

        # Tamper sensitivity check: changing even one byte in one leaf must completely change the Merkle root
        tampered_leaves = list(leaf_hashes)
        tampered_leaves[0] = hashlib.sha256(b"TAMPERED_SAR_RASTER").hexdigest()
        root_tampered = DossierReportGenerator.compute_merkle_root(tampered_leaves)
        self.assertNotEqual(root1, root_tampered)


if __name__ == "__main__":
    unittest.main()
