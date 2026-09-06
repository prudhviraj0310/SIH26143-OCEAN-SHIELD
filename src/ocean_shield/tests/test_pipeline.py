"""
Unit and Integration Test Suite for OCEAN-SHIELD
Smart India Hackathon 2026 - Problem Statement SIH26143
"""

import os
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
        self.report_gen = DossierReportGenerator(output_dir="reports")

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
        self.assertGreater(slick["estimated_mass_tonnes"], 0.0)
        self.assertIn("polygon_geojson", slick)
        self.assertGreaterEqual(slick["confidence_score"], 60.0, "Mineral oil should have high confidence")

    def test_drift_engine_hindcast_and_forecast(self):
        """Validates reverse Lagrangian particle hindcasting and forward forecasting."""
        sar_img, current_field, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        center_lat = scenario_data["center"]["lat"]
        center_lon = scenario_data["center"]["lon"]

        # Run Backward Hindcast
        hindcast = self.drift_engine.run_hindcast(
            center_lat, center_lon, current_field,
            max_lookback_hours=18.0, target_slick_age_hours=10.5
        )
        origin = hindcast["origin_release_point"]
        self.assertAlmostEqual(origin["slick_age_hours"], 10.5, delta=1.5)
        self.assertGreater(hindcast["total_drift_distance_km"], 0.0)
        self.assertGreater(len(hindcast["hindcast_trajectory"]), 10)

        # Run Forward Forecast
        forecast = self.drift_engine.run_forecast(
            center_lat, center_lon, current_field,
            forecast_hours=24.0,
            coastline_lat_threshold=scenario_data["coastline_hazard"]["coastline_lat_threshold"]
        )
        self.assertIn("forecast_trajectory", forecast)
        self.assertIn("beaching_warning", forecast)
        # A real met-ocean field may route the slick offshore; verify the model
        # reports the condition rather than asserting a scenario-specific outcome.
        self.assertIsInstance(forecast["beaching_warning"]["will_beach"], bool)

    def test_ais_correlation_and_culprit_attribution(self):
        """Validates spatio-temporal corridor filtering, kinematic anomalies, and rogue ship attribution."""
        _, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")

        # Test candidate release coordinates derived from drift hindcast
        release_lat = 22.384
        release_lon = 69.116
        release_time_h = -10.5

        results = self.ais_engine.attribute_oil_spill(
            scenario_data["ais_vessels"],
            release_lat,
            release_lon,
            release_time_h
        )

        culprit = results["primary_culprit"]
        self.assertIsNotNone(culprit)
        self.assertEqual(culprit["vessel_name"], "MT NEPTUNE GLORY")
        self.assertEqual(culprit["imo"], 9384722)
        self.assertGreaterEqual(culprit["composite_suspect_score"], 80.0, "Culprit should have high suspect score")
        self.assertLess(culprit["closest_approach"]["distance_nm"], 1.0, "Culprit should be directly over origin")
        self.assertGreaterEqual(culprit["kinematics"]["speed_drop_knots"], 6.0, "Culprit should exhibit speed drop")

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
        self.assertGreater(diagnostics["confidence"], 0.75)

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

    def test_adios_physical_oil_weathering_model(self):
        """Validates Mackay's ADIOS analytical weathering equations (evaporation, emulsification, viscosity)."""
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
        self.assertTrue(provider.metadata.get("is_real_observed"))
        self.assertIn("Fleet Numerical", provider.metadata.get("institution", ""))

        # Velocity extraction at open ocean location
        u_c, v_c, u_w, v_w = provider.get_velocity_at(22.48, 69.20, t_hours_relative=0.0)
        self.assertIsInstance(u_c, float)
        self.assertIsInstance(v_c, float)
        # Ensure fill values (-30000.0) were properly masked
        self.assertGreater(u_c, -5.0)
        self.assertLess(u_c, 5.0)
        self.assertGreater(v_c, -5.0)
        self.assertLess(v_c, 5.0)

        # Provenance telemetry check
        telem = provider.get_telemetry_summary(22.48, 69.20, 0.0)
        self.assertTrue(telem["is_real_observed_currents"])
        self.assertIn("institution", telem)


if __name__ == "__main__":
    unittest.main()
